import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from scipy import sparse
import pandas as pd

from .myBERT import PredictNextItemDataset

def split_user_level_last2(user_seqs):
    """
    Returns:
      train_seqs: 유저별 학습용 prefix 시퀀스 리스트 (마지막 2개 제외)
      valid_targets: (prefix, target_item) 리스트 (target=마지막 2번째)
      test_targets:  (prefix, target_item) 리스트 (target=마지막)
    """
    train_seqs = []
    valid_targets = []
    test_targets = []

    for seq in user_seqs:
        if len(seq) < 3:
            # valid/test 만들기 어려우면 스킵(또는 별도 처리)
            continue
        train_prefix = seq[:-2]
        valid_prefix = seq[:-2]   # 마지막2 제외 prefix를 보고
        valid_t = seq[-2]         # 마지막 2번째를 맞춘다
        test_prefix  = seq[:-1]   # 마지막1 제외 prefix를 보고
        test_t  = seq[-1]         # 마지막을 맞춘다

        train_seqs.append(train_prefix)
        valid_targets.append((valid_prefix, valid_t))
        test_targets.append((test_prefix, test_t))

    return train_seqs, valid_targets, test_targets

def build_user_seqs(df: pd.DataFrame):
    # time 기준 정렬 (없으면 user, item 순으로라도)
    if "time" in df.columns:
        df2 = df.sort_values(["user", "time"])
    else:
        df2 = df.sort_values(["user"])

    user_seqs = df2.groupby("user")["item"].apply(list).to_dict()
    user_order = df["user"].unique()  # 요청: 이 순서를 그대로 유지
    return user_seqs, user_order

@torch.no_grad()
def iter_user_logits(
    model,
    df: pd.DataFrame,
    max_len: int,
    item_num: int,
    batch_size: int = 256,
    device: str = "cuda",
    pad_id: int = 0,
    mask_id: int | None = None,
    filter_seen: bool = True,
):
    """
    Yields:
      users: np.ndarray shape [B]  (원래 user id)
      logits: np.ndarray shape [B, V]  (mask 위치의 전체 logits)
    """
    model.eval()
    model.to(device)

    mask_id = mask_id if mask_id is not None else item_num + 1

    user_seqs, user_order = build_user_seqs(df)

    ds = PredictNextItemDataset(
        user_seqs_by_userid=user_seqs,
        user_order=user_order,
        max_len=max_len,
        mask_id=mask_id,
        pad_id=pad_id,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    for batch in tqdm(loader):
        users = batch["user"]                          # list/array of user ids
        input_ids = batch["input_ids"].to(device)      # [B,L]
        attn = batch["attention_mask"].to(device)      # [B,L]
        mask_pos = batch["mask_pos"].to(device)        # [B]

        logits = model(input_ids, attention_mask=attn)  # [B,L,V]

        # mask position logits만: [B,V]
        b_idx = torch.arange(logits.size(0), device=device)
        mask_logits = logits[b_idx, mask_pos, :]        # [B,V]

        # PAD/MASK 제외
        mask_logits[:, pad_id] = -1e9
        if 0 <= mask_id < mask_logits.size(-1):
            mask_logits[:, mask_id] = -1e9

        # 이미 본 아이템 제외
        if filter_seen:
            mask_logits = mask_logits.clone()
            for i, u in enumerate(users):
                seen = user_seqs.get(int(u), [])
                if len(seen) > 0:
                    seen_t = torch.tensor(seen, device=device, dtype=torch.long)
                    seen_t = seen_t[(seen_t >= 0) & (seen_t < mask_logits.size(-1))]
                    mask_logits[i, seen_t] = -1e9

        yield (
            torch.as_tensor(users).cpu().numpy(),
            mask_logits.detach().float().cpu().numpy()
        )

@torch.no_grad()
def get_user_logits_dict(
    model,
    df: pd.DataFrame,
    max_len: int,
    item_num: int,
    batch_size: int = 256,
    device: str = "cuda",
    pad_id: int = 0,
    mask_id: int | None = None,
    filter_seen: bool = True,
):
    logits_by_user = {}

    for users, logits in iter_user_logits(
        model=model,
        df=df,
        max_len=max_len,
        item_num=item_num,
        batch_size=batch_size,
        device=device,
        pad_id=pad_id,
        mask_id=mask_id,
        filter_seen=filter_seen,
    ):
        for u, vec in zip(users, logits):
            logits_by_user[int(u)] = vec  # vec: np.ndarray [V]

    return logits_by_user

import torch

def verify_state_dict_equal(model, ckpt_state, atol=0.0, rtol=0.0):
    """
    model: load_state_dict 이후의 model
    ckpt_state: ckpt["model_state"]
    atol, rtol: 허용 오차 (float32면 0도 가능)
    """
    model_state = model.state_dict()

    # 1) key 집합 비교
    model_keys = set(model_state.keys())
    ckpt_keys = set(ckpt_state.keys())

    if model_keys != ckpt_keys:
        print("❌ State dict keys mismatch")
        print("Only in model:", model_keys - ckpt_keys)
        print("Only in ckpt:", ckpt_keys - model_keys)
        return False

    # 2) tensor 값 비교
    for k in model_keys:
        t1 = model_state[k].cpu()
        t2 = ckpt_state[k].cpu()

        if t1.shape != t2.shape:
            print(f"❌ Shape mismatch at {k}: {t1.shape} vs {t2.shape}")
            return False

        if not torch.allclose(t1, t2, atol=atol, rtol=rtol):
            max_diff = (t1 - t2).abs().max().item()
            print(f"❌ Value mismatch at {k}, max diff = {max_diff}")
            return False

    print("✅ Model weights are IDENTICAL to checkpoint")
    return True

@torch.no_grad()
def ensemble_topN_topM_make_submission(
    bert_model,
    ease_model,
    df: pd.DataFrame,
    user_item_matrix: sparse.csr_matrix,
    user_id_to_idx: dict,
    max_len: int,
    item_num: int,
    k: int = 10,
    topN_ease: int = 7,          # EASE에서 뽑을 개수
    topM_bert: int = 7,          # BERT에서 뽑을 개수
    batch_size: int = 256,
    device: str = "cuda",
    pad_id: int = 0,
    mask_id: int | None = None,
    filter_seen: bool = True,
    bert_temp: float = 1.0,
    ease_temp: float = 1.0,
    out_csv_path: str | None = None,
):
    """
    Returns:
      sub_df: columns ["user","item"] (각 유저당 k행)
    """
    bert_model.eval().to(device)
    mask_id = mask_id if mask_id is not None else item_num + 1

    # 1) user seq/order
    user_seqs, user_order = build_user_seqs(df)

    ds = PredictNextItemDataset(
        user_seqs_by_userid=user_seqs,
        user_order=user_order,
        max_len=max_len,
        mask_id=mask_id,
        pad_id=pad_id,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # 2) EASE item 축 -> BERT item_id(1..item_num) 축으로 매핑
    ease_col_for_itemid = np.full(item_num + 1, -1, dtype=np.int32)  # 0 unused
    for item_id in range(1, item_num + 1):
        if item_id in ease_model.item_id_to_idx:
            ease_col_for_itemid[item_id] = ease_model.item_id_to_idx[item_id]

    valid_mask = ease_col_for_itemid[1:] >= 0
    valid_cols = ease_col_for_itemid[1:][valid_mask]

    rows = []

    for _, batch in tqdm(enumerate(loader), total=len(loader)):
        users = batch["user"]
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        mask_pos = batch["mask_pos"].to(device)
        B = input_ids.size(0)

        # -------------------------
        # (A) BERT: mask logits -> prob (item 1..item_num)
        # -------------------------
        logits = bert_model(input_ids, attention_mask=attn)   # [B,L,V]
        b_idx = torch.arange(B, device=device)
        mask_logits = logits[b_idx, mask_pos, :]              # [B,V]

        mask_logits[:, pad_id] = -1e9
        if 0 <= mask_id < mask_logits.size(-1):
            mask_logits[:, mask_id] = -1e9

        if filter_seen:
            mask_logits = mask_logits.clone()
            for i, u in enumerate(users):
                seen = user_seqs.get(int(u), [])
                if len(seen) > 0:
                    seen_t = torch.tensor(seen, device=device, dtype=torch.long)
                    seen_t = seen_t[(seen_t >= 0) & (seen_t < mask_logits.size(-1))]
                    mask_logits[i, seen_t] = -1e9

        bert_item_logits = mask_logits[:, 1:item_num+1]                       # [B,item_num]
        bert_prob = torch.softmax(bert_item_logits / bert_temp, dim=-1)       # [B,item_num]

        # 후보는 K보다 넉넉히 뽑아두는 게 안전 (중복/부족 대비)
        bert_need = max(k, topM_bert) + 50
        bert_top_idx = torch.topk(bert_prob, k=min(bert_need, item_num), dim=-1).indices  # [B, ?]
        bert_top_items = (bert_top_idx + 1).detach().cpu().numpy()            # item_id로

        # -------------------------
        # (B) EASE: scores -> prob (item 1..item_num)
        # -------------------------
        user_idx = [user_id_to_idx[int(u)] for u in users]
        Xb = user_item_matrix[user_idx]
        ease_scores = Xb @ ease_model.B

        if sparse.issparse(ease_scores):
            ease_scores = ease_scores.toarray()
        else:
            ease_scores = np.asarray(ease_scores)

        if filter_seen:
            rr, cc = Xb.nonzero()
            ease_scores[rr, cc] = -np.inf

        ease_item_scores = np.full((B, item_num), -np.inf, dtype=np.float32)
        ease_item_scores[:, valid_mask] = ease_scores[:, valid_cols]

        ease_item_scores_t = torch.from_numpy(ease_item_scores).to(device)
        ease_item_scores_t = torch.clamp(ease_item_scores_t, min=-1e9)
        ease_prob = torch.softmax(ease_item_scores_t / ease_temp, dim=-1)

        ease_need = max(k, topN_ease) + 50
        ease_top_idx = torch.topk(ease_prob, k=min(ease_need, item_num), dim=-1).indices
        ease_top_items = (ease_top_idx + 1).detach().cpu().numpy()

        # -------------------------
        # (C) Merge: EASE Top-N + BERT Top-M => total K (중복 제거)
        # -------------------------
        for i, u in enumerate(users):
            u = int(u)
            picked = []
            picked_set = set()

            # 1) EASE Top-N
            for it in ease_top_items[i]:
                it = int(it)
                if it not in picked_set:
                    picked.append(it)
                    picked_set.add(it)
                if len(picked) >= min(topN_ease, k):
                    break

            # 2) BERT Top-M로 채우기
            target_after_ease = min(topN_ease + topM_bert, k)
            for it in bert_top_items[i]:
                it = int(it)
                if it not in picked_set:
                    picked.append(it)
                    picked_set.add(it)
                if len(picked) >= target_after_ease:
                    break

            # 3) 그래도 K가 안 차면(중복이 너무 많거나) 남은 후보로 채우기
            if len(picked) < k:
                for it in ease_top_items[i]:
                    it = int(it)
                    if it not in picked_set:
                        picked.append(it)
                        picked_set.add(it)
                    if len(picked) >= k:
                        break

            if len(picked) < k:
                for it in bert_top_items[i]:
                    it = int(it)
                    if it not in picked_set:
                        picked.append(it)
                        picked_set.add(it)
                    if len(picked) >= k:
                        break

            # 최종 rows
            for it in picked[:k]:
                rows.append((u, it))

    sub_df = pd.DataFrame(rows, columns=["user", "item"])
    if out_csv_path is not None:
        sub_df.to_csv(out_csv_path, index=False)

    return sub_df