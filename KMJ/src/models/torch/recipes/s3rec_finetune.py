from __future__ import annotations

"""
S3Rec/SASRec finetuning recipe (MovieLens seq_topn).

입력:
- cfg.train.*: epochs/batch_size/lr/topk 등
- (선택) cfg.train.pretrained_checkpoint: pretrain run의 last.pt 경로
- bundle.meta:
  - submission.users: 제출 대상 user 순서
  - user_seq: user → item sequence (time 정렬)

출력:
- train_step(): {"loss": torch.Tensor}
- predict_step(): List[List[int]] (batch 단위 topK item ids)
  * 전체 predict 결과는 submission.users 순서와 정렬되어야 함

주의:
- seen-item masking은 `bundle.meta["user_seq"]`의 “전체 히스토리” 기준으로 수행합니다.
"""

from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from src.models.torch.recipes.registry import register_torch_recipe
from src.models.torch.recipes.torch_recipe_base import TorchRecipeBase
from src.models.torch.datasets.sasrec_dataset import SASRecDataset
from src.models.torch.modules.s3rec import S3RecArgs, S3RecModel


def _build_sequences(bundle) -> tuple[list, list[list[int]], int]:
    """
    Returns:
      users_order: list of user tokens (as in submission order)
      seqs: list of item-id sequences aligned with users_order
      max_item: maximum item id observed
    """
    meta = bundle.meta or {}
    users_order = (meta.get("submission", {}) or {}).get("users") or []
    user_seq = meta.get("user_seq") or {}
    if not users_order or not user_seq:
        raise ValueError("seq_topn torch recipe requires bundle.meta['submission']['users'] and meta['user_seq']")

    seqs: list[list[int]] = []
    max_item = 0
    for u in users_order:
        s = user_seq.get(u) or []
        s = [int(x) for x in s]
        seqs.append(s)
        if s:
            max_item = max(max_item, max(s))
    return users_order, seqs, max_item


@register_torch_recipe("s3rec_finetune")
def build(cfg: Any) -> TorchRecipeBase:
    return S3RecFinetuneRecipe(cfg)


class S3RecFinetuneRecipe(TorchRecipeBase):
    """S3Rec finetune 레시피 구현체(Contract는 모듈 docstring 참조)."""

    def __init__(self, cfg: Any):
        super().__init__(cfg)
        # masking strategy for seen items during prediction
        # - "full": mask full history from pipeline meta (submission-friendly)
        # - "input": mask only items present in input_ids (validation-friendly; won't hide GT)
        self._mask_seen_strategy: str = "full"
        self._seen_items_by_index_full: list[list[int]] | None = None

    def set_mask_seen_strategy(self, strategy: str) -> None:
        strategy = str(strategy)
        if strategy not in {"full", "input"}:
            raise ValueError(f"Unknown mask strategy: {strategy}")
        self._mask_seen_strategy = strategy

    def build_model(self, cfg, bundle):
        mcfg = self.model_cfg()
        _, seqs, max_item = _build_sequences(bundle)

        item_size = int(max_item) + 2  # + padding(0) + mask_id
        mask_id = int(max_item) + 1

        args = S3RecArgs(
            hidden_size=int(mcfg.get("hidden_size", 64)),
            num_hidden_layers=int(mcfg.get("num_hidden_layers", 2)),
            num_attention_heads=int(mcfg.get("num_attention_heads", 2)),
            hidden_act=str(mcfg.get("hidden_act", "gelu")),
            attention_probs_dropout_prob=float(mcfg.get("attention_probs_dropout_prob", 0.5)),
            hidden_dropout_prob=float(mcfg.get("hidden_dropout_prob", 0.5)),
            initializer_range=float(mcfg.get("initializer_range", 0.02)),
            max_seq_length=int(mcfg.get("max_seq_length", 50)),
            item_size=item_size,
            mask_id=mask_id,
            attribute_size=int((bundle.meta or {}).get("attribute_size") or 1),
            item2attribute=(bundle.meta or {}).get("item2attributes") or {},
            cuda_condition=False,
        )
        model = S3RecModel(args=args)

        # optional: load pretrained checkpoint
        ckpt = None
        try:
            ckpt = cfg.get("train", {}).get("pretrained_checkpoint", None)
        except Exception:
            ckpt = None
        if ckpt:
            obj = torch.load(str(ckpt), map_location="cpu")
            state = obj.get("model") if isinstance(obj, dict) and "model" in obj else obj
            model.load_state_dict(state, strict=False)
        return model

    def build_optimizer(self, cfg, model):
        tc = self.train_cfg()
        lr = float(getattr(tc, "lr", 1e-3))
        weight_decay = float(getattr(tc, "weight_decay", 0.0) or 0.0)
        beta1 = float(getattr(tc, "adam_beta1", 0.9))
        beta2 = float(getattr(tc, "adam_beta2", 0.999))
        return Adam(model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)

    def build_loaders(self, cfg, bundle):
        tc = self.train_cfg()
        users_order, seqs, max_item = _build_sequences(bundle)

        max_len = int(self.model_cfg().get("max_seq_length", 50))
        item_size = int(max_item) + 2

        bs = int(getattr(tc, "batch_size", 256))
        nw = int(getattr(tc, "num_workers", 0) or 0)

        train_ds = SASRecDataset(user_seqs=seqs, max_len=max_len, item_size=item_size, data_type="train")
        valid_ds = SASRecDataset(user_seqs=seqs, max_len=max_len, item_size=item_size, data_type="valid")
        sub_ds = SASRecDataset(user_seqs=seqs, max_len=max_len, item_size=item_size, data_type="submission")

        # for submission-time masking: keep full seen-items per user index
        # index in dataset == position in `users_order`
        self._seen_items_by_index_full = seqs

        return {
            "train": DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, drop_last=False),
            "valid": DataLoader(valid_ds, batch_size=bs, shuffle=False, num_workers=nw, drop_last=False),
            "test": DataLoader(sub_ds, batch_size=bs, shuffle=False, num_workers=nw, drop_last=False),
        }

    def train_step(self, cfg, batch, model):
        _, input_ids, target_pos, target_neg, _ = batch

        sequence_output = model.finetune(input_ids)  # [B, L, H]
        loss = self._cross_entropy(model, sequence_output, target_pos, target_neg)
        return {"loss": loss}

    def predict_step(self, cfg, batch, model):
        # returns List[List[int]] for this batch
        user_ids, input_ids, _, _, _ = batch
        sequence_output = model.finetune(input_ids)
        recommend_output = sequence_output[:, -1, :]  # [B, H]

        # full sort scores: [B, item_size]
        item_emb = model.item_embeddings.weight  # [item_size, H]
        rating_pred = torch.matmul(recommend_output, item_emb.transpose(0, 1))

        # mask padding + mask_id
        rating_pred[:, 0] = -1e9
        rating_pred[:, model.args.mask_id] = -1e9

        # mask seen items
        # - submission: prefer full history masking
        # - validation: mask only input history so GT item isn't hidden
        if getattr(self, "_mask_seen_strategy", "full") == "full":
            seen_bank = self._seen_items_by_index_full or []
            for i in range(input_ids.size(0)):
                uid = int(user_ids[i].detach().cpu().item())
                if 0 <= uid < len(seen_bank):
                    seen_list = seen_bank[uid] or []
                    for it in seen_list:
                        it = int(it)
                        if 0 < it < rating_pred.size(1):
                            rating_pred[i, it] = -1e9
        else:
            # mask from input_ids only (non-zero)
            for i in range(input_ids.size(0)):
                seen = input_ids[i].detach().cpu().tolist()
                for it in seen:
                    it = int(it)
                    if 0 < it < rating_pred.size(1):
                        rating_pred[i, it] = -1e9

        # topK
        k = 10
        try:
            k = int(cfg.get("train", {}).get("topk", 10))
        except Exception:
            pass

        topk = torch.topk(rating_pred, k=k, dim=1).indices  # [B, K]
        return topk.detach().cpu().tolist()

    @staticmethod
    def _cross_entropy(model: S3RecModel, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        pos_emb = model.item_embeddings(pos_ids)
        neg_emb = model.item_embeddings(neg_ids)

        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, model.args.hidden_size)

        pos_logits = torch.sum(pos * seq_emb, -1)
        neg_logits = torch.sum(neg * seq_emb, -1)

        istarget = (pos_ids > 0).view(pos_ids.size(0) * model.args.max_seq_length).float()
        denom = torch.sum(istarget).clamp(min=1.0)
        loss = torch.sum(
            -torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget
            - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / denom

        return loss


