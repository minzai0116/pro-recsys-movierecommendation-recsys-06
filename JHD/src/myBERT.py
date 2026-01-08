import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset

class Bert4RecModel(nn.Module):
    """
    Minimal BERT-style encoder for masked item prediction.
    - Item embedding + positional embedding
    - TransformerEncoder (PyTorch)
    - Output projection to vocab (items + special tokens)
    """
    def __init__(
        self,
        item_num: int,
        max_len: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        pad_id: int = 0,
        mask_id: int | None = None,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.item_num = item_num
        self.max_len = max_len
        self.pad_id = pad_id
        self.mask_id = mask_id if mask_id is not None else item_num + 1

        # vocab: 0=PAD, 1..item_num=items, item_num+1=MASK
        self.vocab_size = item_num + 2

        self.item_emb = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=hidden_size,
            padding_idx=pad_id,
        )
        self.pos_emb = nn.Embedding(
            num_embeddings=max_len,
            embedding_dim=hidden_size,
        )

        self.emb_dropout = nn.Dropout(dropout)
        self.emb_ln = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        # TransformerEncoder expects [L, B, H]
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=False,   # keep default to be explicit
            norm_first=False,    # post-norm (BERT 원형은 post-norm)
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # output head
        self.out_ln = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_proj = nn.Linear(hidden_size, self.vocab_size, bias=False)

        # weight tying (선택이지만 보통 성능/안정에 도움)
        self.out_proj.weight = self.item_emb.weight

        self._reset_parameters()

    def _reset_parameters(self):
        # BERT 초기화 느낌의 간단한 초기화
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

        # padding row는 0으로
        with torch.no_grad():
            self.item_emb.weight[self.pad_id].fill_(0)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        """
        input_ids: [B, L]
        attention_mask: [B, L] where 1 means keep, 0 means PAD
        return logits: [B, L, vocab_size]
        """
        B, L = input_ids.shape
        if L > self.max_len:
            raise ValueError(f"Sequence length L={L} exceeds max_len={self.max_len}")

        # positions: [L] (0..L-1)
        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)

        x = self.item_emb(input_ids) + self.pos_emb(pos_ids)  # [B, L, H]
        x = self.emb_ln(x)
        x = self.emb_dropout(x)

        # TransformerEncoder: [L, B, H]
        x = x.transpose(0, 1)

        # key_padding_mask: [B, L] with True at PAD positions
        if attention_mask is None:
            key_padding_mask = (input_ids == self.pad_id)
        else:
            key_padding_mask = (attention_mask == 0)

        # Encode
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)  # [L, B, H]
        h = h.transpose(0, 1)  # [B, L, H]

        h = self.out_ln(h)
        logits = self.out_proj(h)  # [B, L, V]
        return logits

    @torch.no_grad()
    def predict_mask(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        """
        Convenience: return logits for mask positions only (for debugging/inference).
        """
        logits = self.forward(input_ids, attention_mask=attention_mask)  # [B, L, V]
        mask_pos = (input_ids == self.mask_id)  # [B, L]
        return logits[mask_pos]  # [num_masked, V]


class TrainMIPDataset(Dataset):
    def __init__(
        self,
        train_seqs,          # List[List[int]]  (유저 prefix)
        max_len: int,
        item_num: int,
        mask_prob: float = 0.15,
        pad_id: int = 0,
        mask_id: int | None = None,
        ignore_index: int = -100,
        seed: int = 42,
    ):
        self.train_seqs = train_seqs
        self.max_len = max_len
        self.item_num = item_num
        self.pad_id = pad_id
        self.mask_id = mask_id if mask_id is not None else item_num + 1
        self.ignore_index = ignore_index
        self.mask_prob = mask_prob
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.train_seqs)

    def _mask_tokens(self, input_ids):
        labels = [self.ignore_index] * len(input_ids)

        cand_pos = [i for i, x in enumerate(input_ids) if x != self.pad_id]
        if len(cand_pos) == 0:
            return input_ids, labels

        n_mask = max(1, int(len(cand_pos) * self.mask_prob))
        mask_pos = self.rng.sample(cand_pos, k=min(n_mask, len(cand_pos)))

        for pos in mask_pos:
            original = input_ids[pos]
            
            labels[pos] = original
            input_ids[pos] = self.mask_id

        return input_ids, labels

    def __getitem__(self, idx):
        seq = self.train_seqs[idx]
                
        input_ids = left_pad_truncate(seq[:], self.max_len, self.pad_id)

        input_ids, labels = self._mask_tokens(input_ids)
        
        attention_mask = [1 if x != self.pad_id else 0 for x in input_ids]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        
class EvalLastMaskDataset(Dataset):
    """
    targets: List[Tuple[prefix_seq, target_item]]
    prefix_seq는 target_item 직전까지의 시퀀스.
    """
    def __init__(
        self,
        targets,
        max_len: int,
        item_num: int,
        pad_id: int = 0,
        mask_id: int | None = None,
        ignore_index: int = -100,
    ):
        self.targets = targets
        self.max_len = max_len
        self.item_num = item_num
        self.pad_id = pad_id
        self.mask_id = mask_id if mask_id is not None else item_num + 1
        self.ignore_index = ignore_index

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        prefix, target = self.targets[idx]

        # prefix의 끝에 mask를 붙여서 "다음 아이템" 예측 형태로 만든다
        seq = prefix[:] + [self.mask_id]
        input_ids = left_pad_truncate(seq, self.max_len, self.pad_id)

        # labels: mask 위치에만 target
        labels = [self.ignore_index] * len(input_ids)
        # mask 위치는 항상 마지막 non-pad 토큰 위치
        # (left pad 했으니, 마지막 위치가 mask라는 보장은 X → 실제 mask 위치를 찾자)
        mask_pos = None
        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i] == self.mask_id:
                mask_pos = i
                break
        if mask_pos is None:
            # 이 경우는 사실상 발생하면 안 됨
            mask_pos = len(input_ids) - 1

        labels[mask_pos] = int(target)

        attention_mask = [1 if x != self.pad_id else 0 for x in input_ids]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        

class PredictNextItemDataset(Dataset):
    def __init__(self, user_seqs_by_userid, user_order, max_len, mask_id, pad_id=0):
        """
        user_seqs_by_userid: dict[user] -> List[item_id]  (시간순)
        user_order: df["user"].unique() 같은 순서의 리스트/array
        """
        self.user_seqs = user_seqs_by_userid
        self.user_order = list(user_order)
        self.max_len = max_len
        self.mask_id = mask_id
        self.pad_id = pad_id

    def __len__(self):
        return len(self.user_order)

    def __getitem__(self, idx):
        user = self.user_order[idx]
        seq = self.user_seqs.get(user, [])

        # prefix + [MASK] (다음 아이템 예측)
        input_ids = (seq + [self.mask_id])[-self.max_len:]
        pad_len = self.max_len - len(input_ids)
        if pad_len > 0:
            input_ids = [self.pad_id] * pad_len + input_ids

        attention_mask = [1 if x != self.pad_id else 0 for x in input_ids]
        # mask position은 항상 마지막 non-pad == mask_id 위치
        mask_pos = len(input_ids) - 1
        # (혹시 패딩 때문에 mask가 마지막이 아닐까 걱정되면 아래로 찾을 수도 있음)
        # mask_pos = max(i for i,x in enumerate(input_ids) if x == self.mask_id)

        return {
            "user": user,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "mask_pos": torch.tensor(mask_pos, dtype=torch.long),
        }

def left_pad_truncate(seq, max_len, pad_id=0):
    seq = seq[-max_len:]
    pad_len = max_len - len(seq)
    if pad_len > 0:
        seq = [pad_id] * pad_len + seq
    return seq


class MIPTrainer:
    def __init__(
        self,
        model,
        train_loader,
        valid_loader,
        device,
        lr=1e-3,
        weight_decay=0.01,
        max_grad_norm=1.0,
        use_amp=True,
        log_every=100,
        save_dir="checkpoints",
        metric_key="recall@10",   # best 기준
        pad_id=0,
        mask_id=None,
        scheduler_type="cosine",  # "cosine" or "linear" or None
        warmup_ratio=0.1,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp
        self.scaler = GradScaler(enabled=use_amp)
        self.log_every = log_every

        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.metric_key = metric_key

        self.pad_id = pad_id
        self.mask_id = mask_id

        self.scheduler = self._build_scheduler(
            scheduler_type=scheduler_type,
            warmup_ratio=warmup_ratio,
        )

        self.global_step = 0
        self.best_metric = -float("inf")

    def _build_scheduler(self, scheduler_type="cosine", warmup_ratio=0.1):
        if scheduler_type is None:
            return None

        total_steps = len(self.train_loader)  # 1 epoch 기준 (epoch마다 새로 만들지 않고 epoch 수 반영하려면 바꾸면 됨)
        warmup_steps = int(total_steps * warmup_ratio)

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
            if scheduler_type == "linear":
                return max(0.0, 1.0 - progress)
            if scheduler_type == "cosine":
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            return 1.0

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _save(self, name="best.pt", extra=None):
        path = os.path.join(self.save_dir, name)
        payload = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "global_step": self.global_step,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        return path

    def train_one_epoch(self, epoch_idx: int):
        self.model.train()
        running_loss = 0.0

        print("Training epoch :", epoch_idx)
        for it, batch in tqdm(enumerate(self.train_loader, start=1)):
            self.global_step += 1
            
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                logits = self.model(input_ids, attention_mask=attention_mask)  # [B,L,V]
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                )

            self.scaler.scale(loss).backward()

            if self.max_grad_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler is not None:
                self.scheduler.step()

            running_loss += loss.item()

            if (it % self.log_every) == 0:
                avg = running_loss / self.log_every
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"[epoch {epoch_idx} | step {it}/{len(self.train_loader)}] loss={avg:.4f} lr={lr:.2e}")
                running_loss = 0.0

    def validate(self, ks=(5, 10, 20)):
        metrics = evaluate_mip(
            self.model,
            self.valid_loader,
            device=self.device,
            ks=ks,
            pad_id=self.pad_id,
            mask_id=self.mask_id,
        )
        return metrics

    def fit(self, epochs: int, ks=(5, 10, 20)):
        for epoch in range(1, epochs + 1):
            
            self.train_one_epoch(epoch)

            metrics = self.validate(ks=ks)
            metric_value = metrics.get(self.metric_key, None)
            print(f"[epoch {epoch}] " + " | ".join([f"{k}={v:.4f}" for k, v in metrics.items()]))

            if metric_value is not None and metric_value > self.best_metric:
                self.best_metric = metric_value
                path = self._save("best.pt", extra={"metrics": metrics, "epoch": epoch})
                print(f"  ↳ best updated: {self.metric_key}={self.best_metric:.4f} saved to {path}")

        # 마지막 상태 저장(선택)
        self._save("last.pt")