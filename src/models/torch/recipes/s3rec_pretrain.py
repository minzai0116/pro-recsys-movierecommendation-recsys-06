from __future__ import annotations

"""
S3Rec self-supervised pretraining recipe.

입력:
- cfg.mode: "pretrain" (권장)
- cfg.train.*: epochs/batch_size/lr 등 공통 학습 하이퍼
- cfg.recipe_args.*: pretrain 전용 하이퍼(mask_p, aap/mip/map/sp weight)
- bundle.meta:
  - submission.users: user 순서(정렬)  (seq_topn pipeline이 제공)
  - user_seq: user → item sequence
  - long_sequence: segment negative sampling용
  - item2attributes/attribute_size: AAP task용(선택)

출력:
- fit(): checkpoint(`last.pt`) 생성 (Engine이 저장)
- predict_step(): pretrain 단계에서는 제출 preds를 만들지 않으므로 빈 list를 반환

비고:
- 일반적으로 `mode: pretrain`로 실행합니다(제출/예측 없음).
"""

from typing import Any

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.models.torch.datasets.s3rec_pretrain_dataset import S3RecPretrainDataset
from src.models.torch.modules.s3rec import S3RecArgs, S3RecModel
from src.models.torch.recipes.registry import register_torch_recipe
from src.models.torch.recipes.torch_recipe_base import TorchRecipeBase


def _build_sequences(bundle) -> tuple[list, list[list[int]], int]:
    meta = bundle.meta or {}
    users_order = (meta.get("submission", {}) or {}).get("users") or []
    user_seq = meta.get("user_seq") or {}
    if not users_order or not user_seq:
        raise ValueError("s3rec_pretrain requires bundle.meta['submission']['users'] and meta['user_seq']")

    seqs: list[list[int]] = []
    max_item = 0
    for u in users_order:
        s = user_seq.get(u) or []
        s = [int(x) for x in s]
        seqs.append(s)
        if s:
            max_item = max(max_item, max(s))
    return users_order, seqs, max_item


@register_torch_recipe("s3rec_pretrain")
def build(cfg: Any) -> TorchRecipeBase:
    return S3RecPretrainRecipe(cfg)


class S3RecPretrainRecipe(TorchRecipeBase):
    """S3Rec pretrain 레시피 구현체(Contract는 모듈 docstring 참조)."""

    @staticmethod
    def _recipe_args(cfg: Any) -> Any:
        """
        recipe-specific hyperparams live under cfg.recipe_args (dict/DictConfig).
        """
        try:
            return getattr(cfg, "recipe_args")
        except Exception:
            return {}

    def build_model(self, cfg, bundle):
        meta = bundle.meta or {}
        item2attributes = meta.get("item2attributes") or {}
        attribute_size = meta.get("attribute_size") or 1
        try:
            attribute_size = int(attribute_size)
        except Exception:
            attribute_size = 1

        mcfg = self.model_cfg()
        _, seqs, max_item = _build_sequences(bundle)

        item_size = int(max_item) + 2
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
            attribute_size=max(1, attribute_size),
            item2attribute=item2attributes,
            cuda_condition=False,
        )
        return S3RecModel(args=args)

    def build_optimizer(self, cfg, model):
        tc = self.train_cfg()
        lr = float(getattr(tc, "lr", 1e-3))
        weight_decay = float(getattr(tc, "weight_decay", 0.0) or 0.0)
        beta1 = float(getattr(tc, "adam_beta1", 0.9))
        beta2 = float(getattr(tc, "adam_beta2", 0.999))
        return Adam(model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)

    def build_loaders(self, cfg, bundle):
        tc = self.train_cfg()
        _, seqs, max_item = _build_sequences(bundle)

        meta = bundle.meta or {}
        long_sequence = meta.get("long_sequence") or []
        item2attributes = meta.get("item2attributes") or {}
        attribute_size = meta.get("attribute_size") or 1
        try:
            attribute_size = int(attribute_size)
        except Exception:
            attribute_size = 1

        max_len = int(self.model_cfg().get("max_seq_length", 50))
        item_size = int(max_item) + 2
        mask_id = int(max_item) + 1

        bs = int(getattr(tc, "batch_size", 512))
        nw = int(getattr(tc, "num_workers", 0) or 0)
        ra = self._recipe_args(cfg)
        try:
            mask_p = float(getattr(ra, "mask_p"))
        except Exception:
            try:
                mask_p = float(ra.get("mask_p", 0.2))
            except Exception:
                mask_p = 0.2

        ds = S3RecPretrainDataset(
            user_seqs=seqs,
            long_sequence=long_sequence,
            max_len=max_len,
            item_size=item_size,
            mask_id=mask_id,
            attribute_size=max(1, attribute_size),
            item2attribute=item2attributes,
            mask_p=mask_p,
        )

        return {
            "train": DataLoader(ds, batch_size=bs, shuffle=True, num_workers=nw, drop_last=False),
            # NOTE: mode=pretrain에서는 predict가 호출되지 않지만, 방어적으로 dummy loader 제공
            "test": DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, drop_last=False),
        }

    def train_step(self, cfg, batch, model: S3RecModel):
        (
            attributes,
            masked_item_sequence,
            pos_items,
            neg_items,
            masked_segment_sequence,
            pos_segment,
            neg_segment,
        ) = batch

        aap_loss, mip_loss, map_loss, sp_loss = model.pretrain(
            attributes,
            masked_item_sequence,
            pos_items,
            neg_items,
            masked_segment_sequence,
            pos_segment,
            neg_segment,
        )

        ra = self._recipe_args(cfg)
        def _getf(key: str, default: float) -> float:
            try:
                return float(getattr(ra, key))
            except Exception:
                try:
                    return float(ra.get(key, default))
                except Exception:
                    return default

        aap_w = _getf("aap_weight", 0.2)
        mip_w = _getf("mip_weight", 1.0)
        map_w = _getf("map_weight", 1.0)
        sp_w = _getf("sp_weight", 0.5)

        joint_loss = aap_w * aap_loss + mip_w * mip_loss + map_w * map_loss + sp_w * sp_loss
        return {"loss": joint_loss}

    def predict_step(self, cfg, batch, model):
        # pretrain stage doesn't produce submission preds; return empty placeholders
        return []


