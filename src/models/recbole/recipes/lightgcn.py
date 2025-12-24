from __future__ import annotations

from typing import Any, Dict

from .base import RecBoleRecipeBase
from .registry import register_recbole_recipe
from src.models.recbole.adapter.fieldmap import build_fieldmap, recbole_field_overrides
from src.models.recbole.adapter.atomic_export import export_inter


@register_recbole_recipe("LightGCN")
def build(cfg: Any) -> RecBoleRecipeBase:
    return LightGCNRecipe(cfg)


class LightGCNRecipe(RecBoleRecipeBase):
    name = "LightGCN"

    def prepare_dataset(self, bundle, *, data_root: str, dataset: str, setting):
        # .inter만 필요 (MVP)
        setting.ensure_dir(data_root)
        setting.ensure_dir(f"{data_root}/{dataset}")

        fm = build_fieldmap(bundle.schema or {})
        return export_inter(bundle=bundle, out_root=data_root, dataset=dataset, fm=fm)

    def build_overrides(self, bundle, *, data_root: str, dataset: str) -> Dict[str, Any]:
        fm = build_fieldmap(bundle.schema or {})

        # 학습 하이퍼 (strict: cfg.train only)
        tcfg = self.cfg.train

        # model_args prune 전/후 모두 견딤
        mcfg = getattr(self.cfg, "model_args", {}) or {}
        if isinstance(mcfg, dict) and self.cfg.model in mcfg:
            mcfg = mcfg[self.cfg.model]

        overrides: Dict[str, Any] = {
            "data_path": str(data_root),
            "dataset": str(dataset),
            "model": str(self.cfg.model),

            "field_separator": "\t",
            **recbole_field_overrides(fm),

            "seed": int(getattr(tcfg, "seed", getattr(self.cfg, "seed", 42))),
            "epochs": int(tcfg.epochs),
            "train_batch_size": int(tcfg.train_batch_size),
            "eval_batch_size": int(tcfg.eval_batch_size),
            "learning_rate": float(tcfg.learning_rate),
        }

        # 모델 파라미터
        if mcfg:
            overrides.update(dict(mcfg))

        # 유저 override (마지막)

        tna = overrides.get("train_neg_sample_args")
        if isinstance(tna, list):
            # 흔한 케이스: [{'distribution': 'uniform', 'sample_num': 5}]
            if len(tna) == 1 and isinstance(tna[0], dict):
                overrides["train_neg_sample_args"] = tna[0]
            else:
                raise ValueError(f"train_neg_sample_args must be dict, got list: {tna}")

        overrides.update({
            "eval_args": {
                "group_by": "user",
                "order": "RO",
                "mode": "full",
            },
            "topk": 10,
        })

        return overrides
