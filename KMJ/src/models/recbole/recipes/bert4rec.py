from __future__ import annotations

from typing import Any, Dict

from .base import RecBoleRecipeBase
from .registry import register_recbole_recipe
from src.models.recbole.adapter.fieldmap import build_fieldmap, recbole_field_overrides
from src.models.recbole.adapter.atomic_export import export_inter, export_item_features


@register_recbole_recipe("BERT4Rec")
def build(cfg: Any) -> RecBoleRecipeBase:
    return BERT4RecRecipe(cfg)


class BERT4RecRecipe(RecBoleRecipeBase):
    name = "BERT4Rec"

    def prepare_dataset(self, bundle, *, data_root: str, dataset: str, setting):
        setting.ensure_dir(data_root)
        setting.ensure_dir(f"{data_root}/{dataset}")

        fm = build_fieldmap(bundle.schema or {})
        spec = export_inter(bundle=bundle, out_root=data_root, dataset=dataset, fm=fm)
        # BERT4Rec이 피처를 안 써도, 전체 아이템 개수(ID 매핑)를 맞추기 위해 내보내는 게 안전
        item_df = bundle.meta.get("item_df")
        if item_df is not None:
            spec = export_item_features(spec, item_df, fm)
        
        return spec

    def build_overrides(self, bundle, *, data_root: str, dataset: str) -> Dict[str, Any]:
        fm = build_fieldmap(bundle.schema or {})
        item_df = bundle.meta.get("item_df")
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
            # --- BERT4Rec 전용 필수 설정 ---
            "loss_type": "CE",   # Cross Entropy 사용
            "mask_ratio": 0.2,   # 아이템의 20%를 가리고 맞추기 게임 진행
        }

        # load_col 설정 추가
        load_col = {
            "inter": [fm.export_user, fm.export_item, fm.export_time]
        }
        
        overrides["load_col"] = load_col

        # 모델 파라미터
        if mcfg:
            overrides.update(dict(mcfg))

        # 유저 config merge (마지막)
        recbole_cfg = getattr(self.cfg.recbole, "config", None)
        if recbole_cfg:
            overrides.update(dict(recbole_cfg))

        user_over = getattr(self.cfg.recbole, "overrides", None)
        if user_over:
            overrides.update(dict(user_over))

        return overrides
