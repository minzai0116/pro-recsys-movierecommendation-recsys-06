from __future__ import annotations

from typing import Any, Dict

from .base import RecBoleRecipeBase
from .registry import register_recbole_recipe
from src.models.recbole.adapter.fieldmap import build_fieldmap, recbole_field_overrides
from src.models.recbole.adapter.atomic_export import export_inter, export_item_features


@register_recbole_recipe("SASRecF")
def build(cfg: Any) -> RecBoleRecipeBase:
    return SASRecfRecipe(cfg)


class SASRecfRecipe(RecBoleRecipeBase):
    name = "SASRecF"

    def prepare_dataset(self, bundle, *, data_root: str, dataset: str, setting):
        setting.ensure_dir(data_root)
        setting.ensure_dir(f"{data_root}/{dataset}")

        fm = build_fieldmap(bundle.schema or {})
        spec = export_inter(bundle=bundle, out_root=data_root, dataset=dataset, fm=fm)

        # SASRecF는 side information을 사용
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
        }

        # load_col 설정 추가
        # RecBole은 명시하지 않으면 .inter 컬럼만 로드할 수 있음
        # SASRecf를 위해 item 피쳐 컬럼들도 로드하도록 설정
        load_col = {
            "inter": [fm.export_user, fm.export_item, fm.export_time]
        }
        if fm.export_target:
            load_col['inter'].append(fm.export_target)

        if item_df is not None:
            # item_df의 컬럼 중 ID 제외한 나머지
            feat_cols = [c for c in item_df.columns if c != fm.item_col]
            # ID 컬럼 + 피쳐 컬럼
            load_col['item'] = [fm.export_item] + feat_cols
            # selected_features 설정 추가 (모델 사용용)
            # SASRecF가 사용할 피처 목록을 명시적으로 지정해줍니다.
            overrides["selected_features"] = feat_cols
        
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
