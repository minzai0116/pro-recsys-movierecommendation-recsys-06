from __future__ import annotations

from typing import Any, Dict

from .base import RecBoleRecipeBase
from .registry import register_recbole_recipe
from src.models.recbole.adapter.fieldmap import build_fieldmap, recbole_field_overrides
from src.models.recbole.adapter.atomic_export import export_inter


@register_recbole_recipe("MultiDAE")
def build(cfg: Any) -> RecBoleRecipeBase:
    return MultiDAERecipe(cfg)


class MultiDAERecipe(RecBoleRecipeBase):
    """
    RecBole MultiDAE recipe.

    RecBole MultiDAE는 내부에서 다음 키를 필수로 참조합니다:
      - mlp_hidden_size
      - latent_dimension
      - dropout_prob

    따라서 cfg.model_args.MultiDAE에 위 3개는 반드시 포함되어야 합니다.
    """

    name = "MultiDAE"

    def prepare_dataset(self, bundle, *, data_root: str, dataset: str, setting):
        setting.ensure_dir(data_root)
        setting.ensure_dir(f"{data_root}/{dataset}")

        fm = build_fieldmap(bundle.schema or {})
        return export_inter(bundle=bundle, out_root=data_root, dataset=dataset, fm=fm)

    def build_overrides(self, bundle, *, data_root: str, dataset: str) -> Dict[str, Any]:
        fm = build_fieldmap(bundle.schema or {})

        # train.* is SSoT
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

        # 유저 config merge (마지막)
        recbole_cfg = getattr(self.cfg.recbole, "config", None)
        if recbole_cfg:
            overrides.update(dict(recbole_cfg))

        user_over = getattr(self.cfg.recbole, "overrides", None)
        if user_over:
            overrides.update(dict(user_over))
        return overrides


