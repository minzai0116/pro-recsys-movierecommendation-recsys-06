# src/factories/sklearn_recipe_factory.py
from __future__ import annotations

from typing import Any

# Single source of truth: recipes registry
from src.models.sklearn.recipes.registry import SKLEARN_RECIPE_REGISTRY


class SklearnRecipeFactory:
    @classmethod
    def _get_model_name(cls, cfg: Any) -> str:
        """
        Supports common config shapes:
          - cfg.model.name
          - cfg.model as string
          - cfg.model as mapping with 'name' (OmegaConf DictConfig 포함)
        """
        m = getattr(cfg, "model", None)
        if m is None:
            raise ValueError("cfg.model is required")

        name = getattr(m, "name", None)
        if name is not None:
            return str(name)

        try:
            # dict / DictConfig
            if isinstance(m, dict) and "name" in m:
                return str(m["name"])
            if "name" in m:  # type: ignore[operator]
                return str(m.get("name"))  # type: ignore[attr-defined]
        except Exception:
            pass

        return str(m)

    @classmethod
    def build(cls, cfg: Any):
        name = cls._get_model_name(cfg)
        if name not in SKLEARN_RECIPE_REGISTRY:
            raise ValueError(
                f"Unknown sklearn recipe: {name}. Available: {list(SKLEARN_RECIPE_REGISTRY.keys())}"
            )
        return SKLEARN_RECIPE_REGISTRY[name](cfg)
