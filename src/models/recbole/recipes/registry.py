from __future__ import annotations

from typing import Any, Callable, Dict

from .base import RecBoleRecipeBase

Builder = Callable[[Any], RecBoleRecipeBase]
_REG: Dict[str, Builder] = {}


def register_recbole_recipe(model_name: str):
    def deco(fn: Builder):
        if model_name in _REG and _REG[model_name] is not fn:
            raise KeyError(f"Duplicate recbole recipe: {model_name}")
        _REG[model_name] = fn
        return fn
    return deco


def build_recbole_recipe(cfg: Any) -> RecBoleRecipeBase:
    model = getattr(cfg, "model", None)
    if not model:
        raise ValueError("cfg.model is required to build recbole recipe")
    if model not in _REG:
        raise KeyError(f"Unknown recbole recipe: {model}. Available={sorted(_REG.keys())}")
    return _REG[model](cfg)

def bootstrap_recbole_recipes() -> list[str]:
    """
    Discover & import recbole recipe modules to trigger @register_recbole_recipe decorators.
    Called by `src.bootstrap.bootstrap_registries()`.
    """
    from src.utils.registry_utils import autodiscover

    return autodiscover(
        "src.models.recbole.recipes",
        exclude=("__init__", "registry", "base"),
        recursive=False,
    )