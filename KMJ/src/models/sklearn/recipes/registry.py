from __future__ import annotations
from typing import Any, Callable, Dict

SklearnBuilder = Callable[[Any], Any]   # registry에서는 Any로 둠
SKLEARN_RECIPE_REGISTRY: Dict[str, SklearnBuilder] = {}

def register_sklearn_recipe(name: str):
    def deco(fn: SklearnBuilder):
        prev = SKLEARN_RECIPE_REGISTRY.get(name)
        if prev is not None and prev is not fn:
            raise KeyError(f"Duplicate sklearn recipe: {name}")
        SKLEARN_RECIPE_REGISTRY[name] = fn
        return fn
    return deco


def bootstrap_sklearn_recipes() -> list[str]:
    """
    Discover & import sklearn recipe modules to trigger @register_sklearn_recipe decorators.
    Called by `src.bootstrap.bootstrap_registries()`.
    """
    from src.utils.registry_utils import autodiscover

    return autodiscover(
        "src.models.sklearn.recipes",
        exclude=("__init__", "registry", "base"),
        recursive=False,
    )