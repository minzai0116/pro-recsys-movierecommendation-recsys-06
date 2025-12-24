# src/models/torch/recipes/registry.py
from __future__ import annotations

from typing import Any, Callable, Dict

from src.models.torch.recipes.torch_recipe_base import TorchRecipeBase

# name -> builder(cfg) -> TorchRecipeBase
_REGISTRY: Dict[str, Callable[[Any], TorchRecipeBase]] = {}


def register_torch_recipe(name: str):
    """
    Register a torch recipe builder.

    Usage:
      @register_torch_recipe("mlp_regression")
      def build(cfg): ...
    """
    def deco(builder: Callable[[Any], TorchRecipeBase]):
        if name in _REGISTRY and _REGISTRY[name] is not builder:
            raise KeyError(f"Duplicate torch recipe: {name}")
        _REGISTRY[name] = builder
        return builder
    return deco


def build_torch_recipe(cfg) -> TorchRecipeBase:
    name = getattr(cfg, "recipe", None)
    if not name:
        raise ValueError("cfg.recipe is required for torch engine")

    if name not in _REGISTRY:
        raise KeyError(f"Unknown torch recipe: {name}. Available={sorted(_REGISTRY.keys())}")

    return _REGISTRY[name](cfg)

def bootstrap_torch_recipes() -> list[str]:
    """
    Discover & import torch recipe modules to trigger @register_torch_recipe decorators.
    Called by `src.bootstrap.bootstrap_registries()`.
    """
    from src.utils.registry_utils import autodiscover

    return autodiscover(
        "src.models.torch.recipes",
        exclude=("__init__", "registry", "torch_recipe_base"),
        recursive=False,
    )
