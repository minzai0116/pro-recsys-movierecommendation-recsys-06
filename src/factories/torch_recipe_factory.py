# src/factories/torch_recipe_factory.py
from __future__ import annotations

from typing import Any

from src.models import TORCH_RECIPE_REGISTRY


class TorchRecipeFactory:
    @classmethod
    def build(cls, cfg: Any):
        model = getattr(cfg, "model", None)
        name = None
        if model is not None and getattr(model, "name", None) is not None:
            name = str(model.name)
        else:
            # fallback if cfg.model is a string-like
            name = str(getattr(cfg, "model", ""))

        if not name:
            raise ValueError("Missing cfg.model.name for torch recipe selection.")

        if name not in TORCH_RECIPE_REGISTRY:
            avail = ", ".join(sorted(TORCH_RECIPE_REGISTRY.keys())) or "<empty>"
            raise ValueError(f"Unknown torch recipe: {name}. Registered: {avail}")

        return TORCH_RECIPE_REGISTRY[name](cfg)
