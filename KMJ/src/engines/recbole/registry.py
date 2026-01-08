# src/engines/recbole/registry.py
from __future__ import annotations

from src.engines.registry import register_engine
from src.engines.recbole.recbole_engine import RecBoleEngine

@register_engine("recbole")
def build(cfg, logger, setting):
    return RecBoleEngine(cfg, logger, setting)
