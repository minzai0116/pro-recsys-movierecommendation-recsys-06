# src/engines/torch/registry.py
from __future__ import annotations

from src.engines.registry import register_engine
from src.engines.torch.torch_base import TorchBaseEngine  # 네 실제 torch 엔진 클래스

@register_engine("torch")
def build_torch_engine(cfg, logger, setting):
    return TorchBaseEngine(cfg, logger, setting)
