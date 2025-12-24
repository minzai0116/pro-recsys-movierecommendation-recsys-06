from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.optim import Optimizer


@dataclass
class TorchCheckpointPayload:
    model: Dict[str, Any]
    optim: Optional[Dict[str, Any]]
    sched: Optional[Dict[str, Any]]
    scaler: Optional[Dict[str, Any]]
    meta: Dict[str, Any]


class TorchCheckpointManager:
    """
    Handles save/load and path conventions.
    Path creation must go through Setting (ensure_dir).
    """

    def __init__(self, setting):
        self.setting = setting

    def ckpt_dir(self) -> str:
        # convention: <run_dir>/checkpoints
        run_dir = getattr(self.setting, "run_dir", ".")
        ckpt_dir = f"{run_dir}/checkpoints"
        # Setting must own dir creation
        self.setting.ensure_dir(ckpt_dir)
        return ckpt_dir

    def default_last_path(self) -> str:
        return f"{self.ckpt_dir()}/last.pt"

    def default_best_path(self) -> str:
        return f"{self.ckpt_dir()}/best.pt"

    def resolve_predict_checkpoint(self, cfg: Any, checkpoint_arg: Optional[str]) -> str:
        # Policy: predict-only requires explicit checkpoint (your stated preference)
        predict_flag = bool(getattr(cfg, "predict", False) or (cfg.get("predict") if hasattr(cfg, "get") else False))
        cfg_ckpt = getattr(cfg, "checkpoint", None) or (cfg.get("checkpoint") if hasattr(cfg, "get") else None)

        ckpt = checkpoint_arg or cfg_ckpt
        if predict_flag and not ckpt:
            raise ValueError("predict-only requires checkpoint (cfg.checkpoint or predict(checkpoint=...))")

        # if not predict-only and still empty, allow default
        return ckpt or self.default_last_path()

    def save(
        self,
        path: str,
        model: nn.Module,
        optimizer: Optional[Optimizer],
        scheduler: Optional[Any],
        scaler: Optional[Any],
        meta: Dict[str, Any],
    ) -> None:
        payload = {
            "model": model.state_dict(),
            "optim": optimizer.state_dict() if optimizer is not None else None,
            "sched": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "meta": meta,
        }
        self.setting.ensure_dir(path.rsplit("/", 1)[0])
        torch.save(payload, path)

    def load(self, path: str, map_location: str = "cpu") -> TorchCheckpointPayload:
        obj = torch.load(path, map_location=map_location)
        return TorchCheckpointPayload(
            model=obj["model"],
            optim=obj.get("optim"),
            sched=obj.get("sched"),
            scaler=obj.get("scaler"),
            meta=obj.get("meta", {}),
        )
