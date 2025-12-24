from __future__ import annotations
from typing import Any, List

"""
Torch 실행 엔진 (공통 루프).

입력:
- cfg.train.*: epochs/batch_size/lr/amp/grad_clip 등
- cfg.predict: bool (predict-only 여부)
- cfg.checkpoint: str (predict-only 시 사용)
- bundle: DataBundle (schema/meta 포함)

출력:
- fit(bundle) -> {"checkpoint_path": str}
- predict(bundle, checkpoint=...) -> preds
  * seq_topn/topn: List[List[int]]
  * regression: np.ndarray (1D) 등

부작용:
- run_dir 아래에 `last.pt` 체크포인트 저장
- logger를 통해 metrics/predict/artifact 이벤트 기록
"""

import torch
import numpy as np

from src.engines.core.engine_base import EngineBase
from src.engines.core.common import PredsValidator
from src.models.torch.recipes.registry import build_torch_recipe


class TorchBaseEngine(EngineBase):
    """
    Torch 공통 실행 엔진 (단 하나).

    책임:
    - train / predict 루프
    - device / AMP / grad clip
    - checkpoint save/load
    - logger / validator 호출

    책임 아님:
    - 태스크 개념
    - 데이터 의미
    - padding / sampling / candidate
    """

    def __init__(self, cfg: Any, logger, setting):
        super().__init__(cfg, logger, setting)

        device_str = str(getattr(cfg, "device", "cpu"))
        self.device = torch.device(
            device_str if device_str.startswith("cuda") and torch.cuda.is_available() else "cpu"
        )

        # train cfg (strict: cfg.train only)
        tc = getattr(cfg, "train")
        try:
            self.use_amp = bool(getattr(tc, "amp"))
        except Exception:
            self.use_amp = bool(getattr(cfg, "amp", False))
        try:
            self.grad_clip = float(getattr(tc, "grad_clip") or 0.0)
        except Exception:
            self.grad_clip = 0.0

        self.recipe = build_torch_recipe(cfg)

        self.model = None
        self.optimizer = None
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    # ---------------- public API ----------------
    def fit(self, bundle):
        # mode=predict 인 경우 fit은 호출되지 않는 것을 전제하지만, 방어적으로 막는다
        if str(getattr(self.cfg, "mode", "")).lower() == "predict" or bool(getattr(self.cfg, "predict", False)):
            return {"checkpoint_path": None}

        loaders = self.recipe.build_loaders(self.cfg, bundle)
        self._init_train_components(bundle)

        tc = self.recipe.train_cfg()
        try:
            epochs = int(getattr(tc, "epochs"))
        except Exception:
            epochs = int(getattr(self.cfg, "epochs", 1))

        for epoch in range(epochs):
            self.model.train()
            losses: List[float] = []
            for batch in loaders["train"]:
                batch = self.recipe.move_batch_to_device(batch, self.device)

                self.optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    out = self.recipe.train_step(self.cfg, batch, self.model)
                    loss = out["loss"]
                    try:
                        losses.append(float(loss.detach().cpu().item()))
                    except Exception:
                        pass

                self.scaler.scale(loss).backward()

                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()

            # epoch logging
            if self.logger:
                lr = None
                try:
                    lr = float(self.optimizer.param_groups[0].get("lr"))  # type: ignore[union-attr]
                except Exception:
                    lr = None
                metrics = {}
                if losses:
                    metrics["loss"] = float(sum(losses) / max(len(losses), 1))
                if lr is not None:
                    metrics["lr"] = lr
                metrics["epoch"] = epoch
                self.logger.log_train_metrics(metrics, step=epoch)

            self._save_checkpoint(epoch)

        # main.py contract: fit() returns dict with checkpoint_path
        return {"checkpoint_path": f"{self.setting.run_dir}/last.pt"}

    def predict(self, bundle, checkpoint: str | None = None):
        loaders = self.recipe.build_loaders(self.cfg, bundle)
        if self.logger:
            self.logger.log_predict_info(
                {"engine": "torch", "model": str(getattr(self.cfg, "model", "")), "stage": "start", "checkpoint": checkpoint or ""},
            )
        self._load_checkpoint(checkpoint)

        self.model.eval()
        preds_chunks: List[Any] = []

        with torch.no_grad():
            for batch in loaders["test"]:
                batch = self.recipe.move_batch_to_device(batch, self.device)
                out = self.recipe.predict_step(self.cfg, batch, self.model)
                preds_chunks.append(out)

        preds = self._merge_preds(preds_chunks)
        PredsValidator.validate(preds, bundle)
        if self.logger:
            try:
                n = len(preds)
            except Exception:
                n = None
            self.logger.log_predict_info(
                {"engine": "torch", "model": str(getattr(self.cfg, "model", "")), "stage": "done", "preds": n},
            )
        return preds

    # ---------------- internal ----------------
    def _init_train_components(self, bundle):
        self.model = self.recipe.build_model(self.cfg, bundle).to(self.device)
        self.optimizer = self.recipe.build_optimizer(self.cfg, self.model)

    def _save_checkpoint(self, epoch: int):
        path = f"{self.setting.run_dir}/last.pt"
        self.setting.ensure_dir(self.setting.run_dir)
        torch.save({"model": self.model.state_dict()}, path)
        if self.logger:
            self.logger.log_artifact(path, name="torch_last_checkpoint")

    def _load_checkpoint(self, path: str | None):
        if not path:
            path = f"{self.setting.run_dir}/last.pt"
        obj = torch.load(path, map_location="cpu")
        self.model.load_state_dict(obj["model"])
        self.model.to(self.device)

    def _merge_preds(self, chunks):
        # 기본: numpy concat (regression)
        if isinstance(chunks[0], np.ndarray):
            return np.concatenate(chunks, axis=0)
        # list-of-list (topn/seq)
        out = []
        for c in chunks:
            out.extend(c)
        return out
