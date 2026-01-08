from __future__ import annotations

"""
Sklearn 엔진 베이스.

입력:
- cfg.train.run_dir: str (run 디렉토리 루트)
- checkpoint: Optional[str]

출력:
- (공통) checkpoint save/load (joblib)
- (공통) logger 이벤트 기록
"""

from abc import ABC
from typing import Any, Optional, Dict
import os
import joblib

from src.engines.core.engine_base import EngineBase
from src.engines.core.common import PredsValidator
from src.utils.cfg_utils import cfg_select


class SklearnEngineBase(EngineBase, ABC):
    """
    Sklearn engines share only:
      - checkpoint save/load (joblib)
      - checkpoint resolution priority (arg > cfg.checkpoint > default)
      - logging
      - preds contract validation (delegated to PredsValidator)
    """

    # -------- run dir --------
    def _run_dir(self) -> str:
        """
        Single source of truth for run directory.
        Uses Setting.get_run_dir() policy.
        """
        # cfg shape (preferred):
        # cfg.train.run_dir, cfg.engine.type, cfg.model, cfg.run_name
        base_dir = cfg_select(self.cfg, "train.run_dir", default="saved/runs")

        model = getattr(self.cfg, "model", None)
        model_name = str(model)

        engine = getattr(self.cfg, "engine", None)
        engine_type = str(getattr(engine, "type", "sklearn"))

        run_name = getattr(self.cfg, "run_name", None) or None
        return self.setting.get_run_dir(
            base_dir=base_dir,
            model=model_name,
            engine_type=engine_type,
            run_name=run_name,
        )

    # -------- checkpoint I/O --------
    def _default_ckpt_path(self, filename: Optional[str] = None) -> str:
        run_dir = self._run_dir()
        self.setting.ensure_dir(run_dir)

        if filename is None:
            filename = f"{self.cfg.model}.joblib"
        return os.path.join(run_dir, filename)

    def _resolve_checkpoint(self, checkpoint: Optional[str], default_filename: Optional[str] = None) -> str:
        if checkpoint:
            return checkpoint

        cfg_ckpt = self.cfg.get("checkpoint", None)
        if cfg_ckpt:
            return cfg_ckpt

        return self._default_ckpt_path(default_filename)

    def _save_checkpoint(self, obj: Dict[str, Any], path: str) -> None:
        joblib.dump(obj, path)
        if self.logger:
            self.logger.log_artifact(path, name="model_checkpoint")

    def _load_checkpoint(self, path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return joblib.load(path)

    # -------- logging helpers --------
    def _log_train(self, payload: Dict[str, Any]) -> None:
        payload = dict(payload)
        payload.setdefault("engine", "sklearn")
        payload.setdefault("mode", "train")
        if self.logger:
            # keep API simple: treat as train metrics at step=0 unless caller provides step elsewhere
            self.logger.log_train_metrics(payload, step=0)

    def _log_predict(self, payload: Dict[str, Any]) -> None:
        payload = dict(payload)
        payload.setdefault("engine", "sklearn")
        payload.setdefault("mode", "predict")
        if self.logger:
            self.logger.log_predict_info(payload)

    # -------- validation helper --------
    def _validate_preds(self, preds, bundle) -> None:
        PredsValidator.validate(preds, bundle)
