from __future__ import annotations

"""
Run 단위 로거.

입력:
- cfg.wandb: bool
- cfg.verbose: bool
- log_dir(run_dir): str

출력:
- config.yaml / config_resolved.yaml
- events.jsonl (표준 이벤트 스트림)
- predict_info.txt (사람용)
- artifacts.txt

비고:
- wandb가 활성화되어 있고 import 가능하면 동일 payload를 wandb에도 기록합니다.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from omegaconf import OmegaConf


class Logger:
    """
    Run-scoped logger.

    Standard outputs under `log_dir`:
      - config.yaml / config_resolved.yaml
      - events.jsonl   (metrics/predict/artifact 모두 JSONL로 통합)
      - artifacts.txt  (간단 인덱스)

    W&B:
      - 가능하면 동일 payload를 wandb.log/config로도 기록
    """

    def __init__(self, cfg: Any, log_dir: str):
        self.cfg = cfg
        self.log_dir = str(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)

        self.use_wandb = bool(cfg.get("wandb", False))
        self.verbose = bool(cfg.get("verbose", True))

        self.events_path = os.path.join(self.log_dir, "events.jsonl")
        self.artifacts_index_path = os.path.join(self.log_dir, "artifacts.txt")

        self.wandb = None
        if self.use_wandb:
            try:
                import wandb  # type: ignore

                self.wandb = wandb
            except Exception:
                # wandb가 설치되지 않았거나 import 실패해도 파일 로깅은 계속 된다
                self.wandb = None
                self._print("wandb import failed; continuing with file logging only.")

    # -------------------------
    # core: JSONL event stream
    # -------------------------
    def log_event(self, kind: str, payload: Dict[str, Any], step: Optional[int] = None) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": str(kind),
            "step": step,
            "payload": payload,
        }
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # console (짧게)
        if self.verbose:
            preview = ", ".join(f"{k}={v}" for k, v in list(payload.items())[:6])
            self._print(f"{kind}" + (f" step={step}" if step is not None else "") + (f" | {preview}" if preview else ""))

    # -------------------------
    # config
    # -------------------------
    def save_args(self) -> None:
        """
        Backward-compatible name. Saves config files + logs to wandb.config (if enabled).
        """
        raw_path = os.path.join(self.log_dir, "config.yaml")
        resolved_path = os.path.join(self.log_dir, "config_resolved.yaml")

        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(OmegaConf.to_yaml(self.cfg))

        with open(resolved_path, "w", encoding="utf-8") as f:
            f.write(OmegaConf.to_yaml(OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=True))))

        if self.wandb:
            self.wandb.config.update(
                OmegaConf.to_container(self.cfg, resolve=True),
                allow_val_change=True,
            )

        self.log_event("config/saved", {"config_path": raw_path, "config_resolved_path": resolved_path})

    # -------------------------
    # metrics
    # -------------------------
    def log_train_metrics(self, metrics: dict, step: int) -> None:
        # wandb는 prefix를 유지 (기존 스타일 호환)
        if self.wandb:
            payload = {f"train/{k}": v for k, v in metrics.items()}
            self.wandb.log(payload, step=step)

        self.log_event("metrics/train", dict(metrics), step=step)

    def log_valid_metrics(self, metrics: dict, step: int) -> None:
        if self.wandb:
            payload = {f"valid/{k}": v for k, v in metrics.items()}
            self.wandb.log(payload, step=step)

        self.log_event("metrics/valid", dict(metrics), step=step)

    # -------------------------
    # predict
    # -------------------------
    def log_predict_info(self, info: dict, step: int | None = None) -> None:
        if self.wandb:
            payload = {f"predict/{k}": v for k, v in info.items()}
            if step is None:
                self.wandb.log(payload)
            else:
                self.wandb.log(payload, step=step)

        # 기존 txt도 유지(사람이 보기 쉬움)
        txt_path = os.path.join(self.log_dir, "predict_info.txt")
        with open(txt_path, "a", encoding="utf-8") as f:
            f.write(f"\n[{datetime.now()}]\n")
            for k, v in info.items():
                f.write(f"{k}: {v}\n")

        self.log_event("predict/info", dict(info), step=step)

    # -------------------------
    # artifact
    # -------------------------
    def log_artifact(self, path: str, name: str) -> None:
        path = str(path)
        name = str(name)

        if self.wandb:
            try:
                self.wandb.save(path)
            except Exception:
                pass

        with open(self.artifacts_index_path, "a", encoding="utf-8") as f:
            f.write(f"{name}: {path}\n")

        self.log_event("artifact", {"name": name, "path": path})

    def _print(self, msg: str) -> None:
        if self.verbose:
            print(f"[LOG] {msg}")

