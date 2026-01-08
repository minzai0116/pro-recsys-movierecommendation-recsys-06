from __future__ import annotations

"""
Engine 공통 유틸/Contract 검사.

입력:
- cfg / setting / checkpoint_arg 등
- preds / DataBundle

출력:
- CheckpointPolicy: checkpoint 경로 결정
- PredsValidator: task별 preds 형식 검증(예: seq_topn은 List[List[int]])
"""

from typing import Any, Dict, List, Optional, Union
import os
import numpy as np


class CheckpointPolicy:
    """
    Centralizes checkpoint resolution rules.

    Priority for predict():
      1) explicit checkpoint argument
      2) cfg.checkpoint
      3) default path in run_dir (model-specific filename)
    """

    def __init__(self, cfg: Any, setting):
        self.cfg = cfg
        self.setting = setting

    def default_ckpt_path(self, run_dir: str, filename: str) -> str:
        self.setting.ensure_dir(run_dir)
        return os.path.join(run_dir, filename)

    def resolve_predict_checkpoint(
            self,
            checkpoint_arg: Optional[str],
            run_dir: str,
            default_filename: str,
    ) -> str:
        if checkpoint_arg:
            return checkpoint_arg

        cfg_ckpt = self.cfg.get("checkpoint", None)  # 일단 그대로 둠
        if cfg_ckpt:
            return cfg_ckpt

        return self.default_ckpt_path(run_dir, default_filename)


class PredsValidator:
    """
    Hard contract checks for engine outputs.
    """

    @staticmethod
    def validate(preds, bundle) -> None:
        task = bundle.schema["task"]

        if task == "regression":
            # expect 1D array-like length == len(test)
            arr = np.asarray(preds)
            if arr.ndim != 1:
                raise ValueError(f"regression preds must be 1D, got shape={arr.shape}")
            if len(arr) != len(bundle.test):
                raise ValueError(f"regression preds length mismatch: preds={len(arr)} test={len(bundle.test)}")
            return

        if task == "topn":
            # expect List[List[item]] length == len(submission.users)
            sub = bundle.meta.get("submission", None)
            if not sub or "users" not in sub:
                raise ValueError("topn bundle must have meta['submission']['users']")
            users: List = sub["users"]
            if not isinstance(preds, list):
                raise ValueError("topn preds must be a list (outer)")
            if len(preds) != len(users):
                raise ValueError(f"topn preds length mismatch: preds={len(preds)} users={len(users)}")
            # inner lists
            for i, row in enumerate(preds[:5]):  # sample-check first few
                if not isinstance(row, list):
                    raise ValueError(f"topn preds[{i}] must be a list (inner)")
            return

        if task == "seq_topn":
            sub = bundle.meta.get("submission", None)
            if not sub or "users" not in sub:
                raise ValueError("seq_topn bundle must have meta['submission']['users']")
            if "user_seq" not in (bundle.meta or {}):
                raise ValueError("seq_topn bundle must have meta['user_seq']")
            users: List = sub["users"]
            if not isinstance(preds, list):
                raise ValueError("seq_topn preds must be a list (outer)")
            if len(preds) != len(users):
                raise ValueError(f"seq_topn preds length mismatch: preds={len(preds)} users={len(users)}")
            for i, row in enumerate(preds[:5]):
                if not isinstance(row, list):
                    raise ValueError(f"seq_topn preds[{i}] must be a list (inner)")
            return

        raise ValueError(f"Unknown task: {task}")
