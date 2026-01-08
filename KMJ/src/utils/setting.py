"""
실행(run) 단위의 경로/재현성 유틸.

입력:
- seed_everything(seed): int
- get_run_dir(base_dir, model, engine_type, run_name)

출력:
- Setting.run_dir: 현재 run의 저장 디렉토리
- get_submit_path(...): 제출 파일 저장 경로(str)
"""

import os
import time
import random
import numpy as np
import torch


class Setting:
    @staticmethod
    def seed_everything(seed: int):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def __init__(self):
        now = time.localtime()
        self.run_id = time.strftime("%Y%m%d_%H%M%S", now)

    @staticmethod
    def ensure_dir(path: str) -> str:
        os.makedirs(path, exist_ok=True)
        return path

    def get_run_dir(self, base_dir: str, model: str, engine_type: str, run_name: str | None = None) -> str:
        """
        실험(run) 단위 디렉토리.
        - base_dir: 예) saved/runs
        - model: 예) LightGBM
        - engine_type: sklearn|torch|recbole
        - run_name: optional label (wandb run_name 같은 것)
        """
        name = f"{self.run_id}_{engine_type}_{model}"
        if run_name:
            safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in run_name)
            name = f"{name}_{safe}"
        path = os.path.join(base_dir, name)
        path = self.ensure_dir(path)
        # engines/logger may refer to setting.run_dir as run-scoped single source of truth
        self.run_dir = path
        return path

    def get_submit_path(
            self,
            submit_dir: str | None = None,
            base_dir: str | None = None,  # 과거 호출 호환
            model: str | None = None,  # 과거 호출 호환
            filename: str | None = None,
            ext: str = "csv",
            **kwargs,
    ) -> str:
        """
        Backward-compatible submit path builder.
        - base_dir / model 키워드가 와도 안 죽게 처리
        """
        # 기본 submit_dir
        if submit_dir is None:
            submit_dir = getattr(self, "submit_dir", None) or getattr(self, "run_dir", ".")

        self.ensure_dir(submit_dir)

        # 파일명 기본값: model 기반으로 만들고 싶으면 model 사용
        if filename is None:
            if model is None:
                filename = f"submission.{ext}"
            else:
                filename = f"{model}.{ext}"

        import os
        return os.path.join(submit_dir, filename)

    @staticmethod
    def infer_checkpoint_tag(checkpoint_path: str) -> str:
        """
        predict-only에서 제출 파일명에 쓸 태그 (checkpoint basename 기반).
        """
        base = os.path.basename(checkpoint_path)
        # 확장자까지 포함해도 되고, 빼고 싶으면 splitext 사용
        return base
