from __future__ import annotations

"""
Problem 계층: "대회/업무 문제 정의"의 최상위 Contract.

입력:
- cfg: OmegaConf(DictConfig)
- DataPipeline이 만든 DataBundle

출력(Contract):
- run() -> DataBundle
- save_submission(preds, ...) -> Optional[str] (저장 경로)
- (선택) evaluate_preds(preds, ...) -> dict|None (wandb/logging용 지표)

핵심:
- Problem은 "어떤 DataPipeline을 쓸지"와 "제출 포맷 정책"을 소유합니다.
"""

from typing import Any, Optional, List

import pandas as pd

from src.data.data_bundle import DataBundle


class ProblemBase:
    """
    Problem responsibilities:
      - choose data pipeline
      - define submission policy

    Pipeline responsibilities:
      - load / normalize / split
      - build DataBundle

    Engine responsibilities:
      - consume DataBundle
      - produce preds matching submission contract
    """

    name: str = "base"

    def __init__(self, cfg: Any):
        self.cfg = cfg

    def run(self) -> DataBundle:
        bundle = self.build_data_bundle()
        self.validate_bundle(bundle)
        return bundle

    def build_data_bundle(self) -> DataBundle:
        raise NotImplementedError

    def save_submission(self, preds, cfg: Any, setting, bundle) -> Optional[str]:
        raise NotImplementedError

    def evaluate_preds(self, preds, cfg: Any, bundle: DataBundle) -> dict | None:
        """
        Optional evaluation hook.
        - Called from main after `engine.predict()` and before `save_submission()`
        - Return a flat dict of metrics to be logged (e.g., {"Recall@10": 0.123})
        """
        return None

    # --------------------------------------------------
    # Contract validation
    # --------------------------------------------------
    def validate_bundle(self, bundle: DataBundle) -> None:
        self._validate_bundle_types(bundle)
        self._validate_schema_common(bundle)

        task = bundle.schema["task"]
        if task == "regression":
            self._validate_regression(bundle)
        elif task == "topn":
            self._validate_topn(bundle)
        elif task == "seq_topn":
            self._validate_seq_topn(bundle)
        else:
            raise ValueError("schema.task must be regression|topn|seq_topn")

    # ------------------------
    # Type checks
    # ------------------------
    @staticmethod
    def _validate_bundle_types(bundle: DataBundle) -> None:
        if not isinstance(bundle, DataBundle):
            raise TypeError("Problem.run() must return DataBundle")

        if not isinstance(bundle.train, pd.DataFrame):
            raise TypeError("DataBundle.train must be DataFrame")

        if bundle.valid is not None and not isinstance(bundle.valid, pd.DataFrame):
            raise TypeError("DataBundle.valid must be DataFrame or None")

        if bundle.test is not None and not isinstance(bundle.test, pd.DataFrame):
            raise TypeError("DataBundle.test must be DataFrame or None")

        for key in ["schema", "meta"]:
            if not isinstance(getattr(bundle, key), dict):
                raise TypeError(f"DataBundle.{key} must be dict")

    # ------------------------
    # Schema common
    # ------------------------
    def _validate_schema_common(self, bundle: DataBundle) -> None:
        schema = bundle.schema

        # 공통 키 강제 (time_col/target_col은 task별로 meaning이 달라서 None 허용)
        for k in ["task", "user_col", "item_col", "time_col", "target_col"]:
            if k not in schema:
                raise ValueError(f"schema missing required key: {k}")

        task = schema["task"]
        if task not in ("regression", "topn", "seq_topn"):
            raise ValueError("schema.task must be regression|topn|seq_topn")

        user_col = schema["user_col"]
        item_col = schema["item_col"]

        # train에는 최소 user/item은 있어야 한다
        self._require_columns(bundle.train, [user_col, item_col], "train")

    # ------------------------
    # Task-specific validation
    # ------------------------
    def _validate_regression(self, bundle: DataBundle) -> None:
        schema = bundle.schema
        target_col = schema.get("target_col", None)
        if not target_col:
            raise ValueError("regression requires schema.target_col")

        target_col_str: str = target_col
        self._require_columns(bundle.train, [target_col_str], "train")

        # feature_cols는 optional (없으면 엔진/레시피에서 추론할 수 있음)
        feature_cols = schema.get("feature_cols", None)
        if feature_cols is not None:
            if not isinstance(feature_cols, list):
                raise ValueError("schema.feature_cols must be list or None")
            self._require_columns(bundle.train, feature_cols, "train")

    def _validate_topn(self, bundle: DataBundle) -> None:
        schema = bundle.schema

        # topn은 비시퀀셜일 수 있으므로 time_col 필수로 강제하지 않는다
        # time_col이 제공되면 train에 존재해야 함
        time_col = schema.get("time_col", None)
        if time_col is not None:
            self._require_columns(bundle.train, [time_col], "train")

        # submission contract (topn 계열 공통)
        self._validate_submission_users(bundle)

    def _validate_seq_topn(self, bundle: DataBundle) -> None:
        schema = bundle.schema

        # seq_topn은 time_col 필수
        time_col = schema.get("time_col", None)
        if not time_col:
            raise ValueError("seq_topn requires schema.time_col")

        time_col_str: str = time_col
        self._require_columns(bundle.train, [time_col_str], "train")

        # submission contract (topn 계열 공통)
        self._validate_submission_users(bundle)

        # sequential-specific meta
        user_seq = bundle.meta.get("user_seq", None)
        if user_seq is None:
            raise ValueError("seq_topn requires meta['user_seq']")

    # ------------------------
    # Meta validation helpers
    # ------------------------
    @staticmethod
    def _validate_submission_users(bundle: DataBundle) -> None:
        submission = bundle.meta.get("submission", None)
        if submission is None or not isinstance(submission, dict):
            raise ValueError("topn/seq_topn requires meta['submission'] dict")

        users = submission.get("users", None)
        if users is None or not isinstance(users, list):
            raise ValueError("meta['submission']['users'] must be a list")

    # ------------------------
    # DataFrame helper
    # ------------------------
    @staticmethod
    def _require_columns(df: pd.DataFrame, cols: List[str], where: str) -> None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{where} missing columns: {missing}")
