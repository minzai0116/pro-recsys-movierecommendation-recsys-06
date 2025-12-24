# src/engines/sklearn/sklearn_topn_engine.py
from __future__ import annotations

from typing import Optional
from src.data.data_bundle import DataBundle
from src.engines.sklearn.sklearn_base import SklearnEngineBase

class SklearnTopNEngine(SklearnEngineBase):
    def fit(self, bundle: DataBundle) -> None:
        raise NotImplementedError("SklearnTopNEngine will be implemented when Top-N protocol is defined.")

    def predict(self, bundle: DataBundle, checkpoint: Optional[str] = None):
        raise NotImplementedError("SklearnTopNEngine will be implemented when Top-N protocol is defined.")
