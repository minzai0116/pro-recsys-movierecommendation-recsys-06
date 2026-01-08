# src/models/sklearn/recipes/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


class SklearnRegressionRecipeSpec:
    """
    Recipe-first regression spec.
    Engine(=task-family runner)은 모델명을 모르고, 이 인터페이스만 호출한다.
    """
    name: str = "base"

    def __init__(self, cfg: Any):
        self.cfg = cfg

    # --- schema/feature contract ---
    def prepare_schema(self, bundle) -> Tuple[List[str], str]:
        target_col = bundle.schema.get("target_col")
        if not target_col:
            raise ValueError("Regression requires bundle.schema['target_col']")
        feature_cols = bundle.schema.get("feature_cols")
        if not feature_cols:
            feature_cols = [c for c in bundle.train.columns if c != target_col]
        return list(feature_cols), str(target_col)

    # --- pipeline builders ---
    def build_preprocessor(self, bundle, feature_cols: List[str]):
        """
        Default: OHE for categorical + median/mode impute.
        CatBoost 같은 모델은 override해서 passthrough/None로 바꾼다.
        """
        meta = bundle.meta or {}
        cat_cols = meta.get("categorical_features", None)
        num_cols = meta.get("numeric_features", None)

        if cat_cols is None or num_cols is None:
            sample = bundle.train[feature_cols]
            inferred_cat = [c for c in feature_cols if sample[c].dtype == "object"]
            inferred_num = [c for c in feature_cols if c not in inferred_cat]
            cat_cols = inferred_cat if cat_cols is None else cat_cols
            num_cols = inferred_num if num_cols is None else num_cols

        cat_cols = [c for c in cat_cols if c in feature_cols]
        num_cols = [c for c in num_cols if c in feature_cols]

        cat_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ])
        num_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ])

        return ColumnTransformer(
            transformers=[
                ("cat", cat_pipe, cat_cols),
                ("num", num_pipe, num_cols),
            ],
            remainder="drop",
        )

    def build_estimator(self, cfg: Any, bundle):
        raise NotImplementedError

    def configure_estimator(self, bundle, estimator, feature_cols: List[str]):
        # data-dependent configuration hook (e.g., CatBoost cat_features)
        return estimator

    def fit_params(self, cfg: Any, bundle) -> Dict[str, Any]:
        return {}

    def predict(self, pipeline, X_test, bundle) -> np.ndarray:
        return pipeline.predict(X_test)
