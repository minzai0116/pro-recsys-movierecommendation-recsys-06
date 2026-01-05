# src/engines/sklearn/sklearn_regression_engine.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.data.data_bundle import DataBundle
from src.engines.sklearn.sklearn_base import SklearnEngineBase
from src.utils.cfg_utils import cfg_select

class SklearnRegressionEngine(SklearnEngineBase):
    """
    sklearn 계열(비딥러닝) 회귀/랭킹 엔진.
    - 예: CatBoost Ranker 기반 2-stage 재랭킹

    Contract:
    - fit(DataBundle) -> checkpoint 저장
    - predict(DataBundle) -> (topn/seq_topn)일 때 List[List[item_id]] 반환
    """

    def fit(self, bundle: DataBundle) -> Dict[str, Any]:
        model_name = self._model_name()
        mcfg = self._model_args(model_name)

        # 1. Prepare Recipe
        recipe = self._get_recipe(model_name)
        
        # 2. Fit
        print(f"[Sklearn] 학습 시작 | model={model_name}, recipe={recipe.name}")
        
        # Recipe handles feature engineering and model fitting
        result = recipe.fit(bundle, self)
        
        ckpt_path = self._default_ckpt_path(f"{model_name}.joblib")
        self._save_checkpoint(
            {
                "model": model_name,
                "recipe_name": recipe.name,
                "ckpt": result["model"],
                "schema": dict(bundle.schema),
                "feature_cols": result.get("feature_cols"),
                "meta": result.get("meta", {}),
            },
            ckpt_path,
        )

        self._log_train({
            "engine": "sklearn_regression",
            "model": model_name,
            "checkpoint_saved": ckpt_path,
            **result.get("metrics", {})
        })

        return {"checkpoint_path": ckpt_path}

    def predict(self, bundle: DataBundle, checkpoint: Optional[str] = None) -> Any:
        model_name = self._model_name()
        ckpt_path = self._resolve_checkpoint(checkpoint, f"{model_name}.joblib")
        obj = self._load_checkpoint(ckpt_path)
        
        recipe = self._get_recipe(model_name)
        
        print(f"[Sklearn] 예측 시작 | model={model_name}, recipe={recipe.name}")
        preds = recipe.predict(obj["ckpt"], bundle, self)
        
        # If it's a ranking task, preds should be List[List[int]]
        if bundle.schema.get("task") in ("topn", "seq_topn"):
            self._validate_preds(preds, bundle)

        self._log_predict({
            "model": model_name,
            "checkpoint_used": ckpt_path,
        })
        
        return preds

    # ---------- helpers ----------
    def _model_name(self) -> str:
        m = getattr(self.cfg, "model", None)
        if m is None: return "model"
        name = getattr(m, "name", None)
        return str(name) if name is not None else str(m)

    def _model_args(self, model_name: str) -> Dict[str, Any]:
        margs = getattr(self.cfg, "model_args", {}) or {}
        if isinstance(margs, dict) and model_name in margs:
            return dict(margs[model_name] or {})
        return {}

    def _get_recipe(self, model_name: str):
        from src.models.sklearn.recipes.registry import SKLEARN_RECIPE_REGISTRY
        # Fallback to model name if recipe not explicitly provided
        recipe_key = cfg_select(self.cfg, "recipe", default=model_name)
        if recipe_key not in SKLEARN_RECIPE_REGISTRY:
            # Try finding a generic regression recipe if specific one doesn't exist
            if "catboost" in model_name.lower():
                recipe_key = "catboost_ranker"
            else:
                raise ValueError(f"No sklearn recipe registered for {recipe_key} or {model_name}")
        
        return SKLEARN_RECIPE_REGISTRY[recipe_key](self.cfg)

