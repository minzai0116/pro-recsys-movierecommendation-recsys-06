# src/data/pipelines/random_masked_ml_v1.py
from __future__ import annotations

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional

from src.data.data_bundle import DataBundle
from src.data.pipelines.base import DataPipelineBase
from src.data.pipelines.registry import register_pipeline
from src.utils.cfg_utils import cfg_select

@register_pipeline("random_masked_ml_v1")
class RandomMaskedPipeline(DataPipelineBase):
    """
    랜덤 마스킹 기반 split 파이프라인.
    - 최초 1회 split 결과를 cache_path에 저장하고,
    - 이후 실행은 동일 cache를 재사용하여 모델 실행 시점이 달라도 동일한 마스킹을 보장합니다.

    Base Protocol(DataPipelineBase.build)을 따릅니다:
      load_raw -> global transforms -> prepare_data -> post transforms -> to_bundle
    """

    def load_raw(self, cfg: Any) -> Dict[str, Any]:
        from src.data.loaders.ml_train_dir import load_ml_train_dir
        return load_ml_train_dir(cfg)

    def prepare_data(
        self, cfg: Any, raw: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame, Dict[str, Any]]:
        # 1. Check for Cache at the start of prepare_data
        cache_dir = cfg_select(cfg, "dataset.cache_dir", default="saved/data")
        cache_name = cfg_select(cfg, "dataset.cache_name", default="random_split_v1.pkl")
        cache_path = os.path.join(cache_dir, cache_name)

        if os.path.exists(cache_path):
            print(f"[Pipeline] 캐시 로드 | path={cache_path}")
            bundle = joblib.load(cache_path)
            # Ensure test_df is at least an empty DataFrame
            test_df = bundle.test if bundle.test is not None else pd.DataFrame()
            return bundle.train, bundle.valid, test_df, bundle.meta

        # 2. If no cache, perform random masking
        print("[Pipeline] 캐시 없음: 랜덤 마스킹 split 생성")
        df = raw["ratings"]
        user_col, item_col, time_col = "user", "item", "time"

        # 결정론적 split (seed 고정)
        seed = cfg_select(cfg, "seed", default=42)
        np.random.seed(seed)
        
        df = df.sort_values([user_col, time_col]).reset_index(drop=True)
        mask_ratio = cfg_select(cfg, "dataset.mask_ratio", default=0.1)
        
        valid_indices = []
        for _, group in df.groupby(user_col):
            n = len(group)
            if n > 5:
                n_mask = max(1, int(n * mask_ratio))
                idx = np.random.choice(group.index[:-1], size=n_mask, replace=False)
                valid_indices.extend(idx)
        
        valid_df = df.loc[valid_indices].copy()
        train_df = df.drop(valid_indices).copy()
        test_df = pd.DataFrame()  # seq_topn 파이프라인 호환을 위한 빈 test

        # Engine/Exporter 호환을 위해 target_col(label)을 명시적으로 추가
        train_df['label'] = 1.0
        valid_df['label'] = 1.0

        # meta 생성 (다른 파이프라인 인스턴스화/순환참조 없이 자체 생성)
        meta = self._build_meta(df, raw.get("sample_submission"))

        return train_df, valid_df, test_df, meta

    def to_bundle(
        self,
        cfg: Any,
        train_df: pd.DataFrame,
        valid_df: Optional[pd.DataFrame],
        test_df: Optional[pd.DataFrame],
        meta: Dict[str, Any],
    ) -> DataBundle:
        # Get column names from config or fallback to standard
        user_col = cfg_select(cfg, "dataset.user_column", default="user")
        item_col = cfg_select(cfg, "dataset.item_column", default="item")
        time_col = cfg_select(cfg, "dataset.time_column", default="time")

        bundle = DataBundle(
            train=train_df,
            valid=valid_df,
            test=test_df,
            schema={
                "user_col": user_col,
                "item_col": item_col,
                "time_col": time_col,
                "target_col": "label", # Added to satisfy problem validation
                "task": "seq_topn"
            },
            meta=meta
        )

        # 최종 번들 캐시 저장
        cache_dir = cfg_select(cfg, "dataset.cache_dir", default="saved/data")
        cache_name = cfg_select(cfg, "dataset.cache_name", default="random_split_v1.pkl")
        cache_path = os.path.join(cache_dir, cache_name)
        
        if not os.path.exists(cache_path):
            os.makedirs(cache_dir, exist_ok=True)
            joblib.dump(bundle, cache_path)
            print(f"[Pipeline] 캐시 저장 | path={cache_path}")
            
        return bundle

    def _build_meta(self, df: pd.DataFrame, sample_submission: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """제출/평가에 필요한 meta를 생성합니다."""
        meta: Dict[str, Any] = {}
        
        # 1. User Sequence (for various models)
        user_seq = df.sort_values(["user", "time"]).groupby("user")["item"].apply(list).to_dict()
        meta["user_seq"] = user_seq
        
        # 2. Submission Target Users
        if sample_submission is not None:
            target_users = sample_submission["user"].unique().tolist()
            meta["submission"] = {
                "users": target_users,
                "k": 10
            }
        else:
            meta["submission"] = {
                "users": sorted(df["user"].unique().tolist()),
                "k": 10
            }
            
        return meta
