from __future__ import annotations

"""
MovieLens Seq Top-K 파이프라인 (v1).

입력:
- raw["ratings"]: pd.DataFrame (columns: user,item,time)
- (선택) raw["sample_submission"]: pd.DataFrame (columns: user,item)
- (선택) raw["item2attributes"]: dict[str, list[int]]

출력:
- DataBundle
  - train: ratings 그대로
  - test: 빈 DF(컬럼만 유지)  (Engine/Problem contract 맞추기용)
  - meta:
    - submission.users: 제출 대상 user 리스트(순서 포함)
      * sample_submission이 있으면 그 user 순서를 SSoT로 사용
    - submission.k: sample_submission에서 추정한 user당 추천 개수(K) (없으면 None)
    - user_seq: dict[user, list[item]] (time 정렬)
    - long_sequence: list[int] (pretrain negative segment sampling용)
    - item2attributes / attribute_size: (선택) S3Rec pretrain용

주요 cfg:
- cfg.dataset.{data_path,user_col,item_col,time_col,load_aux_tables,sample_submission_path}
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from src.data.data_bundle import DataBundle
from src.data.pipelines.base import DataPipelineBase
from src.data.loaders.ml_train_dir import load_ml_train_dir
from src.data.pipelines.registry import register_pipeline

@register_pipeline("seq_topn_ml_v1")
class SeqTopNMLV1Pipeline(DataPipelineBase):
    name = "seq_topn_ml_v1"

    def __init__(self, cfg: Any):
        super().__init__(cfg)

    def load_raw(self, cfg: Any) -> Dict[str, Any]:
        """입력(cfg)으로부터 raw dict를 생성합니다."""
        return load_ml_train_dir(cfg)


    def prepare_data(
        self, cfg: Any, raw: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame, Dict[str, Any]]:
        """raw -> (train_df, valid_df, test_df, meta) 로 변환합니다."""
        ratings: pd.DataFrame = raw["ratings"].copy()

        user_col = str(cfg.dataset.get("user_col", "user"))
        item_col = str(cfg.dataset.get("item_col", "item"))
        time_col = str(cfg.dataset.get("time_col", "time"))

        # train users (for building sequences / long_sequence)
        train_users: List[int] = ratings[user_col].drop_duplicates().tolist()

        # user_seq: user -> list[item] (time 정렬)
        ratings_sorted = ratings.sort_values([user_col, time_col], ascending=True)
        user_seq = ratings_sorted.groupby(user_col)[item_col].apply(list).to_dict()
        # long sequence (for pretraining negative segment sampling)
        long_sequence: List[int] = []
        for u in train_users:
            long_sequence.extend(user_seq.get(u, []) or [])

        # submission users order:
        # - 대회 템플릿 방식이면 sample_submission의 user 순서를 SSoT로 사용
        users: List[int] = train_users
        k_from_sample = None
        ss = raw.get("sample_submission", None)
        if isinstance(ss, pd.DataFrame) and (user_col in ss.columns):
            users = ss[user_col].drop_duplicates().tolist()
            try:
                vc = ss[user_col].value_counts()
                if len(vc) > 0:
                    k_from_sample = int(vc.iloc[0])
            except Exception:
                k_from_sample = None

        # optional item2attributes (for S3Rec pretraining tasks)
        item2attributes = raw.get("item2attributes", None)
        attribute_size = None
        if isinstance(item2attributes, dict) and item2attributes:
            try:
                mx = 0
                for _, attrs in item2attributes.items():
                    if not attrs:
                        continue
                    mx = max(mx, max(int(a) for a in attrs))
                # +1 for 0 padding
                attribute_size = int(mx) + 1
            except Exception:
                attribute_size = None

        tr = ratings.reset_index(drop=True)
        va = None

        # test는 Problem/Engine 계약상 항상 존재해야 하므로
        # 현재 스켈레톤에서는 "빈 DF(컬럼 유지)"로 채워 둠.
        te = ratings.iloc[0:0].copy().reset_index(drop=True)

        meta = {
            "pipeline": self.name,
            "submission": {"users": users, "k": k_from_sample},
            "user_seq": user_seq,
            "long_sequence": long_sequence,
            # optional: for pretraining
            "item2attributes": item2attributes,
            "attribute_size": attribute_size,
        }
        return tr, va, te, meta

    '''
    def get_post_transforms(self, cfg: Any) -> List[PostTransform]:
        # config로 조립하는 게 최종형이지만, 우선은 코드로 박아도 됨
        return [
            UserItemCountTransform(user_col="user_id", item_col="isbn"),
        ]
    '''

    def to_bundle(
        self,
        cfg: Any,
        train_df: pd.DataFrame,
        valid_df: Optional[pd.DataFrame],
        test_df: pd.DataFrame,
        meta: Dict[str, Any],
    ) -> DataBundle:
        user_col = str(cfg.dataset.get("user_col", "user"))
        item_col = str(cfg.dataset.get("item_col", "item"))
        time_col = str(cfg.dataset.get("time_col", "time"))

        schema = {
            "task": "seq_topn",
            "user_col": user_col,
            "item_col": item_col,
            "time_col": time_col,
            "target_col": None,
            "feature_cols": None,
        }
        return DataBundle(
            train=train_df,
            valid=valid_df,
            test=test_df,
            schema=schema,
            meta=meta,
        )
