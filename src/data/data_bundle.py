from __future__ import annotations

"""
DataBundle: DataPipeline → Engine 계약 객체.

입력:
- train/valid/test: pd.DataFrame
- schema: dict (task/user_col/item_col/time_col/target_col/...)
- meta: dict (free-form)

출력:
- Engine이 consume 가능한 표준 데이터 컨테이너
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Literal

import pandas as pd

TaskType = Literal["regression", "topn", "seq_topn"]


@dataclass(frozen=True)
class DataBundle:
    """
    Data → Engine contract (via Problem.run()).

    Required:
      - train/test: DataFrame
      - valid: DataFrame or None
      - schema: task/user_col/item_col/time_col/target_col (+ optional feature_cols)
      - meta: free-form
    """
    train: pd.DataFrame
    valid: Optional[pd.DataFrame]
    test: pd.DataFrame
    schema: Dict[str, Any]
    meta: Dict[str, Any]
