from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd


class GlobalTransform(ABC):
    """Step (2): raw 전체에 적용되는 전처리 (대개 stateless)."""

    name: str = "global_base"

    @abstractmethod
    def __call__(self, cfg: Any, raw: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        ...


class PostTransform(ABC):
    """
    Step (4): prepare 이후 전처리 (train에서 fit한 state를 valid/test에 적용).
    state는 '한 실행 내 메모리 전달'만 가정.
    """

    name: str = "post_base"

    @abstractmethod
    def fit(self, cfg: Any, raw: Dict[str, pd.DataFrame], train_df: pd.DataFrame) -> Any:
        ...

    @abstractmethod
    def transform(
        self,
        cfg: Any,
        raw: Dict[str, pd.DataFrame],
        df: pd.DataFrame,
        state: Any,
    ) -> pd.DataFrame:
        ...
