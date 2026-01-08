from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from src.data.data_bundle import DataBundle


class RecBoleRecipeBase(ABC):
    """
    RecBoleEngine 전용 Recipe 계약.
    """

    name: str = "base"

    def __init__(self, cfg: Any):
        self.cfg = cfg

    @abstractmethod
    def prepare_dataset(
        self,
        bundle: DataBundle,
        *,
        data_root: str,
        dataset: str,
        setting,
    ):
        """bundle -> atomic files. return DatasetSpec"""

    @abstractmethod
    def build_overrides(
        self,
        bundle: DataBundle,
        *,
        data_root: str,
        dataset: str,
    ) -> Dict[str, Any]:
        """RecBole Config(config_dict)에 넣을 override dict 생성"""
