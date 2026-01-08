# src/engines/core/engine_base.py
from __future__ import annotations

"""
Engine 추상 인터페이스.

입력:
- cfg: OmegaConf(DictConfig)
- bundle: DataBundle

출력(Contract):
- fit(bundle) -> (엔진별) 학습 결과/체크포인트 정보
- predict(bundle, checkpoint=...) -> preds (task별 contract 준수)
"""

from typing import Any, Optional, Dict
from abc import ABC, abstractmethod

from src.data.data_bundle import DataBundle


class EngineBase(ABC):
    def __init__(self, cfg: Any, logger, setting):
        self.cfg = cfg
        self.logger = logger
        self.setting = setting

    @abstractmethod
    def fit(self, data_bundle: DataBundle) -> Dict[str, Any]:
        """
        입력:
        - data_bundle: DataBundle

        출력(Contract):
        - dict with at least:
          - checkpoint_path: Optional[str]
        """
        ...

    @abstractmethod
    def predict(self, data_bundle: DataBundle, checkpoint: Optional[str] = None):
        ...
