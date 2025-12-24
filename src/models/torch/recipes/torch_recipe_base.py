from __future__ import annotations

"""
Torch 레시피(Recipe) 기본 인터페이스.

입력:
- cfg: OmegaConf(DictConfig) 형태를 전제 (train.*, model, model_args, recipe_args 등)
- bundle: DataBundle (Problem.run() 결과)

출력(계약):
- train_step(): {"loss": torch.Tensor} (스칼라 loss)
- predict_step(): task별 preds 조각
  - seq_topn/topn: List[List[int]]
  - regression: np.ndarray 또는 torch.Tensor 등(Engine이 merge 후 validate)

비고:
- TorchBaseEngine이 이 인터페이스를 호출해 학습/예측 루프를 구성합니다.
"""

from abc import ABC, abstractmethod
from typing import Any
import torch


class TorchRecipeBase(ABC):
    """
    입력/출력(Contract) 중심 설계:
    - 입력(cfg):
      * cfg.train.*: 공통 학습 하이퍼(epochs, batch_size, lr, ...)
      * cfg.model: 모델 키(문자열)
      * cfg.model_args[cfg.model]: 모델 구조 하이퍼
      * (선택) cfg.recipe_args: 레시피 전용 하이퍼
    """

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self._validate_cfg_contract()

    def _validate_cfg_contract(self) -> None:
        if not hasattr(self.cfg, "train"):
            raise KeyError("Missing cfg.train (train.* is required)")
        if not hasattr(self.cfg, "model"):
            raise KeyError("Missing cfg.model")
        if not hasattr(self.cfg, "model_args"):
            raise KeyError("Missing cfg.model_args")
        if self.cfg.model not in self.cfg.model_args:
            raise KeyError(f"Missing cfg.model_args['{self.cfg.model}']")

    def train_cfg(self):
        """입력: cfg.train.* / 출력: 학습 하이퍼 객체(DictConfig)."""
        return self.cfg.train

    def model_cfg(self):
        """입력: cfg.model, cfg.model_args / 출력: cfg.model_args[cfg.model]."""
        return self.cfg.model_args[self.cfg.model]

    @abstractmethod
    def build_model(self, cfg, bundle): ...
    @abstractmethod
    def build_optimizer(self, cfg, model): ...
    @abstractmethod
    def build_loaders(self, cfg, bundle): ...
    @abstractmethod
    def train_step(self, cfg, batch, model): ...
    @abstractmethod
    def predict_step(self, cfg, batch, model): ...

    def move_batch_to_device(self, batch, device):
        if torch.is_tensor(batch):
            return batch.to(device)
        if isinstance(batch, list):
            return [self.move_batch_to_device(b, device) for b in batch]
        if isinstance(batch, tuple):
            return tuple(self.move_batch_to_device(b, device) for b in batch)
        if isinstance(batch, dict):
            return {k: self.move_batch_to_device(v, device) for k, v in batch.items()}
        return batch
