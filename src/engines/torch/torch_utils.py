from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import torch


def move_to_device(x: Any, device: torch.device) -> Any:
    """
    Recursively move tensors in common container types to device.
    Does NOT interpret semantics; just structure.
    """
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)

    if isinstance(x, np.ndarray):
        # leave numpy arrays as-is; recipe/collate should tensorize if needed
        return x

    if isinstance(x, Mapping):
        return {k: move_to_device(v, device) for k, v in x.items()}

    if isinstance(x, tuple):
        return tuple(move_to_device(v, device) for v in x)

    if isinstance(x, list):
        return [move_to_device(v, device) for v in x]

    return x


class AverageMeter:
    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * int(n)
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)
