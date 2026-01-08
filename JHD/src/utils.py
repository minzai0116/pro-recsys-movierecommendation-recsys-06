"""유틸리티 함수들."""
import numpy as np
import random
import torch

def set_seed(seed: int = 42):
    """시드 고정."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

