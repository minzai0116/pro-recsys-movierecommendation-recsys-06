# src/models/torch/collates/regression.py
import numpy as np
import torch

def regression_collate(batch):
    if isinstance(batch[0], tuple):
        X, y = zip(*batch)
        return (
            torch.tensor(np.stack(X)),
            torch.tensor(np.array(y)),
        )
    return torch.tensor(np.stack(batch))
