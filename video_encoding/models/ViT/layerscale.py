from __future__ import annotations

import torch
from torch import nn


class LayerScale(nn.Module):
    """Learnable per-channel residual-branch scaling (Touvron et al. 2021,
    CaiT), initialised small so each block starts close to an identity
    residual and gradients from the untrained deep stack don't dominate
    early training."""

    def __init__(self, dim: int, init_value: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma
