from __future__ import annotations

import torch
from torch import nn


class DropPath(nn.Module):
    """Per-sample stochastic depth: drops the whole residual branch for a
    random subset of samples in the batch, scaling survivors by 1/keep so
    the expected output is unchanged (standard inverted-dropout scaling)."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, dtype=x.dtype, device=x.device)
        mask = torch.floor(mask + keep)  # binary: 1 w.p. keep, 0 w.p. drop_prob
        return x / keep * mask
