from __future__ import annotations

import torch
from torch import nn


class Projector(nn.Module):
    """Linear -> BatchNorm1d -> GELU -> Linear head mapping [cls] embeddings
    into the SIGReg/invariance objective's embedding space."""

    def __init__(self, embed_dim: int, hidden_dim: int = 2048, output_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
