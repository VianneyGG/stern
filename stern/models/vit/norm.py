"""Normalisation layers for ViT blocks, selected by `norm_type`."""

from __future__ import annotations

import torch
from torch import nn


class CenteredNorm(nn.Module):
    """LayerNorm without the variance normalisation: centers each token's
    feature vector (subtracts its mean) but does not rescale by its standard
    deviation."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        centered = x - x.mean(dim=-1, keepdim=True)
        return centered * self.weight + self.bias


NORMS: dict[str, type[nn.Module]] = {
    "layernorm": nn.LayerNorm,
    "rmsnorm": nn.RMSNorm,  # ships in torch >= 2.4, no need to hand-roll it
    "centerednorm": CenteredNorm,
}


def build_norm(norm_type: str, dim: int) -> nn.Module:
    if norm_type not in NORMS:
        raise ValueError(f"unknown norm_type {norm_type!r}; choices: {sorted(NORMS)}")
    return NORMS[norm_type](dim)
