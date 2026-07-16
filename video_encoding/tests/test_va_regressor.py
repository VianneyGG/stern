"""VideoMAEVARegressor: shape/bounds + embed-dim guard, with a stub backbone.

A stub backbone is injected to keep this test focused on the composition
(pool -> head -> tanh) and the dim-mismatch guard; the real from-scratch encoder
is covered in tests/test_videomae_encoder.py.
"""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from video_encoding.models import VideoMAEVARegressor


class _StubBackbone(nn.Module):
    """Maps (B, C, T, H, W) -> (B, out_dim), like the pooled v2 feature."""

    def __init__(self, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(3, out_dim)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.lin(pixel_values.mean(dim=[2, 3, 4]))


def _cfg(embed_dim: int):
    return OmegaConf.create({"embed_dim": embed_dim, "num_frames": 4})


def test_forward_returns_bounded_va_pairs():
    model = VideoMAEVARegressor(_cfg(8), backbone=_StubBackbone(8))
    out = model(torch.randn(5, 3, 4, 8, 8))

    assert out.shape == (5, 2)
    assert torch.all(out >= -1.0) and torch.all(out <= 1.0)


def test_forward_raises_on_embed_dim_mismatch():
    # backbone emits 16-d features but config claims 8 -> fail closed.
    model = VideoMAEVARegressor(_cfg(8), backbone=_StubBackbone(16))
    with pytest.raises(ValueError, match="feature dim"):
        model(torch.randn(2, 3, 4, 8, 8))
