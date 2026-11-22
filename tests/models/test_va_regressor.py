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

from stern.models import VideoMAEVARegressor


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
    torch.manual_seed(0)
    model = VideoMAEVARegressor(_cfg(12), backbone=_StubBackbone(12))
    out = model(torch.randn(5, 3, 4, 8, 8))

    assert out.shape == (5, 2)
    assert torch.all(out >= -1.0) and torch.all(out <= 1.0)
    # Measured 0.15 at seed 0. A tanh pinned at +-1 would still pass the bound
    # check above, so also require the outputs to sit strictly inside it.
    assert out.abs().max().item() < 0.99


def test_forward_raises_on_embed_dim_mismatch():
    # backbone emits 16-d features but config claims 12 -> fail closed.
    model = VideoMAEVARegressor(_cfg(12), backbone=_StubBackbone(16))
    with pytest.raises(ValueError, match="feature dim"):
        model(torch.randn(2, 3, 4, 8, 8))


def test_regressor_can_freeze_backbone_after_pretraining():
    backbone = nn.Sequential(nn.Linear(3, 12))
    model = VideoMAEVARegressor(_cfg(12), backbone=backbone)

    model.freeze_backbone()

    assert all(not p.requires_grad for p in model.backbone.parameters())


def test_videomae_va_regressor_accepts_levjepa_encoder_as_backbone():
    """Proves the design's drop-in claim: VideoMAEVARegressor needs zero code
    changes to run on a LeVJEPAEncoder backbone, via the existing injectable
    `backbone=` seam and _backbone_tokens's hasattr fallback."""
    from omegaconf import OmegaConf

    from stern.models.levjepa.encoder import LeVJEPAEncoder

    model_cfg = OmegaConf.create(
        dict(
            num_frames=4,
            image_size=32,
            in_channels=3,
            patch_size=16,
            embed_dim=12,
            depth=2,
            num_heads=2,
            mlp_ratio=2.0,
            dropout=0.0,
            drop_path_rate=0.0,
            drop_ratio=0.0,
        )
    )
    backbone = LeVJEPAEncoder(model_cfg)
    regressor = VideoMAEVARegressor(model_cfg, backbone=backbone)
    regressor.freeze_backbone()
    x = torch.randn(2, 3, 4, 32, 32)
    out = regressor(x)
    assert out.shape == (2, 2)
    assert torch.all(out >= -1.0) and torch.all(out <= 1.0)


def test_head_is_bounded_unless_the_model_config_says_otherwise():
    default = VideoMAEVARegressor(_cfg(12), backbone=_StubBackbone(12))
    cfg = OmegaConf.create({"embed_dim": 12, "num_frames": 4, "head_bounded": False})
    unbounded = VideoMAEVARegressor(cfg, backbone=_StubBackbone(12))

    assert default.head.bounded is True
    assert unbounded.head.bounded is False


def test_unbounded_head_trains_past_where_tanh_saturates():
    # The ssl-diag-08 collapse: a pre-activation of ~7 pins tanh at +-1 and
    # zeroes the output's spread. Unbounded, the training output keeps it.
    cfg = OmegaConf.create({"embed_dim": 12, "num_frames": 4, "head_bounded": False})
    model = VideoMAEVARegressor(cfg, backbone=_StubBackbone(12))
    with torch.no_grad():
        model.head.unembed.bias.fill_(7.0)
    model.train()

    out = model(torch.randn(8, 3, 4, 8, 8))

    assert out.min() > 1.0
    assert out.std(dim=0).min() > 0.0
