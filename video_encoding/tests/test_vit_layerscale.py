from __future__ import annotations

import torch

from video_encoding.models.ViT.layerscale import LayerScale


def test_layerscale_scales_by_init_value():
    ls = LayerScale(dim=8, init_value=0.1)
    x = torch.ones(2, 3, 8)
    out = ls(x)
    assert torch.allclose(out, torch.full_like(x, 0.1))


def test_layerscale_gamma_is_trainable():
    ls = LayerScale(dim=8, init_value=1e-5)
    x = torch.randn(2, 3, 8, requires_grad=True)
    out = ls(x)
    out.sum().backward()
    assert ls.gamma.grad.shape == ls.gamma.shape
