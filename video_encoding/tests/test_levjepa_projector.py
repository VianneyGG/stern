from __future__ import annotations

import pytest
import torch

from video_encoding.models.levjepa.projector import Projector


def test_projector_output_shape():
    proj = Projector(embed_dim=768, hidden_dim=2048, output_dim=256)
    x = torch.randn(4, 768)
    out = proj(x)
    assert out.shape == (4, 256)


def test_projector_uses_batchnorm_and_gelu():
    proj = Projector(embed_dim=32, hidden_dim=64, output_dim=8)
    modules = [type(m).__name__ for m in proj.modules()]
    assert "BatchNorm1d" in modules
    assert "GELU" in modules


def test_projector_rejects_wrong_input_dim():
    proj = Projector(embed_dim=32, hidden_dim=64, output_dim=8)
    with pytest.raises(RuntimeError):
        proj(torch.randn(4, 16))
