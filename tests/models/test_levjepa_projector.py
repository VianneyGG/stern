from __future__ import annotations

import torch

from stern.models.levjepa.projector import Projector


def test_projector_output_shape():
    proj = Projector(embed_dim=768, hidden_dim=2048, output_dim=256)
    x = torch.randn(4, 768)
    out = proj(x)
    assert out.shape == (4, 256)
