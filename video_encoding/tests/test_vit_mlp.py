from __future__ import annotations

import pytest
import torch

from video_encoding.models.ViT.mlp import MLPS, build_mlp


@pytest.mark.parametrize("mlp_type", ["gelu", "swiglu"])
def test_build_mlp_shape(mlp_type):
    mlp = build_mlp(mlp_type, embed_dim=8, mlp_ratio=2.0)
    x = torch.randn(3, 5, 8)
    out = mlp(x)
    assert out.shape == x.shape


def test_build_mlp_rejects_unknown_type():
    with pytest.raises(ValueError, match="unknown mlp_type"):
        build_mlp("nope", embed_dim=8, mlp_ratio=2.0)


def test_mlps_registry_has_expected_keys():
    assert set(MLPS.keys()) == {"gelu", "swiglu"}


def test_gelu_and_swiglu_are_trainable():
    for mlp_type in ("gelu", "swiglu"):
        mlp = build_mlp(mlp_type, embed_dim=8, mlp_ratio=2.0)
        x = torch.randn(2, 3, 8, requires_grad=True)
        out = mlp(x)
        out.sum().backward()
        assert x.grad is not None
