from __future__ import annotations

import pytest
import torch

from stern.models.vit.mlp import build_mlp


@pytest.mark.parametrize("mlp_type", ["gelu", "swiglu"])
def test_build_mlp_shape(mlp_type):
    mlp = build_mlp(mlp_type, embed_dim=8, mlp_ratio=2.0)
    x = torch.randn(3, 5, 8)
    out = mlp(x)
    assert out.shape == x.shape


def test_build_mlp_rejects_unknown_type():
    with pytest.raises(ValueError, match="unknown mlp_type"):
        build_mlp("nope", embed_dim=8, mlp_ratio=2.0)


def test_swiglu_hidden_dim_applies_the_two_thirds_rule():
    """SwiGLU has three projections where GELU has two, so a naive
    hidden = embed_dim * mlp_ratio gives it 1.5x the parameters. The standard
    2/3 correction keeps `mlp_type` a clean ablation axis instead of a
    capacity confound."""
    mlp = build_mlp("swiglu", embed_dim=768, mlp_ratio=4.0)
    assert mlp.gate.out_features == 2048  # int(768 * 4.0 * 2/3)
    assert mlp.up.out_features == 2048
    assert mlp.down.in_features == 2048


def test_swiglu_and_gelu_have_matched_weight_parameter_counts():
    """The property the 2/3 rule exists to guarantee: at equal mlp_ratio the
    two MLP variants must be capacity-matched, so an ablation measures the
    activation function rather than a bigger model."""
    embed_dim, mlp_ratio = 768, 4.0
    gelu = build_mlp("gelu", embed_dim=embed_dim, mlp_ratio=mlp_ratio)
    swiglu = build_mlp("swiglu", embed_dim=embed_dim, mlp_ratio=mlp_ratio)

    def weight_params(m):
        return sum(p.numel() for n, p in m.named_parameters() if n.endswith("weight"))

    assert weight_params(gelu) == 4_718_592  # 2 * 768 * 3072
    assert weight_params(swiglu) == 4_718_592  # 3 * 768 * 2048


@pytest.mark.parametrize("mlp_type", ["gelu", "swiglu"])
def test_gelu_and_swiglu_are_trainable(mlp_type):
    """Every weight must receive a non-zero gradient. SwiGLU's gate and up
    branches are the ones worth pinning: a wiring slip that dropped either
    from the product would still backprop to the input and still produce the
    right output shape."""
    torch.manual_seed(0)
    mlp = build_mlp(mlp_type, embed_dim=8, mlp_ratio=2.0)
    x = torch.randn(2, 3, 8, requires_grad=True)

    mlp(x).sum().backward()

    assert x.grad.shape == (2, 3, 8)
    assert x.grad.abs().max().item() > 0.0
    # `grad is None` is the unused-parameter case, all-zeros the
    # disconnected-but-reachable one; both mean the weight cannot train.
    dead = [
        n
        for n, p in mlp.named_parameters()
        if p.grad is None or p.grad.abs().max().item() == 0.0
    ]
    assert dead == []
