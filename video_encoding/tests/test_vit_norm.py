from __future__ import annotations

import pytest
import torch

from video_encoding.models.ViT.norm import NORMS, CenteredNorm, RMSNorm, build_norm


@pytest.mark.parametrize("norm_type", ["layernorm", "rmsnorm", "centerednorm"])
def test_build_norm_shape(norm_type):
    norm = build_norm(norm_type, dim=8)
    x = torch.randn(3, 5, 8)
    out = norm(x)
    assert out.shape == x.shape


def test_build_norm_rejects_unknown_type():
    with pytest.raises(ValueError, match="unknown norm_type"):
        build_norm("nope", dim=8)


def test_rmsnorm_normalises_to_unit_rms():
    norm = RMSNorm(dim=8)
    x = torch.randn(4, 8) * 10.0  # large scale, should still normalise
    out = norm(x)
    rms = out.pow(2).mean(dim=-1).sqrt()
    # weight starts at all-ones, so output RMS should be ~1
    assert torch.allclose(rms, torch.ones(4), atol=1e-4)


def test_centerednorm_centers_without_rescaling_variance():
    norm = CenteredNorm(dim=8)
    x = torch.randn(4, 8) * 10.0
    out = norm(x)
    # weight=1, bias=0 at init -> output mean per row should be ~0
    assert torch.allclose(out.mean(dim=-1), torch.zeros(4), atol=1e-4)
    # but variance is NOT normalised to 1 (unlike LayerNorm) -- centering only
    assert out.std(dim=-1).mean().item() > 2.0  # input std was ~10, stays large


def test_norms_registry_has_expected_keys():
    assert set(NORMS.keys()) == {"layernorm", "rmsnorm", "centerednorm"}
