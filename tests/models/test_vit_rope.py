from __future__ import annotations

import pytest
import torch

from stern.models.vit.rope import Rotary3D, apply_rotary


def test_rotary_rejects_odd_head_dim():
    with pytest.raises(ValueError, match="even"):
        Rotary3D(head_dim=11)


def test_rotary_accepts_head_dim_not_divisible_by_six():
    """ViT-B geometry (embed_dim=768, num_heads=12) gives head_dim=64, which
    is not divisible by 6 -- the axis split must tolerate uneven (but each
    still even, for rotate-half pairing) group sizes."""
    head_dim = 64
    rope = Rotary3D(head_dim=head_dim)
    coords = torch.randint(0, 8, (1, 3, 3), dtype=torch.long)
    cos, sin = rope(coords)
    assert cos.shape == (1, 3, head_dim)
    assert sin.shape == (1, 3, head_dim)


def test_rotary_zero_coords_is_identity_rotation():
    head_dim = 12
    rope = Rotary3D(head_dim=head_dim)
    coords = torch.zeros(1, 3, 3, dtype=torch.long)
    cos, sin = rope(coords)
    assert cos.shape == (1, 3, head_dim)
    assert sin.shape == (1, 3, head_dim)
    assert torch.allclose(cos, torch.ones_like(cos))
    assert torch.allclose(sin, torch.zeros_like(sin), atol=1e-6)


def test_apply_rotary_preserves_shapes():
    head_dim = 12
    q = torch.randn(2, 4, 5, head_dim)
    k = torch.randn(2, 4, 5, head_dim)
    rope = Rotary3D(head_dim=head_dim)
    coords = torch.randint(0, 8, (2, 5, 3), dtype=torch.long)
    cos, sin = rope(coords)
    q_out, k_out = apply_rotary(q, k, cos.unsqueeze(1), sin.unsqueeze(1))
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape


def test_rotary_relative_position_invariance():
    """Dot product of a rotated (q, k) pair depends only on the *difference*
    between their coordinates, not their absolute position -- the defining
    property of RoPE. Shifting both tokens' coordinates by the same offset
    must leave q . k unchanged."""
    head_dim = 12
    torch.manual_seed(0)
    q = torch.randn(1, 1, 2, head_dim)
    k = torch.randn(1, 1, 2, head_dim)
    rope = Rotary3D(head_dim=head_dim)

    coords_a = torch.tensor([[[0, 0, 0], [1, 2, 3]]], dtype=torch.long)
    offset = torch.tensor([2, 1, 4], dtype=torch.long)
    coords_b = coords_a + offset

    cos_a, sin_a = rope(coords_a)
    cos_b, sin_b = rope(coords_b)
    q_a, k_a = apply_rotary(q, k, cos_a.unsqueeze(1), sin_a.unsqueeze(1))
    q_b, k_b = apply_rotary(q, k, cos_b.unsqueeze(1), sin_b.unsqueeze(1))

    dot_a = (q_a[:, :, 0] * k_a[:, :, 1]).sum()
    dot_b = (q_b[:, :, 0] * k_b[:, :, 1]).sum()
    assert torch.allclose(dot_a, dot_b, atol=1e-4)
