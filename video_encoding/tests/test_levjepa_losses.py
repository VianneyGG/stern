from __future__ import annotations

import torch

from video_encoding.models.levjepa.losses import (
    invariance_loss,
    levjepa_loss,
    sigreg_loss,
)


def test_invariance_loss_zero_when_all_views_identical():
    z_global = torch.randn(4, 8)
    z_locals = z_global.unsqueeze(1).repeat(1, 3, 1)
    loss = invariance_loss(z_global, z_locals)
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)


def test_invariance_loss_positive_when_views_differ():
    torch.manual_seed(0)
    z_global = torch.randn(4, 8)
    z_locals = torch.randn(4, 3, 8)
    loss = invariance_loss(z_global, z_locals)
    assert loss.item() > 0


def test_sigreg_loss_low_for_true_gaussian_samples():
    torch.manual_seed(0)
    z = torch.randn(2048, 16)  # true N(0, I) samples
    loss = sigreg_loss(z, num_directions=256)
    assert loss.item() < 0.05


def test_sigreg_loss_high_for_collapsed_batch():
    """The core collapse-guarantee regression test: a batch where every
    embedding is identical (zero variance along every direction) must score
    far worse than true Gaussian samples."""
    z_collapsed = torch.ones(2048, 16) * 3.0
    z_gaussian = torch.randn(2048, 16)
    loss_collapsed = sigreg_loss(z_collapsed, num_directions=256)
    loss_gaussian = sigreg_loss(z_gaussian, num_directions=256)
    assert loss_collapsed.item() > 10 * loss_gaussian.item()


def test_sigreg_loss_is_nonnegative():
    z = torch.randn(64, 8)
    assert sigreg_loss(z, num_directions=32).item() >= 0.0


def test_levjepa_loss_combines_terms_with_fixed_lambda():
    torch.manual_seed(0)
    z_global = torch.randn(8, 16)
    z_locals = torch.randn(8, 4, 16)
    total, components = levjepa_loss(z_global, z_locals, lam=0.02, num_directions=64)
    expected = components["invariance"] + 0.02 * components["sigreg"]
    assert torch.allclose(total, expected, atol=1e-6)
    assert set(components) == {"invariance", "sigreg"}
