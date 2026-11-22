from __future__ import annotations

import pytest
import torch

from video_encoding.models.levjepa.losses import (
    invariance_loss,
    levjepa_loss,
    sigreg_empirical_floor,
    sigreg_loss,
)


def test_invariance_loss_zero_when_all_views_identical():
    z_global = torch.randn(4, 8)
    z_locals = z_global.unsqueeze(1).repeat(1, 3, 1)
    loss = invariance_loss(z_global, z_locals)
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)


def test_invariance_loss_is_a_per_dimension_mean_not_a_sum():
    """LeJEPA's `(center - proj).square().mean()` averages over the embedding
    dims too. Summing over K=256 inflated the invariance term 256x against
    SIGReg, so lam=0.02 no longer meant what the paper tuned it to mean."""
    z_global = torch.zeros(2, 256)
    z_locals = torch.ones(2, 3, 256)
    assert invariance_loss(z_global, z_locals).item() == pytest.approx(1.0)


# Golden values produced by running the upstream `EppsPulley().forward` (defaults
# t_max=3, n_points=17, world_size=1) from rbalestr-lab/lejepa @ c293d291,
# lejepa/univariate/epps_pulley.py, on the same inputs and projections.
@pytest.mark.parametrize(
    "batch_size, upstream",
    [(2, 0.7728983163833618), (16, 4.675853729248047), (128, 36.51273727416992)],
)
def test_sigreg_loss_matches_official_epps_pulley(batch_size, upstream):
    torch.manual_seed(1)
    z = 0.3 * torch.randn(batch_size, 32)
    torch.manual_seed(7)
    assert sigreg_loss(z, num_directions=64).item() == pytest.approx(upstream, rel=1e-5)


def test_sigreg_collapse_penalty_scales_with_batch_size():
    """The official statistic is multiplied by N: its value on N(0, I) stays
    O(1), while a collapsed batch's penalty grows linearly with N. Without the
    factor, SIGReg is ~N times too weak against the invariance term."""
    torch.manual_seed(0)
    small = sigreg_loss(0.05 * torch.randn(64, 16), num_directions=256)
    large = sigreg_loss(0.05 * torch.randn(256, 16), num_directions=256)
    assert large.item() / small.item() == pytest.approx(4.0, rel=0.05)


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


def test_sigreg_empirical_floor_is_batch_size_independent():
    """The N factor cancels the O(1/N) finite-sample bias of the empirical
    characteristic function, so the floor on true N(0, I) samples is an O(1)
    constant rather than shrinking with batch size."""
    torch.manual_seed(0)
    small = sigreg_empirical_floor(
        batch_size=16, dim=8, num_directions=256, num_trials=8
    )
    large = sigreg_empirical_floor(
        batch_size=1024, dim=8, num_directions=256, num_trials=8
    )
    assert 0.3 < small < 3.0
    assert large == pytest.approx(small, rel=0.3)


def test_levjepa_loss_combines_terms_with_fixed_lambda():
    torch.manual_seed(0)
    z_global = torch.randn(8, 16)
    z_locals = torch.randn(8, 4, 16)
    total, components = levjepa_loss(z_global, z_locals, lam=0.02, num_directions=64)
    expected = (1 - 0.02) * components["invariance"] + 0.02 * components["sigreg"]
    assert torch.allclose(total, expected, atol=1e-6)
    assert set(components) == {"invariance", "sigreg"}
