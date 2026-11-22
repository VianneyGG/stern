"""The arm-invariant reconstruction metric.

The four factorial arms do not optimise the same loss, so the only valid
cross-arm number is one computed identically for all of them. These tests pin
both ends of its scale and guard the two bugs already made in this
investigation.
"""

import torch
from omegaconf import DictConfig

from stern.models.videomae.mae import VideoMAEForPreTraining
from stern.evaluation.diagnostics.reconstruction import (
    collapse_to_time_avg_z,
    holdout_mse_z,
    patch_stats,
    temporal_split,
    temporal_correlation,
    temporal_variance_ratio,
)

GRID = (4, 2, 2)
# The temporal metrics are statistics over t_p * cols * D values; the small GRID
# leaves too few for a noise correlation to be reliably near zero.
BIG_GRID = (8, 6, 6)


def _tiny_cfg(**overrides) -> DictConfig:
    base = {
        "num_frames": 8,
        "image_size": 32,
        "in_channels": 3,
        "tubelet_size": 2,
        "patch_size": 16,
        "embed_dim": 32,
        "depth": 1,
        "num_heads": 4,
        "mlp_ratio": 4.0,
        "dropout": 0.0,
        "decoder_depth": 1,
        "decoder_embed_dim": 16,
        "decoder_num_heads": 4,
        "mask_ratio": 0.5,
    }
    base.update(overrides)
    return DictConfig(base)


class TestTemporalSplit:
    def test_partitions_by_the_cut(self):
        keep, mask, cut = temporal_split(GRID, torch.device("cpu"), keep_frac=0.5)
        n_sp = GRID[1] * GRID[2]
        assert cut == 2
        assert torch.equal(keep, torch.arange(cut * n_sp))
        assert torch.equal(mask, torch.arange(cut * n_sp, GRID[0] * n_sp))

    def test_is_complete_and_disjoint(self):
        keep, mask, _ = temporal_split(GRID, torch.device("cpu"))
        n = GRID[0] * GRID[1] * GRID[2]
        assert torch.equal(torch.sort(torch.cat([keep, mask]))[0], torch.arange(n))


class TestZeroPredictionIsArmInvariant:
    """A model emitting zeros predicts each patch's own mean, so its z-space MSE
    sits at the fixed zero-point (D-1)/D. Both arms must land on that same
    number, otherwise their MSEs are on different scales and cannot be compared.

    Not exactly 1.0: _patchify_targets z-scores with torch.std, which is
    Bessel-corrected, so mean(z**2) = (D-1)/D over the D=3*p*p target vector.
    """

    ZERO_POINT = (3 * 16 * 16 - 1) / (3 * 16 * 16)

    def _zeroed(self, cfg):
        model = VideoMAEForPreTraining(cfg).eval()
        torch.nn.init.zeros_(model.predict_head.weight)
        torch.nn.init.zeros_(model.predict_head.bias)
        return model

    def _both(self, pixel_values):
        return (
            holdout_mse_z(self._zeroed(_tiny_cfg()), pixel_values, torch.device("cpu")),
            holdout_mse_z(
                self._zeroed(_tiny_cfg(target_mode="per_frame")),
                pixel_values,
                torch.device("cpu"),
            ),
        )

    def test_both_modes_score_the_zero_point(self):
        avg, per = self._both(torch.randn(1, 3, 8, 32, 32))
        assert abs(avg - self.ZERO_POINT) < 1e-4
        assert abs(per - self.ZERO_POINT) < 1e-4

    def test_the_two_arms_agree(self):
        """The invariance property itself, independent of what the zero-point is."""
        avg, per = self._both(torch.randn(1, 3, 8, 32, 32))
        assert abs(avg - per) < 1e-6


class TestCollapse:
    def test_time_avg_passes_through_unchanged(self):
        pred = torch.randn(1, 16, 3 * 16 * 16)
        out = collapse_to_time_avg_z(
            pred,
            target_mode="time_avg",
            tubelet_size=2,
            sigma_per_frame=torch.ones(1, 16, 1),
            sigma_time_avg=torch.ones(1, 16, 1),
        )
        assert torch.equal(out, pred)

    def test_per_frame_collapses_and_rescales(self):
        """mu is identical in both layouts (same values, same weights), so the
        collapse is a frame-average rescaled by the sigma ratio."""
        d = 3 * 16 * 16
        pred = torch.randn(1, 16, 2 * d)
        s_pf = torch.full((1, 16, 1), 2.0)
        s_ta = torch.full((1, 16, 1), 1.0)
        out = collapse_to_time_avg_z(
            pred,
            target_mode="per_frame",
            tubelet_size=2,
            sigma_per_frame=s_pf,
            sigma_time_avg=s_ta,
        )
        expected = pred.view(1, 16, 2, d).mean(dim=2) * 2.0
        torch.testing.assert_close(out, expected)
        assert out.shape == (1, 16, d)

    def test_zero_maps_to_zero(self):
        """The property the arm-invariance rests on."""
        out = collapse_to_time_avg_z(
            torch.zeros(1, 16, 2 * 3 * 16 * 16),
            target_mode="per_frame",
            tubelet_size=2,
            sigma_per_frame=torch.rand(1, 16, 1) + 0.5,
            sigma_time_avg=torch.rand(1, 16, 1) + 0.5,
        )
        assert torch.count_nonzero(out) == 0


class TestPatchStats:
    def test_mu_is_identical_across_layouts(self):
        """collapse_to_time_avg_z assumes mu_per_frame == mu_time_avg. If that
        ever stops holding, the collapse silently acquires an offset."""
        pixel_values = torch.randn(1, 3, 8, 32, 32)
        _, mu_a, _, _ = patch_stats(pixel_values, 2, 16, per_frame=False)
        _, mu_p, _, _ = patch_stats(pixel_values, 2, 16, per_frame=True)
        torch.testing.assert_close(mu_a, mu_p)

    def test_per_frame_sigma_is_at_least_time_avg_sigma(self):
        """Averaging over the tubelet can only remove variance."""
        pixel_values = torch.randn(1, 3, 8, 32, 32)
        _, _, s_a, _ = patch_stats(pixel_values, 2, 16, per_frame=False)
        _, _, s_p, _ = patch_stats(pixel_values, 2, 16, per_frame=True)
        assert bool((s_p >= s_a - 1e-6).all())


class TestTemporalVarianceRatio:
    def test_a_temporally_flat_prediction_scores_zero(self):
        """The regression test for the pixel-space contamination bug: measured in
        pixel space a noise-emitting model scored 0.538 here."""
        n_sp = GRID[1] * GRID[2]
        target = torch.randn(1, GRID[0] * n_sp, 8)
        flat = torch.ones(1, GRID[0] * n_sp, 8)
        mask = torch.arange(GRID[0] * n_sp)
        assert temporal_variance_ratio(flat, target, GRID, mask) < 1e-6

    def test_a_perfect_prediction_scores_one(self):
        n_sp = GRID[1] * GRID[2]
        target = torch.randn(1, GRID[0] * n_sp, 8)
        mask = torch.arange(GRID[0] * n_sp)
        assert abs(temporal_variance_ratio(target, target, GRID, mask) - 1.0) < 1e-5


class TestTemporalCorrelation:
    """temporal_variance_ratio is a ratio of magnitudes and cannot see direction.
    These pin the direction half, and the pair that documents why one number is
    not enough.
    """

    N_SP = BIG_GRID[1] * BIG_GRID[2]
    MASK = torch.arange(BIG_GRID[0] * (BIG_GRID[1] * BIG_GRID[2]))

    def _target(self, seed=0):
        torch.manual_seed(seed)
        return torch.randn(1, BIG_GRID[0] * self.N_SP, 8)

    def test_a_perfect_prediction_correlates_at_one(self):
        target = self._target()
        assert (
            abs(temporal_correlation(target, target, BIG_GRID, self.MASK) - 1.0) < 1e-5
        )

    def test_uncorrelated_temporal_noise_correlates_at_zero(self):
        target = self._target()
        noise = torch.randn(1, BIG_GRID[0] * self.N_SP, 8)
        assert abs(temporal_correlation(noise, target, BIG_GRID, self.MASK)) < 0.1

    def test_a_temporally_flat_prediction_correlates_at_zero(self):
        """No deviation at all to correlate; must be 0.0, not a 0/0 nan."""
        target = self._target()
        flat = torch.ones(1, BIG_GRID[0] * self.N_SP, 8)
        assert temporal_correlation(flat, target, BIG_GRID, self.MASK) == 0.0

    def test_a_partial_signal_lands_between_the_two_ends(self):
        target = self._target()
        half = 0.5 * target + 0.5 * torch.randn_like(target)
        assert 0.3 < temporal_correlation(half, target, BIG_GRID, self.MASK) < 0.95

    def test_sign_is_recovered(self):
        """An anti-correlated prediction has the right magnitude and the wrong
        direction -- exactly what the variance ratio cannot express."""
        target = self._target()
        assert temporal_correlation(-target, target, BIG_GRID, self.MASK) < -0.99


class TestTheTwoTemporalMetricsTogether:
    """The regression test for the defect these were written to close.

    Measured in z-space, a noise-emitting model scores temporal_variance_ratio
    ~1.0 -- indistinguishable from a perfect model, and *better* than a model
    that captures half the signal (~0.5). The ratio alone is not evidence of
    motion tracking; only the pair is.
    """

    N_SP = BIG_GRID[1] * BIG_GRID[2]
    MASK = torch.arange(BIG_GRID[0] * (BIG_GRID[1] * BIG_GRID[2]))

    def _pair(self, pred, target):
        return (
            temporal_variance_ratio(pred, target, BIG_GRID, self.MASK),
            temporal_correlation(pred, target, BIG_GRID, self.MASK),
        )

    def test_noise_and_perfect_are_separated_only_by_the_correlation(self):
        torch.manual_seed(0)
        target = torch.randn(1, BIG_GRID[0] * self.N_SP, 8)
        noise_ratio, noise_corr = self._pair(torch.randn_like(target), target)
        perfect_ratio, perfect_corr = self._pair(target, target)

        # The ratio cannot tell them apart ...
        assert abs(noise_ratio - perfect_ratio) < 0.1
        assert noise_ratio > 0.9
        # ... the correlation can.
        assert abs(noise_corr) < 0.1
        assert perfect_corr > 0.99

    def test_the_ratio_ranks_half_signal_below_noise(self):
        """Stated as a test so the inversion is documented, not rediscovered:
        on the ratio alone, a genuinely half-right model looks worse than a
        model emitting pure noise."""
        torch.manual_seed(0)
        target = torch.randn(1, BIG_GRID[0] * self.N_SP, 8)
        half_ratio, half_corr = self._pair(
            0.5 * target + 0.5 * torch.randn_like(target), target
        )
        noise_ratio, noise_corr = self._pair(torch.randn_like(target), target)

        assert half_ratio < noise_ratio
        assert half_corr > noise_corr


class TestPerfectPredictionIsArmInvariantToo:
    """The other end of holdout_mse_z's scale.

    Every arm-invariance test above uses a *zeroed* predict_head, and zero is a
    fixed point of collapse_to_time_avg_z -- it is preserved for any sigma
    ratio, including an inverted one. So those tests pin one end of the scale
    and verify nothing about the collapse's correctness for a real prediction.

    The algebra: a perfect per_frame prediction is (x_pf - mu)/sigma_pf; its
    frame-average is (x_ta - mu)/sigma_pf; multiplying by sigma_pf/sigma_ta
    gives exactly (x_ta - mu)/sigma_ta, the time_avg z-target. So a perfect
    model scores 0.0 in both arms. That claim is what makes 0.9987 readable as
    "useless" rather than just "some number".
    """

    def _perfect(self, pixel_values, *, per_frame: bool):
        x, mu, sigma, _ = patch_stats(pixel_values, 2, 16, per_frame=per_frame)
        return (x - mu) / (sigma + 1e-6), sigma

    def test_a_perfect_per_frame_prediction_lands_on_the_time_avg_target(self):
        pixel_values = torch.randn(1, 3, 8, 32, 32)
        pred_pf, sigma_p = self._perfect(pixel_values, per_frame=True)
        target_ta, sigma_a = self._perfect(pixel_values, per_frame=False)

        collapsed = collapse_to_time_avg_z(
            pred_pf,
            target_mode="per_frame",
            tubelet_size=2,
            sigma_per_frame=sigma_p,
            sigma_time_avg=sigma_a,
        )
        torch.testing.assert_close(collapsed, target_ta, atol=1e-4, rtol=1e-4)

    def test_both_arms_score_zero_for_a_perfect_prediction(self):
        pixel_values = torch.randn(1, 3, 8, 32, 32)
        pred_pf, sigma_p = self._perfect(pixel_values, per_frame=True)
        target_ta, sigma_a = self._perfect(pixel_values, per_frame=False)

        per_frame_mse = float(
            (
                (
                    collapse_to_time_avg_z(
                        pred_pf,
                        target_mode="per_frame",
                        tubelet_size=2,
                        sigma_per_frame=sigma_p,
                        sigma_time_avg=sigma_a,
                    )
                    - target_ta
                )
                ** 2
            ).mean()
        )
        time_avg_mse = float(
            (
                (
                    collapse_to_time_avg_z(
                        target_ta,
                        target_mode="time_avg",
                        tubelet_size=2,
                        sigma_per_frame=sigma_p,
                        sigma_time_avg=sigma_a,
                    )
                    - target_ta
                )
                ** 2
            ).mean()
        )
        assert per_frame_mse < 1e-6
        assert time_avg_mse == 0.0

    def test_an_inverted_sigma_ratio_would_not_land_on_zero(self):
        """What the zeroed-head tests structurally cannot catch: swapping the
        two sigmas still maps zero to zero, so only a non-zero prediction
        exposes it."""
        pixel_values = torch.randn(1, 3, 8, 32, 32)
        pred_pf, sigma_p = self._perfect(pixel_values, per_frame=True)
        target_ta, sigma_a = self._perfect(pixel_values, per_frame=False)

        inverted = collapse_to_time_avg_z(
            pred_pf,
            target_mode="per_frame",
            tubelet_size=2,
            sigma_per_frame=sigma_a,  # swapped
            sigma_time_avg=sigma_p,
        )
        assert float(((inverted - target_ta) ** 2).mean()) > 1e-3
