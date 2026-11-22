"""How much motion is there to lose in the first place?

Both factorial hypotheses assume the training data contains inter-frame change
worth preserving. Nothing measured that. Time-averaging is a linear operator,
so the variance it destroys is computable in closed form from the target
statistics -- no training required.
"""

import pytest
import torch

from stern.models.videomae.encoder import _IMAGENET_MEAN, _IMAGENET_STD
from stern.evaluation.diagnostics.motion_budget import (
    h1_verdict,
    motion_budget,
    subsample_stride,
)

TS, PS = 2, 16


def _clip(motion: float, *, n_frames: int = 8, seed: int = 0) -> torch.Tensor:
    """A static scene plus inter-frame change of the given amplitude."""
    g = torch.Generator().manual_seed(seed)
    still = torch.randn(1, 3, 1, 32, 32, generator=g).repeat(1, 1, n_frames, 1, 1)
    return still + motion * torch.randn(1, 3, n_frames, 32, 32, generator=g)


class TestMotionBudget:
    def test_a_static_clip_loses_nothing_to_time_averaging(self):
        """The zero-point. Averaging identical frames destroys no variance."""
        assert abs(motion_budget(_clip(0.0), TS, PS)["frac_var_lost"]) < 1e-3

    def test_a_static_clip_has_a_sigma_ratio_of_one(self):
        assert abs(motion_budget(_clip(0.0), TS, PS)["sigma_ratio"] - 1.0) < 1e-3

    def test_it_rises_monotonically_with_inter_frame_change(self):
        lost = [
            motion_budget(_clip(m), TS, PS)["frac_var_lost"]
            for m in (0.0, 0.1, 0.3, 1.0)
        ]
        assert lost == sorted(lost)
        assert lost[-1] > 0.1

    def test_it_is_invariant_to_brightness_and_contrast(self):
        """A ratio of per-token sigmas: rescaling the clip must not move it.
        Otherwise the number would track exposure rather than motion.
        """
        clip = _clip(0.3)
        base = motion_budget(clip, TS, PS)["frac_var_lost"]
        scaled = motion_budget(3.0 * clip + 7.0, TS, PS)["frac_var_lost"]
        assert abs(base - scaled) < 1e-3

    def test_tubelet_size_one_has_nothing_to_average_away(self):
        """The structural zero: with one frame per tubelet the two layouts are
        the same tensor, so no motion can be lost regardless of the clip.

        Not exactly 0.0 -- the division guard adds 1e-6 to sigma, which biases
        the ratio down by ~1e-5 at unit sigma. That bias is negligible beside
        the 0.02 decision threshold but is not zero.
        """
        assert abs(motion_budget(_clip(1.0), 1, PS)["frac_var_lost"]) < 1e-4

    def test_it_matches_the_analytic_value_for_a_known_ratio(self):
        """Pins the functional form, not just its monotonicity.

        With frames drawn iid, averaging two of them halves the variance, so
        sigma_time_avg = sigma_per_frame / sqrt(2) and the *variance* fraction
        lost is exactly 1 - 1/2 = 0.5. A ratio-based form such as 1 - 1/ratio
        would give 0.293 here and pass every other test in this class.
        """
        g = torch.Generator().manual_seed(5)
        iid = torch.randn(1, 3, 8, 32, 32, generator=g)
        report = motion_budget(iid, TS, PS)
        # Tolerances are finite-sample slack: each sigma is estimated over 1536
        # values. Wide enough to be stable, far tighter than the 0.207 gap to
        # the 1 - 1/ratio form.
        assert abs(report["sigma_ratio"] - 2.0**0.5) < 0.03
        assert abs(report["frac_var_lost"] - 0.5) < 0.02

    def _blacken(self, clip, rows=slice(0, 16)):
        """Set a region to true black.

        patch_stats denormalises with the ImageNet statistics, so a constant
        *normalised* value still leaves three distinct channel values and a
        non-zero sigma. Only genuine black -- (0 - mean_c) / std_c per channel
        -- denormalises to all-zero and drives sigma to exactly 0, which is the
        regime real footage reaches with letterboxing and dark scenes.
        """
        for c, (m, sd) in enumerate(zip(_IMAGENET_MEAN, _IMAGENET_STD)):
            clip[:, c, :, rows, :] = -m / sd
        return clip

    def test_flat_patches_cannot_drive_the_budget_negative(self):
        """The regression test for the first real-data run, which returned
        frac_var_lost = -1.12 on VoxCeleb2 shard-000007.

        Averaging can only remove variance, so the budget is bounded in [0, 1)
        by construction. Real footage is full of near-constant patches -- black
        bars, flat background -- where both sigmas approach zero and the
        division guard dominates the per-token ratio, producing huge negative
        outliers that a mean over tokens then inherits. Synthetic clips have no
        such patches, so no test in this class could see it.
        """
        report = motion_budget(self._blacken(_clip(0.3)), TS, PS)
        assert 0.0 <= report["frac_var_lost"] < 1.0

    def test_the_budget_is_bounded_for_a_wholly_constant_clip(self):
        """The limit of the case above: every patch degenerate."""
        report = motion_budget(
            self._blacken(torch.zeros(1, 3, 8, 32, 32), rows=slice(None)), TS, PS
        )
        assert 0.0 <= report["frac_var_lost"] < 1.0

    def test_flat_patches_do_not_dilute_a_real_signal(self):
        """A pooled statistic weights by variance, so adding dead background
        must not wash out the moving region the way an unweighted mean would."""
        moving = _clip(1.0)
        padded = self._blacken(moving.clone())
        assert (
            abs(
                motion_budget(padded, TS, PS)["frac_var_lost"]
                - motion_budget(moving, TS, PS)["frac_var_lost"]
            )
            < 0.15
        )

class TestStrideSubsampling:
    """H3: window_starts builds stride-1 windows, so a tubelet spans two
    adjacent frames -- 80ms at 25fps. If that is why motion is absent, widening
    the stride restores it, and this is the cheapest of the three hypotheses to
    test.
    """

    def test_stride_one_is_the_identity(self):
        clip = _clip(0.3, n_frames=8)
        assert torch.equal(subsample_stride(clip, 1, 8), clip)

    def test_stride_two_takes_every_other_frame(self):
        clip = _clip(0.3, n_frames=16)
        out = subsample_stride(clip, 2, 8)
        assert out.shape[2] == 8
        torch.testing.assert_close(out, clip[:, :, 0:16:2])

    def test_too_few_frames_for_the_stride_raises(self):
        with pytest.raises(ValueError, match="stride"):
            subsample_stride(_clip(0.3, n_frames=8), 4, 8)

    def test_a_wider_stride_exposes_more_motion(self):
        """The H3 measurement itself: on a clip whose frames drift steadily,
        sampling further apart must show strictly more inter-frame change."""
        g = torch.Generator().manual_seed(1)
        drift = torch.cumsum(0.1 * torch.randn(1, 3, 32, 32, 32, generator=g), dim=2)
        wide = motion_budget(subsample_stride(drift, 4, 8), TS, PS)["frac_var_lost"]
        narrow = motion_budget(subsample_stride(drift, 1, 8), TS, PS)["frac_var_lost"]
        assert wide > narrow


class TestH1Verdict:
    def test_a_negligible_budget_refutes_the_premise(self):
        assert h1_verdict(0.004) == "refuted"

    def test_a_large_budget_supports_it(self):
        assert h1_verdict(0.35) == "holds"

    @pytest.mark.parametrize(
        "budget, verdict",
        [
            pytest.param(0.05, "weak", id="middle_is_not_forced"),
            pytest.param(0.02, "weak", id="lower_boundary_is_weak"),
            pytest.param(0.0199, "refuted", id="just_below_lower_boundary"),
            pytest.param(0.10, "holds", id="upper_boundary_holds"),
            pytest.param(0.0999, "weak", id="just_below_upper_boundary"),
        ],
    )
    def test_the_middle_and_boundaries_are_where_the_docstring_says(self, budget, verdict):
        assert h1_verdict(budget) == verdict
