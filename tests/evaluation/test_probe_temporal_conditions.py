"""The gate statistic. 124 clips is small, so a fixed CCC threshold would be
noise -- the confidence interval is the decision."""

from pathlib import Path

import numpy as np
import pytest

from stern.evaluation.probes.temporal_conditions import (
    gate_verdict,
    paired_bootstrap_ccc_delta,
    parse_prediction_path,
    prediction_path,
)


class TestPairedBootstrap:
    def test_the_resampling_is_actually_paired(self):
        """Both arms are scored on the same windows, so a resample must draw the
        same window indices for both. When the two predictions are identical the
        per-resample delta is then exactly zero and the interval collapses --
        under independent (unpaired) resampling it would have real width.

        Asserting only that the interval spans zero cannot detect this: an
        unpaired interval spans zero too, just far too wide.
        """
        rng = np.random.default_rng(4)
        target = rng.normal(size=200)
        pred = target + rng.normal(scale=0.3, size=200)
        delta, lo, hi = paired_bootstrap_ccc_delta(pred, pred, target, seed=0)
        # Every resample scores both arms on one index draw, so each per-resample
        # delta is x - x: exactly 0.0, hence both percentiles are exactly 0.0.
        assert (delta, lo, hi) == (0.0, 0.0, 0.0)

    def test_a_clearly_better_condition_gives_a_positive_interval(self):
        rng = np.random.default_rng(1)
        target = rng.normal(size=400)
        good = target + rng.normal(scale=0.2, size=400)
        bad = rng.normal(size=400)
        delta, lo, hi = paired_bootstrap_ccc_delta(good, bad, target, seed=0)
        assert delta > 0.5
        assert lo > 0.0

    def test_pure_noise_in_both_arms_gives_an_interval_spanning_zero(self):
        rng = np.random.default_rng(2)
        target = rng.normal(size=300)
        delta, lo, hi = paired_bootstrap_ccc_delta(
            rng.normal(size=300), rng.normal(size=300), target, seed=0
        )
        assert lo < 0.0 < hi

    def test_is_deterministic_for_a_seed(self):
        rng = np.random.default_rng(3)
        target = rng.normal(size=150)
        a, b = target + rng.normal(scale=0.4, size=150), rng.normal(size=150)
        assert paired_bootstrap_ccc_delta(a, b, target, seed=7) == (
            paired_bootstrap_ccc_delta(a, b, target, seed=7)
        )


class TestGateVerdict:
    def test_interval_above_zero_opens_the_gate(self):
        assert gate_verdict(0.04, 0.19) == "open"

    def test_interval_spanning_zero_closes_the_gate(self):
        assert gate_verdict(-0.03, 0.08) == "closed"

    def test_interval_below_zero_means_the_probe_is_broken(self):
        """Shuffling frames cannot legitimately *improve* CCC."""
        assert gate_verdict(-0.20, -0.05) == "broken"


class TestPredictionPath:
    @pytest.mark.parametrize(
        "checkpoint", ["latest", "r120", "random_init", "master_round20"]
    )
    @pytest.mark.parametrize("condition", ["normal", "shuffled", "frozen"])
    def test_round_trips(self, checkpoint, condition):
        """Checkpoint names contain underscores, so a naive split on '_' would
        parse random_init_normal as ('random', 'init_normal')."""
        path = prediction_path(Path("/tmp/preds"), checkpoint, condition)
        assert parse_prediction_path(path) == (checkpoint, condition)
