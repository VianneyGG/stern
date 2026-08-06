from __future__ import annotations

import math

import pytest

from video_encoding.mesh.quorum import NodeStats, ready_for_merge, token_weights


def test_ready_for_merge_true_once_quorum_reached_even_before_grace_expires():
    checked_in = {"a", "b", "c"}
    assert ready_for_merge(checked_in, quorum_k=3, grace_s=900.0, now=1.0) is True


def test_ready_for_merge_waits_below_quorum_before_grace_expires():
    checked_in = {"a", "b"}
    assert ready_for_merge(checked_in, quorum_k=3, grace_s=900.0, now=1.0) is False


def test_ready_for_merge_below_quorum_merges_with_whats_present_after_grace_expires():
    checked_in = {"a", "b"}
    assert ready_for_merge(checked_in, quorum_k=3, grace_s=900.0, now=900.0) is True


def test_token_weights_zero_token_node_weighted_out_without_nan():
    stats = {
        "a": NodeStats(c_tokens=1000, c_steps=200),
        "b": NodeStats(c_tokens=0, c_steps=0),  # fetched no shards
    }
    weights = token_weights(stats)
    assert weights["b"] == 0.0
    assert not math.isnan(weights["a"])
    assert not math.isnan(weights["b"])


def test_token_weights_sum_to_one():
    stats = {
        "a": NodeStats(c_tokens=1000, c_steps=200),
        "b": NodeStats(c_tokens=500, c_steps=200),
        "c": NodeStats(c_tokens=0, c_steps=0),
    }
    weights = token_weights(stats)
    assert math.isclose(sum(weights.values()), 1.0)


def test_token_weights_matches_the_quantity_times_quality_formula():
    """Pins the actual w_i = c_tokens * (c_tokens / c_steps) formula, not
    just the normalization invariant -- a bug that dropped the quality term
    entirely (raw = c_tokens) would still sum to 1.0 and pass the test above."""
    stats = {
        "a": NodeStats(c_tokens=1000, c_steps=200),  # raw = 1000*5    = 5000
        "b": NodeStats(c_tokens=500, c_steps=200),  # raw = 500*2.5   = 1250
    }
    weights = token_weights(stats)
    assert math.isclose(weights["a"], 0.8)
    assert math.isclose(weights["b"], 0.2)


def test_token_weights_all_zero_raises():
    stats = {
        "a": NodeStats(c_tokens=0, c_steps=0),
        "b": NodeStats(c_tokens=0, c_steps=0),
    }
    with pytest.raises(ValueError):
        token_weights(stats)
