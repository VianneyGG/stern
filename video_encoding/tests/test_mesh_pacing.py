from __future__ import annotations

import math

import pytest

from video_encoding.mesh.pacing import steps_for_budget, update_rate_ema


def test_steps_for_budget_is_rate_times_budget():
    assert (
        steps_for_budget(
            rate_steps_per_s=10.0, budget_s=900.0, floor_steps=2400, ceil_steps=19200
        )
        == 9000
    )


def test_steps_for_budget_clamps_a_degraded_node_up_to_the_floor():
    """A node running at 1 step/s would contribute a 900-step round -- small
    enough that its pseudo-gradient is mostly noise. The floor keeps its
    contribution meaningful (it just finishes late, which quorum handles)."""
    assert (
        steps_for_budget(
            rate_steps_per_s=1.0, budget_s=900.0, floor_steps=2400, ceil_steps=19200
        )
        == 2400
    )


def test_steps_for_budget_clamps_a_runaway_estimate_down_to_the_ceiling():
    assert (
        steps_for_budget(
            rate_steps_per_s=50.0, budget_s=900.0, floor_steps=2400, ceil_steps=19200
        )
        == 19200
    )


def test_steps_for_budget_rejects_a_non_positive_rate():
    """rate<=0 means the throughput measurement failed, not that the node is
    slow. Silently returning floor_steps would bury that."""
    with pytest.raises(ValueError, match="positive"):
        steps_for_budget(
            rate_steps_per_s=0.0, budget_s=900.0, floor_steps=2400, ceil_steps=19200
        )


def test_update_rate_ema_adopts_the_first_observation_outright():
    assert update_rate_ema(None, 9.9) == 9.9


def test_update_rate_ema_blends_against_history():
    """alpha=0.3 (the real default), prev != observed enough that a role
    swap (alpha*prev + (1-alpha)*observed) lands on a different, equally
    plausible-looking number -- alpha=0.5 with symmetric inputs can't tell
    the two formulas apart."""
    assert math.isclose(update_rate_ema(10.0, 4.0, alpha=0.3), 8.2)


def test_update_rate_ema_defaults_alpha_to_point_three():
    assert math.isclose(update_rate_ema(10.0, 4.0), 8.2)
