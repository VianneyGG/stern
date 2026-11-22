"""The worker's per-job round bound (`mesh.chunk_rounds`)."""

from __future__ import annotations

import pytest

from stern.training.mesh_round import chunk_stop_round


@pytest.mark.parametrize(
    "start_round",
    [
        # The default for every run registered before the 2h cap existed.
        pytest.param(0, id="fresh_run"),
        pytest.param(17, id="mid_run_relaunch"),
    ],
)
def test_no_chunking_runs_the_rest_of_the_run_in_one_job(start_round):
    stop = chunk_stop_round(rounds=40, start_round=start_round, chunk_rounds=None)
    assert stop == 40


def test_first_chunk_stops_at_the_chunk_length():
    assert chunk_stop_round(rounds=40, start_round=0, chunk_rounds=5) == 5


def test_a_later_chunk_is_relative_to_its_own_start():
    # Round 5..9 -- the resumed job must not restart the count at zero.
    assert chunk_stop_round(rounds=40, start_round=5, chunk_rounds=5) == 10


def test_the_final_chunk_stops_at_the_run_end_not_past_it():
    # 38 + 5 = 43 would overshoot; the run is 40 rounds long.
    assert chunk_stop_round(rounds=40, start_round=38, chunk_rounds=5) == 40


def test_a_chunk_starting_at_the_run_end_runs_nothing():
    assert chunk_stop_round(rounds=40, start_round=40, chunk_rounds=5) == 40


@pytest.mark.parametrize("chunk_rounds", [0, -1])
def test_a_non_positive_chunk_exits_cleanly_rather_than_reversing(chunk_rounds):
    """`range(start, stop)` with stop < start silently skips the round.

    Clamping to start_round turns a misconfiguration into an immediate clean
    exit -- which the watcher sees as a finished chunk and relaunches -- not
    into a worker that reports success having trained nothing.
    """
    stop = chunk_stop_round(rounds=40, start_round=12, chunk_rounds=chunk_rounds)
    assert stop == 12
    assert list(range(12, stop)) == []
