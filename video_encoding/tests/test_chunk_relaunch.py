"""Chunk-boundary detection: relaunch only when every worker is really gone."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from video_encoding.watcher.chunk_relaunch import (
    SUBMIT_VISIBILITY_S,
    cancellable_job_ids,
    needs_worker_relaunch,
)
from video_encoding.watcher.registry import JobRecord

NOW = 1_787_500_000.0
OLD = datetime.fromtimestamp(NOW - 10_000, tz=timezone.utc).isoformat()


def _job(job_id: str, run_id: str = "run-a", *, submitted_at: str = OLD) -> JobRecord:
    return JobRecord(
        slurm_job_id=job_id,
        experiment="videomae_mesh_ssl",
        run_tag=run_id,
        submitted_at=submitted_at,
        nodelist=f"node{job_id}",
    )


def _live(*job_ids: str) -> dict[str, dict[str, str]]:
    return {j: {"state": "RUNNING", "nodelist": "n"} for j in job_ids}


def _call(**kw):
    base = dict(
        run_id="run-a",
        status="RUNNING",
        registry=[_job("1"), _job("2")],
        snapshot={},
        snapshot_available=True,
        now=NOW,
    )
    return needs_worker_relaunch(**{**base, **kw})


def test_relaunches_when_every_worker_job_has_left_the_queue():
    assert _call(snapshot={}) is True


def test_does_not_relaunch_while_any_worker_is_still_running():
    # The chunk is not over until the LAST worker exits; relaunching here
    # double-submits --exclusive jobs onto nodes already running them.
    assert _call(snapshot=_live("2")) is False


def test_does_not_relaunch_when_the_snapshot_is_unavailable():
    """The most dangerous input: it makes every run look finished at once."""
    assert _call(snapshot={}, snapshot_available=False) is False


def test_does_not_relaunch_a_run_that_was_never_submitted():
    assert _call(registry=[]) is False


def test_ignores_other_runs_jobs():
    # Another run's live jobs must not keep this run's finished chunk waiting.
    assert _call(registry=[_job("1")], snapshot=_live("99")) is True


def test_another_runs_live_job_does_not_mask_this_runs_departure():
    assert _call(registry=[_job("1", "run-b")], snapshot={}) is False


@pytest.mark.parametrize(
    "status", ["MERGING", "SYNCING", "DONE", "DEGRADED", "ASSIGNING"]
)
def test_only_relaunches_from_states_where_workers_should_be_running(status):
    assert _call(status=status) is False


@pytest.mark.parametrize("status", ["RUNNING", "AWAITING_CHECKPOINTS"])
def test_relaunches_from_both_worker_running_states(status):
    assert _call(status=status) is True


def test_a_just_submitted_chunk_is_not_immediately_relaunched():
    """sbatch -> squeue visibility lags the snapshot taken at tick start.

    Without this the very first tick after submitting a chunk would see no
    live jobs and submit it a second time.
    """
    fresh = datetime.fromtimestamp(NOW - 5, tz=timezone.utc).isoformat()
    assert _call(registry=[_job("1", submitted_at=fresh)], snapshot={}) is False


def test_the_visibility_grace_expires():
    stale = datetime.fromtimestamp(
        NOW - SUBMIT_VISIBILITY_S - 1, tz=timezone.utc
    ).isoformat()
    assert _call(registry=[_job("1", submitted_at=stale)], snapshot={}) is True


def test_an_unparseable_submitted_at_falls_back_to_the_snapshot():
    # Must not crash the tick, and must not grant an indefinite grace.
    assert _call(registry=[_job("1", submitted_at="not-a-date")], snapshot={}) is True
    assert (
        _call(registry=[_job("1", submitted_at="not-a-date")], snapshot=_live("1"))
        is False
    )


def test_a_job_caught_in_a_terminal_state_does_not_count_as_alive():
    snapshot = {"1": {"state": "COMPLETING"}, "2": {"state": "CANCELLED"}}
    # COMPLETING is still alive; CANCELLED is not -- one live job is enough.
    assert _call(snapshot=snapshot) is False
    assert _call(snapshot={"1": {"state": "FAILED"}, "2": {"state": "TIMEOUT"}}) is True


def _rec(job_id: str, node: str, run_tag: str = "run-a") -> JobRecord:
    return JobRecord(
        slurm_job_id=job_id,
        experiment="videomae_mesh_ssl",
        run_tag=run_tag,
        submitted_at="2026-08-23T09:00:00+00:00",
        nodelist=node,
    )


def test_cancellable_picks_only_live_jobs_of_this_run():
    registry = [_rec("1", "node055"), _rec("2", "node057"), _rec("3", "node051", "other")]
    snapshot = {
        "1": {"state": "RUNNING", "nodelist": "node055"},
        "2": {"state": "COMPLETED", "nodelist": "node057"},
        "3": {"state": "RUNNING", "nodelist": "node051"},
    }
    assert cancellable_job_ids(
        run_id="run-a", registry=registry, snapshot=snapshot
    ) == ["1"]


def test_cancellable_skips_job_id_absent_from_snapshot():
    registry = [_rec("7", "node055")]
    assert cancellable_job_ids(run_id="run-a", registry=registry, snapshot={}) == []


def test_cancellable_rejects_recycled_job_id_on_a_different_node():
    """The 2026-08-22 rebuild reset the job counter, so a stale registry
    record can collide with a live id belonging to someone else's job."""
    registry = [_rec("42", "node055")]
    snapshot = {"42": {"state": "RUNNING", "nodelist": "someone-elses-node"}}
    assert (
        cancellable_job_ids(run_id="run-a", registry=registry, snapshot=snapshot) == []
    )


def test_cancellable_returns_every_live_straggler():
    registry = [_rec("10", "node055"), _rec("11", "node032"), _rec("12", "node067")]
    snapshot = {
        "10": {"state": "RUNNING", "nodelist": "node055"},
        "11": {"state": "COMPLETING", "nodelist": "node032"},
        "12": {"state": "RUNNING", "nodelist": "node067"},
    }
    assert cancellable_job_ids(
        run_id="run-a", registry=registry, snapshot=snapshot
    ) == ["10", "11", "12"]
