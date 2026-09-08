import os
import subprocess
import time

import pytest

pytestmark = pytest.mark.system

GATE = "video_encoding/scripts/mesh_continue_allowed.sh"


def _run_gate(env, run_id="RUN1", node="node001"):
    result = subprocess.run(
        ["bash", GATE, run_id, node],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
    )
    return result.returncode


def test_stops_past_absolute_deadline(tmp_path):
    rc = _run_gate(
        {
            "MESH_SHARED": str(tmp_path),
            "MESH_DEADLINE_EPOCH": str(int(time.time()) - 10),
            "MESH_MAX_REQUEUES": "100",
            "SLURM_RESTART_COUNT": "0",
        }
    )
    assert rc == 1


def test_continues_before_deadline_with_no_other_stop_condition(tmp_path):
    rc = _run_gate(
        {
            "MESH_SHARED": str(tmp_path),
            "MESH_DEADLINE_EPOCH": str(int(time.time()) + 3600),
            "MESH_MAX_REQUEUES": "100",
            "SLURM_RESTART_COUNT": "0",
        }
    )
    assert rc == 0


def test_stops_when_restart_count_exceeds_budget(tmp_path):
    rc = _run_gate(
        {
            "MESH_SHARED": str(tmp_path),
            "MESH_DEADLINE_EPOCH": str(int(time.time()) + 3600),
            "MESH_MAX_REQUEUES": "3",
            "SLURM_RESTART_COUNT": "3",
        }
    )
    assert rc == 1


def test_stops_on_halt_file(tmp_path):
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "HALT").touch()
    rc = _run_gate(
        {
            "MESH_SHARED": str(tmp_path),
            "MESH_DEADLINE_EPOCH": str(int(time.time()) + 3600),
            "MESH_MAX_REQUEUES": "100",
            "SLURM_RESTART_COUNT": "0",
        }
    )
    assert rc == 1


def test_stops_on_per_run_halt_file(tmp_path):
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "HALT-RUN1").touch()
    rc = _run_gate(
        {
            "MESH_SHARED": str(tmp_path),
            "MESH_DEADLINE_EPOCH": str(int(time.time()) + 3600),
            "MESH_MAX_REQUEUES": "100",
            "SLURM_RESTART_COUNT": "0",
        }
    )
    assert rc == 1


def test_stops_on_quarantine_marker(tmp_path):
    qdir = tmp_path / "state" / "quarantine" / "RUN1"
    qdir.mkdir(parents=True)
    (qdir / "node001").touch()
    rc = _run_gate(
        {
            "MESH_SHARED": str(tmp_path),
            "MESH_DEADLINE_EPOCH": str(int(time.time()) + 3600),
            "MESH_MAX_REQUEUES": "100",
            "SLURM_RESTART_COUNT": "0",
        }
    )
    assert rc == 1


def test_stops_when_watcher_state_is_stale(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    stale_ts = time.time() - 700  # past the default 600s (10 min) bound
    (state_dir / "watcher_state.json").write_text(f'{{"last_tick_at": {stale_ts}}}')
    rc = _run_gate(
        {
            "MESH_SHARED": str(tmp_path),
            "MESH_DEADLINE_EPOCH": str(int(time.time()) + 3600),
            "MESH_MAX_REQUEUES": "100",
            "SLURM_RESTART_COUNT": "0",
        }
    )
    assert rc == 1


def test_continues_when_watcher_state_is_missing_fail_open(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    # No watcher_state.json at all -- best-effort, fail-open.
    rc = _run_gate(
        {
            "MESH_SHARED": str(tmp_path),
            "MESH_DEADLINE_EPOCH": str(int(time.time()) + 3600),
            "MESH_MAX_REQUEUES": "100",
            "SLURM_RESTART_COUNT": "0",
        }
    )
    assert rc == 0


def test_stops_after_futile_requeue_bound_and_writes_escalation(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    requeue_dir = state_dir / "requeue"
    requeue_dir.mkdir()
    # Two prior requeues at the same round (no progress) -- default bound is 2.
    (requeue_dir / "RUN1-node001.json").write_text(
        '{"round": 5, "consecutive_futile": 2}'
    )
    (requeue_dir / "RUN1-node001-current_round").write_text("5")
    rc = _run_gate(
        {
            "MESH_SHARED": str(tmp_path),
            "MESH_DEADLINE_EPOCH": str(int(time.time()) + 3600),
            "MESH_MAX_REQUEUES": "100",
            "SLURM_RESTART_COUNT": "0",
        }
    )
    assert rc == 1
    assert list(
        (tmp_path.parent / "runs" / "state" / "escalations").glob("watcher_stalled-*")
    )


def test_continues_when_round_has_advanced(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "requeue").mkdir()
    (state_dir / "requeue" / "RUN1-node001.json").write_text(
        '{"round": 4, "consecutive_futile": 1}'
    )
    (state_dir / "requeue" / "RUN1-node001-current_round").write_text("5")
    rc = _run_gate(
        {
            "MESH_SHARED": str(tmp_path),
            "MESH_DEADLINE_EPOCH": str(int(time.time()) + 3600),
            "MESH_MAX_REQUEUES": "100",
            "SLURM_RESTART_COUNT": "0",
        }
    )
    assert rc == 0
    import json

    journal = json.loads((state_dir / "requeue" / "RUN1-node001.json").read_text())
    assert journal == {"round": 5, "consecutive_futile": 0}
