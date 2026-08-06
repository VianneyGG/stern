from __future__ import annotations

import json
from unittest.mock import patch


def test_main_fast_path_refreshes_squeue_and_exits_without_full_check(
    tmp_path, monkeypatch
):
    import video_encoding.scripts.watch_training as watch_mod

    monkeypatch.setattr(watch_mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(watch_mod, "STATE_DIR", tmp_path / "runs" / "state")
    monkeypatch.setattr(watch_mod, "RESULTS_DIR", tmp_path / "runs" / "results")
    monkeypatch.setattr(
        watch_mod, "LOG_PATH", tmp_path / "runs" / "state" / "watcher.log"
    )
    monkeypatch.setattr(
        watch_mod, "LOCK_PATH", tmp_path / "runs" / "state" / "watch_training.lock"
    )

    # Pre-seed state so next_full_check_at is far in the future -- fast path.
    (tmp_path / "runs" / "state").mkdir(parents=True)
    (tmp_path / "runs" / "state" / "watcher_state.json").write_text(
        json.dumps(
            {
                "next_full_check_at": 9_999_999_999.0,
                "current_interval_s": 1800.0,
                "jobs": {},
                "escalation_day": None,
                "escalation_count_today": 0,
            }
        )
    )

    with (
        patch(
            "video_encoding.scripts.watch_training.refresh_squeue_snapshot"
        ) as mock_refresh,
        patch(
            "video_encoding.scripts.watch_training.run_full_check"
        ) as mock_full_check,
    ):
        rc = watch_mod.main()

    assert rc == 0
    mock_refresh.assert_called_once()
    mock_full_check.assert_not_called()
    assert not (tmp_path / "runs" / "state" / "watcher.log").exists()


def test_main_runs_full_check_and_logs_when_due(tmp_path, monkeypatch):
    import video_encoding.scripts.watch_training as watch_mod
    from video_encoding.watcher.tick import TickResult

    monkeypatch.setattr(watch_mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(watch_mod, "STATE_DIR", tmp_path / "runs" / "state")
    monkeypatch.setattr(watch_mod, "RESULTS_DIR", tmp_path / "runs" / "results")
    monkeypatch.setattr(
        watch_mod, "LOG_PATH", tmp_path / "runs" / "state" / "watcher.log"
    )
    monkeypatch.setattr(
        watch_mod, "LOCK_PATH", tmp_path / "runs" / "state" / "watch_training.lock"
    )

    with (
        patch("video_encoding.scripts.watch_training.refresh_squeue_snapshot"),
        patch(
            "video_encoding.scripts.watch_training.run_full_check",
            return_value=TickResult(num_flags=0, actions=[], action_taken=False),
        ) as mock_full_check,
    ):
        rc = watch_mod.main()

    assert rc == 0
    mock_full_check.assert_called_once()
    log_text = (tmp_path / "runs" / "state" / "watcher.log").read_text()
    assert "flags=0" in log_text


def test_main_survives_mesh_advancement_exception(tmp_path, monkeypatch):
    """A single bad mesh run config (bad hostname, n_shards mismatched
    against shard_map.json, etc.) must not crash main() before
    run_full_check -- that would silently disable dead/stalled-job
    detection for the entire fleet, not just the misconfigured mesh run."""
    import video_encoding.scripts.watch_training as watch_mod
    from video_encoding.watcher.tick import TickResult

    monkeypatch.setattr(watch_mod, "REPO_ROOT", tmp_path)
    state_dir = tmp_path / "runs" / "state"
    monkeypatch.setattr(watch_mod, "STATE_DIR", state_dir)
    monkeypatch.setattr(watch_mod, "RESULTS_DIR", tmp_path / "runs" / "results")
    monkeypatch.setattr(watch_mod, "LOG_PATH", state_dir / "watcher.log")
    monkeypatch.setattr(watch_mod, "LOCK_PATH", state_dir / "watch_training.lock")

    state_dir.mkdir(parents=True)
    (state_dir / "mesh_run_configs.json").write_text(
        json.dumps(
            [
                {
                    "run_id": "RUN1",
                    "experiment": "videomae_mesh_ssl",
                    "nodes": ["node001"],
                    "n_shards": 9,
                    "outer_lr": 0.7,
                    "outer_momentum": 0.9,
                    "weighting": "data",
                    "rounds": 10,
                    "merge": True,
                }
            ]
        )
    )

    with (
        patch("video_encoding.scripts.watch_training.refresh_squeue_snapshot"),
        patch(
            "video_encoding.scripts.watch_training.advance_all_runs",
            side_effect=KeyError("bad-hostname"),
        ),
        patch(
            "video_encoding.scripts.watch_training.run_full_check",
            return_value=TickResult(num_flags=0, actions=[], action_taken=False),
        ) as mock_full_check,
    ):
        rc = watch_mod.main()

    assert rc == 0
    mock_full_check.assert_called_once()
    assert mock_full_check.call_args.kwargs["mesh_run_ids"] == frozenset({"RUN1"})
    log_text = (state_dir / "watcher.log").read_text()
    assert "mesh advancement FAILED" in log_text


def test_main_exempts_mesh_runs_no_longer_in_config_from_remediation(
    tmp_path, monkeypatch
):
    """A mesh run that finished (or was dropped from mesh_run_configs.json)
    still has dead worker jobs in jobs_registry.json. If the exemption set
    only covers the live config, those jobs reach remediate_dead_job, which
    hardcodes train.sbatch -- that is how 39 junk stern-train jobs appeared on
    2026-07-27, queued ahead of the mesh workers _replenish_pool needed."""
    import video_encoding.scripts.watch_training as watch_mod
    from video_encoding.watcher.tick import TickResult

    monkeypatch.setattr(watch_mod, "REPO_ROOT", tmp_path)
    state_dir = tmp_path / "runs" / "state"
    monkeypatch.setattr(watch_mod, "STATE_DIR", state_dir)
    monkeypatch.setattr(watch_mod, "RESULTS_DIR", tmp_path / "runs" / "results")
    monkeypatch.setattr(watch_mod, "LOG_PATH", state_dir / "watcher.log")
    monkeypatch.setattr(watch_mod, "LOCK_PATH", state_dir / "watch_training.lock")

    state_dir.mkdir(parents=True)
    # LIVE is configured; FINISHED is not -- it survives only in state.runs,
    # exactly like cand2a-8h / videomae_mesh_ssl-diloco-cand2b on the cluster.
    (state_dir / "watcher_state.json").write_text(
        json.dumps(
            {
                "next_full_check_at": 0.0,
                "current_interval_s": 1800.0,
                "jobs": {},
                "escalation_day": None,
                "escalation_count_today": 0,
                "runs": {
                    "FINISHED": {
                        "run_id": "FINISHED",
                        "round": 5,
                        "status": "DONE",
                        "missed_rounds": {},
                        "target_pool_size": 16,
                    }
                },
            }
        )
    )
    (state_dir / "mesh_run_configs.json").write_text(
        json.dumps(
            [
                {
                    "run_id": "LIVE",
                    "experiment": "videomae_mesh_ssl",
                    "nodes": ["node001"],
                    "n_shards": 9,
                    "outer_lr": 0.7,
                    "outer_momentum": 0.9,
                    "weighting": "data",
                    "rounds": 10,
                    "merge": True,
                }
            ]
        )
    )

    with (
        patch("video_encoding.scripts.watch_training.refresh_squeue_snapshot"),
        patch(
            "video_encoding.scripts.watch_training.advance_all_runs", return_value=[]
        ),
        patch(
            "video_encoding.scripts.watch_training.run_full_check",
            return_value=TickResult(num_flags=0, actions=[], action_taken=False),
        ) as mock_full_check,
    ):
        rc = watch_mod.main()

    assert rc == 0
    assert mock_full_check.call_args.kwargs["mesh_run_ids"] == frozenset(
        {"LIVE", "FINISHED"}
    )


def test_main_sweeps_orphaned_pending_checkpoints_each_tick(tmp_path, monkeypatch):
    """0.2: pending/ accumulates checkpoints from runs that died before
    reaching SYNCING/DONE (the only other sweep site). This must run every
    tick, unconditional like advance_all_runs, not gated by the full-check
    backoff timer."""
    import video_encoding.scripts.watch_training as watch_mod
    from video_encoding.watcher.tick import TickResult

    monkeypatch.setattr(watch_mod, "REPO_ROOT", tmp_path)
    state_dir = tmp_path / "runs" / "state"
    monkeypatch.setattr(watch_mod, "STATE_DIR", state_dir)
    monkeypatch.setattr(watch_mod, "RESULTS_DIR", tmp_path / "runs" / "results")
    monkeypatch.setattr(watch_mod, "LOG_PATH", state_dir / "watcher.log")
    monkeypatch.setattr(watch_mod, "LOCK_PATH", state_dir / "watch_training.lock")
    mesh_shared = tmp_path / "mesh_shared"
    monkeypatch.setenv("MESH_SHARED", str(mesh_shared))

    state_dir.mkdir(parents=True)
    (state_dir / "mesh_run_configs.json").write_text(
        json.dumps(
            [
                {
                    "run_id": "RUN1",
                    "experiment": "videomae_mesh_ssl",
                    "nodes": ["node001"],
                    "n_shards": 9,
                    "outer_lr": 0.7,
                    "outer_momentum": 0.9,
                    "weighting": "data",
                    "rounds": 10,
                    "merge": True,
                }
            ]
        )
    )

    with (
        patch("video_encoding.scripts.watch_training.refresh_squeue_snapshot"),
        patch(
            "video_encoding.scripts.watch_training.advance_all_runs", return_value=[]
        ),
        patch(
            "video_encoding.scripts.watch_training.sweep_orphaned_pending",
            return_value=3,
        ) as mock_sweep,
        patch(
            "video_encoding.scripts.watch_training.run_full_check",
            return_value=TickResult(num_flags=0, actions=[], action_taken=False),
        ),
    ):
        rc = watch_mod.main()

    assert rc == 0
    mock_sweep.assert_called_once()
    assert mock_sweep.call_args.args[0] == mesh_shared / "checkpoints" / "pending"
    assert mock_sweep.call_args.args[1] == {"RUN1"}
    log_text = (state_dir / "watcher.log").read_text()
    assert "swept 3 orphaned pending checkpoint(s)" in log_text


def test_main_detects_and_remediates_nan_nodes_each_tick(tmp_path, monkeypatch):
    """1.: per-round NaN can't wait for the health-check backoff timer --
    detect_nan/remediate_nan_node run unconditionally each tick, same call
    site as sweep_orphaned_pending above."""
    import video_encoding.scripts.watch_training as watch_mod
    from video_encoding.watcher.detectors import Flag
    from video_encoding.watcher.tick import TickResult

    monkeypatch.setattr(watch_mod, "REPO_ROOT", tmp_path)
    state_dir = tmp_path / "runs" / "state"
    monkeypatch.setattr(watch_mod, "STATE_DIR", state_dir)
    monkeypatch.setattr(watch_mod, "RESULTS_DIR", tmp_path / "runs" / "results")
    monkeypatch.setattr(watch_mod, "LOG_PATH", state_dir / "watcher.log")
    monkeypatch.setattr(watch_mod, "LOCK_PATH", state_dir / "watch_training.lock")
    monkeypatch.setenv("MESH_SHARED", str(tmp_path / "mesh_shared"))
    monkeypatch.setenv("NAN_REMEDIATION_DRY_RUN", "false")

    state_dir.mkdir(parents=True)
    (state_dir / "mesh_run_configs.json").write_text(
        json.dumps(
            [
                {
                    "run_id": "RUN1",
                    "experiment": "videomae_mesh_ssl",
                    "nodes": ["node001"],
                    "n_shards": 9,
                    "outer_lr": 0.7,
                    "outer_momentum": 0.9,
                    "weighting": "data",
                    "rounds": 10,
                    "merge": True,
                }
            ]
        )
    )
    flag = Flag(
        "nan:RUN1:node001", "nan", "node001: train_loss=nan round=1 run_id=RUN1"
    )

    with (
        patch("video_encoding.scripts.watch_training.refresh_squeue_snapshot"),
        patch(
            "video_encoding.scripts.watch_training.advance_all_runs", return_value=[]
        ),
        patch(
            "video_encoding.scripts.watch_training.sweep_orphaned_pending",
            return_value=0,
        ),
        patch(
            "video_encoding.scripts.watch_training.detect_nan",
            return_value=[flag],
        ) as mock_detect,
        patch(
            "video_encoding.scripts.watch_training.remediate_nan_node",
            return_value="retried",
        ) as mock_remediate,
        patch(
            "video_encoding.scripts.watch_training.run_full_check",
            return_value=TickResult(num_flags=0, actions=[], action_taken=False),
        ),
    ):
        rc = watch_mod.main()

    assert rc == 0
    mock_detect.assert_called_once()
    assert mock_detect.call_args.args[0] == "RUN1"
    mock_remediate.assert_called_once()
    assert mock_remediate.call_args.kwargs["run_id"] == "RUN1"
    assert mock_remediate.call_args.kwargs["node"] == "node001"
    assert mock_remediate.call_args.kwargs["dry_run"] is False
    log_text = (state_dir / "watcher.log").read_text()
    assert "nan: nan:RUN1:node001 -> retried" in log_text


def test_main_skips_when_lock_held(tmp_path, monkeypatch):
    import video_encoding.scripts.watch_training as watch_mod

    monkeypatch.setattr(watch_mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(watch_mod, "STATE_DIR", tmp_path / "runs" / "state")
    monkeypatch.setattr(watch_mod, "RESULTS_DIR", tmp_path / "runs" / "results")
    monkeypatch.setattr(
        watch_mod, "LOG_PATH", tmp_path / "runs" / "state" / "watcher.log"
    )
    lock_path = tmp_path / "runs" / "state" / "watch_training.lock"
    monkeypatch.setattr(watch_mod, "LOCK_PATH", lock_path)

    lock_path.parent.mkdir(parents=True)
    import fcntl

    held_fh = open(lock_path, "w")
    fcntl.flock(held_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with patch(
            "video_encoding.scripts.watch_training.refresh_squeue_snapshot"
        ) as mock_refresh:
            rc = watch_mod.main()
        assert rc == 0
        mock_refresh.assert_not_called()
    finally:
        fcntl.flock(held_fh, fcntl.LOCK_UN)
        held_fh.close()
