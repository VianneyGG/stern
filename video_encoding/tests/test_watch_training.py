from __future__ import annotations

import json
from unittest.mock import patch


def test_main_fast_path_refreshes_squeue_and_exits_without_full_check(
    tmp_path, monkeypatch
):
    import video_encoding.scripts.watch_training as watch_mod

    monkeypatch.setattr(watch_mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(watch_mod, "STATE_DIR", tmp_path / ".claude" / "state")
    monkeypatch.setattr(watch_mod, "RESULTS_DIR", tmp_path / ".claude" / "results")
    monkeypatch.setattr(
        watch_mod, "LOG_PATH", tmp_path / ".claude" / "state" / "watcher.log"
    )
    monkeypatch.setattr(
        watch_mod, "LOCK_PATH", tmp_path / ".claude" / "state" / "watch_training.lock"
    )

    # Pre-seed state so next_full_check_at is far in the future -- fast path.
    (tmp_path / ".claude" / "state").mkdir(parents=True)
    (tmp_path / ".claude" / "state" / "watcher_state.json").write_text(
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
    assert not (tmp_path / ".claude" / "state" / "watcher.log").exists()


def test_main_runs_full_check_and_logs_when_due(tmp_path, monkeypatch):
    import video_encoding.scripts.watch_training as watch_mod
    from video_encoding.watcher.tick import TickResult

    monkeypatch.setattr(watch_mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(watch_mod, "STATE_DIR", tmp_path / ".claude" / "state")
    monkeypatch.setattr(watch_mod, "RESULTS_DIR", tmp_path / ".claude" / "results")
    monkeypatch.setattr(
        watch_mod, "LOG_PATH", tmp_path / ".claude" / "state" / "watcher.log"
    )
    monkeypatch.setattr(
        watch_mod, "LOCK_PATH", tmp_path / ".claude" / "state" / "watch_training.lock"
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
    log_text = (tmp_path / ".claude" / "state" / "watcher.log").read_text()
    assert "flags=0" in log_text


def test_main_skips_when_lock_held(tmp_path, monkeypatch):
    import video_encoding.scripts.watch_training as watch_mod

    monkeypatch.setattr(watch_mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(watch_mod, "STATE_DIR", tmp_path / ".claude" / "state")
    monkeypatch.setattr(watch_mod, "RESULTS_DIR", tmp_path / ".claude" / "results")
    monkeypatch.setattr(
        watch_mod, "LOG_PATH", tmp_path / ".claude" / "state" / "watcher.log"
    )
    lock_path = tmp_path / ".claude" / "state" / "watch_training.lock"
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
