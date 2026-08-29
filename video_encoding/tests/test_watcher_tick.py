from __future__ import annotations

import json
from unittest.mock import patch

from video_encoding.watcher.registry import JobRecord, append_job_record
from video_encoding.watcher.state import WatcherState
from video_encoding.watcher.tick import run_full_check


def _setup(tmp_path, *, squeue_jobs=None, registry_records=None):
    state_dir = tmp_path / "state"
    results_root = tmp_path / "results"
    state_dir.mkdir()
    results_root.mkdir()
    (state_dir / "squeue.json").write_text(json.dumps({"jobs": squeue_jobs or []}))
    for record in registry_records or []:
        append_job_record(state_dir / "jobs_registry.json", record)
    return state_dir, results_root


def test_run_full_check_no_flags_advances_backoff_without_actions(tmp_path):
    state_dir, results_root = _setup(tmp_path)
    state = WatcherState(current_interval_s=60.0)

    with patch("video_encoding.watcher.tick.collect_disk_usage_pct", return_value={}):
        result = run_full_check(
            repo_root=tmp_path,
            state_dir=state_dir,
            results_root=results_root,
            now=1000.0,
            state=state,
        )

    assert result.num_flags == 0
    assert result.actions == []
    assert result.action_taken is False
    assert state.current_interval_s == 90.0  # 60 * 1.5


def test_run_full_check_resubmits_dead_job_first_time(tmp_path):
    record = JobRecord("100", "exp_a", "tagA", "2026-07-15T09:00:00Z", "node001")
    state_dir, results_root = _setup(tmp_path, registry_records=[record])
    state = WatcherState()

    with (
        patch("video_encoding.watcher.tick.collect_disk_usage_pct", return_value={}),
        patch(
            "video_encoding.watcher.tick.remediate_dead_job", return_value="101"
        ) as mock_remediate,
    ):
        result = run_full_check(
            repo_root=tmp_path,
            state_dir=state_dir,
            results_root=results_root,
            now=2000.0,
            state=state,
        )

    mock_remediate.assert_called_once_with(record, repo_root=tmp_path)
    assert result.action_taken is True
    assert "resubmit 100 -> 101" in result.actions[0]
    assert state.jobs["100"].status == "resolved"
    assert state.jobs["101"].auto_resubmit_count == 1
    assert state.jobs["101"].status == "healthy"
    assert state.current_interval_s == 60.0  # reset to fast floor


def test_run_full_check_escalates_when_resubmit_attempt_fails(tmp_path):
    record = JobRecord("100", "exp_a", "tagA", "2026-07-15T09:00:00Z", "node001")
    state_dir, results_root = _setup(tmp_path, registry_records=[record])
    state = WatcherState()

    with (
        patch("video_encoding.watcher.tick.collect_disk_usage_pct", return_value={}),
        patch(
            "video_encoding.watcher.tick.remediate_dead_job", return_value=None
        ) as mock_remediate,
        patch(
            "video_encoding.watcher.tick.run_escalation",
            return_value=tmp_path / "transcript.md",
        ) as mock_escalate,
    ):
        result = run_full_check(
            repo_root=tmp_path,
            state_dir=state_dir,
            results_root=results_root,
            now=2000.0,
            state=state,
        )

    mock_remediate.assert_called_once_with(record, repo_root=tmp_path)
    mock_escalate.assert_called_once()
    assert state.jobs["100"].status != "resolved"
    assert state.jobs["100"].status == "escalated"
    assert result.action_taken is True


def test_run_full_check_escalates_dead_after_resubmit_instead_of_resubmitting_again(
    tmp_path,
):
    record = JobRecord("100", "exp_a", "tagA", "2026-07-15T09:00:00Z", "node001")
    state_dir, results_root = _setup(tmp_path, registry_records=[record])
    state = WatcherState()
    state.job("100").auto_resubmit_count = 1  # already resubmitted once

    with (
        patch("video_encoding.watcher.tick.collect_disk_usage_pct", return_value={}),
        patch("video_encoding.watcher.tick.remediate_dead_job") as mock_remediate,
        patch(
            "video_encoding.watcher.tick.run_escalation",
            return_value=tmp_path / "transcript.md",
        ) as mock_escalate,
    ):
        result = run_full_check(
            repo_root=tmp_path,
            state_dir=state_dir,
            results_root=results_root,
            now=2000.0,
            state=state,
        )

    mock_remediate.assert_not_called()
    mock_escalate.assert_called_once()
    assert state.jobs["100"].status == "escalated"
    assert result.action_taken is True


def test_run_full_check_escalates_stalled_job(tmp_path):
    record = JobRecord("100", "exp_a", None, "2026-07-15T09:00:00Z", "node001")
    state_dir, results_root = _setup(
        tmp_path,
        squeue_jobs=[{"job_id": 100, "job_state": ["RUNNING"], "nodes": "node001"}],
        registry_records=[record],
    )
    (results_root / "exp_a.json").write_text("{}")
    import os

    old_time = 1000.0
    os.utime(results_root / "exp_a.json", (old_time, old_time))
    state = WatcherState()

    with (
        patch("video_encoding.watcher.tick.collect_disk_usage_pct", return_value={}),
        patch(
            "video_encoding.watcher.tick.run_escalation",
            return_value=tmp_path / "transcript.md",
        ) as mock_escalate,
    ):
        result = run_full_check(
            repo_root=tmp_path,
            state_dir=state_dir,
            results_root=results_root,
            now=old_time + 2701,
            state=state,
        )

    mock_escalate.assert_called_once()
    assert any("stalled" in a for a in result.actions)
    assert result.action_taken is True


def test_run_full_check_skips_remediation_for_mesh_job(tmp_path):
    record = JobRecord(
        "100",
        "videomae_mesh_ssl",
        "videomae_mesh_ssl-pool-fish",
        "2026-07-15T09:00:00Z",
        "node001",
    )
    state_dir, results_root = _setup(tmp_path, registry_records=[record])
    state = WatcherState()

    with (
        patch("video_encoding.watcher.tick.collect_disk_usage_pct", return_value={}),
        patch("video_encoding.watcher.tick.remediate_dead_job") as mock_remediate,
        patch("video_encoding.watcher.tick.run_escalation") as mock_escalate,
    ):
        result = run_full_check(
            repo_root=tmp_path,
            state_dir=state_dir,
            results_root=results_root,
            now=2000.0,
            state=state,
            mesh_run_ids=frozenset({"videomae_mesh_ssl-pool-fish"}),
        )

    mock_remediate.assert_not_called()
    mock_escalate.assert_not_called()
    assert result.num_flags == 0
    assert result.actions == []
    assert "100" not in state.jobs


def test_run_full_check_still_remediates_non_mesh_job_when_mesh_run_ids_present(
    tmp_path,
):
    record = JobRecord("100", "exp_a", "tagA", "2026-07-15T09:00:00Z", "node001")
    state_dir, results_root = _setup(tmp_path, registry_records=[record])
    state = WatcherState()

    with (
        patch("video_encoding.watcher.tick.collect_disk_usage_pct", return_value={}),
        patch(
            "video_encoding.watcher.tick.remediate_dead_job", return_value="101"
        ) as mock_remediate,
    ):
        result = run_full_check(
            repo_root=tmp_path,
            state_dir=state_dir,
            results_root=results_root,
            now=2000.0,
            state=state,
            mesh_run_ids=frozenset({"videomae_mesh_ssl-pool-fish"}),
        )

    mock_remediate.assert_called_once_with(record, repo_root=tmp_path)
    assert "resubmit 100 -> 101" in result.actions[0]


def test_run_full_check_respects_escalation_cooldown(tmp_path):
    record = JobRecord("100", "exp_a", "tagA", "2026-07-15T09:00:00Z", "node001")
    state_dir, results_root = _setup(tmp_path, registry_records=[record])
    state = WatcherState()
    state.job("100").auto_resubmit_count = 1
    state.job("100").last_escalation_at = 1999.0  # escalated 1s ago -- inside cooldown

    with (
        patch("video_encoding.watcher.tick.collect_disk_usage_pct", return_value={}),
        patch("video_encoding.watcher.tick.run_escalation") as mock_escalate,
    ):
        result = run_full_check(
            repo_root=tmp_path,
            state_dir=state_dir,
            results_root=results_root,
            now=2000.0,
            state=state,
        )

    mock_escalate.assert_not_called()
    assert result.action_taken is False


def test_run_full_check_cleans_up_quota_pressure_and_skips_escalation(tmp_path):
    state_dir, results_root = _setup(tmp_path)
    state = WatcherState()

    with (
        patch(
            "video_encoding.watcher.tick.collect_disk_usage_pct",
            return_value={"node:node058": 95.0},
        ),
        patch(
            "video_encoding.watcher.tick.remediate_quota_pressure",
            return_value="deleted 2 item(s) on node058: done-run.pt, tb",
        ) as mock_cleanup,
        patch("video_encoding.watcher.tick.run_escalation") as mock_escalate,
    ):
        result = run_full_check(
            repo_root=tmp_path,
            state_dir=state_dir,
            results_root=results_root,
            now=2000.0,
            state=state,
        )

    mock_cleanup.assert_called_once()
    assert mock_cleanup.call_args.args[0] == "node:node058"
    mock_escalate.assert_not_called()
    assert any("cleanup quota:node:node058" in a for a in result.actions)
    assert result.action_taken is True


def test_run_full_check_notifies_instead_of_escalating_on_quota_with_nothing_to_delete(
    tmp_path,
):
    state_dir, results_root = _setup(tmp_path)
    state = WatcherState()
    notified = []

    with (
        patch(
            "video_encoding.watcher.tick.collect_disk_usage_pct",
            return_value={"home": 95.0},
        ),
        patch(
            "video_encoding.watcher.tick.remediate_quota_pressure", return_value=None
        ) as mock_cleanup,
        patch(
            "video_encoding.watcher.tick.notify",
            side_effect=lambda subject, detail: notified.append((subject, detail)),
        ),
        patch("video_encoding.watcher.tick.run_escalation") as mock_escalate,
    ):
        result = run_full_check(
            repo_root=tmp_path,
            state_dir=state_dir,
            results_root=results_root,
            now=2000.0,
            state=state,
        )

    mock_cleanup.assert_called_once()
    mock_escalate.assert_not_called()  # claude -p never spawned for this path
    assert notified
    assert result.action_taken is True


def test_run_full_check_escalates_stalled_flag_independently_of_quota_cleanup(
    tmp_path,
):
    record = JobRecord("100", "exp_a", None, "2026-07-15T09:00:00Z", "node001")
    state_dir, results_root = _setup(
        tmp_path,
        squeue_jobs=[{"job_id": 100, "job_state": ["RUNNING"], "nodes": "node001"}],
        registry_records=[record],
    )
    (results_root / "exp_a.json").write_text("{}")
    import os

    old_time = 1000.0
    os.utime(results_root / "exp_a.json", (old_time, old_time))
    state = WatcherState()

    with (
        patch(
            "video_encoding.watcher.tick.collect_disk_usage_pct",
            return_value={"node:node058": 95.0},
        ),
        patch(
            "video_encoding.watcher.tick.remediate_quota_pressure",
            return_value="deleted 1 item(s) on node058: tb",
        ),
        patch(
            "video_encoding.watcher.tick.run_escalation",
            return_value=tmp_path / "transcript.md",
        ) as mock_escalate,
    ):
        result = run_full_check(
            repo_root=tmp_path,
            state_dir=state_dir,
            results_root=results_root,
            now=old_time + 2701,
            state=state,
        )

    mock_escalate.assert_called_once()
    assert mock_escalate.call_args.args[0].kind == "stalled"
    assert any("cleanup quota:node:node058" in a for a in result.actions)
    assert any("stalled" in a for a in result.actions)
