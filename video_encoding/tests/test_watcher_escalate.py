# video_encoding/tests/test_watcher_escalate.py
from __future__ import annotations

from unittest.mock import patch

from video_encoding.watcher.detectors import Flag
from video_encoding.watcher.escalate import (
    DEFAULT_ESCALATION_BUDGET_PER_DAY,
    DEFAULT_ESCALATION_COOLDOWN_S,
    build_prompt,
    record_escalation,
    run_escalation,
    should_escalate,
)
from video_encoding.watcher.registry import JobRecord
from video_encoding.watcher.state import WatcherState


def test_should_escalate_true_when_never_escalated():
    state = WatcherState()
    assert should_escalate(state, "36901", now=1000.0) is True


def test_should_escalate_false_within_cooldown():
    state = WatcherState()
    state.job("36901").last_escalation_at = 1000.0
    assert (
        should_escalate(state, "36901", now=1000.0 + DEFAULT_ESCALATION_COOLDOWN_S - 1)
        is False
    )


def test_should_escalate_true_after_cooldown_elapses():
    state = WatcherState()
    state.job("36901").last_escalation_at = 1000.0
    assert (
        should_escalate(state, "36901", now=1000.0 + DEFAULT_ESCALATION_COOLDOWN_S + 1)
        is True
    )


def test_should_escalate_false_when_daily_budget_exhausted():
    state = WatcherState(
        escalation_day="2026-07-15",
        escalation_count_today=DEFAULT_ESCALATION_BUDGET_PER_DAY,
    )
    now = 1_800_000_000.0  # 2026-01-15-ish; exact date doesn't matter, only that it matches escalation_day
    with patch("video_encoding.watcher.escalate._today_utc", return_value="2026-07-15"):
        assert should_escalate(state, "new-subject", now=now) is False


def test_should_escalate_resets_budget_on_new_day():
    state = WatcherState(
        escalation_day="2026-07-14",
        escalation_count_today=DEFAULT_ESCALATION_BUDGET_PER_DAY,
    )
    with patch("video_encoding.watcher.escalate._today_utc", return_value="2026-07-15"):
        assert should_escalate(state, "new-subject", now=0.0) is True


def test_record_escalation_increments_budget_and_sets_last_escalation_at():
    state = WatcherState()
    with patch("video_encoding.watcher.escalate._today_utc", return_value="2026-07-15"):
        record_escalation(state, "36901", now=1234.0)
    assert state.escalation_day == "2026-07-15"
    assert state.escalation_count_today == 1
    assert state.jobs["36901"].last_escalation_at == 1234.0
    assert state.jobs["36901"].status == "escalated"


def test_record_escalation_rolls_over_on_new_day():
    state = WatcherState(escalation_day="2026-07-14", escalation_count_today=5)
    with patch("video_encoding.watcher.escalate._today_utc", return_value="2026-07-15"):
        record_escalation(state, "36901", now=0.0)
    assert state.escalation_count_today == 1


def test_build_prompt_includes_job_and_registry_context():
    flag = Flag("36901", "stalled", "job 36901 RUNNING but no progress in 3000s")
    record = JobRecord(
        "36901",
        "videomae_ssl_pretrain_voxmini",
        "p2-eb256",
        "2026-07-15T09:00:00Z",
        "node001",
    )
    prompt = build_prompt(flag, registry_entry=record)
    assert "36901" in prompt
    assert "stalled" in prompt
    assert "videomae_ssl_pretrain_voxmini" in prompt
    assert "node001" in prompt


def test_build_prompt_handles_missing_registry_entry():
    flag = Flag("quota:home", "quota", "home at 92.0% (threshold 90%)")
    prompt = build_prompt(flag, registry_entry=None)
    assert "quota:home" in prompt
    assert "92.0%" in prompt


def test_run_escalation_writes_transcript_before_returning(tmp_path):
    flag = Flag("36901", "stalled", "no progress in 3000s")
    transcript_dir = tmp_path / "escalations"

    def _fake_popen(cmd, *, stdout, stderr, text, cwd=None):
        stdout.write("simulated claude -p transcript\n")
        stdout.flush()

        class _Proc:
            def wait(self_inner):
                return 0

        return _Proc()

    with patch("subprocess.Popen", side_effect=_fake_popen):
        path = run_escalation(
            flag,
            repo_root=tmp_path,
            transcript_dir=transcript_dir,
            registry_entry=None,
            now_str="20260715-100000",
            max_turns=15,
        )

    assert path.exists()
    assert "simulated claude -p transcript" in path.read_text()
    assert path.name == "20260715-100000-36901.md"


def test_run_escalation_builds_scoped_allowed_tools(tmp_path):
    flag = Flag("36901", "dead_after_resubmit", "died again")
    captured = {}

    def _fake_popen(cmd, *, stdout, stderr, text, cwd=None):
        captured["cmd"] = cmd

        class _Proc:
            def wait(self_inner):
                return 0

        return _Proc()

    with patch("subprocess.Popen", side_effect=_fake_popen):
        run_escalation(
            flag,
            repo_root=tmp_path,
            transcript_dir=tmp_path / "escalations",
            registry_entry=None,
            now_str="20260715-100000",
        )

    cmd = captured["cmd"]
    assert cmd[0] == "claude"
    assert "-p" in cmd
    idx = cmd.index("--allowed-tools")
    tools = cmd[idx + 1]
    assert "Read" in tools and "Grep" in tools and "Glob" in tools
    assert "Edit(video_encoding/conf/**)" in tools
    assert "Bash(video_encoding/scripts/slurm_wrapper.sh:*)" in tools
    assert "--max-turns" in cmd
    assert "15" in cmd


def test_run_escalation_handles_popen_launch_failure(tmp_path):
    flag = Flag("36901", "stalled", "no progress in 3000s")
    transcript_dir = tmp_path / "escalations"

    with patch(
        "subprocess.Popen",
        side_effect=FileNotFoundError("claude: command not found"),
    ):
        path = run_escalation(
            flag,
            repo_root=tmp_path,
            transcript_dir=transcript_dir,
            registry_entry=None,
            now_str="20260715-100000",
            max_turns=15,
        )

    assert path.exists()
    assert "ERROR: failed to launch claude -p" in path.read_text()
    assert "claude: command not found" in path.read_text()
