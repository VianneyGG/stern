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
from video_encoding.watcher.state import JobState, WatcherState


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

    from video_encoding.watcher.escalate import _claude_bin

    cmd = captured["cmd"]
    assert cmd[0] == _claude_bin()
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


def _fake_detached_popen(captured):
    def _popen(cmd, *, stdout, stderr, text, cwd=None, start_new_session=None):
        captured["cmd"] = cmd
        captured["start_new_session"] = start_new_session
        stdout.write("simulated claude -p transcript\n")
        stdout.close()
        return object()  # remediate_nan_node must never call .wait() on this

    return _popen


def test_remediate_nan_node_excludes_node_and_retries_once(tmp_path):
    from video_encoding.watcher.escalate import remediate_nan_node

    flag = Flag("nan:run1:nodeA", "nan", "nodeA: train_loss=nan round=3 run_id=run1")
    state = WatcherState()
    captured = {}

    with patch("subprocess.Popen", side_effect=_fake_detached_popen(captured)):
        result = remediate_nan_node(
            flag,
            run_id="run1",
            node="nodeA",
            round_num=3,
            state=state,
            repo_root=tmp_path,
            transcript_dir=tmp_path / "escalations",
            now=1000.0,
            now_str="20260805-100000",
        )

    assert result == "retried"
    assert "nodeA" in state.excluded_nodes
    assert state.excluded_nodes["nodeA"].endswith("of run1")
    assert state.jobs["nan:run1:nodeA"].auto_resubmit_count == 1
    assert state.nan_escalations_in_flight["run1"] == 1000.0
    assert "cmd" in captured  # claude -p was actually spawned


def test_remediate_nan_node_never_waits_on_the_spawned_process(tmp_path):
    from video_encoding.watcher.escalate import remediate_nan_node

    flag = Flag("nan:run1:nodeA", "nan", "nodeA: train_loss=nan round=1 run_id=run1")
    state = WatcherState()

    class _NoWaitAllowed:
        def wait(self):
            raise AssertionError(
                "remediate_nan_node must spawn detached, never block on .wait()"
            )

    def _popen(cmd, *, stdout, stderr, text, cwd=None, start_new_session=None):
        stdout.write("ok\n")
        stdout.close()
        return _NoWaitAllowed()

    with patch("subprocess.Popen", side_effect=_popen):
        remediate_nan_node(
            flag,
            run_id="run1",
            node="nodeA",
            round_num=1,
            state=state,
            repo_root=tmp_path,
            transcript_dir=tmp_path / "escalations",
            now=1000.0,
            now_str="20260805-100000",
        )


def test_remediate_nan_node_second_nan_drains_and_never_resubmits(tmp_path):
    from video_encoding.watcher.escalate import remediate_nan_node

    state = WatcherState()
    state.jobs["nan:run1:nodeA"] = JobState("nan:run1:nodeA", auto_resubmit_count=1)
    flag = Flag(
        "nan:run1:nodeA", "nan_after_retry", "nodeA: train_loss=nan round=5 run_id=run1"
    )

    with patch("subprocess.Popen") as mock_popen:
        result = remediate_nan_node(
            flag,
            run_id="run1",
            node="nodeA",
            round_num=5,
            state=state,
            repo_root=tmp_path,
            transcript_dir=tmp_path / "escalations",
            now=2000.0,
            now_str="20260805-110000",
        )

    assert result == "drained"
    assert "nodeA" in state.excluded_nodes  # still excluded
    assert state.jobs["nan:run1:nodeA"].status == "resolved"
    mock_popen.assert_not_called()


def test_remediate_nan_node_concurrency_cap_blocks_second_simultaneous_node(tmp_path):
    from video_encoding.watcher.escalate import remediate_nan_node

    state = WatcherState()
    captured = {}
    flag_a = Flag("nan:run1:nodeA", "nan", "nodeA: train_loss=nan round=3 run_id=run1")
    flag_b = Flag("nan:run1:nodeB", "nan", "nodeB: train_loss=nan round=3 run_id=run1")

    with patch("subprocess.Popen", side_effect=_fake_detached_popen(captured)):
        first = remediate_nan_node(
            flag_a,
            run_id="run1",
            node="nodeA",
            round_num=3,
            state=state,
            repo_root=tmp_path,
            transcript_dir=tmp_path / "escalations",
            now=1000.0,
            now_str="20260805-100000",
        )
        with patch("subprocess.Popen") as mock_popen_second:
            second = remediate_nan_node(
                flag_b,
                run_id="run1",
                node="nodeB",
                round_num=3,
                state=state,
                repo_root=tmp_path,
                transcript_dir=tmp_path / "escalations",
                now=1005.0,
                now_str="20260805-100005",
            )

    assert first == "retried"
    assert second == "capped"
    mock_popen_second.assert_not_called()
    # nodeB is still excluded from the round even though escalation was capped
    assert "nodeB" in state.excluded_nodes
    assert (
        "nodeB" not in state.jobs
        or state.jobs["nan:run1:nodeB"].auto_resubmit_count == 0
    )


def test_remediate_nan_node_concurrency_cap_clears_after_timeout(tmp_path):
    from video_encoding.watcher.escalate import (
        DEFAULT_NAN_CONCURRENCY_TIMEOUT_S,
        remediate_nan_node,
    )

    state = WatcherState()
    state.nan_escalations_in_flight["run1"] = 1000.0
    flag = Flag("nan:run1:nodeB", "nan", "nodeB: train_loss=nan round=3 run_id=run1")
    captured = {}

    with patch("subprocess.Popen", side_effect=_fake_detached_popen(captured)):
        result = remediate_nan_node(
            flag,
            run_id="run1",
            node="nodeB",
            round_num=3,
            state=state,
            repo_root=tmp_path,
            transcript_dir=tmp_path / "escalations",
            now=1000.0 + DEFAULT_NAN_CONCURRENCY_TIMEOUT_S + 1,
            now_str="20260805-120000",
        )

    assert result == "retried"
    assert "cmd" in captured


def test_remediate_nan_node_dry_run_excludes_but_takes_no_action(tmp_path):
    from video_encoding.watcher.escalate import remediate_nan_node

    state = WatcherState()
    flag = Flag("nan:run1:nodeA", "nan", "nodeA: train_loss=nan round=3 run_id=run1")
    transcript_dir = tmp_path / "escalations"

    with patch("subprocess.Popen") as mock_popen:
        result = remediate_nan_node(
            flag,
            run_id="run1",
            node="nodeA",
            round_num=3,
            state=state,
            repo_root=tmp_path,
            transcript_dir=transcript_dir,
            now=1000.0,
            now_str="20260805-100000",
            dry_run=True,
        )

    report = transcript_dir / "nan-run1-nodeA-20260805-100000.md"
    assert report.exists()
    assert "DRY RUN" in report.read_text()
    assert flag.detail in report.read_text()

    assert result == "dry_run"
    assert "nodeA" in state.excluded_nodes
    mock_popen.assert_not_called()
    assert state.jobs.get("nan:run1:nodeA") is None or (
        state.jobs["nan:run1:nodeA"].auto_resubmit_count == 0
    )
