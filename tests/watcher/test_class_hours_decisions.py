"""Watcher decisions under the real class-hours gate, and the gate's exit codes.

conftest.py pins class_hours.blocked open for this module, so each test here
sets the gate explicitly: CLOSED (a reason string) or OPEN (None).
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import pytest

from stern.watcher import class_hours
from stern.watcher.class_hours import PARIS
from stern.watcher.detectors import Flag
from stern.watcher.registry import JobRecord
from stern.watcher.remediate import remediate_dead_job
from stern.watcher.state import WatcherState
from stern.watcher.tick import run_full_check

REASON = "class hours: test"
# Captured at import; conftest only swaps the gate inside a fixture, so this is the real rule.
REAL_BLOCKED = class_hours.blocked
NOW = 1_000_000.0


def _set_gate(monkeypatch, closed: bool):
    reason = REASON if closed else None
    monkeypatch.setattr(class_hours, "blocked", lambda *a, **k: reason)


def _record():
    return JobRecord("100", "exp_a", "tagA", "2026-07-15T09:00:00Z", "node001")


@pytest.fixture
def ticked(tmp_path, monkeypatch):
    """Returns a callable running one tick; records remediate/escalate calls."""
    from stern.watcher import tick

    calls = {"resubmit": [], "escalate": []}

    def fake_remediate(record, *, repo_root):
        calls["resubmit"].append(record.slurm_job_id)
        return "200"

    def fake_escalate(flag, **_kwargs):
        calls["escalate"].append(flag.subject)
        return tmp_path / "transcript.md"

    monkeypatch.setattr(tick, "load_registry", lambda _p: [_record()])
    monkeypatch.setattr(tick, "load_squeue_snapshot", lambda _p: {})
    monkeypatch.setattr(tick, "result_json_mtimes", lambda *_a: {})
    monkeypatch.setattr(tick, "collect_disk_usage_pct", lambda _nodes: {})
    monkeypatch.setattr(tick, "remediate_dead_job", fake_remediate)
    monkeypatch.setattr(tick, "run_escalation", fake_escalate)

    def run(*, dead=(), stalled=(), state=None):
        monkeypatch.setattr(tick, "detect_dead", lambda *a, **k: list(dead))
        monkeypatch.setattr(tick, "detect_stalled", lambda *a, **k: list(stalled))
        state = state if state is not None else WatcherState()
        state_dir = tmp_path / "state"
        state_dir.mkdir(exist_ok=True)
        result = run_full_check(
            repo_root=tmp_path,
            state_dir=state_dir,
            results_root=tmp_path,
            now=NOW,
            state=state,
        )
        return result, state

    return run, calls


DEAD = Flag(subject="100", kind="dead", detail="job gone")
STALLED = Flag(subject="100", kind="stalled", detail="no progress")


def test_dead_job_is_neither_resubmitted_nor_escalated_while_gate_closed(
    monkeypatch, ticked
):
    run, calls = ticked
    _set_gate(monkeypatch, closed=True)
    result, state = run(dead=[DEAD])
    assert calls == {"resubmit": [], "escalate": []}
    assert result.actions == []
    assert result.action_taken is False
    assert result.num_flags == 0
    assert state.jobs == {}


def test_dead_job_is_resubmitted_once_gate_opens(monkeypatch, ticked):
    run, calls = ticked
    _set_gate(monkeypatch, closed=False)
    result, state = run(dead=[DEAD])
    assert calls == {"resubmit": ["100"], "escalate": []}
    assert result.actions == ["resubmit 100 -> 200"]
    assert state.jobs["200"].auto_resubmit_count == 1
    assert state.jobs["100"].status == "resolved"


def test_stalled_job_is_not_escalated_while_gate_closed(monkeypatch, ticked):
    run, calls = ticked
    _set_gate(monkeypatch, closed=True)
    result, _ = run(stalled=[STALLED])
    assert calls["escalate"] == []
    assert result.actions == []


def test_stalled_job_is_escalated_once_gate_opens(monkeypatch, ticked):
    run, calls = ticked
    _set_gate(monkeypatch, closed=False)
    result, _ = run(stalled=[STALLED])
    assert calls["escalate"] == ["100"]
    assert result.actions[0].startswith("escalate 100 (stalled)")


def test_dead_job_already_resubmitted_goes_to_escalation_only_when_open(
    monkeypatch, ticked
):
    run, calls = ticked
    state = WatcherState()
    state.job("100").auto_resubmit_count = 1
    _set_gate(monkeypatch, closed=True)
    run(dead=[DEAD], state=state)
    assert calls == {"resubmit": [], "escalate": []}

    _set_gate(monkeypatch, closed=False)
    run(dead=[DEAD], state=state)
    assert calls == {"resubmit": [], "escalate": ["100"]}


def test_remediate_dead_job_submits_nothing_while_gate_closed(monkeypatch, tmp_path):
    _set_gate(monkeypatch, closed=True)
    with patch("subprocess.run") as run:
        assert remediate_dead_job(_record(), repo_root=tmp_path) is None
    run.assert_not_called()


def test_remediate_dead_job_runs_sbatch_on_same_node_once_gate_opens(
    monkeypatch, tmp_path
):
    _set_gate(monkeypatch, closed=False)
    done = type("R", (), {"returncode": 0, "stdout": "Submitted batch job 201\n"})
    with patch("subprocess.run", return_value=done) as run:
        assert remediate_dead_job(_record(), repo_root=tmp_path) == "201"
    cmd = run.call_args.args[0]
    assert cmd[0] == "sbatch"
    assert "--nodelist=node001" in cmd
    assert cmd[-1] == "+run_tag=tagA"


class _FixedDatetime(datetime):
    """datetime whose now() returns a fixed Paris wall-clock time."""

    fixed: datetime

    @classmethod
    def now(cls, tz=None):
        return cls.fixed


def _exit_code_at(monkeypatch, when: datetime, argv: list[str]) -> int:
    _FixedDatetime.fixed = when.replace(tzinfo=PARIS)
    monkeypatch.setattr(class_hours, "datetime", _FixedDatetime)
    monkeypatch.setattr(class_hours, "blocked", REAL_BLOCKED)
    return class_hours.main(argv)


# 2026-09-21 is a Monday.
@pytest.mark.parametrize(
    ("hour", "minute", "argv", "exit_code"),
    [
        pytest.param(10, 0, [], 1, id="monday-10h-default-walltime-refused"),
        pytest.param(21, 0, [], 0, id="monday-21h-allowed"),
        pytest.param(7, 0, [], 1, id="monday-7h-2h-job-runs-into-class"),
        pytest.param(7, 0, ["--walltime", "00:30:00"], 0, id="monday-7h-30min-ends-before-8"),
        pytest.param(7, 0, ["--walltime", "UNLIMITED"], 1, id="unparseable-walltime-fails-closed"),
    ],
)
def test_script_exit_code_follows_the_gate(monkeypatch, hour, minute, argv, exit_code):
    assert _exit_code_at(monkeypatch, datetime(2026, 9, 21, hour, minute), argv) == exit_code


def test_script_refusal_message_goes_to_stderr(monkeypatch, capsys):
    _exit_code_at(monkeypatch, datetime(2026, 9, 21, 10, 0), [])
    err = capsys.readouterr().err
    assert err.startswith("REFUSED -- class hours:")
    assert "would overlap Mon 08:00-20:00" in err
