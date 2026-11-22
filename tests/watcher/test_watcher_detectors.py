from __future__ import annotations

import json

from stern.watcher.detectors import (
    detect_dead,
    detect_drained_nodes,
    detect_foreign_gpu_process,
    detect_nan,
    detect_quota,
    detect_stalled,
)
from stern.watcher.registry import JobRecord
from stern.watcher.state import JobState, WatcherState

REGISTRY = [
    JobRecord("1", "exp_a", "tagA", "2026-07-15T09:00:00Z", "node001"),
    JobRecord("2", "exp_b", None, "2026-07-15T09:05:00Z", "node047"),
]


def test_detect_dead_flags_job_not_in_squeue_with_no_result():
    flags = detect_dead(
        REGISTRY, squeue={}, result_mtimes={}, job_states={}, now=2000.0
    )
    ids = {f.subject for f in flags}
    assert ids == {"1", "2"}
    assert all(f.kind == "dead" for f in flags)


def test_detect_dead_skips_job_present_in_squeue():
    squeue = {"1": {"state": "RUNNING", "nodelist": "node001"}}
    flags = detect_dead(
        REGISTRY, squeue=squeue, result_mtimes={}, job_states={}, now=2000.0
    )
    assert {f.subject for f in flags} == {"2"}


def test_detect_dead_does_not_flag_a_requeued_pending_job():
    """Pins existing-and-correct behaviour once restart_cnt is threaded
    through the snapshot: a requeued job is still in squeue, so the
    existing `if jid in squeue: continue` already treats it as not dead."""
    squeue = {"1": {"state": "PENDING", "nodelist": "", "restart_cnt": 1}}
    flags = detect_dead(
        REGISTRY, squeue=squeue, result_mtimes={}, job_states={}, now=2000.0
    )
    assert {f.subject for f in flags} == {"2"}


def test_detect_dead_skips_job_that_finished_with_a_result():
    flags = detect_dead(
        REGISTRY, squeue={}, result_mtimes={"1": 1900.0}, job_states={}, now=2000.0
    )
    assert {f.subject for f in flags} == {"2"}


def test_detect_dead_classifies_as_dead_after_resubmit_when_already_resubmitted_once():
    job_states = {"1": JobState("1", auto_resubmit_count=1)}
    flags = detect_dead(
        REGISTRY, squeue={}, result_mtimes={}, job_states=job_states, now=2000.0
    )
    flag_by_id = {f.subject: f for f in flags}
    assert flag_by_id["1"].kind == "dead_after_resubmit"
    assert flag_by_id["2"].kind == "dead"


def test_detect_dead_skips_job_already_resolved_or_escalated():
    job_states = {
        "1": JobState("1", status="resolved"),
        "2": JobState("2", status="escalated"),
    }
    flags = detect_dead(
        REGISTRY, squeue={}, result_mtimes={}, job_states=job_states, now=2000.0
    )
    assert flags == []


def test_detect_stalled_flags_running_job_with_old_result():
    squeue = {"1": {"state": "RUNNING", "nodelist": "node001"}}
    flags = detect_stalled(
        squeue,
        result_mtimes={"1": 0.0},
        registry=REGISTRY,
        stall_threshold_s=2700,
        now=3000.0,
    )
    assert len(flags) == 1
    assert flags[0].subject == "1"
    assert flags[0].kind == "stalled"


def test_detect_stalled_does_not_flag_fresh_result():
    squeue = {"1": {"state": "RUNNING", "nodelist": "node001"}}
    flags = detect_stalled(
        squeue,
        result_mtimes={"1": 2900.0},
        registry=REGISTRY,
        stall_threshold_s=2700,
        now=3000.0,
    )
    assert flags == []


def test_detect_stalled_ignores_non_running_jobs():
    squeue = {"1": {"state": "PENDING", "nodelist": ""}}
    flags = detect_stalled(squeue, result_mtimes={}, registry=REGISTRY, now=100000.0)
    assert flags == []


def test_detect_stalled_uses_submitted_at_before_first_result_exists():
    # Job started at 2026-07-15T09:00:00Z with no result yet -- must not be
    # flagged before stall_threshold_s has elapsed since submission.
    squeue = {"1": {"state": "RUNNING", "nodelist": "node001"}}
    submitted_epoch = 1_800_000_000.0  # arbitrary reference instant
    import stern.watcher.detectors as det

    orig = det._parse_iso
    det._parse_iso = lambda _: submitted_epoch
    try:
        flags_before = detect_stalled(
            squeue,
            result_mtimes={},
            registry=REGISTRY,
            stall_threshold_s=2700,
            now=submitted_epoch + 100,
        )
        flags_after = detect_stalled(
            squeue,
            result_mtimes={},
            registry=REGISTRY,
            stall_threshold_s=2700,
            now=submitted_epoch + 2701,
        )
    finally:
        det._parse_iso = orig
    assert flags_before == []
    assert len(flags_after) == 1


def test_detect_quota_flags_above_threshold():
    flags = detect_quota({"home": 92.0, "node:node001": 40.0}, threshold_pct=90)
    assert len(flags) == 1
    assert flags[0].subject == "quota:home"
    assert flags[0].kind == "quota"


def test_detect_quota_no_flags_below_threshold():
    assert detect_quota({"home": 50.0}, threshold_pct=90) == []


def _write_metrics(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")


def _round_record(node, round_num, train_loss):
    return {
        "run_id": "run1",
        "node": node,
        "round": round_num,
        "train_loss": train_loss,
        "val_loss": 0.1,
        "samp_s": 10.0,
        "n_shards": 4,
        "ts": 1000.0 + round_num,
    }


def test_detect_nan_flags_node_with_nan_train_loss(tmp_path):
    _write_metrics(
        tmp_path / "metrics" / "run1" / "nodeA.jsonl",
        [_round_record("nodeA", 1, float("nan"))],
    )
    flags = detect_nan("run1", tmp_path, WatcherState())
    assert len(flags) == 1
    assert flags[0].subject == "nan:run1:nodeA"
    assert flags[0].kind == "nan"


def test_detect_nan_flags_node_with_inf_train_loss(tmp_path):
    _write_metrics(
        tmp_path / "metrics" / "run1" / "nodeA.jsonl",
        [_round_record("nodeA", 1, float("inf"))],
    )
    flags = detect_nan("run1", tmp_path, WatcherState())
    assert len(flags) == 1


def test_detect_nan_ignores_node_with_normal_loss(tmp_path):
    _write_metrics(
        tmp_path / "metrics" / "run1" / "nodeA.jsonl",
        [_round_record("nodeA", 1, 0.42)],
    )
    assert detect_nan("run1", tmp_path, WatcherState()) == []


def test_detect_nan_uses_only_latest_round_not_a_stale_earlier_nan(tmp_path):
    _write_metrics(
        tmp_path / "metrics" / "run1" / "nodeA.jsonl",
        [_round_record("nodeA", 1, float("nan")), _round_record("nodeA", 2, 0.3)],
    )
    assert detect_nan("run1", tmp_path, WatcherState()) == []


def test_detect_nan_returns_empty_when_no_metrics_dir(tmp_path):
    assert detect_nan("run1", tmp_path, WatcherState()) == []


def test_detect_nan_skips_node_already_resolved(tmp_path):
    _write_metrics(
        tmp_path / "metrics" / "run1" / "nodeA.jsonl",
        [_round_record("nodeA", 1, float("nan"))],
    )
    state = WatcherState()
    state.jobs["nan:run1:nodeA"] = JobState("nan:run1:nodeA", status="resolved")
    assert detect_nan("run1", tmp_path, state) == []


def test_detect_nan_classifies_as_nan_after_retry_once_already_retried(tmp_path):
    _write_metrics(
        tmp_path / "metrics" / "run1" / "nodeA.jsonl",
        [_round_record("nodeA", 2, float("nan"))],
    )
    state = WatcherState()
    state.jobs["nan:run1:nodeA"] = JobState("nan:run1:nodeA", auto_resubmit_count=1)
    flags = detect_nan("run1", tmp_path, state)
    assert len(flags) == 1
    assert flags[0].kind == "nan_after_retry"


def test_detect_nan_annotates_poisoned_merge_after_syncing_transition(tmp_path):
    _write_metrics(
        tmp_path / "metrics" / "run1" / "nodeA.jsonl",
        [_round_record("nodeA", 1, float("nan"))],
    )
    state = WatcherState()
    run = state.run("run1")
    run.status = "SYNCING"
    run.round = 1
    flags = detect_nan("run1", tmp_path, state)
    assert len(flags) == 1
    assert "MERGING->SYNCING" in flags[0].detail


def test_detect_nan_does_not_annotate_poisoned_merge_before_round_one(tmp_path):
    _write_metrics(
        tmp_path / "metrics" / "run1" / "nodeA.jsonl",
        [_round_record("nodeA", 1, float("nan"))],
    )
    state = WatcherState()
    run = state.run("run1")
    run.status = "SYNCING"
    run.round = 0
    flags = detect_nan("run1", tmp_path, state)
    assert len(flags) == 1
    assert "MERGING->SYNCING" not in flags[0].detail


def test_detect_foreign_gpu_process_flags_unowned_pid():
    flag = detect_foreign_gpu_process(
        "node001", "123", gpu_pids={111, 222}, job_pid_tree={111}
    )
    assert flag is not None
    assert flag.kind == "foreign_gpu_process"


def test_detect_foreign_gpu_process_none_when_all_pids_owned():
    assert (
        detect_foreign_gpu_process("node001", "123", gpu_pids={111}, job_pid_tree={111})
        is None
    )


def test_detect_foreign_gpu_process_none_on_probe_failure():
    assert (
        detect_foreign_gpu_process("node001", "123", gpu_pids=None, job_pid_tree={111})
        is None
    )


def test_detect_nan_skips_run_already_marked_done(tmp_path):
    """A DONE run's workers are gone -- its last metrics line is whatever
    was written at finish time and will never change again. Without this
    guard, a run that happened to finish on a NaN'd round gets re-flagged
    (and re-escalated) on every single future tick forever, observed live
    on login-node 2026-08-06 for pool-cars-2r/node052."""
    _write_metrics(
        tmp_path / "metrics" / "run1" / "nodeA.jsonl",
        [_round_record("nodeA", 5, float("nan"))],
    )
    state = WatcherState()
    state.run("run1").status = "DONE"
    assert detect_nan("run1", tmp_path, state) == []


def test_detect_nan_ignores_a_line_left_over_from_a_previous_attempt(tmp_path):
    """A relaunch reuses the metrics dir, so before a worker appends its
    first round the last line is still the failed attempt's NaN. Flagging it
    excluded 8 healthy nodes on login-node 2026-08-21 (node039's file held
    round 38 NaN from job 60 next to a healthy round 1 from job 162)."""
    _write_metrics(
        tmp_path / "metrics" / "run1" / "nodeA.jsonl",
        [_round_record("nodeA", 38, float("nan"))],  # ts = 1038.0
    )
    state = WatcherState()
    state.run("run1").round_started_at = 2000.0
    assert detect_nan("run1", tmp_path, state) == []


def test_detect_nan_still_flags_a_nan_written_during_the_current_round(tmp_path):
    _write_metrics(
        tmp_path / "metrics" / "run1" / "nodeA.jsonl",
        [_round_record("nodeA", 1, float("nan"))],  # ts = 1001.0
    )
    state = WatcherState()
    state.run("run1").round_started_at = 1000.0
    flags = detect_nan("run1", tmp_path, state)
    assert len(flags) == 1
    assert flags[0].subject == "nan:run1:nodeA"


def test_detect_drained_nodes_flags_drained_state():
    flags = detect_drained_nodes(
        {"node001": "drained", "node002": "idle"}, roster=["node001", "node002"]
    )
    assert len(flags) == 1
    assert flags[0].subject == "drain:node001"
    assert flags[0].kind == "node_drained"


def test_detect_drained_nodes_flags_down_state():
    flags = detect_drained_nodes({"node001": "down"}, roster=["node001"])
    assert len(flags) == 1
    assert flags[0].subject == "drain:node001"


def test_detect_drained_nodes_flags_draining_state():
    flags = detect_drained_nodes({"node001": "draining"}, roster=["node001"])
    assert len(flags) == 1
    assert flags[0].subject == "drain:node001"


def test_detect_drained_nodes_ignores_idle_and_mixed_states():
    flags = detect_drained_nodes(
        {"node001": "idle", "node002": "mixed"}, roster=["node001", "node002"]
    )
    assert flags == []


def test_detect_drained_nodes_only_checks_roster_nodes():
    flags = detect_drained_nodes(
        {"node001": "drained", "node002": "drained"}, roster=["node002"]
    )
    assert {f.subject for f in flags} == {"drain:node002"}


def test_detect_drained_nodes_ignores_node_absent_from_sinfo():
    flags = detect_drained_nodes({}, roster=["node001"])
    assert flags == []
