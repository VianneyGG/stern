from __future__ import annotations

from video_encoding.watcher.detectors import detect_dead, detect_quota, detect_stalled
from video_encoding.watcher.registry import JobRecord
from video_encoding.watcher.state import JobState

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
    import video_encoding.watcher.detectors as det

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
