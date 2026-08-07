from datetime import datetime, timezone

from video_encoding.scripts.audit_watcher_log import (
    detect_watcher_gaps,
    parse_merge_event,
    summarize_merge_events,
)


def test_parse_merge_event_full_checkin():
    line = (
        "2026-08-05T14:30:00+00:00 mesh: sw2-birds-lr070m080: "
        "AWAITING_CHECKPOINTS -> MERGING (16 nodes)"
    )
    event = parse_merge_event(line)
    assert event == {
        "timestamp": datetime(2026, 8, 5, 14, 30, 0, tzinfo=timezone.utc),
        "run_id": "sw2-birds-lr070m080",
        "kind": "full",
    }


def test_parse_merge_event_quorum():
    line = (
        "2026-08-05T14:31:00+00:00 mesh: sw3-birds-quorum12: "
        "AWAITING_CHECKPOINTS -> MERGING (quorum 12/16 reached, 12 nodes)"
    )
    event = parse_merge_event(line)
    assert event["kind"] == "quorum"
    assert event["run_id"] == "sw3-birds-quorum12"


def test_parse_merge_event_timeout():
    line = (
        "2026-08-05T14:32:00+00:00 mesh: sw2-fish-lr090m090: "
        "AWAITING_CHECKPOINTS -> MERGING (timeout, missing ['node009', 'node011'])"
    )
    event = parse_merge_event(line)
    assert event["kind"] == "timeout"


def test_parse_merge_event_ignores_unrelated_lines():
    assert (
        parse_merge_event(
            "2026-08-05T14:33:00+00:00 swept 2 orphaned pending checkpoint(s)"
        )
        is None
    )


def test_summarize_counts_kinds_and_wait_gap_per_run():
    lines = [
        "2026-08-05T14:00:00+00:00 mesh: run-a: AWAITING_CHECKPOINTS -> MERGING (16 nodes)",
        "2026-08-05T14:20:00+00:00 mesh: run-a: AWAITING_CHECKPOINTS -> MERGING (quorum 12/16 reached, 12 nodes)",
    ]
    summary = summarize_merge_events(lines)
    assert summary["run-a"]["full"] == 1
    assert summary["run-a"]["quorum"] == 1
    assert summary["run-a"]["timeout"] == 0
    assert summary["run-a"]["wait_s_between_merges"] == [1200.0]


def test_detect_watcher_gaps_flags_outage():
    lines = [
        "2026-08-05T14:00:00+00:00 mesh: run-a: AWAITING_CHECKPOINTS -> MERGING (16 nodes)",
        "2026-08-05T18:00:00+00:00 mesh: run-a: AWAITING_CHECKPOINTS -> MERGING (16 nodes)",
    ]
    gaps = detect_watcher_gaps(lines, threshold_s=120.0)
    assert len(gaps) == 1
    assert gaps[0]["gap_s"] == 4 * 3600.0


def test_detect_watcher_gaps_ignores_normal_cadence():
    lines = [
        "2026-08-05T14:00:00+00:00 flags=0 actions=none next_interval_s=60",
        "2026-08-05T14:01:00+00:00 flags=0 actions=none next_interval_s=60",
    ]
    assert detect_watcher_gaps(lines, threshold_s=120.0) == []
