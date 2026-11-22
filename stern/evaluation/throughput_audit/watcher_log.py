"""Parse runs/state/watcher.log for MERGING events and cron-outage gaps.

The line format comes from `_log()` in stern/watcher/watch_training.py:
    f"{iso_ts} mesh: {run_id}: AWAITING_CHECKPOINTS -> MERGING ({detail})"
`detail` is written by `_do_awaiting_checkpoints` in stern/watcher/mesh_tick.py
and has three shapes: "{N} nodes" (full check-in), "quorum {k}/{eligible}
reached, {N} nodes" (quorum merge), and "timeout, missing {list}..." (timeout).
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

_MERGE_RE = re.compile(
    r"^(?P<ts>\S+) mesh: (?P<run_id>[\w.-]+): "
    r"AWAITING_CHECKPOINTS -> MERGING \((?P<detail>.*)\)$"
)
_ANY_LINE_RE = re.compile(r"^(?P<ts>\S+) ")


def parse_merge_event(line: str) -> dict | None:
    match = _MERGE_RE.match(line)
    if match is None:
        return None
    detail = match.group("detail")
    if detail.startswith("quorum "):
        kind = "quorum"
    elif detail.startswith("timeout"):
        kind = "timeout"
    else:
        kind = "full"
    return {
        "timestamp": datetime.fromisoformat(match.group("ts")),
        "run_id": match.group("run_id"),
        "kind": kind,
    }


def summarize_merge_events(lines: list[str]) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    last_ts: dict[str, datetime] = {}
    for line in lines:
        event = parse_merge_event(line)
        if event is None:
            continue
        run_id = event["run_id"]
        entry = summary.setdefault(
            run_id, {"full": 0, "quorum": 0, "timeout": 0, "wait_s_between_merges": []}
        )
        entry[event["kind"]] += 1
        if run_id in last_ts:
            entry["wait_s_between_merges"].append(
                (event["timestamp"] - last_ts[run_id]).total_seconds()
            )
        last_ts[run_id] = event["timestamp"]
    return summary


def detect_watcher_gaps(lines: list[str], threshold_s: float = 120.0) -> list[dict]:
    timestamps: list[datetime] = []
    for line in lines:
        match = _ANY_LINE_RE.match(line)
        if match is None:
            continue
        try:
            timestamps.append(datetime.fromisoformat(match.group("ts")))
        except ValueError:
            continue
    gaps = []
    for prev, curr in zip(timestamps, timestamps[1:]):
        gap_s = (curr - prev).total_seconds()
        if gap_s > threshold_s:
            gaps.append({"gap_start": prev, "gap_end": curr, "gap_s": gap_s})
    return gaps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("watcher_log_path", type=Path)
    args = parser.parse_args()

    lines = args.watcher_log_path.read_text().splitlines()
    print(
        json.dumps(
            {
                "merge_summary": summarize_merge_events(lines),
                "watcher_gaps": detect_watcher_gaps(lines),
            },
            default=str,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
