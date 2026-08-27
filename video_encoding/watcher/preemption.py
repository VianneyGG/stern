"""Surface nodes that preemption has quietly turned into freeloaders.

A node whose round is merged without it abandons that round and resyncs
(`mesh_round.round_already_merged`), which is what stops one slow node from
dragging its pool's cadence. The cost is that the node keeps a whole GPU
allocated while contributing to no merge, and nothing about that is visible
in the state machine -- the run looks perfectly healthy, because it is: it
just isn't using 1/16th of its roster.

This reads what the workers already write. No new state file.
"""

from __future__ import annotations

import json
from pathlib import Path

# Matches mesh_round._ABANDON_ALERT_STREAK. Three consecutive is past the
# point where a transient explains it -- node055 was contended for 23 rounds
# and node032 for 19 on 2026-08-22, both recovering unaided afterwards.
DEFAULT_STREAK = 3


def trailing_abandons(records: list[dict]) -> int:
    """How many of the most recent rounds this node abandoned, unbroken.

    Counts backwards from the newest round, so a node that was preempted
    early and has since recovered reports 0 -- the number answers "is it
    contributing now", not "has it ever failed". Ordered by round rather
    than by file order: a chunk relaunch starts a fresh process appending to
    the same file.
    """
    streak = 0
    for record in sorted(records, key=lambda r: r.get("round", 0), reverse=True):
        if not record.get("abandoned"):
            break
        streak += 1
    return streak


def starving_nodes(
    metrics_dir: Path, run_id: str, *, streak: int = DEFAULT_STREAK
) -> dict[str, int]:
    """{node: consecutive abandoned rounds} for nodes at or past `streak`.

    Never raises: a malformed or half-written JSONL line must not take the
    watcher tick down, so unparseable lines are skipped and an unreadable
    file is treated as "nothing to report". This is a diagnostic, and a
    diagnostic that can crash the supervisor is worse than no diagnostic.
    """
    run_dir = Path(metrics_dir) / run_id
    if not run_dir.is_dir():
        return {}
    starving: dict[str, int] = {}
    for path in sorted(run_dir.glob("*.jsonl")):
        records = []
        try:
            for line in path.read_text().splitlines():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except OSError:
            continue
        if not records:
            continue
        count = trailing_abandons(records)
        if count >= streak:
            starving[path.stem] = count
    return starving
