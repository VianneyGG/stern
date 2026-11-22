"""Filename contract for checkpoints/pending/: `{run_id}_{round}_{node}.pt`.

The round segment is what lets every consumer (the watcher's round clock in
mesh_tick.py, quorum/checked-in counting in epoch_advance.py, and the merge
in run_outer_step.py) tell a genuine current-round submission apart from a
straggler still pushing a PRIOR round's checkpoint after that round's merge
already cleared pending/. mtime cannot make that distinction: a straggler
necessarily writes in real wall-clock time AFTER the new round officially
started (it's a separate SLURM process, still mid-training), so any
mtime-relative-to-round-start filter passes it through no matter which
round it actually belongs to. Only the round the worker was actually
training for -- carried in the filename -- resolves that ambiguity.

Both `stern/training/mesh_round.py` (the writer, via `pending_filename`)
and every reader import this instead of building/parsing the pattern by
hand, so the format can't drift between them.
"""

from __future__ import annotations

from pathlib import Path


def pending_filename(run_id: str, round_num: int, node: str) -> str:
    return f"{run_id}_{round_num}_{node}.pt"


def parse_pending_filename(stem: str, run_id: str) -> tuple[int, str] | None:
    """`(round_num, node)` if `stem` matches `run_id`'s pending-file
    pattern, else `None`.

    Returns `None` rather than raising on any non-matching or malformed
    remainder. Callers glob `"{run_id}_*.pt"` first, a plain prefix match that
    also catches sibling runs whose ids extend this one (`pool-fish` matches
    `pool-fish-5r_...`). A bad parse therefore means "not this run's file" or
    "predates round tagging", not an error.
    """
    prefix = f"{run_id}_"
    if not stem.startswith(prefix):
        return None
    round_str, sep, node = stem[len(prefix) :].partition("_")
    if not sep or not round_str.isdigit() or not node:
        return None
    return int(round_str), node


def pending_files_for_round(
    pending_dir: Path, run_id: str, round_num: int
) -> list[Path]:
    """Every `{run_id}_{round_num}_*.pt` file currently in `pending_dir`."""
    matches = []
    for f in pending_dir.glob(f"{run_id}_*.pt"):
        parsed = parse_pending_filename(f.stem, run_id)
        if parsed is not None and parsed[0] == round_num:
            matches.append(f)
    return matches
