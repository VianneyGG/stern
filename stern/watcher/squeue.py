"""Parse `squeue --json` output into a normalized snapshot.

Slurm's JSON schema nests state under `job_state` as a list of strings (e.g.
`["RUNNING"]`) and the job id under `job_id` (int). This module is the only
place that knows that shape -- detectors (stern/watcher/detectors.py)
work against the flat dict this produces.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def refresh_squeue_snapshot(out_path: Path, *, user: str) -> None:
    """Run `squeue -u <user>`, write a squeue-`--json`-shaped snapshot to `out_path`.

    This cluster's Slurm build has no JSON serializer plugin (`squeue --json`
    fails with `serializer_required: could not find plugin for
    application/json`), so we parse squeue's plain `-o` format instead and
    reassemble the same `{"jobs": [{"job_id", "job_state", "nodes"}, ...]}`
    shape `load_squeue_snapshot` expects -- that on-disk contract stays
    unchanged even though the source command differs.

    Tmp-write + rename mirrors poll_squeue.sh / checkpoint.save -- a crash
    mid-write must never leave a truncated squeue.json for the next tick.
    Raises `subprocess.CalledProcessError` on failure; the caller
    (scripts/watch_training.py) decides how to log/skip that tick.
    """
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    result = subprocess.run(
        ["squeue", "-u", user, "-h", "-o", "%i|%T|%N|%r"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    jobs = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        job_id, state, nodes, restart_cnt = line.split("|", 3)
        jobs.append(
            {
                "job_id": int(job_id),
                "job_state": [state],
                "nodes": nodes,
                "restart_cnt": int(restart_cnt) if restart_cnt.isdigit() else 0,
            }
        )
    tmp.write_text(json.dumps({"jobs": jobs}))
    tmp.replace(out_path)


def load_squeue_snapshot(path: Path) -> dict[str, dict[str, Any]]:
    """Load squeue.json -> {slurm_job_id (str): {"state", "nodelist", "restart_cnt"}}.

    Missing file (before the first refresh) returns {} rather than raising --
    callers treat "no snapshot yet" the same as "no jobs running".
    """
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    snapshot: dict[str, dict[str, Any]] = {}
    for job in payload.get("jobs", []):
        job_id = str(job["job_id"])
        state = job.get("job_state", ["UNKNOWN"])
        state = state[0] if isinstance(state, list) else state
        snapshot[job_id] = {
            "state": state,
            "nodelist": job.get("nodes", ""),
            "restart_cnt": job.get("restart_cnt", 0),
        }
    return snapshot
