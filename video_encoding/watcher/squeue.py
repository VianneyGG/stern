"""Parse `squeue --json` output into a normalized snapshot.

Slurm's JSON schema nests state under `job_state` as a list of strings (e.g.
`["RUNNING"]`) and the job id under `job_id` (int). This module is the only
place that knows that shape -- detectors (video_encoding/watcher/detectors.py)
work against the flat dict this produces.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def refresh_squeue_snapshot(out_path: Path, *, user: str) -> None:
    """Run `squeue -u <user> --json`, write atomically to `out_path`.

    Tmp-write + rename mirrors poll_squeue.sh / checkpoint.save -- a crash
    mid-write must never leave a truncated squeue.json for the next tick.
    Raises `subprocess.CalledProcessError` on failure; the caller
    (scripts/watch_training.py) decides how to log/skip that tick.
    """
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    result = subprocess.run(
        ["squeue", "-u", user, "--json"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    tmp.write_text(result.stdout)
    tmp.replace(out_path)


def load_squeue_snapshot(path: Path) -> dict[str, dict[str, Any]]:
    """Load squeue.json -> {slurm_job_id (str): {"state": str, "nodelist": str}}.

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
        snapshot[job_id] = {"state": state, "nodelist": job.get("nodes", "")}
    return snapshot
