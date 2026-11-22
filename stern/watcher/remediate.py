"""Auto-resubmit a dead job exactly once.

Calls `sbatch` directly, mirroring the remote block of
cluster/submit_to_cluster.sh, pinned to the same nodelist as the dead job via
sbatch's --nodelist flag. The watcher runs from its checkout on the cluster
host, so the rsync/ssh hop that submit_to_cluster.sh makes from a laptop is
skipped.

Also skips the remote `uv sync` step. The checkout and venv live on the
NFS-shared home (~/stern), so every node sees the same environment.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from stern.watcher import class_hours
from stern.watcher.registry import JobRecord


def remediate_dead_job(record: JobRecord, *, repo_root: Path) -> str | None:
    if class_hours.blocked(time.time()) is not None:
        return None  # no job may run Mon-Fri 08:00-20:00 (class_hours.py)
    n = len(record.nodelist.split(",")) if record.nodelist else 1
    cmd = ["sbatch", f"--nodes={n}", f"--ntasks={n}"]
    if record.nodelist:
        cmd.append(f"--nodelist={record.nodelist}")
    cmd += ["cluster/slurm/train.sbatch", record.experiment]
    if record.run_tag:
        cmd.append(f"+run_tag={record.run_tag}")

    try:
        result = subprocess.run(
            cmd, cwd=repo_root, capture_output=True, text=True, timeout=300
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if result.returncode != 0:
        return None

    for line in result.stdout.splitlines():
        if "Submitted batch job" in line:
            tail = line.strip().split()[-1]
            return tail if tail.isdigit() else None
    return None
