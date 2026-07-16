"""Auto-resubmit a dead job exactly once (spec §3 Remediator).

Calls `sbatch` directly, mirroring the remote block of submit_login-node.sh
(video_encoding/scripts/submit_login-node.sh), pinned to the SAME nodelist the
dead job used via sbatch's own --nodelist flag. This runs the watcher's
already-checked-out code on login-node in place, without the rsync/ssh round
trip submit_login-node.sh performs for a developer-laptop -> cluster submission
(that round trip is unnecessary here, and self-referential/untested when the
watcher itself runs on login-node).

Deliberately skips submit_login-node.sh's remote `uv sync --quiet` step: the
checkout/venv this watcher runs from lives on the NFS-shared home
(~/stern), not node-local /Data, so any fish node picks up the same
already-synced environment without re-syncing per resubmit.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from video_encoding.watcher.registry import JobRecord


def remediate_dead_job(record: JobRecord, *, repo_root: Path) -> str | None:
    n = len(record.nodelist.split(",")) if record.nodelist else 1
    cmd = ["sbatch", f"--nodes={n}", f"--ntasks={n}"]
    if record.nodelist:
        cmd.append(f"--nodelist={record.nodelist}")
    cmd += ["video_encoding/slurm/train.sbatch", record.experiment]
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
