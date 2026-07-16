"""Disk-usage probes for the `quota` detector.

Home-dir usage is a plain local `df` (the watcher runs on login-node, the NFS
home's mount point). Node-local /Data usage is per-node -- VoxCeleb2 frames
and checkpoints live there (train.sbatch's STERN_CHECKPOINT_ROOT /
VOXMINI_FRAMES_ROOT), not on the NFS home -- so it requires one `ssh <node>`
per currently-running job's node, same pattern as sync_login-node.sh's Bash(ssh
*) precedent.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def home_quota_pct() -> float | None:
    try:
        result = subprocess.run(
            ["df", "--output=pcent", str(Path.home())],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    return float(lines[-1].rstrip("%"))


def node_data_usage_pct(nodes: list[str]) -> dict[str, float]:
    usage: dict[str, float] = {}
    for node in nodes:
        try:
            result = subprocess.run(
                ["ssh", node, "df", "--output=pcent", "/Data"],
                capture_output=True,
                text=True,
                check=True,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError):
            continue  # unreachable node -- skip, don't fail the whole tick
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if len(lines) >= 2:
            usage[node] = float(lines[-1].rstrip("%"))
    return usage


def collect_disk_usage_pct(running_nodes: list[str]) -> dict[str, float]:
    merged: dict[str, float] = {}
    home = home_quota_pct()
    if home is not None:
        merged["home"] = home
    for node, pct in node_data_usage_pct(running_nodes).items():
        merged[f"node:{node}"] = pct
    return merged
