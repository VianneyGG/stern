"""Disk-usage probes for the `quota` detector.

Home-dir usage is the user's own NFS quota (`quota -w -p`, local -- the watcher
runs on login-node, the NFS home's mount point), NOT `df`: `df` reports the
whole shared filesystem's occupancy, which is a different number from this
user's allowance and stayed near 51% while a real quota-exceeded incident
was happening at 100% of a 30G limit (2026-08-14). `df` is kept only as a
fallback for when `quota` itself is unavailable. Node-local /Data usage is
per-node -- VoxCeleb2 frames and checkpoints live there (train.sbatch's
STERN_CHECKPOINT_ROOT / VOXMINI_FRAMES_ROOT), not on the NFS home -- so it
requires one `ssh <node>` per currently-running job's node, same pattern as
sync_to_cluster.sh's Bash(ssh *) precedent.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def parse_quota_output(stdout: str) -> float | None:
    """Highest of block-% and inode-% from `quota -w -p` stdout.

    `quota -p`'s per-mount data row ends in 8 whitespace-separated numeric
    fields: blocks_used blocks_soft blocks_hard blocks_grace files_used
    files_soft files_hard files_grace, optionally preceded by the
    filesystem name on the same line (with `-w`/--no-wrap) -- without `-w`
    a long name instead wraps onto its own line, leaving the data row at
    exactly 8 fields with no name prefix. Matching on the trailing 8 fields
    (not an exact field count) handles both, since `home_quota_pct` always
    passes `-w` but this function must still parse pre-`-w` fixtures/output.
    Header lines and the bare filesystem-name line don't end in 8 integers
    and are skipped. Returning on the FIRST such row is deliberate: `quota
    -p` repeats the same mount once per .snapshot/* entry, all identical to
    the real one, so later rows must be ignored rather than averaged in. A
    hard limit of 0 means "unlimited" for that dimension and is skipped
    rather than divided by (seen for inode limits on some mounts).
    """
    for line in stdout.splitlines():
        fields = line.split()
        if len(fields) < 8:
            continue
        tail = fields[-8:]
        try:
            blocks_used, _, blocks_hard = (int(tail[i]) for i in (0, 1, 2))
            files_used, _, files_hard = (int(tail[i]) for i in (4, 5, 6))
        except ValueError:
            continue
        pcts = []
        if blocks_hard > 0:
            pcts.append(100.0 * blocks_used / blocks_hard)
        if files_hard > 0:
            pcts.append(100.0 * files_used / files_hard)
        return max(pcts) if pcts else None
    return None


def _df_home_pct() -> float | None:
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


def home_quota_pct() -> float | None:
    try:
        result = subprocess.run(
            ["quota", "-w", "-p"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return _df_home_pct()
    pct = parse_quota_output(result.stdout)
    return pct if pct is not None else _df_home_pct()


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
