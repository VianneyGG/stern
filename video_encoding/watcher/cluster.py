"""SLURM cluster query utilities for mesh pool replenishment."""

from __future__ import annotations

import subprocess


def query_idle_nodes(*, timeout_s: float = 30.0) -> set[str]:
    """Runs `sinfo -N -o "%N %T" --state=idle`, parses stdout into a
    hostname set (mirrors pick_nodes.sh's IDLE snapshot: each line is
    "<host> <state>", keep lines where state == "idle"). Returns an empty
    set on any subprocess failure (nonzero exit, timeout, sinfo missing) --
    fail-safe, so a cluster hiccup skips this tick's replenishment instead
    of crashing the watcher tick."""
    try:
        result = subprocess.run(
            ["sinfo", "-N", "-o", "%N %T", "--state=idle"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if result.returncode != 0:
            return set()

        idle_hosts = set()
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                host, state = parts[0], parts[1]
                if state == "idle":
                    idle_hosts.add(host)
        return idle_hosts
    except (OSError, subprocess.SubprocessError):
        return set()
