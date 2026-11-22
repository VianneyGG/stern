"""Cluster-specific settings (host names, roster, data root), kept out of the code.

Real values live in gitignored files next to the shell launchers:
`cluster/site.conf` (KEY=VALUE lines) and `cluster/tiers.tsv` (the node roster).
The committed `*.example` copies hold placeholder names, so a fresh clone still
imports and tests cleanly. `cluster/cluster_env.sh` applies the same lookup order
for the shell scripts.
"""

from __future__ import annotations

import getpass
import os
import shlex
from pathlib import Path

CLUSTER_DIR = Path(__file__).resolve().parents[1] / "cluster"


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw = line.split("=", 1)
        # shlex handles the quoting a shell `source` would accept.
        parsed = shlex.split(raw, comments=True)
        values[key.strip()] = " ".join(parsed)
    return values


def setting(name: str) -> str:
    """Return a cluster setting: process env, then site.conf, then site.conf.example."""
    if name in os.environ:
        return os.environ[name]
    for filename in ("site.conf", "site.conf.example"):
        values = _read_env_file(CLUSTER_DIR / filename)
        if name in values:
            return values[name]
    raise KeyError(f"cluster setting {name} is not set (see cluster/site.conf.example)")


def tiers_path() -> Path:
    """Return the node roster: $STERN_TIERS_FILE, then cluster/tiers.tsv, then the example."""
    if "STERN_TIERS_FILE" in os.environ:
        return Path(os.environ["STERN_TIERS_FILE"])
    real = CLUSTER_DIR / "tiers.tsv"
    return real if real.is_file() else CLUSTER_DIR / "tiers.example.tsv"


def data_root() -> Path:
    """Return the per-user data directory on the nodes' local disks ($STERN_DATA_ROOT, else /Data/<user>)."""
    return Path(os.environ.get("STERN_DATA_ROOT") or Path("/Data") / getpass.getuser())
