"""Cluster tier topology: which hosts exist, and which tier/subnet each is on.

Reads `cluster/tiers.tsv`, the single source of truth for the roster. The
shell launchers (`cluster/pick_nodes.sh`, `cluster/clean_tier_data.sh`,
`cluster/launch_mesh_pool.sh`) read the same file, so SLURM node picking and
mesh seeding cannot disagree about which nodes exist.

Lives in `mesh/` because `watcher/mesh_tick.py` and `training/mesh_round.py`
both need it; a library should not import constants from a CLI script. The
shell scripts read the TSV directly rather than shelling out to Python, so
bash has no dependency on this module.

TSV rather than YAML: bash parses it with `read -r` and no external tooling,
which matters because `pick_nodes.sh` runs on the login node before any venv
is guaranteed to be active.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from stern.cluster_env import tiers_path

TIERS_PATH = tiers_path()


@dataclass(frozen=True)
class Tier:
    name: str
    subnet: str
    nodes: tuple[str, ...]


def _load(path: Path = TIERS_PATH) -> tuple[Tier, ...]:
    tiers: list[Tier] = []
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        # Split the un-stripped line: stripping first would eat a trailing tab
        # and report an empty node field as a field-count error instead of the
        # clearer "tier has no nodes".
        fields = raw.split("\t")
        if len(fields) != 3:
            raise ValueError(
                f"{path}:{lineno}: expected 3 tab-separated fields "
                f"(tier, subnet, nodes), got {len(fields)}: {raw!r}"
            )
        name, subnet, nodes_csv = (f.strip() for f in fields)
        nodes = tuple(h for h in (n.strip() for n in nodes_csv.split(",")) if h)
        if not nodes:
            raise ValueError(f"{path}:{lineno}: tier {name!r} has no nodes")
        tiers.append(Tier(name=name, subnet=subnet, nodes=nodes))
    if not tiers:
        raise ValueError(f"{path}: no tiers defined")
    return tuple(tiers)


#: Tiers in preference order (best GPU first). `cluster/pick_nodes.sh` walks the same order.
TIERS: tuple[Tier, ...] = _load()

#: host -> tier name (e.g. "node001" -> "fish")
NODE_GROUPS: dict[str, str] = {h: t.name for t in TIERS for h in t.nodes}

#: host -> subnet (e.g. "node001" -> "/254"). Holder selection prefers a
#: same-subnet holder, so a missing entry silently degrades every fetch to
#: cross-subnet. Callers should raise on an unknown host rather than default.
NODE_SUBNETS: dict[str, str] = {h: t.subnet for t in TIERS for h in t.nodes}


def tier_nodes(name: str) -> list[str]:
    """Hosts in `name`, in roster order. Raises KeyError naming the valid tiers."""
    for tier in TIERS:
        if tier.name == name:
            return list(tier.nodes)
    raise KeyError(f"unknown tier {name!r} -- known tiers: {[t.name for t in TIERS]}")
