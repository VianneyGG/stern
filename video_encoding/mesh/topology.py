"""Cluster tier topology: which hosts exist, and which tier/subnet each is on.

Reads `tiers.tsv` next to this module -- the single source of truth. The same
roster used to be declared verbatim in four places (`scripts/pick_nodes.sh`,
`scripts/seed_mesh.py`, `scripts/clean_tier_data.sh`,
`scripts/launch_mesh_pool.sh`), kept in sync only by a comment telling the
reader to update the others. Adding a node to one and not the rest would
desync SLURM node picking from mesh seeding, so a node could be picked for a
job and then have no seeded shard payload.

This lives in `mesh/` rather than `scripts/` because `watcher/mesh_tick.py` and
`training/mesh_round.py` both need it: importing cluster constants *from a CLI
script* inverted the dependency direction, making a library depend on an
entrypoint. The shell scripts read `tiers.tsv` directly rather than shelling
out to Python, so bash keeps no dependency on this module.

TSV over YAML deliberately: bash parses it with `read -r` and no external
tooling, which matters because `pick_nodes.sh` runs on the login node before
any venv is guaranteed active.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

TIERS_PATH = Path(__file__).with_name("tiers.tsv")


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


#: Tiers in preference order (best GPU first) -- pick_nodes.sh walks the same order.
TIERS: tuple[Tier, ...] = _load()

#: host -> tier name (e.g. "node001" -> "fish")
NODE_GROUPS: dict[str, str] = {h: t.name for t in TIERS for h in t.nodes}

#: host -> subnet (e.g. "node001" -> "/254"). `shard_map.pick_holder` prefers a
#: same-subnet holder, so a missing entry silently degrades every fetch to
#: cross-subnet -- callers should raise on an unknown host rather than default.
NODE_SUBNETS: dict[str, str] = {h: t.subnet for t in TIERS for h in t.nodes}


def tier_nodes(name: str) -> list[str]:
    """Hosts in `name`, in roster order. Raises KeyError naming the valid tiers."""
    for tier in TIERS:
        if tier.name == name:
            return list(tier.nodes)
    raise KeyError(f"unknown tier {name!r} -- known tiers: {[t.name for t in TIERS]}")
