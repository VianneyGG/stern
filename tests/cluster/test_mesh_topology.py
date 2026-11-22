"""Unit tests for the cluster tier roster (stern/mesh/topology.py).

The roster was previously pasted verbatim into four files (pick_nodes.sh,
seed_mesh.py, clean_tier_data.sh, launch_mesh_pool.sh) and kept in sync only by
comments telling the reader to update the others. These tests pin the two things
that collapse into one source of truth is supposed to buy: Python and bash read
the *same* rows, and nothing re-declares a roster of its own.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from stern.mesh.topology import (
    NODE_GROUPS,
    NODE_SUBNETS,
    TIERS,
    _load,
    tier_nodes,
)

pytestmark = pytest.mark.system

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "cluster"


def _bash_tier_nodes(tier: str) -> str:
    """Call tiers_lib.sh the way the shell scripts do."""
    result = subprocess.run(
        ["bash", "-c", f'source "{SCRIPTS / "tiers_lib.sh"}"; tier_nodes {tier}'],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@pytest.mark.parametrize("tier", ["fish", "birds", "cars", "countries", "orchids"])
def test_bash_and_python_read_the_same_roster(tier):
    """The whole point of tiers.tsv: `tiers_lib.sh` and `topology.py` must agree
    node-for-node. If they ever diverge, SLURM picks nodes the mesh never
    seeded, and a worker dies on read_my_shards with an empty shard dir."""
    assert _bash_tier_nodes(tier) == ",".join(tier_nodes(tier))


def test_every_host_maps_to_exactly_one_tier_and_subnet():
    assert NODE_GROUPS["node001"] == "fish"
    assert NODE_GROUPS["node024"] == "birds"
    assert NODE_GROUPS["node089"] == "orchids"
    assert NODE_GROUPS["node070"] == "countries"
    assert NODE_SUBNETS["node001"] == "/254"
    assert NODE_SUBNETS["node047"] == "/252"
    assert set(NODE_GROUPS) == set(NODE_SUBNETS)


def test_hosts_are_unique_across_tiers():
    """A host in two tiers would make NODE_GROUPS order-dependent and could put
    one node in two pools at once."""
    seen: dict[str, str] = {}
    for tier in TIERS:
        for host in tier.nodes:
            assert host not in seen, f"{host} in both {seen[host]} and {tier.name}"
            seen[host] = tier.name


def test_unknown_tier_raises_naming_the_valid_ones():
    with pytest.raises(KeyError, match="unknown tier"):
        tier_nodes("penguins")


def test_bash_rejects_an_unknown_tier_instead_of_returning_empty():
    """An empty nodelist would flow into `xargs ssh` in clean_tier_data.sh as a
    no-op, or into sbatch --nodelist as a cluster-wide request."""
    result = subprocess.run(
        ["bash", "-c", f'source "{SCRIPTS / "tiers_lib.sh"}"; tier_nodes penguins'],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "unknown tier" in result.stderr


def test_malformed_row_is_rejected_with_the_line_number(tmp_path):
    bad = tmp_path / "tiers.tsv"
    bad.write_text("# comment\nfish\t/254\tablette\nbirds\t/253\n")
    with pytest.raises(ValueError, match=r"tiers\.tsv:3: expected 3 tab-separated"):
        _load(bad)


def test_tier_with_no_nodes_is_rejected(tmp_path):
    bad = tmp_path / "tiers.tsv"
    bad.write_text("fish\t/254\t\n")
    with pytest.raises(ValueError, match="has no nodes"):
        _load(bad)
