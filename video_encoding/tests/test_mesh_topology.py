"""Unit tests for the cluster tier roster (video_encoding/mesh/topology.py).

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

from video_encoding.mesh.topology import (
    NODE_GROUPS,
    NODE_SUBNETS,
    TIERS,
    TIERS_PATH,
    _load,
    tier_nodes,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "video_encoding" / "scripts"


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


def test_no_script_redeclares_a_roster():
    """Regression guard for the duplication itself. A hostname list pasted back
    into any script silently reintroduces the drift this file exists to stop --
    tiers.tsv is the only place 'node001' may appear alongside its tier-mates."""
    offenders = []
    for path in list(SCRIPTS.glob("*.sh")) + list(SCRIPTS.glob("*.py")):
        for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
            line = raw.strip()
            if line.startswith("#") or line.startswith("*"):
                continue  # prose may cite "node001,node002,..." as an example
            # The signature of a pasted roster: two adjacent tier-mates on a
            # line that is actually code.
            if "node001,node002" in line or "node024,node025" in line:
                offenders.append(f"{path.name}:{lineno}")
    assert offenders == [], f"tier roster re-declared in: {offenders}"


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


def test_tiers_are_in_gpu_preference_order():
    """pick_nodes.sh walks TIERS top-down and takes the first tier with enough
    idle nodes, so row order is load-bearing, not cosmetic."""
    assert [t.name for t in TIERS] == [
        "fish",
        "birds",
        "cars",
        "countries",
        "orchids",
    ]


def test_subnets_that_are_shared_between_tiers_stay_shared():
    """countries shares /254 with fish, orchids shares /253 with birds. That is
    intentional and documented; pinning it here means a subnet typo shows up as
    a test failure instead of as cross-switch fetches at 0.94 Gbps."""
    by_name = {t.name: t.subnet for t in TIERS}
    assert by_name["countries"] == by_name["fish"] == "/254"
    assert by_name["orchids"] == by_name["birds"] == "/253"
    assert by_name["cars"] == "/252"


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


def test_roster_file_ships_next_to_the_module():
    """topology.py resolves tiers.tsv relative to __file__; a move that leaves
    the data behind must fail here, not at 3am on the cluster."""
    assert TIERS_PATH.is_file()
    assert TIERS_PATH.parent == (REPO_ROOT / "video_encoding" / "mesh")
