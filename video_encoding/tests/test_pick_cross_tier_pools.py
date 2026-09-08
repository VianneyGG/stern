"""Behavioural tests for pick_cross_tier_pools.sh.

Run against a mirrored repo layout with `sinfo` stubbed on PATH, the same way
test_launch_mesh_pool_script.py works: the script only runs on login-node, and
what matters is the roster it deals, not the source text that deals it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.system

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def picker(tmp_path):
    """Returns a `run(*args, idle=...)` callable -> CompletedProcess."""
    scripts = tmp_path / "video_encoding" / "scripts"
    scripts.mkdir(parents=True)
    mesh = tmp_path / "video_encoding" / "mesh"
    mesh.mkdir(parents=True)
    for name in ("pick_cross_tier_pools.sh", "tiers_lib.sh"):
        shutil.copy2(REPO_ROOT / "video_encoding" / "scripts" / name, scripts / name)
    shutil.copy2(
        REPO_ROOT / "video_encoding" / "mesh" / "tiers.tsv", mesh / "tiers.tsv"
    )

    bindir = tmp_path / "bin"
    bindir.mkdir()
    sinfo = bindir / "sinfo"
    # Reads STUB_IDLE so each test controls which nodes look idle.
    sinfo.write_text(
        "#!/usr/bin/env bash\n"
        'tr "," "\\n" <<< "${STUB_IDLE}" | while read -r n; do '
        '[[ -n "$n" ]] && echo "$n idle"; done\n'
    )
    sinfo.chmod(0o755)
    # A poll loop that would otherwise block the suite forever.
    (bindir / "sleep").write_text("#!/usr/bin/env bash\nexit 1\n")
    (bindir / "sleep").chmod(0o755)

    def run(*args, idle=""):
        return subprocess.run(
            ["bash", str(scripts / "pick_cross_tier_pools.sh"), *args],
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "PATH": f"{bindir}:{os.environ['PATH']}",
                "STUB_IDLE": idle,
            },
            timeout=60,
        )

    return run


ALL_IDLE = ",".join(
    [
        "node001,node002,node003,node004",
        "node024,node025,node026,node027",
        "node047,node048,node049,node050",
        "node070,node071,node072,node073",
    ]
)


def test_deals_disjoint_pools_with_equal_tier_representation(picker):
    """Every pool gets the same count from every tier, and no node is in two.

    This is the whole point: an arm that draws unevenly across tiers is still
    partly a measurement of its GPUs.
    """
    result = picker("2", "2", idle=ALL_IDLE)
    assert result.returncode == 0, result.stderr

    pools = [ln.split(",") for ln in result.stdout.split()]
    assert len(pools) == 2
    assert all(len(p) == 8 for p in pools), pools  # 4 tiers x 2

    flat = [n for p in pools for n in p]
    assert len(set(flat)) == len(flat), "pools overlap"

    from video_encoding.mesh.topology import NODE_GROUPS

    for pool in pools:
        by_tier: dict[str, int] = {}
        for node in pool:
            by_tier[NODE_GROUPS[node]] = by_tier.get(NODE_GROUPS[node], 0) + 1
        assert by_tier == {"fish": 2, "birds": 2, "cars": 2, "countries": 2}, pool


def test_restricting_the_tier_list_restricts_the_pools(picker):
    result = picker("2", "2", "fish", "birds", idle=ALL_IDLE)
    assert result.returncode == 0, result.stderr

    from video_encoding.mesh.topology import NODE_GROUPS

    for line in result.stdout.split():
        assert {NODE_GROUPS[n] for n in line.split(",")} == {"fish", "birds"}


def test_demand_larger_than_a_tier_fails_instead_of_polling_forever(picker):
    """countries has 19 nodes; 5 pools x 4 can never be satisfied."""
    result = picker("5", "4", idle=ALL_IDLE)
    assert result.returncode != 0
    assert "countries" in result.stderr and "19" in result.stderr, result.stderr


def test_waits_rather_than_dealing_a_short_roster(picker):
    """A half-idle cluster must not yield undersized pools -- that silently
    reintroduces the tier imbalance the script exists to remove."""
    result = picker("2", "2", idle="node001,node002,node024,node025")
    # The stubbed `sleep` exits 1, so `set -e` aborts the poll loop instead of
    # spinning; what matters is that nothing was printed to stdout.
    assert result.returncode != 0
    assert result.stdout.strip() == "", result.stdout
    assert "fish=2/4" in result.stderr, result.stderr
