"""The factorial launcher's dry-run output is the contract.

These assertions exist because the four arms are a ~40 GPU-hour spend that
fails late and quietly when it fails at all: a wrong Hydra prefix aborts every
job at submit, and a wrong override trains four jobs that are secretly two
conditions.
"""

import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.system

SCRIPT = Path("video_encoding/scripts/run_factorial.sh")
ARMS = ["tube-time_avg", "tube-per_frame", "mixed-time_avg", "mixed-per_frame"]


@pytest.fixture(scope="module")
def dry_run() -> str:
    result = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_script_has_no_syntax_errors():
    result = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_all_four_cells_are_submitted(dry_run):
    for arm in ARMS:
        mask_type, target_mode = arm.split("-", 1)
        assert f"model.mask_type={mask_type}" in dry_run
        assert f"model.target_mode={target_mode}" in dry_run
        assert f"factorial-{arm}" in dry_run


def test_each_arm_gets_its_own_node(dry_run):
    nodes = [
        line.split("-> ")[1].strip() for line in dry_run.splitlines() if "] -> " in line
    ]
    assert len(nodes) == 4
    assert len(set(nodes)) == 4, f"arms share a node: {nodes}"


def test_mesh_keys_are_overridden_not_appended(dry_run):
    """videomae_mesh_diloco.yaml already declares mesh.merge and mesh.rounds,
    and Hydra refuses '+key=' on an existing key -- with the + prefix every
    job dies at submit with 'An item is already at mesh.merge'."""
    assert "mesh.merge=false" in dry_run
    assert "+mesh.merge" not in dry_run
    assert "+mesh.rounds" not in dry_run


def test_arms_share_one_seed_and_one_shard_list(dry_run):
    """The factorial is only interpretable if the arms differ in exactly the
    two knobs under test, so seed and data must be identical across them."""
    assert dry_run.count("train.seed=0") == 4
    assert "shard_assignments_factorial.json" in dry_run
    # One shard list is built once and reused; a per-node list would let the
    # arms diverge on data.
    assert dry_run.count("shard-000000.tar") >= 4


def test_solo_arms_do_not_merge(dry_run):
    """Each arm is one node; a merge would block forever on a quorum that can
    never form."""
    assert dry_run.count("mesh.merge=false") == 4
