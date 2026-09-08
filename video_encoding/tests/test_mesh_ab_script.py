from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.system

SCRIPT = (
    Path(__file__).resolve().parents[2] / "video_encoding" / "scripts" / "mesh_ab.sh"
)


def test_ab_script_has_no_syntax_errors():
    result = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_ab_script_covers_all_three_arms():
    text = SCRIPT.read_text()
    assert "single_node" in text
    assert "naive_avg" in text
    assert "diloco" in text


def test_naive_arm_uses_the_lr1_momentum0_equivalence():
    text = SCRIPT.read_text()
    assert "outer_lr=1.0" in text
    assert "outer_momentum=0.0" in text


def test_rejects_a_nodelist_exceeding_the_shared_home_quota():
    nodelist = ",".join(f"n{i}" for i in range(17))
    result = subprocess.run(
        ["bash", str(SCRIPT), "some_experiment", nodelist],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "MAX_MESH_NODES" in result.stderr
