from __future__ import annotations

import subprocess
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "video_encoding"
    / "scripts"
    / "sweep_mesh_outer.sh"
)


def test_sweep_script_has_no_syntax_errors():
    result = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_sweep_includes_the_naive_anchor_cell():
    text = SCRIPT.read_text()
    assert "1.0" in text
    assert "0.0" in text


def test_sweep_covers_all_three_axes():
    text = SCRIPT.read_text()
    assert "OUTER_LRS" in text
    assert "OUTER_MOMENTUMS" in text
    assert "MERGE_EVERY" in text


def test_sweep_tags_each_cell_for_result_joining():
    text = SCRIPT.read_text()
    assert "+run_tag=" in text


def test_rejects_a_nodelist_exceeding_the_shared_home_quota():
    nodelist = ",".join(f"n{i}" for i in range(17))
    result = subprocess.run(
        ["bash", str(SCRIPT), "some_experiment", nodelist],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "MAX_MESH_NODES" in result.stderr
