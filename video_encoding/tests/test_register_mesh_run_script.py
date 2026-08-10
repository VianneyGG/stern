from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REAL_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "video_encoding"
    / "scripts"
    / "register_mesh_run.sh"
)


@pytest.fixture
def script(tmp_path) -> Path:
    """register_mesh_run.sh derives REPO_ROOT from its own two-levels-up
    location (video_encoding/scripts/<script> -> repo root), not from an
    env var or arg -- so testing it against a real target requires mirroring
    that layout under tmp_path rather than pointing it at the real repo."""
    dest = tmp_path / "video_encoding" / "scripts" / "register_mesh_run.sh"
    dest.parent.mkdir(parents=True)
    shutil.copy2(REAL_SCRIPT, dest)
    dest.chmod(0o755)
    return dest


def test_script_has_no_syntax_errors():
    result = subprocess.run(
        ["bash", "-n", str(REAL_SCRIPT)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_registers_without_quorum_k_matches_prior_behavior(script, tmp_path):
    result = subprocess.run(
        ["bash", str(script), "RUN1", "a,b,c", "10", "0.7", "0.9"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    cfgs = json.loads(
        (tmp_path / "runs" / "state" / "mesh_run_configs.json").read_text()
    )
    assert len(cfgs) == 1
    assert cfgs[0]["weighting"] == "data"
    assert "quorum_k" not in cfgs[0]


def test_registers_quorum_k_and_token_weighting_when_provided(script, tmp_path):
    result = subprocess.run(
        ["bash", str(script), "RUN1", "a,b,c", "10", "0.7", "0.9", "8", "token"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    cfgs = json.loads(
        (tmp_path / "runs" / "state" / "mesh_run_configs.json").read_text()
    )
    assert cfgs[0]["quorum_k"] == 8
    assert cfgs[0]["weighting"] == "token"


def test_rejects_an_unknown_weighting_scheme(script, tmp_path):
    """A typo'd or stale weighting value must fail registration loudly here
    -- not get written into mesh_run_configs.json and only surface later as
    a ValueError inside run_outer_step.py's merge subprocess, where
    advance_all_runs' while-loop retries the same failing merge every tick
    indefinitely (mesh_tick.py's while status != DONE)."""
    result = subprocess.run(
        ["bash", str(script), "RUN1", "a,b,c", "10", "0.7", "0.9", "", "bogus"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "bogus" in result.stderr
    assert not (tmp_path / "runs" / "state" / "mesh_run_configs.json").exists()


def test_re_registering_the_same_run_id_replaces_not_appends(script, tmp_path):
    subprocess.run(
        ["bash", str(script), "RUN1", "a,b,c", "10", "0.7", "0.9"], check=True
    )
    subprocess.run(
        ["bash", str(script), "RUN1", "a,b,c,d", "10", "0.7", "0.9", "8", "token"],
        check=True,
    )

    cfgs = json.loads(
        (tmp_path / "runs" / "state" / "mesh_run_configs.json").read_text()
    )
    assert len(cfgs) == 1
    assert cfgs[0]["quorum_k"] == 8
    assert cfgs[0]["nodes"] == ["a", "b", "c", "d"]
