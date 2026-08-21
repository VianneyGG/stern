from __future__ import annotations

import json
import os
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
    # Pre-guard behavior is now reachable only via the escape hatch.
    result = subprocess.run(
        ["bash", str(script), "RUN1", "a,b,c", "10", "0.7", "0.9"],
        capture_output=True,
        text=True,
        env={**os.environ, "ALLOW_NO_QUORUM": "1"},
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
        ["bash", str(script), "RUN1", "a,b,c", "10", "0.7", "0.9", "2", "token"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    cfgs = json.loads(
        (tmp_path / "runs" / "state" / "mesh_run_configs.json").read_text()
    )
    assert cfgs[0]["quorum_k"] == 2
    assert cfgs[0]["weighting"] == "token"


def test_rejects_an_unknown_weighting_scheme(script, tmp_path):
    """A typo'd or stale weighting value must fail registration loudly here
    -- not get written into mesh_run_configs.json and only surface later as
    a ValueError inside run_outer_step.py's merge subprocess, where
    advance_all_runs' while-loop retries the same failing merge every tick
    indefinitely (mesh_tick.py's while status != DONE)."""
    result = subprocess.run(
        ["bash", str(script), "RUN1", "a,b,c", "10", "0.7", "0.9", "2", "bogus"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "bogus" in result.stderr
    assert not (tmp_path / "runs" / "state" / "mesh_run_configs.json").exists()


def test_re_registering_the_same_run_id_replaces_not_appends(script, tmp_path):
    subprocess.run(
        ["bash", str(script), "RUN1", "a,b,c", "10", "0.7", "0.9", "2", "token"],
        check=True,
    )
    subprocess.run(
        ["bash", str(script), "RUN1", "a,b,c,d", "10", "0.7", "0.9", "3", "token"],
        check=True,
    )

    cfgs = json.loads(
        (tmp_path / "runs" / "state" / "mesh_run_configs.json").read_text()
    )
    assert len(cfgs) == 1
    assert cfgs[0]["quorum_k"] == 3
    assert cfgs[0]["nodes"] == ["a", "b", "c", "d"]


def test_registers_extra_overrides_when_provided(script, tmp_path):
    result = subprocess.run(
        [
            "bash",
            str(script),
            "RUN1",
            "a,b,c",
            "10",
            "0.7",
            "0.9",
            "2",
            "token",
            "train.accum_steps=128 train.lr_mult=1.5",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    cfgs = json.loads(
        (tmp_path / "runs" / "state" / "mesh_run_configs.json").read_text()
    )
    assert cfgs[0]["extra_overrides"] == [
        "train.accum_steps=128",
        "train.lr_mult=1.5",
    ]


def test_omits_extra_overrides_key_when_not_provided(script, tmp_path):
    result = subprocess.run(
        ["bash", str(script), "RUN1", "a,b,c", "10", "0.7", "0.9", "2", "token"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    cfgs = json.loads(
        (tmp_path / "runs" / "state" / "mesh_run_configs.json").read_text()
    )
    assert "extra_overrides" not in cfgs[0]


def test_rejects_a_malformed_extra_override(script, tmp_path):
    result = subprocess.run(
        [
            "bash",
            str(script),
            "RUN1",
            "a,b,c",
            "10",
            "0.7",
            "0.9",
            "2",
            "token",
            "not_a_key_value_pair",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "not_a_key_value_pair" in result.stderr
    assert not (tmp_path / "runs" / "state" / "mesh_run_configs.json").exists()


def test_remove_deletes_only_the_named_run(script, tmp_path):
    """A sweep wave is torn down at its deadline by dropping its cells from
    the live configs file; every neighbouring run must survive untouched."""
    subprocess.run(
        ["bash", str(script), "RUN1", "a,b,c", "10", "0.7", "0.9", "2", "token"],
        check=True,
    )
    subprocess.run(
        ["bash", str(script), "RUN2", "d,e", "24", "0.7", "0.9", "2", "token"],
        check=True,
    )
    configs = tmp_path / "runs" / "state" / "mesh_run_configs.json"
    kept_before = [c for c in json.loads(configs.read_text()) if c["run_id"] == "RUN2"]

    result = subprocess.run(
        ["bash", str(script), "--remove", "RUN1"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr

    cfgs = json.loads(configs.read_text())
    assert [c["run_id"] for c in cfgs] == ["RUN2"]
    assert cfgs == kept_before


def test_remove_is_idempotent_for_an_absent_run(script, tmp_path):
    """The teardown runbook covers several cells and may be re-run; a
    non-zero exit on an already-removed cell would abort it midway."""
    subprocess.run(
        ["bash", str(script), "RUN1", "a,b,c", "10", "0.7", "0.9", "2", "token"],
        check=True,
    )
    configs = tmp_path / "runs" / "state" / "mesh_run_configs.json"
    before = configs.read_text()

    result = subprocess.run(
        ["bash", str(script), "--remove", "NEVER_REGISTERED"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert configs.read_text() == before


def test_remove_exits_nonzero_when_configs_file_missing(script, tmp_path):
    """No configs file at teardown time is a broken deployment, not a no-op.

    The message must name the missing file: without it this exit is
    indistinguishable from the positional-argument USAGE guard firing,
    which is a different bug with a different fix."""
    result = subprocess.run(
        ["bash", str(script), "--remove", "RUN1"], capture_output=True, text=True
    )
    assert result.returncode == 1
    assert "mesh_run_configs.json" in result.stderr


def test_remove_rejects_extra_registration_arguments(script, tmp_path):
    """Likewise: the rejection must come from the --remove parser, not from
    int() choking on a shifted positional argument."""
    result = subprocess.run(
        ["bash", str(script), "--remove", "RUN1", "a,b,c", "10", "0.7", "0.9"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "--remove" in result.stderr


# --- quorum_k guard (openspec mesh-seeding-validation, group 4) -------------


def test_empty_quorum_k_is_refused_and_writes_no_config(script, tmp_path):
    """Every run this script registers has merge=True hardcoded, so a missing
    quorum_k means the round waits for all-or-ROUND_TIMEOUT_S. One dead node
    then costs the whole pool a 30-minute stall per round."""
    result = subprocess.run(
        ["bash", str(script), "RUN1", "a,b,c", "10", "0.7", "0.9"],
        capture_output=True,
        text=True,
        env={**os.environ, "ALLOW_NO_QUORUM": ""},
    )
    assert result.returncode != 0
    assert "quorum_k" in result.stderr
    assert not (tmp_path / "runs" / "state" / "mesh_run_configs.json").exists()


def test_quorum_k_larger_than_the_roster_is_refused(script, tmp_path):
    result = subprocess.run(
        ["bash", str(script), "RUN1", "a,b,c", "10", "0.7", "0.9", "4", "token"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "4" in result.stderr and "3" in result.stderr
    assert not (tmp_path / "runs" / "state" / "mesh_run_configs.json").exists()


def test_allow_no_quorum_escape_hatch_registers_without_the_key(script, tmp_path):
    result = subprocess.run(
        ["bash", str(script), "RUN1", "a,b,c", "10", "0.7", "0.9"],
        capture_output=True,
        text=True,
        env={**os.environ, "ALLOW_NO_QUORUM": "1"},
    )
    assert result.returncode == 0, result.stderr

    cfgs = json.loads(
        (tmp_path / "runs" / "state" / "mesh_run_configs.json").read_text()
    )
    assert "quorum_k" not in cfgs[0]


def test_every_queued_sweep_cell_carries_quorum_k_and_weighting():
    """sweep_queue.sh `exit 1`s the whole queue when a register call fails, so
    one under-specified cell strands every cell behind it -- the guard and the
    queue files have to be consistent or the queue stops draining."""
    repo_root = Path(__file__).resolve().parents[2]
    queues = sorted((repo_root / "runs" / "state").glob("sweep_queue_*.txt"))
    if not queues:
        # runs/state/ is gitignored and machine-local by policy (see
        # CLAUDE.md), so a fresh clone legitimately has no queues to check.
        pytest.skip("no sweep queue files on this machine")

    for queue in queues:
        for lineno, line in enumerate(queue.read_text().splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            assert len(fields) >= 6, (
                f"{queue.name}:{lineno} has {len(fields)} fields, needs at least "
                f"6 (run_id rounds outer_lr outer_momentum quorum_k weighting): "
                f"{stripped!r}"
            )
