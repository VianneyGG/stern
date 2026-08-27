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
    # EXPERIMENT is validated against the config tree, so the mirror needs
    # one too -- names only, since the script never reads their contents.
    conf = tmp_path / "video_encoding" / "conf" / "experiment"
    conf.mkdir(parents=True)
    for real in (REAL_SCRIPT.parents[1] / "conf" / "experiment").glob("*.yaml"):
        (conf / real.name).touch()
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


# --- GRACE_S: the below-quorum round deadline ------------------------------


def _configs(tmp_path):
    return json.loads(
        (tmp_path / "runs" / "state" / "mesh_run_configs.json").read_text()
    )


def _register(
    script, tmp_path, *, grace=None, chunk=None, run_id="RUN1", experiment=None
):
    env = {**os.environ}
    if grace is not None:
        env["GRACE_S"] = grace
    if chunk is not None:
        env["CHUNK_ROUNDS"] = chunk
    if experiment is not None:
        env["EXPERIMENT"] = experiment
    return subprocess.run(
        ["bash", str(script), run_id, "a,b,c", "10", "0.7", "0.9", "3", "token"],
        capture_output=True,
        text=True,
        env=env,
    )


def test_grace_s_is_written_when_set(script, tmp_path):
    """With mesh.round_budget_s=null every node runs the same step count, and
    9600 steps needs 912s on birds -- the 900s default would time out the slow
    tier every round."""
    result = _register(script, tmp_path, grace="1800")
    assert result.returncode == 0, result.stderr
    assert _configs(tmp_path)[0]["grace_s"] == 1800.0


def test_grace_s_key_is_absent_when_unset(script, tmp_path):
    """Every run registered before this field existed has no value for it;
    the key must stay absent so the dataclass default applies."""
    result = _register(script, tmp_path)
    assert result.returncode == 0, result.stderr
    assert "grace_s" not in _configs(tmp_path)[0]


@pytest.mark.parametrize("bad", ["not-a-number", "0", "-5"])
def test_a_nonsensical_grace_s_is_refused(script, tmp_path, bad):
    """A silently-dropped bad value would leave the run on the 900s default
    and time out the slow tier -- the exact failure GRACE_S exists to prevent."""
    result = _register(script, tmp_path, grace=bad)
    assert result.returncode != 0
    assert not (tmp_path / "runs" / "state" / "mesh_run_configs.json").exists()


# --- CHUNK_ROUNDS: rounds per worker JOB, not per run -----------------------


def test_chunk_rounds_is_written_when_set(script, tmp_path):
    """SallesInfo caps a job at 2h since the 2026-08-22 rebuild, so a ~11h
    40-round run has to be split across successive jobs."""
    result = _register(script, tmp_path, chunk="5")
    assert result.returncode == 0, result.stderr
    assert _configs(tmp_path)[0]["chunk_rounds"] == 5


def test_chunk_rounds_key_is_absent_when_unset(script, tmp_path):
    """Absent means one job per run -- the behaviour of every run registered
    before the walltime cap existed, and the dataclass default."""
    result = _register(script, tmp_path)
    assert result.returncode == 0, result.stderr
    assert "chunk_rounds" not in _configs(tmp_path)[0]


@pytest.mark.parametrize("bad", ["not-a-number", "0", "-5"])
def test_chunk_rounds_rejects_values_that_would_train_nothing(script, tmp_path, bad):
    result = _register(script, tmp_path, chunk=bad)
    assert result.returncode != 0
    assert "CHUNK_ROUNDS" in result.stdout + result.stderr


def test_chunk_rounds_longer_than_the_run_is_rejected(script, tmp_path):
    """It would silently mean "no chunking", which is exactly the
    configuration the partition walltime cap makes unschedulable."""
    result = _register(script, tmp_path, chunk="11")  # _register passes rounds=10
    assert result.returncode != 0
    assert "exceeds rounds" in result.stdout + result.stderr


def test_a_chunk_equal_to_the_run_length_is_allowed(script, tmp_path):
    """Legitimate for a short run that does fit in one job."""
    result = _register(script, tmp_path, chunk="10")
    assert result.returncode == 0, result.stderr
    assert _configs(tmp_path)[0]["chunk_rounds"] == 10


# --- EXPERIMENT: which conf/experiment/*.yaml the workers actually load ------


def test_experiment_defaults_to_the_legacy_mesh_config(script, tmp_path):
    """Every run registered before this env var existed loaded
    videomae_mesh_ssl, so an unset EXPERIMENT must keep doing that."""
    result = _register(script, tmp_path)
    assert result.returncode == 0, result.stderr
    assert _configs(tmp_path)[0]["experiment"] == "videomae_mesh_ssl"


def test_experiment_is_overridable_from_the_environment(script, tmp_path):
    """The experiment name was hardcoded, so a run launched with a
    paper-recipe config would silently train the OLD recipe -- the arm label
    would say one thing and every worker would do another."""
    result = _register(
        script, tmp_path, experiment="videomae_mesh_paper_a_stride1_timeavg"
    )
    assert result.returncode == 0, result.stderr
    assert (
        _configs(tmp_path)[0]["experiment"] == "videomae_mesh_paper_a_stride1_timeavg"
    )


def test_experiment_must_name_a_real_config(script, tmp_path):
    """A typo would otherwise register fine and only surface as a Hydra
    crash on every worker, after the pool has already been seeded."""
    result = _register(script, tmp_path, experiment="videomae_mesh_typo")
    assert result.returncode != 0
    assert "EXPERIMENT" in result.stdout + result.stderr
