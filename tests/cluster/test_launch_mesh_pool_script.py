"""Behavioural tests for launch_mesh_pool.sh.

The script only ever runs on login-node and shells out to sinfo/ssh/uv/sbatch,
so it is exercised here against a mirrored repo layout under tmp_path with
those commands stubbed on PATH. Stubs record their argv, which is what the
ordering and gating assertions read -- nothing here asserts on source text
except the two arithmetic tests, which extract the QUORUM_K lines and
*evaluate* them rather than pattern-matching them.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.system

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_SCRIPT = REPO_ROOT / "cluster" / "launch_mesh_pool.sh"

_STUB = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "${CALL_LOG}"
"""


def _write_stub(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(0o755)


@pytest.fixture
def harness(tmp_path):
    """Mirrored repo + stubbed PATH. Returns (env, call_log, tmp_path)."""
    scripts = tmp_path / "cluster"
    scripts.mkdir(parents=True)
    for name in ("launch_mesh_pool.sh", "tiers_lib.sh", "cluster_env.sh", "site.conf.example", "tiers.example.tsv"):
        shutil.copy2(REPO_ROOT / "cluster" / name, scripts / name)
    (scripts / "launch_mesh_pool.sh").chmod(0o755)
    (tmp_path / "runs" / "state").mkdir(parents=True)

    call_log = tmp_path / "calls.log"
    call_log.touch()

    bindir = tmp_path / "bin"
    bindir.mkdir()
    # Non-empty squeue -> the script assumes the watcher is already up and
    # never touches crontab.
    _write_stub(bindir / "squeue", "#!/usr/bin/env bash\necho 'job 1'\n")
    # Every node in the tier reports idle, so the poll loop exits immediately.
    _write_stub(
        bindir / "sinfo",
        "#!/usr/bin/env bash\n"
        'tr "," "\\n" <<< "${STUB_TIER_NODES}" | while read -r n; do '
        'echo "$n idle"; done\n',
    )
    # The source node already holds the full dataset -> no 223 GB reseed.
    _write_stub(bindir / "ssh", "#!/usr/bin/env bash\necho 260\n")
    _write_stub(bindir / "sleep", "#!/usr/bin/env bash\nexit 0\n")
    # uv dispatches both seed_mesh and validate_seeding; the latter's exit
    # code is what the gate is supposed to honour.
    _write_stub(
        bindir / "uv",
        "#!/usr/bin/env bash\n"
        'printf "uv %s\\n" "$*" >> "${CALL_LOG}"\n'
        'if [[ "$*" == *validate_seeding* ]]; then exit "${STUB_VALIDATE_RC:-0}"; fi\n'
        "exit 0\n",
    )
    _write_stub(
        scripts / "register_mesh_run.sh",
        '#!/usr/bin/env bash\nprintf "register %s\\n" "$*" >> "${CALL_LOG}"\n',
    )

    tier_nodes = subprocess.run(
        ["bash", "-c", f'source "{scripts}/tiers_lib.sh"; tier_nodes fish'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    env = {
        **os.environ,
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "CALL_LOG": str(call_log),
        "STUB_TIER_NODES": tier_nodes,
        "HOME": str(tmp_path / "home"),
        "NEED": "22",
    }
    (tmp_path / "home").mkdir()
    return env, call_log, tmp_path


def _run(harness, extra_env=None, tier="fish", run_id="RUN1"):
    env, call_log, tmp_path = harness
    result = subprocess.run(
        [
            "bash",
            str(tmp_path / "cluster" / "launch_mesh_pool.sh"),
            tier,
            run_id,
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={**env, **(extra_env or {})},
        timeout=120,
    )
    return result, call_log.read_text()


# --- group 3: the validation gate -----------------------------------------


def test_seeding_is_validated_before_the_run_is_registered(harness):
    result, log = _run(harness)
    assert result.returncode == 0, result.stderr

    lines = log.splitlines()
    validate = next(i for i, ln in enumerate(lines) if "validate_seeding" in ln)
    register = next(i for i, ln in enumerate(lines) if ln.startswith("register "))
    assert validate < register, log


def test_failed_validation_blocks_registration(harness):
    """An unvalidated pool must not launch: _do_assigning would either refuse
    (no verified_holdings.json) or assign against a stale one."""
    result, log = _run(harness, {"STUB_VALIDATE_RC": "1"})

    assert result.returncode != 0
    assert "validate_seeding" in log
    assert not any(ln.startswith("register ") for ln in log.splitlines()), log


def test_skip_seed_validation_warns_and_continues(harness):
    result, log = _run(harness, {"SKIP_SEED_VALIDATION": "1", "STUB_VALIDATE_RC": "1"})

    assert result.returncode == 0, result.stderr
    assert "SKIP_SEED_VALIDATION" in result.stderr
    assert "validate_seeding" not in log
    assert any(ln.startswith("register ") for ln in log.splitlines()), log


# --- group 5: quorum and weighting defaults --------------------------------


def test_launcher_passes_quorum_k_and_weighting_to_register(harness):
    result, log = _run(harness)
    assert result.returncode == 0, result.stderr

    register = next(ln for ln in log.splitlines() if ln.startswith("register "))
    fields = register.split()
    # register <run_id> <csv> <rounds> <outer_lr> <outer_momentum> <quorum_k> <weighting>
    assert len(fields) >= 8, register
    assert fields[6] == "12", register  # floor(0.75 * 16 active)
    assert fields[7] == "token", register


def test_quorum_k_and_weighting_env_vars_override_the_defaults(harness):
    result, log = _run(harness, {"QUORUM_K": "5", "WEIGHTING": "inverse_loss"})
    assert result.returncode == 0, result.stderr

    register = next(ln for ln in log.splitlines() if ln.startswith("register "))
    fields = register.split()
    assert fields[6] == "5", register
    assert fields[7] == "inverse_loss", register


def _eval_quorum_arithmetic(n_active: int) -> str:
    """Evaluate the launcher's real QUORUM_K lines for a given pool size.

    Extracts them from the script and runs them, so the test exercises the
    arithmetic itself rather than asserting the source contains some string.
    """
    lines = [
        ln
        for ln in REAL_SCRIPT.read_text().splitlines()
        if ln.startswith("N_ACTIVE=")
        or ln.startswith("QUORUM_K=")
        or ("QUORUM_K" in ln and ln.startswith("[["))
    ]
    assert lines, "no QUORUM_K arithmetic found in launch_mesh_pool.sh"
    snippet = "\n".join(
        [
            f'ACTIVE_CSV="{",".join(f"n{i}" for i in range(n_active))}"',
            *lines,
            'echo "${QUORUM_K}"',
        ]
    )
    return subprocess.run(
        ["bash", "-c", snippet], capture_output=True, text=True, check=True
    ).stdout.strip()


def test_quorum_default_is_three_quarters_of_the_active_pool():
    assert _eval_quorum_arithmetic(16) == "12"
    assert _eval_quorum_arithmetic(22) == "16"
    assert _eval_quorum_arithmetic(8) == "6"


def test_quorum_default_never_falls_below_one():
    """countries has 19 nodes total and a 1-node smoke run is a real case;
    floor(0.75 * 1) is 0, which no node count could ever reach."""
    assert _eval_quorum_arithmetic(1) == "1"


def test_extra_overrides_env_var_reaches_register(harness):
    """Per-arm Hydra knobs must survive the launcher.

    register_mesh_run.sh has accepted extra_overrides as argument 8 all
    along, but the launcher never forwarded it, so factorial arms
    (mask_type/target_mode) could not be launched through the normal path
    and were hand-written into mesh_run_configs.json instead.
    """
    overrides = "model.mask_type=mixed model.target_mode=per_frame"
    result, log = _run(harness, {"EXTRA_OVERRIDES": overrides})
    assert result.returncode == 0, result.stderr

    register = next(ln for ln in log.splitlines() if ln.startswith("register "))
    fields = register.split()
    # register <run_id> <csv> <rounds> <lr> <momentum> <quorum_k> <weighting> <overrides...>
    assert fields[8] == "model.mask_type=mixed", register
    assert fields[9] == "model.target_mode=per_frame", register


def test_extra_overrides_absent_by_default(harness):
    """An unset EXTRA_OVERRIDES must not add a stray argument."""
    result, log = _run(harness)
    assert result.returncode == 0, result.stderr

    register = next(ln for ln in log.splitlines() if ln.startswith("register "))
    # "register" + 7 real args; the empty 8th collapses in "$*".
    assert len(register.split()) == 8, register


# --- group 4: POOL_NODES, the cross-tier roster ----------------------------

# A roster spanning three tiers under a label that is deliberately absent from
# tiers.tsv -- an arm name, not a tier.
_CROSS = "node001,node002,node024,node025,node047,node048"


def _seed_line(log):
    return next(ln for ln in log.splitlines() if "seed_nodes" in ln)


def _validate_line(log):
    return next(ln for ln in log.splitlines() if "validate_seeding" in ln)


def test_pool_nodes_bypasses_the_tier_poll_for_a_non_tier_label(harness):
    """The label keys the pool's state; only POOL_NODES names hosts.

    Without the guard, tier_nodes aborts the script on an unknown tier under
    `set -e` before anything is seeded.
    """
    result, log = _run(harness, {"POOL_NODES": _CROSS}, tier="arm-a")
    assert result.returncode == 0, result.stderr
    assert "no idle poll" in result.stdout, result.stdout
    assert any(ln.startswith("register ") for ln in log.splitlines()), log


def test_pool_nodes_seeds_and_activates_the_identical_roster(harness):
    """Placement must span exactly the training set.

    seed_mesh plans over --nodes but only --active-nodes train, so a --nodes
    superset spreads shards onto nodes no worker reads -- which is what left
    the 2026-08-21 arms holding 192-250 of 260 shards each.
    """
    result, log = _run(harness, {"POOL_NODES": _CROSS}, tier="arm-a")
    assert result.returncode == 0, result.stderr

    seed = _seed_line(log)
    assert f"--nodes {_CROSS}" in seed, seed
    assert f"--active-nodes {_CROSS}" in seed, seed


def test_pool_nodes_validates_exactly_the_roster(harness):
    """A cross-tier pool has no tier to scan, so the gate must be told the hosts."""
    result, log = _run(harness, {"POOL_NODES": _CROSS}, tier="arm-a")
    assert result.returncode == 0, result.stderr
    assert f"--nodes {_CROSS}" in _validate_line(log)


def test_tier_pool_validates_its_own_roster_not_the_whole_tier(harness):
    """22 of the tier's 23 nodes are the pool; the 23rd must not gate the run."""
    result, log = _run(harness)
    assert result.returncode == 0, result.stderr

    validate = _validate_line(log)
    roster = validate.split("--nodes ")[1].split()[0]
    assert len(roster.split(",")) == 22, validate


def test_pool_source_node_overrides_the_first_roster_entry(harness):
    """Lets seeding start from a node that already holds all 260 shards
    instead of pre-staging 223 GB onto whichever host sorted first."""
    result, log = _run(
        harness, {"POOL_NODES": _CROSS, "POOL_SOURCE_NODE": "node048"}, tier="arm-a"
    )
    assert result.returncode == 0, result.stderr
    assert "--source-node node048" in _seed_line(log)
