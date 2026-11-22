from __future__ import annotations

import contextlib
import difflib
import json
import os
import subprocess
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.system

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "cluster"
    / "sweep_queue.sh"
)
NODES = [f"n{i:02d}" for i in range(16)]
TIER = "fish"
NAN = float("nan")

REGISTER_STUB = """#!/usr/bin/env bash
set -euo pipefail
echo "register $*" >> "${STUB_CALLS}"
python3 - "${STUB_CONFIGS}" "$@" <<'PYEOF'
import json
import sys
from pathlib import Path

argv = sys.argv[1:]
path, run_id, active_csv, rounds, lr, mom = argv[:6]
# launch_item always passes 8 args to REGISTER_CMD, even when quorum_k/
# weighting/extra_overrides are the empty string -- so "argv[7] is present
# but empty" must default the same way bash's ${7:-data} does (empty
# counts as unset), not just "argv[7] is absent".
quorum_k = argv[6] if len(argv) > 6 else ""
weighting = (argv[7] if len(argv) > 7 else "") or "data"
extra_overrides = argv[8] if len(argv) > 8 else ""
path = Path(path)
cfgs = json.loads(path.read_text()) if path.exists() else []
cfgs = [c for c in cfgs if c["run_id"] != run_id]
entry = {
    "run_id": run_id,
    "experiment": "videomae_mesh_ssl",
    "nodes": active_csv.split(","),
    "n_shards": 260,
    "outer_lr": float(lr),
    "outer_momentum": float(mom),
    "weighting": weighting,
    "rounds": int(rounds),
    "merge": True,
}
if quorum_k:
    entry["quorum_k"] = int(quorum_k)
if extra_overrides:
    entry["extra_overrides"] = extra_overrides.split()
cfgs.append(entry)
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(cfgs, indent=2))
tmp.replace(path)
PYEOF
"""

# Registers nothing -- stands in for a state where the run never lands in
# mesh_run_configs.json, so cap_rounds has nothing to update.
REGISTER_NOOP_STUB = """#!/usr/bin/env bash
set -euo pipefail
echo "register $*" >> "${STUB_CALLS}"
"""

LAUNCH_STUB = """#!/usr/bin/env bash
set -euo pipefail
echo "launch $* cwd=$(pwd)" >> "${STUB_CALLS}"
"""

SINFO_STUB = """#!/usr/bin/env bash
cat "${STUB_SINFO_OUT}"
"""

SQUEUE_STUB = """#!/usr/bin/env bash
cat "${STUB_SQUEUE_OUT}"
"""

FAILING_STUB = """#!/usr/bin/env bash
echo "slurm_load_node error" >&2
exit 1
"""

SCANCEL_STUB = """#!/usr/bin/env bash
echo "scancel $*" >> "${STUB_CALLS}"
"""


class Sweep:
    def __init__(self, tmp_path: Path):
        self.tmp = tmp_path
        self.root = tmp_path / "repo"
        self.state = self.root / "runs" / "state"
        self.mesh = tmp_path / "mesh_shared"
        self.bin = tmp_path / "bin"
        self.state.mkdir(parents=True)
        self.bin.mkdir()
        self.calls = tmp_path / "calls.log"
        self.calls.touch()
        self.sinfo_out = tmp_path / "sinfo.txt"
        self.squeue_out = tmp_path / "squeue.txt"
        self.set_cluster(idle=NODES, mine=[])
        self._stub("register.sh", REGISTER_STUB)
        self._stub("register_noop.sh", REGISTER_NOOP_STUB)
        self._stub("launch.sh", LAUNCH_STUB)
        self._stub("sinfo", SINFO_STUB)
        self._stub("squeue", SQUEUE_STUB)
        self._stub("sinfo_broken", FAILING_STUB)
        # On PATH, so an abort implemented via scancel would be recorded --
        # the "no scancel" assertion then has something real to catch.
        self._stub("scancel", SCANCEL_STUB)
        self.register_cmd = self.bin / "register.sh"
        self.sinfo_cmd = self.bin / "sinfo"
        self.stall_s = "3600"
        self.deadline_epoch: str | None = None
        (self.state / f"mesh_pool_ready_{TIER}.txt").write_text("\n".join(NODES) + "\n")
        (self.state / f"mesh_pool_active_{TIER}.txt").write_text(",".join(NODES) + "\n")
        self.set_watcher({})
        # Every test runs as the designated host; the guard is exercised
        # explicitly by test_sweep_refuses_to_run_off_the_designated_host.
        self.set_designation("sweephost")
        self.set_hostname("sweephost")

    def _stub(self, name: str, body: str) -> Path:
        path = self.bin / name
        path.write_text(body)
        path.chmod(0o755)
        return path

    def set_designation(self, host: str) -> None:
        """runs/state/watcher_host names the one host allowed to write mesh
        control-plane state; sweep_queue.sh refuses to run anywhere else."""
        (self.state / "watcher_host").write_text(f"{host}\n")

    def set_hostname(self, host: str) -> None:
        self._stub("hostname", f'#!/bin/sh\necho "{host}"\n')

    def set_cluster(self, idle: list[str], mine: list[str]) -> None:
        """`mine` nodes get synthetic job ids with no jobs_registry.json
        record -- untracked jobs, so busy_nodes() in sweep_queue.sh leaves
        them available (matches the pre-fix "previous run tearing down"
        assumption). Use set_registry()/set_watcher() together to simulate a
        DIFFERENT tracked mesh run actively holding a node instead."""
        self.sinfo_out.write_text("".join(f"{n} idle\n" for n in idle))
        self.squeue_out.write_text(
            "".join(f"{9000 + i}|{n}\n" for i, n in enumerate(mine))
        )

    def set_registry(self, records: list[dict]) -> None:
        (self.state / "jobs_registry.json").write_text(json.dumps(records, indent=2))

    def set_queue(self, items: list[str]) -> None:
        text = "# tier sweep\n\n" + "\n".join(items) + "\n"
        (self.state / f"sweep_queue_{TIER}.txt").write_text(text)

    def set_watcher(self, runs: dict) -> None:
        (self.state / "watcher_state.json").write_text(json.dumps({"runs": runs}))

    def set_configs(self, cfgs: list[dict]) -> None:
        (self.state / "mesh_run_configs.json").write_text(json.dumps(cfgs, indent=2))

    def write_metrics(self, run_id: str, per_round: dict[int, list[float]]) -> None:
        for round_num, losses in per_round.items():
            assert len(losses) <= len(NODES)
            for node, loss in zip(NODES, losses):
                path = self.mesh / "metrics" / run_id / f"{node}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                rec = {
                    "run_id": run_id,
                    "node": node,
                    "round": round_num,
                    "train_loss": loss,
                    "val_loss": loss,
                    "samp_s": 10.0,
                    "n_shards": 32,
                    "ts": 0.0,
                }
                with open(path, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")

    @property
    def configs_path(self) -> Path:
        return self.state / "mesh_run_configs.json"

    def configs(self) -> list[dict]:
        return json.loads(self.configs_path.read_text())

    def log(self) -> str:
        path = self.state / f"sweep_queue_{TIER}.log"
        return path.read_text() if path.exists() else ""

    def call_log(self) -> str:
        return self.calls.read_text()

    def stop(self) -> None:
        (self.state / f"sweep_queue_{TIER}.stop").touch()

    def env(self) -> dict:
        env = dict(os.environ)
        env.update(
            PATH=f"{self.bin}:{env['PATH']}",
            SWEEP_REPO_ROOT=str(self.root),
            SWEEP_MESH_SHARED=str(self.mesh),
            SWEEP_POLL_S="1",
            SWEEP_STALL_S=self.stall_s,
            SWEEP_LAUNCH_CMD=str(self.bin / "launch.sh"),
            SWEEP_REGISTER_CMD=str(self.register_cmd),
            SWEEP_SINFO=str(self.sinfo_cmd),
            SWEEP_SQUEUE=str(self.bin / "squeue"),
            STUB_CALLS=str(self.calls),
            STUB_CONFIGS=str(self.configs_path),
            STUB_SINFO_OUT=str(self.sinfo_out),
            STUB_SQUEUE_OUT=str(self.squeue_out),
        )
        if self.deadline_epoch is not None:
            env["SWEEP_DEADLINE_EPOCH"] = self.deadline_epoch
        return env

    def run(self, timeout: int = 30) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(SCRIPT), TIER],
            env=self.env(),
            cwd=str(self.tmp),
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def popen(self) -> subprocess.Popen:
        # cwd deliberately NOT the repo root: a nohup'd daemon has no reason to
        # inherit one, and launch_mesh_pool.sh writes runs/state relatively.
        return subprocess.Popen(
            ["bash", str(SCRIPT), TIER],
            env=self.env(),
            cwd=str(self.tmp),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    @contextlib.contextmanager
    def running(self):
        proc = self.popen()
        try:
            yield proc
        finally:
            if proc.poll() is None:
                self.stop()
                try:
                    proc.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


def wait_until(predicate, timeout: float = 90.0):
    # Generous on purpose: returns as soon as the condition holds, and the script
    # under test polls once a second, which stretches badly on a loaded machine.
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            value = predicate()
        except (OSError, ValueError, StopIteration):
            value = None
        if value:
            return value
        time.sleep(0.2)
    raise AssertionError("condition not met within timeout")


def _entry(cfgs: list[dict], run_id: str) -> dict:
    return next(c for c in cfgs if c["run_id"] == run_id)


def _registered(sweep: Sweep, run_id: str):
    if not sweep.configs_path.exists():
        return None
    return next((c for c in sweep.configs() if c["run_id"] == run_id), None)


@pytest.fixture()
def sweep(tmp_path):
    return Sweep(tmp_path)


def test_next_item_is_registered_once_the_current_run_reaches_done(sweep):
    sweep.set_queue(["run_a 25 0.7 0.9", "run_b 30 0.45 0.8"])
    sweep.set_watcher({"run_a": {"status": "RUNNING", "round": 1}})
    with sweep.running():
        wait_until(lambda: _registered(sweep, "run_a"))
        assert "run_b" not in sweep.call_log()
        sweep.set_watcher({"run_a": {"status": "DONE", "round": 25}})
        entry = wait_until(lambda: _registered(sweep, "run_b"))
    assert entry["rounds"] == 30
    assert entry["outer_lr"] == 0.45
    assert entry["outer_momentum"] == 0.8
    assert entry["nodes"] == NODES
    assert "run_a DONE" in sweep.log()


def test_second_instance_exits_nonzero_and_changes_nothing(sweep):
    sweep.set_queue(["run_a 25 0.7 0.9"])
    sweep.set_watcher({"run_a": {"status": "RUNNING", "round": 1}})
    with sweep.running():
        wait_until(lambda: _registered(sweep, "run_a"))
        calls_before = sweep.call_log()
        second = sweep.run(timeout=90)
        assert second.returncode != 0
        assert "another sweep_queue" in second.stderr
        # The first instance keeps polling at SWEEP_POLL_S=1 throughout, so its
        # own heartbeat lines may land while the second one starts up --
        # comparing the whole log byte-for-byte makes this test a race. What
        # must hold is that the second instance advanced nothing: no new run
        # was registered and no queue item was consumed.
        assert sweep.call_log() == calls_before
        assert "run_a DONE" not in sweep.log()


def test_stop_file_exits_cleanly_and_leaves_the_run_untouched(sweep):
    sweep.set_queue(["run_a 25 0.7 0.9", "run_b 30 0.45 0.8"])
    sweep.set_watcher({"run_a": {"status": "RUNNING", "round": 2}})
    proc = sweep.popen()
    wait_until(lambda: _registered(sweep, "run_a"))
    sweep.stop()
    assert proc.wait(timeout=90) == 0
    assert _entry(sweep.configs(), "run_a")["rounds"] == 25
    assert "run_b" not in sweep.call_log()


def test_a_stale_done_entry_is_not_accepted_as_completion(sweep):
    sweep.set_queue(["run_a 25 0.7 0.9", "run_b 30 0.45 0.8"])
    # watcher_state.json is never pruned: run_a reads DONE from a previous run
    # before the watcher has picked the newly-registered one up.
    sweep.set_watcher({"run_a": {"status": "DONE", "round": 25}})
    with sweep.running() as proc:
        wait_until(lambda: _registered(sweep, "run_a"))
        wait_until(lambda: "stale" in sweep.log())
        time.sleep(4)
        assert _registered(sweep, "run_b") is None
        assert proc.poll() is None
        sweep.set_watcher({"run_a": {"status": "ASSIGNING", "round": 0}})
        wait_until(lambda: "run_a ASSIGNING" in sweep.log())
        sweep.set_watcher({"run_a": {"status": "DONE", "round": 25}})
        assert wait_until(lambda: _registered(sweep, "run_b"))


def test_divergence_aborts_by_capping_rounds_and_preserves_other_entries(sweep):
    sweep.set_configs(
        [
            {
                "run_id": "other_run",
                "experiment": "videomae_mesh_ssl",
                "nodes": ["z00"],
                "n_shards": 260,
                "outer_lr": 0.7,
                "outer_momentum": 0.9,
                "weighting": "data",
                "rounds": 25,
                "merge": True,
            }
        ]
    )
    sweep.set_queue(["run_a 25 0.7 0.9"])
    sweep.set_watcher({"run_a": {"status": "RUNNING", "round": 0}})
    with sweep.running():
        wait_until(lambda: _registered(sweep, "run_a"))
        before = sweep.configs_path.read_text()
        sweep.write_metrics("run_a", {1: [1.0] * 16, 2: [0.8] * 16, 3: [2.0] * 16})
        sweep.set_watcher({"run_a": {"status": "RUNNING", "round": 3}})
        wait_until(lambda: _entry(sweep.configs(), "run_a")["rounds"] == 3)
        after = sweep.configs_path.read_text()

    assert _entry(sweep.configs(), "other_run")["rounds"] == 25
    changed = [
        line
        for line in difflib.ndiff(before.splitlines(), after.splitlines())
        if line.startswith(("+ ", "- "))
    ]
    assert len(changed) == 2, changed
    assert all("rounds" in line for line in changed), changed
    assert "ABORT run_a: round 3" in sweep.log()
    assert "mean val_loss=2.0" in sweep.log()
    assert "best prior=0.8" in sweep.log()
    assert "scancel" not in sweep.call_log()


def test_a_partial_round_without_quorum_does_not_abort(sweep):
    sweep.set_queue(["run_a 25 0.7 0.9"])
    sweep.set_watcher({"run_a": {"status": "RUNNING", "round": 0}})
    with sweep.running():
        wait_until(lambda: _registered(sweep, "run_a"))
        # Round 4 caught mid-write from a single node, at a loss that would
        # trip the 1.5x rule if it were judged as a completed round.
        sweep.write_metrics(
            "run_a", {1: [1.0] * 16, 2: [0.8] * 16, 3: [0.7] * 16, 4: [1.4]}
        )
        sweep.set_watcher({"run_a": {"status": "RUNNING", "round": 4}})
        wait_until(lambda: "run_a RUNNING round=4" in sweep.log())
        time.sleep(4)
        assert _entry(sweep.configs(), "run_a")["rounds"] == 25
    assert "ABORT" not in sweep.log()


def test_a_single_non_finite_node_does_not_abort(sweep):
    sweep.set_queue(["run_a 25 0.7 0.9"])
    sweep.set_watcher({"run_a": {"status": "RUNNING", "round": 0}})
    with sweep.running():
        wait_until(lambda: _registered(sweep, "run_a"))
        sweep.write_metrics(
            "run_a", {1: [1.0] * 16, 2: [0.8] * 16, 3: [NAN] + [0.7] * 15}
        )
        sweep.set_watcher({"run_a": {"status": "RUNNING", "round": 3}})
        wait_until(lambda: "run_a RUNNING round=3" in sweep.log())
        time.sleep(4)
        assert _entry(sweep.configs(), "run_a")["rounds"] == 25
    assert "ABORT" not in sweep.log()


def test_majority_non_finite_aborts(sweep):
    sweep.set_queue(["run_a 25 0.7 0.9"])
    sweep.set_watcher({"run_a": {"status": "RUNNING", "round": 0}})
    with sweep.running():
        wait_until(lambda: _registered(sweep, "run_a"))
        sweep.write_metrics(
            "run_a", {1: [1.0] * 16, 2: [0.8] * 16, 3: [NAN] * 9 + [0.7] * 7}
        )
        sweep.set_watcher({"run_a": {"status": "RUNNING", "round": 4}})
        wait_until(lambda: _entry(sweep.configs(), "run_a")["rounds"] == 4)
    assert "ABORT run_a: round 3 mean val_loss=nan" in sweep.log()


def test_a_failed_cap_is_retried_on_the_next_poll(sweep):
    sweep.register_cmd = sweep.bin / "register_noop.sh"
    sweep.set_queue(["run_a 25 0.7 0.9"])
    sweep.set_watcher({"run_a": {"status": "RUNNING", "round": 3}})
    sweep.write_metrics("run_a", {1: [1.0] * 16, 2: [0.8] * 16, 3: [2.0] * 16})
    with sweep.running():
        wait_until(lambda: "could not cap rounds" in sweep.log())
        # The entry finally appears; the abort must still fire.
        sweep.set_configs(
            [
                {
                    "run_id": "run_a",
                    "experiment": "videomae_mesh_ssl",
                    "nodes": NODES,
                    "n_shards": 260,
                    "outer_lr": 0.7,
                    "outer_momentum": 0.9,
                    "weighting": "data",
                    "rounds": 25,
                    "merge": True,
                }
            ]
        )
        wait_until(lambda: _entry(sweep.configs(), "run_a")["rounds"] == 3)
    assert "ABORT run_a: round 3" in sweep.log()


def test_healthy_run_is_not_aborted(sweep):
    sweep.set_queue(["run_a 25 0.7 0.9"])
    sweep.set_watcher({"run_a": {"status": "RUNNING", "round": 0}})
    with sweep.running():
        wait_until(lambda: _registered(sweep, "run_a"))
        sweep.write_metrics(
            "run_a", {1: [1.0] * 16, 2: [0.8] * 16, 3: [0.7] * 16, 4: [0.6] * 16}
        )
        sweep.set_watcher({"run_a": {"status": "RUNNING", "round": 4}})
        wait_until(lambda: "run_a RUNNING round=4" in sweep.log())
        time.sleep(4)
        assert _entry(sweep.configs(), "run_a")["rounds"] == 25
    assert "ABORT" not in sweep.log()


def test_a_frozen_watcher_is_reported_as_a_stall(sweep):
    sweep.stall_s = "2"
    sweep.set_queue(["run_a 25 0.7 0.9"])
    sweep.set_watcher({"run_a": {"status": "MERGING", "round": 5}})
    with sweep.running() as proc:
        wait_until(lambda: "STALL run_a: MERGING round=5 unchanged for" in sweep.log())
        assert proc.poll() is None
        assert _entry(sweep.configs(), "run_a")["rounds"] == 25
    assert "ABORT" not in sweep.log()


def test_too_few_available_nodes_forces_a_full_launch(sweep):
    sweep.set_cluster(idle=NODES[:10], mine=[])
    sweep.set_queue(["run_a 25 0.7 0.9"])
    sweep.set_watcher({"run_a": {"status": "RUNNING", "round": 1}})
    with sweep.running():
        wait_until(lambda: "launch" in sweep.call_log())
    assert (
        sweep.call_log().strip() == f"launch {TIER} run_a 25 0.7 0.9 cwd={sweep.root}"
    )
    assert "register" not in sweep.call_log()
    assert "only 10/16 pool nodes available" in sweep.log()
    assert not sweep.configs_path.exists()


def test_a_probe_failure_is_logged_as_such_not_as_zero_nodes(sweep):
    sweep.sinfo_cmd = sweep.bin / "sinfo_broken"
    sweep.set_queue(["run_a 25 0.7 0.9"])
    sweep.set_watcher({"run_a": {"status": "RUNNING", "round": 1}})
    with sweep.running():
        wait_until(lambda: "launch" in sweep.call_log())
    assert "sinfo/squeue error, availability unknown" in sweep.log()
    assert "0/16 pool nodes available" not in sweep.log()


def test_only_the_available_subset_is_registered(sweep):
    sweep.set_cluster(idle=NODES[:12], mine=NODES[12:14])
    sweep.set_queue(["run_a 25 0.7 0.9"])
    sweep.set_watcher({"run_a": {"status": "RUNNING", "round": 1}})
    with sweep.running():
        entry = wait_until(lambda: _registered(sweep, "run_a"))
    assert entry["nodes"] == NODES[:14]
    assert "14/16 pool nodes available" in sweep.log()
    assert "launch" not in sweep.call_log()


def test_nodes_running_our_jobs_count_as_available(sweep):
    sweep.set_cluster(idle=NODES[:6], mine=NODES[6:])
    sweep.set_queue(["run_a 25 0.7 0.9"])
    sweep.set_watcher({"run_a": {"status": "RUNNING", "round": 1}})
    with sweep.running():
        entry = wait_until(lambda: _registered(sweep, "run_a"))
    assert entry["nodes"] == NODES
    assert "16/16 pool nodes available" in sweep.log()
    assert "launch" not in sweep.call_log()


def test_nodes_running_a_different_tracked_runs_jobs_are_not_available(sweep):
    """Reproduces the 2026-09-07 incident: lev-window-200 register-launched
    straight onto lev-fix16c's still-RUNNING nodes because the old
    available_nodes() only asked "is this my job", not "is this job's run
    actually done". A node running one of OUR jobs for a DIFFERENT run this
    sweep doesn't manage -- whose watcher_state status is not DONE -- must be
    excluded, forcing register-only to wait below quorum rather than launch
    straight into the collision."""
    sweep.set_cluster(idle=NODES[:10], mine=NODES[10:])
    sweep.set_registry(
        [
            {"slurm_job_id": str(9000 + i), "run_tag": "other_run", "nodelist": n}
            for i, n in enumerate(NODES[10:])
        ]
    )
    sweep.set_queue(["run_a 25 0.7 0.9 8 token train.accum_steps=128"])
    sweep.set_watcher(
        {
            "run_a": {"status": "RUNNING", "round": 1},
            "other_run": {"status": "RUNNING", "round": 5},
        }
    )
    with sweep.running() as proc:
        wait_until(lambda: proc.poll() is not None, timeout=30)
    assert proc.returncode != 0
    assert "only 10/16 available" in sweep.log()
    assert "cannot full-launch" in sweep.log()
    assert "register" not in sweep.call_log()
    assert not sweep.configs_path.exists()


def test_nodes_running_a_now_done_tracked_runs_jobs_are_available(sweep):
    """The teardown case the original comment was written for: the SAME
    node's job now belongs to a run that has reached DONE -- must still
    count as available, same as an untracked job would."""
    sweep.set_cluster(idle=NODES[:12], mine=NODES[12:])
    sweep.set_registry(
        [
            {"slurm_job_id": str(9000 + i), "run_tag": "prior_run", "nodelist": n}
            for i, n in enumerate(NODES[12:])
        ]
    )
    sweep.set_queue(["run_a 25 0.7 0.9"])
    sweep.set_watcher(
        {
            "run_a": {"status": "RUNNING", "round": 1},
            "prior_run": {"status": "DONE", "round": 25},
        }
    )
    with sweep.running():
        entry = wait_until(lambda: _registered(sweep, "run_a"))
    assert entry["nodes"] == NODES
    assert "16/16 pool nodes available" in sweep.log()


def test_registry_record_missing_run_tag_leaves_node_available(sweep):
    """busy_nodes()'s comment claims a registry record with no run_tag is
    left available, same as no record at all -- only "no record" was ever
    actually exercised (test_nodes_running_our_jobs_count_as_available).
    Here the record exists (so `latest` gets an entry) but is malformed:
    no `run_tag` key. `rec.get("run_tag") or ""` must fall through to the
    untracked/available branch rather than raising or misreading the tag."""
    sweep.set_cluster(idle=NODES[:12], mine=NODES[12:])
    sweep.set_registry(
        [
            {"slurm_job_id": str(9000 + i), "nodelist": n}
            for i, n in enumerate(NODES[12:])
        ]
    )
    sweep.set_queue(["run_a 25 0.7 0.9"])
    sweep.set_watcher({"run_a": {"status": "RUNNING", "round": 1}})
    with sweep.running():
        entry = wait_until(lambda: _registered(sweep, "run_a"))
    assert entry["nodes"] == NODES
    assert "16/16 pool nodes available" in sweep.log()


def test_quorum_k_weighting_and_extra_overrides_reach_register(sweep):
    sweep.set_queue(["run_a 40 0.6 0.9 8 token train.accum_steps=128"])
    sweep.set_watcher({"run_a": {"status": "RUNNING", "round": 1}})
    with sweep.running():
        entry = wait_until(lambda: _registered(sweep, "run_a"))
    assert entry["quorum_k"] == 8
    assert entry["weighting"] == "token"
    assert entry["extra_overrides"] == ["train.accum_steps=128"]


def test_legacy_four_field_queue_line_still_works(sweep):
    """Regression: fish/birds' existing sweep_queue_*.txt files use the old
    4-column format ('run_id rounds outer_lr outer_momentum') -- adding the
    3 new optional fields must not require rewriting those files."""
    sweep.set_queue(["run_a 25 0.7 0.9"])
    sweep.set_watcher({"run_a": {"status": "RUNNING", "round": 1}})
    with sweep.running():
        entry = wait_until(lambda: _registered(sweep, "run_a"))
    assert "quorum_k" not in entry
    assert entry["weighting"] == "data"


def test_full_launch_fallback_errors_instead_of_silently_dropping_overrides(sweep):
    """Below-quorum availability that never recovers must still error out
    eventually, after exhausting retries -- not fall back to launch_mesh_pool.sh
    (silently dropping quorum_k/token weighting/extra_overrides) and not hang
    forever."""
    sweep.set_cluster(idle=NODES[:10], mine=[])  # stays below quorum forever
    sweep.set_queue(["run_a 25 0.7 0.9 8 token train.accum_steps=128"])
    sweep.set_watcher({"run_a": {"status": "RUNNING", "round": 1}})
    with sweep.running() as proc:
        wait_until(lambda: proc.poll() is not None, timeout=30)
    assert proc.returncode != 0
    assert "retrying in" in sweep.log()
    assert "cannot full-launch" in sweep.log()
    assert "launch" not in sweep.call_log()
    assert not sweep.configs_path.exists()


def test_transient_shortage_retries_then_succeeds_once_pool_recovers(sweep):
    """Cars historically sees 15/16 nodes report per round, comfortably above
    the 13-node full-launch threshold -- but not guaranteed. A cell that needs
    quorum_k/weighting/extra_overrides must survive one below-quorum poll by
    retrying, rather than erroring out (or silently full-launching) on the
    first bad reading."""
    sweep.set_cluster(idle=NODES[:10], mine=[])
    sweep.set_queue(["run_a 40 0.6 0.9 8 token train.accum_steps=128"])
    sweep.set_watcher({"run_a": {"status": "RUNNING", "round": 1}})
    with sweep.running():
        wait_until(lambda: "retrying in" in sweep.log())
        sweep.set_cluster(idle=NODES, mine=[])
        entry = wait_until(lambda: _registered(sweep, "run_a"), timeout=30)
    assert entry["quorum_k"] == 8
    assert entry["weighting"] == "token"
    assert entry["extra_overrides"] == ["train.accum_steps=128"]
    assert "launch" not in sweep.call_log()


def test_deadline_reached_exits_before_starting_the_next_item(sweep):
    """SWEEP_DEADLINE_EPOCH is optional and unset by default -- existing
    fish/birds usage (which never sets it) is unaffected. When set and
    already past, the drain loop must exit cleanly before launching anything,
    matching the design's tail-cut convention (whatever hasn't started by the
    deadline is dropped, not pre-shrunk)."""
    sweep.deadline_epoch = str(int(time.time()) - 1)
    sweep.set_queue(["run_a 25 0.7 0.9", "run_b 30 0.45 0.8"])
    with sweep.running() as proc:
        assert proc.wait(timeout=90) == 0
    assert "deadline" in sweep.log()
    assert not sweep.configs_path.exists()
    assert "register" not in sweep.call_log()
    assert "launch" not in sweep.call_log()


def test_sweep_refuses_to_run_off_the_designated_host(sweep):
    """sweep_queue.sh and the watcher both read-modify-write
    mesh_run_configs.json, and flock does not exclude them from each other
    across the shared NFSv3 home -- so they have to be on the same machine."""
    sweep.set_configs([])
    sweep.set_hostname("login-node.example.edu")
    sweep.set_queue(["sw3-fish-a 10 0.7 0.9"])
    before = sweep.configs_path.read_text()

    result = sweep.run()

    assert result.returncode != 0
    assert "designated" in (result.stderr + result.stdout)
    assert sweep.configs_path.read_text() == before


def test_sweep_refuses_when_no_host_is_designated(sweep):
    (sweep.state / "watcher_host").unlink()
    sweep.set_queue(["sw3-fish-a 10 0.7 0.9"])

    result = sweep.run()

    assert result.returncode != 0
