from __future__ import annotations

import contextlib
import difflib
import json
import os
import subprocess
import time
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "video_encoding"
    / "scripts"
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

path, run_id, active_csv, rounds, lr, mom = sys.argv[1:7]
path = Path(path)
cfgs = json.loads(path.read_text()) if path.exists() else []
cfgs = [c for c in cfgs if c["run_id"] != run_id]
cfgs.append(
    {
        "run_id": run_id,
        "experiment": "videomae_mesh_ssl",
        "nodes": active_csv.split(","),
        "n_shards": 260,
        "outer_lr": float(lr),
        "outer_momentum": float(mom),
        "weighting": "data",
        "rounds": int(rounds),
        "merge": True,
    }
)
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
        self.state = self.root / ".claude" / "state"
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
        (self.state / f"mesh_pool_ready_{TIER}.txt").write_text("\n".join(NODES) + "\n")
        (self.state / f"mesh_pool_active_{TIER}.txt").write_text(",".join(NODES) + "\n")
        self.set_watcher({})

    def _stub(self, name: str, body: str) -> Path:
        path = self.bin / name
        path.write_text(body)
        path.chmod(0o755)
        return path

    def set_cluster(self, idle: list[str], mine: list[str]) -> None:
        self.sinfo_out.write_text("".join(f"{n} idle\n" for n in idle))
        self.squeue_out.write_text("".join(f"{n}\n" for n in mine))

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
        # inherit one, and launch_mesh_pool.sh writes .claude/state relatively.
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
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


def wait_until(predicate, timeout: float = 25.0):
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


def test_script_has_no_syntax_errors():
    result = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


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
        log_before, calls_before = sweep.log(), sweep.call_log()
        second = sweep.run(timeout=20)
        assert second.returncode != 0
        assert "another sweep_queue" in second.stderr
        assert sweep.log() == log_before
        assert sweep.call_log() == calls_before


def test_stop_file_exits_cleanly_and_leaves_the_run_untouched(sweep):
    sweep.set_queue(["run_a 25 0.7 0.9", "run_b 30 0.45 0.8"])
    sweep.set_watcher({"run_a": {"status": "RUNNING", "round": 2}})
    proc = sweep.popen()
    wait_until(lambda: _registered(sweep, "run_a"))
    sweep.stop()
    assert proc.wait(timeout=20) == 0
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
