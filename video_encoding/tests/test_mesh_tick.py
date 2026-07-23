from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import torch

from video_encoding.mesh.shard_map import Holder, save_shard_map
from video_encoding.watcher.epoch_advance import exclude_node
from video_encoding.watcher.mesh_tick import (
    MeshRunConfig,
    _replenish_pool,
    _submit_worker_job,
    advance_all_runs,
    advance_one_run,
)
from video_encoding.watcher.state import WatcherState

# All three fish-tier, equally weighted (SCORE_TABLE["fish"]=100) so a
# 9-shard split lands exactly 3/3/3 with no rounding drift.
_NODES = ["node001", "node002", "node003"]


def _write_pending_checkpoint(path, *, n_samples: int = 100) -> None:
    """Real torch checkpoint (not `.write_text("x")`) -- _do_merging now
    reads pending files with torch.load when master.pt is missing, to
    bootstrap the initial master (see _bootstrap_master)."""
    torch.save(
        {"model_state_dict": {"w": torch.zeros(2)}, "n_samples": n_samples}, path
    )


def _run_config(**overrides) -> MeshRunConfig:
    defaults = dict(
        run_id="RUN1",
        experiment="videomae_mesh_ssl",
        nodes=list(_NODES),
        n_shards=9,
        outer_lr=0.7,
        outer_momentum=0.9,
        weighting="data",
        rounds=10,
        merge=True,
    )
    defaults.update(overrides)
    return MeshRunConfig(**defaults)


def _seed_shard_map(mesh_shared) -> None:
    """9 shards, all seeded on node001 -- phase-1 locality then spreads the
    remaining 6 across node002/node003 in assign_locality_first's phase 2."""
    shard_map = {
        f"shard-{i:06d}.tar": [Holder(host="node001", subnet="/254")] for i in range(9)
    }
    save_shard_map(mesh_shared / "metadata" / "shard_map.json", shard_map)


def test_assigning_writes_assignments_file_and_submits_jobs(tmp_path):
    mesh_shared = tmp_path / "mesh_shared"
    _seed_shard_map(mesh_shared)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    run_config = _run_config()
    state = WatcherState()
    fake = MagicMock(returncode=0, stdout="Submitted batch job 101\n")

    with patch(
        "video_encoding.watcher.mesh_tick.subprocess.run", return_value=fake
    ) as mock_run:
        action = advance_one_run(
            run_config,
            state=state,
            repo_root=repo_root,
            mesh_shared=mesh_shared,
            mesh_root=tmp_path / "mesh_root",
            now=1000.0,
        )

    assert mock_run.call_count == 3
    assignments_path = mesh_shared / "metadata" / "shard_assignments_RUN1.json"
    assignments = json.loads(assignments_path.read_text())
    assert set(assignments.keys()) == set(_NODES)
    assert sum(len(v) for v in assignments.values()) == 9

    # +run_tag must disambiguate the result-JSON filename
    # (tb_logger.result_json_filename) between mesh_tick-driven runs that
    # share the same `experiment` but a different `run_id` -- without it
    # they'd silently overwrite each other's result file.
    submitted_cmd = mock_run.call_args_list[0][0][0]
    assert "+run_tag=RUN1" in submitted_cmd

    assert state.run("RUN1").status == "RUNNING"
    assert "RUNNING" in action


def test_running_transitions_on_first_pending_checkpoint(tmp_path):
    mesh_shared = tmp_path / "mesh_shared"
    pending_dir = mesh_shared / "checkpoints" / "pending"
    pending_dir.mkdir(parents=True)
    (pending_dir / "RUN1_node001.pt").write_text("x")

    run_config = _run_config()
    state = WatcherState()
    state.run("RUN1").status = "RUNNING"

    action = advance_one_run(
        run_config,
        state=state,
        repo_root=tmp_path,
        mesh_shared=mesh_shared,
        mesh_root=tmp_path,
        now=1000.0,
    )

    assert state.run("RUN1").status == "AWAITING_CHECKPOINTS"
    assert action is not None


def test_awaiting_checkpoints_transitions_when_all_arrive(tmp_path):
    mesh_shared = tmp_path / "mesh_shared"
    pending_dir = mesh_shared / "checkpoints" / "pending"
    pending_dir.mkdir(parents=True)
    for node in _NODES:
        (pending_dir / f"RUN1_{node}.pt").write_text("x")

    run_config = _run_config()
    state = WatcherState()
    state.run("RUN1").status = "AWAITING_CHECKPOINTS"

    action = advance_one_run(
        run_config,
        state=state,
        repo_root=tmp_path,
        mesh_shared=mesh_shared,
        mesh_root=tmp_path,
        now=1000.0,
    )

    assert state.run("RUN1").status == "MERGING"
    assert action is not None


def test_awaiting_checkpoints_proceeds_on_timeout_with_partial_set(tmp_path):
    mesh_shared = tmp_path / "mesh_shared"
    pending_dir = mesh_shared / "checkpoints" / "pending"
    pending_dir.mkdir(parents=True)
    # Only 2 of 3 nodes checked in; node003 never showed up.
    old_mtime = 1000.0
    for node in ("node001", "node002"):
        path = pending_dir / f"RUN1_{node}.pt"
        path.write_text("x")
        os.utime(path, (old_mtime, old_mtime))

    run_config = _run_config()
    state = WatcherState()
    state.run("RUN1").status = "AWAITING_CHECKPOINTS"

    now = old_mtime + 4000.0  # well past ROUND_TIMEOUT_S (3600s)
    action = advance_one_run(
        run_config,
        state=state,
        repo_root=tmp_path,
        mesh_shared=mesh_shared,
        mesh_root=tmp_path,
        now=now,
    )

    assert state.run("RUN1").status == "MERGING"
    assert state.run("RUN1").missed_rounds.get("node003") == 1
    assert "node003" not in state.excluded_nodes  # only 1 miss so far
    assert action is not None


def test_merging_calls_run_outer_step_with_correct_args(tmp_path):
    mesh_shared = tmp_path / "mesh_shared"
    (mesh_shared / "checkpoints" / "pending").mkdir(parents=True)

    run_config = _run_config(outer_lr=0.7, outer_momentum=0.9, weighting="data")
    state = WatcherState()
    state.run("RUN1").status = "MERGING"

    fake = MagicMock(returncode=0, stdout="", stderr="")
    with patch(
        "video_encoding.watcher.mesh_tick.subprocess.run", return_value=fake
    ) as mock_run:
        advance_one_run(
            run_config,
            state=state,
            repo_root=tmp_path,
            mesh_shared=mesh_shared,
            mesh_root=tmp_path,
            now=1000.0,
        )

    cmd = mock_run.call_args[0][0]
    # Must invoke as a module (`-m pkg.mod`), not a bare script path -- a
    # bare `python path/to/script.py` puts the script's own directory on
    # sys.path[0], not the repo root, so run_outer_step.py's
    # `from video_encoding... import` can never resolve and every merge
    # fails with ModuleNotFoundError regardless of args below.
    assert cmd[:5] == [
        "uv",
        "run",
        "python",
        "-m",
        "video_encoding.scripts.run_outer_step",
    ]
    assert "--master" in cmd
    master_idx = cmd.index("--master")
    assert cmd[master_idx + 1] == str(
        mesh_shared / "checkpoints" / "RUN1" / "master.pt"
    )
    outer_lr_idx = cmd.index("--outer-lr")
    assert cmd[outer_lr_idx + 1] == "0.7"
    outer_momentum_idx = cmd.index("--outer-momentum")
    assert cmd[outer_momentum_idx + 1] == "0.9"
    weighting_idx = cmd.index("--weighting")
    assert cmd[weighting_idx + 1] == "data"


def test_merging_bootstraps_master_from_pending_when_missing(tmp_path):
    """Round 1 of any run: master.pt has never been written by anyone.
    merge_round (run_outer_step.py) unconditionally torch.loads it with no
    cold-start fallback and this plan forbids modifying that file, so
    _do_merging must seed it itself before invoking the subprocess --
    otherwise every run's first merge crashes and retries forever."""
    mesh_shared = tmp_path / "mesh_shared"
    pending_dir = mesh_shared / "checkpoints" / "pending"
    pending_dir.mkdir(parents=True)
    for node in _NODES:
        _write_pending_checkpoint(pending_dir / f"RUN1_{node}.pt")

    run_config = _run_config()
    state = WatcherState()
    state.run("RUN1").status = "MERGING"

    master_path = mesh_shared / "checkpoints" / "RUN1" / "master.pt"
    fake = MagicMock(returncode=0, stdout="", stderr="")
    with patch("video_encoding.watcher.mesh_tick.subprocess.run", return_value=fake):
        advance_one_run(
            run_config,
            state=state,
            repo_root=tmp_path,
            mesh_shared=mesh_shared,
            mesh_root=tmp_path,
            now=1000.0,
        )

    assert master_path.exists()
    payload = torch.load(master_path, map_location="cpu", weights_only=False)
    assert "model_state_dict" in payload


def test_merging_does_not_overwrite_existing_master(tmp_path):
    mesh_shared = tmp_path / "mesh_shared"
    pending_dir = mesh_shared / "checkpoints" / "pending"
    pending_dir.mkdir(parents=True)
    for node in _NODES:
        _write_pending_checkpoint(pending_dir / f"RUN1_{node}.pt")

    master_path = mesh_shared / "checkpoints" / "RUN1" / "master.pt"
    master_path.parent.mkdir(parents=True)
    torch.save({"model_state_dict": {"w": torch.ones(2)}}, master_path)
    original_mtime = master_path.stat().st_mtime_ns

    run_config = _run_config()
    state = WatcherState()
    state.run("RUN1").status = "MERGING"

    fake = MagicMock(returncode=0, stdout="", stderr="")
    with patch("video_encoding.watcher.mesh_tick.subprocess.run", return_value=fake):
        advance_one_run(
            run_config,
            state=state,
            repo_root=tmp_path,
            mesh_shared=mesh_shared,
            mesh_root=tmp_path,
            now=1000.0,
        )

    assert master_path.stat().st_mtime_ns == original_mtime


def test_merging_deletes_pending_on_success(tmp_path):
    mesh_shared = tmp_path / "mesh_shared"
    pending_dir = mesh_shared / "checkpoints" / "pending"
    pending_dir.mkdir(parents=True)
    for node in _NODES:
        _write_pending_checkpoint(pending_dir / f"RUN1_{node}.pt")
    other_run_file = pending_dir / "RUN2_node001.pt"
    other_run_file.write_text("keep me")

    run_config = _run_config()
    state = WatcherState()
    state.run("RUN1").status = "MERGING"

    fake = MagicMock(returncode=0, stdout="", stderr="")
    with patch("video_encoding.watcher.mesh_tick.subprocess.run", return_value=fake):
        advance_one_run(
            run_config,
            state=state,
            repo_root=tmp_path,
            mesh_shared=mesh_shared,
            mesh_root=tmp_path,
            now=1000.0,
        )

    remaining = sorted(p.name for p in pending_dir.glob("*.pt"))
    assert remaining == ["RUN2_node001.pt"]
    assert other_run_file.exists()
    assert state.run("RUN1").status == "SYNCING"


def test_merging_retries_on_failure_without_deleting_pending(tmp_path):
    mesh_shared = tmp_path / "mesh_shared"
    pending_dir = mesh_shared / "checkpoints" / "pending"
    pending_dir.mkdir(parents=True)
    for node in _NODES:
        _write_pending_checkpoint(pending_dir / f"RUN1_{node}.pt")

    run_config = _run_config()
    state = WatcherState()
    state.run("RUN1").status = "MERGING"

    fake = MagicMock(returncode=1, stdout="", stderr="Traceback...\nValueError: boom")
    with patch("video_encoding.watcher.mesh_tick.subprocess.run", return_value=fake):
        result_msg = advance_one_run(
            run_config,
            state=state,
            repo_root=tmp_path,
            mesh_shared=mesh_shared,
            mesh_root=tmp_path,
            now=1000.0,
        )

    remaining = sorted(p.name for p in pending_dir.glob("*.pt"))
    assert remaining == [f"RUN1_{node}.pt" for node in sorted(_NODES)]
    assert state.run("RUN1").status == "MERGING"
    # The failure line must carry the actual error, not just the exit code --
    # a bare "exit 1" is what let this class of bug go undiagnosed for
    # several ticks in production before someone had to SSH in and
    # reproduce it manually.
    assert result_msg is not None and "ValueError: boom" in result_msg


def test_merge_skipped_for_single_node_arm(tmp_path):
    mesh_shared = tmp_path / "mesh_shared"
    (mesh_shared / "checkpoints" / "pending").mkdir(parents=True)

    run_config = _run_config(merge=False, nodes=["node001"])
    state = WatcherState()
    state.run("RUN1").status = "RUNNING"

    with patch("video_encoding.watcher.mesh_tick.subprocess.run") as mock_run:
        action = advance_one_run(
            run_config,
            state=state,
            repo_root=tmp_path,
            mesh_shared=mesh_shared,
            mesh_root=tmp_path,
            now=1000.0,
        )

    mock_run.assert_not_called()
    assert state.run("RUN1").status == "DONE"
    assert action is not None


def test_syncing_writes_flag_and_advances_round(tmp_path):
    mesh_shared = tmp_path / "mesh_shared"
    run_config = _run_config(rounds=10)
    state = WatcherState()
    state.run("RUN1").status = "SYNCING"
    state.run("RUN1").round = 0

    advance_one_run(
        run_config,
        state=state,
        repo_root=tmp_path,
        mesh_shared=mesh_shared,
        mesh_root=tmp_path,
        now=1000.0,
    )

    flag_path = mesh_shared / "checkpoints" / "flags" / "sync_complete_RUN1_0.flag"
    assert flag_path.exists()
    assert state.run("RUN1").round == 1
    # Not ASSIGNING: Task 2's worker job is a single long-lived SLURM
    # allocation that loops over every round internally and never exits
    # between rounds, so re-entering ASSIGNING here would submit a second,
    # duplicate job per node on top of the still-alive original. RUNNING
    # correctly waits for the next round's checkpoints instead.
    assert state.run("RUN1").status == "RUNNING"


def test_syncing_reaches_done_at_final_round(tmp_path):
    mesh_shared = tmp_path / "mesh_shared"
    run_config = _run_config(rounds=3)
    state = WatcherState()
    state.run("RUN1").status = "SYNCING"
    state.run("RUN1").round = 2  # entering the final round's sync

    advance_one_run(
        run_config,
        state=state,
        repo_root=tmp_path,
        mesh_shared=mesh_shared,
        mesh_root=tmp_path,
        now=1000.0,
    )

    flag_path = mesh_shared / "checkpoints" / "flags" / "sync_complete_RUN1_2.flag"
    assert flag_path.exists()
    assert state.run("RUN1").round == 3
    assert state.run("RUN1").status == "DONE"


def test_advance_all_runs_skips_done_runs(tmp_path):
    mesh_shared = tmp_path / "mesh_shared"
    pending_dir = mesh_shared / "checkpoints" / "pending"
    pending_dir.mkdir(parents=True)
    (pending_dir / "RUN2_node001.pt").write_text("x")

    done_config = _run_config(run_id="RUN1")
    running_config = _run_config(run_id="RUN2")

    state = WatcherState()
    state.run("RUN1").status = "DONE"
    state.run("RUN1").round = 5
    state.run("RUN2").status = "RUNNING"

    actions = advance_all_runs(
        [done_config, running_config],
        state=state,
        repo_root=tmp_path,
        mesh_shared=mesh_shared,
        mesh_root=tmp_path,
        now=1000.0,
    )

    assert state.run("RUN1").status == "DONE"
    assert state.run("RUN1").round == 5  # untouched
    assert state.run("RUN2").status == "AWAITING_CHECKPOINTS"
    assert len(actions) == 1
    assert "RUN2" in actions[0]


def test_submit_worker_job_includes_start_round_zero_by_default(tmp_path):
    run_config = _run_config()
    registry_path = tmp_path / "jobs_registry.json"
    fake = MagicMock(returncode=0, stdout="Submitted batch job 555\n")
    with patch(
        "video_encoding.watcher.mesh_tick.subprocess.run", return_value=fake
    ) as mock_run:
        job_id = _submit_worker_job(
            run_config, "node001", tmp_path, registry_path, "2026-01-01T00:00:00Z"
        )
    assert job_id == "555"
    submitted_cmd = mock_run.call_args[0][0]
    assert "+mesh.start_round=0" in submitted_cmd


def test_submit_worker_job_passes_nonzero_start_round(tmp_path):
    run_config = _run_config()
    registry_path = tmp_path / "jobs_registry.json"
    fake = MagicMock(returncode=0, stdout="Submitted batch job 556\n")
    with patch(
        "video_encoding.watcher.mesh_tick.subprocess.run", return_value=fake
    ) as mock_run:
        _submit_worker_job(
            run_config,
            "node001",
            tmp_path,
            registry_path,
            "2026-01-01T00:00:00Z",
            start_round=5,
        )
    submitted_cmd = mock_run.call_args[0][0]
    assert "+mesh.start_round=5" in submitted_cmd


def test_replenish_pool_noop_when_deficit_zero(tmp_path):
    run_config = _run_config()  # 3 nodes
    state = WatcherState()
    run = state.run("RUN1")
    run.target_pool_size = 3

    result = _replenish_pool(
        run_config,
        run,
        state,
        tmp_path,
        tmp_path,
        tmp_path / "mesh_run_configs.json",
        "node001",
    )
    assert result is None
    assert not (tmp_path / "mesh_run_configs.json").exists()


def test_replenish_pool_returns_none_when_run_configs_path_missing(tmp_path):
    run_config = _run_config()
    state = WatcherState()
    run = state.run("RUN1")
    run.target_pool_size = 3
    exclude_node(state, "node003", "crash-loop")

    with patch("video_encoding.watcher.mesh_tick.subprocess.run") as mock_run:
        result = _replenish_pool(
            run_config, run, state, tmp_path, tmp_path, None, "node003"
        )
    assert result is None
    mock_run.assert_not_called()


def test_replenish_pool_appends_idle_node_and_reassigns_shards(tmp_path):
    mesh_shared = tmp_path / "mesh_shared"
    run_config = _run_config()  # node001, node002, node003 -- fish tier
    state = WatcherState()
    run = state.run("RUN1")
    run.target_pool_size = 3
    run.round = 4
    exclude_node(state, "node003", "crash-loop: missed 2 consecutive rounds of RUN1")

    shard_map = {
        "s1": [Holder(host="node001", subnet="/254")],
        "s2": [Holder(host="node002", subnet="/254")],
        "s3": [Holder(host="node003", subnet="/254")],
    }
    save_shard_map(mesh_shared / "metadata" / "shard_map.json", shard_map)
    assignments_path = mesh_shared / "metadata" / "shard_assignments_RUN1.json"
    assignments_path.write_text(
        json.dumps({"node001": ["s1"], "node002": ["s2"], "node003": ["s3"]})
    )

    run_configs_path = tmp_path / "mesh_run_configs.json"
    run_configs_path.write_text(
        json.dumps(
            [
                {
                    "run_id": "RUN1",
                    "experiment": "videomae_mesh_ssl",
                    "nodes": list(_NODES),
                    "n_shards": 3,
                    "outer_lr": 0.7,
                    "outer_momentum": 0.9,
                    "weighting": "data",
                    "rounds": 10,
                    "merge": True,
                }
            ]
        )
    )

    idle_sinfo = MagicMock(returncode=0, stdout="node004 idle\n")
    sbatch_result = MagicMock(returncode=0, stdout="Submitted batch job 999\n")

    # video_encoding.watcher.cluster and video_encoding.watcher.mesh_tick both
    # do a bare `import subprocess`, so they share the single `subprocess`
    # module object -- patching subprocess.run via either module's dotted
    # path mutates the same global attribute, not an isolated per-module
    # view. A single patch with a command-dispatching side_effect (instead
    # of two colliding patches, whichever is entered last silently winning
    # for both call sites) is what actually isolates the sinfo vs sbatch
    # responses.
    def _fake_run(cmd, *args, **kwargs):
        return idle_sinfo if cmd[0] == "sinfo" else sbatch_result

    with patch(
        "video_encoding.watcher.mesh_tick.subprocess.run", side_effect=_fake_run
    ) as mock_sbatch:
        result = _replenish_pool(
            run_config,
            run,
            state,
            mesh_shared,
            tmp_path,
            run_configs_path,
            "node003",
        )

    assert "node004" in result
    submitted_cmd = mock_sbatch.call_args[0][0]
    assert "+mesh.start_round=5" in submitted_cmd  # run.round(4) + 1

    grown = json.loads(run_configs_path.read_text())
    assert "node004" in grown[0]["nodes"]

    assignments = json.loads(assignments_path.read_text())
    assert "node003" not in assignments
    assert assignments["node004"] == ["s3"]

    registry = json.loads(
        (tmp_path / ".claude" / "state" / "jobs_registry.json").read_text()
    )
    assert any(r["nodelist"] == "node004" for r in registry)


def test_replenish_pool_partial_fill_when_not_enough_idle(tmp_path):
    mesh_shared = tmp_path / "mesh_shared"
    run_config = _run_config()
    state = WatcherState()
    run = state.run("RUN1")
    run.target_pool_size = 5  # eligible drops to 2 once node003 is excluded,
    exclude_node(state, "node003", "crash-loop")  # so deficit = 5 - 2 = 3

    run_configs_path = tmp_path / "mesh_run_configs.json"
    run_configs_path.write_text(
        json.dumps(
            [
                {
                    "run_id": "RUN1",
                    "experiment": "videomae_mesh_ssl",
                    "nodes": list(_NODES),
                    "n_shards": 3,
                    "outer_lr": 0.7,
                    "outer_momentum": 0.9,
                    "weighting": "data",
                    "rounds": 10,
                    "merge": True,
                }
            ]
        )
    )

    idle_sinfo = MagicMock(returncode=0, stdout="node004 idle\n")  # only 1 idle
    sbatch_result = MagicMock(returncode=0, stdout="Submitted batch job 999\n")

    # See the sibling test above for why this must be a single patch with a
    # dispatching side_effect rather than two colliding patches on the
    # shared `subprocess` module.
    def _fake_run(cmd, *args, **kwargs):
        return idle_sinfo if cmd[0] == "sinfo" else sbatch_result

    with patch(
        "video_encoding.watcher.mesh_tick.subprocess.run", side_effect=_fake_run
    ):
        result = _replenish_pool(
            run_config,
            run,
            state,
            mesh_shared,
            tmp_path,
            run_configs_path,
            "node003",
        )

    grown = json.loads(run_configs_path.read_text())
    assert grown[0]["nodes"].count("node004") == 1
    assert len(grown[0]["nodes"]) == len(_NODES) + 1  # only 1 of the 2 needed was idle
    assert result is not None


def test_advance_one_run_sets_target_pool_size_on_first_sight(tmp_path):
    run_config = _run_config()  # 3 nodes
    state = WatcherState()
    # advance_one_run's default status is ASSIGNING, which requires a shard
    # map to plan assignments -- unrelated to target_pool_size itself, but
    # needed so the dispatch below doesn't raise before reaching the
    # assertion. sbatch isn't mocked: _submit_worker_job's own
    # OSError/SubprocessError catch (sbatch not on PATH here) already makes
    # that a silent no-op, matching its documented best-effort handling.
    _seed_shard_map(tmp_path / "mesh_shared")

    advance_one_run(
        run_config,
        state=state,
        repo_root=tmp_path,
        mesh_shared=tmp_path / "mesh_shared",
        mesh_root=tmp_path,
        now=1000.0,
    )
    assert state.run("RUN1").target_pool_size == 3

    # Simulate a prior replenishment having grown the roster; target must
    # not move on a later tick.
    grown_config = _run_config(nodes=_NODES + ["node004"])
    advance_one_run(
        grown_config,
        state=state,
        repo_root=tmp_path,
        mesh_shared=tmp_path / "mesh_shared",
        mesh_root=tmp_path,
        now=2000.0,
    )
    assert state.run("RUN1").target_pool_size == 3


def test_do_awaiting_checkpoints_triggers_replenish_on_new_exclusion(tmp_path):
    mesh_shared = tmp_path / "mesh_shared"
    pending_dir = mesh_shared / "checkpoints" / "pending"
    pending_dir.mkdir(parents=True)
    old_mtime = 1000.0
    for node in ("node001", "node002"):
        path = pending_dir / f"RUN1_{node}.pt"
        path.write_text("x")
        os.utime(path, (old_mtime, old_mtime))

    save_shard_map(
        mesh_shared / "metadata" / "shard_map.json",
        {"s1": [Holder(host="node001", subnet="/254")]},
    )
    (mesh_shared / "metadata" / "shard_assignments_RUN1.json").write_text(
        json.dumps({"node001": ["s1"], "node002": [], "node003": []})
    )

    run_configs_path = tmp_path / "mesh_run_configs.json"
    run_config = _run_config()
    run_configs_path.write_text(
        json.dumps(
            [
                {
                    "run_id": "RUN1",
                    "experiment": run_config.experiment,
                    "nodes": list(_NODES),
                    "n_shards": 9,
                    "outer_lr": 0.7,
                    "outer_momentum": 0.9,
                    "weighting": "data",
                    "rounds": 10,
                    "merge": True,
                }
            ]
        )
    )

    state = WatcherState()
    state.run("RUN1").status = "AWAITING_CHECKPOINTS"
    state.run("RUN1").target_pool_size = 3
    state.run("RUN1").missed_rounds["node003"] = 1  # one miss already recorded

    idle_sinfo = MagicMock(returncode=0, stdout="node004 idle\n")
    sbatch_result = MagicMock(returncode=0, stdout="Submitted batch job 999\n")
    now = old_mtime + 4000.0  # past ROUND_TIMEOUT_S

    # See test_replenish_pool_appends_idle_node_and_reassigns_shards for why
    # this must be a single patch with a dispatching side_effect rather than
    # two colliding patches on the shared `subprocess` module.
    def _fake_run(cmd, *args, **kwargs):
        return idle_sinfo if cmd[0] == "sinfo" else sbatch_result

    with patch(
        "video_encoding.watcher.mesh_tick.subprocess.run", side_effect=_fake_run
    ):
        action = advance_one_run(
            run_config,
            state=state,
            repo_root=tmp_path,
            mesh_shared=mesh_shared,
            mesh_root=tmp_path,
            now=now,
            run_configs_path=run_configs_path,
        )

    assert "node003" in state.excluded_nodes  # 2nd miss -> excluded
    assert state.run("RUN1").status == "MERGING"
    assert "node004" in action
    grown = json.loads(run_configs_path.read_text())
    assert "node004" in grown[0]["nodes"]


def test_do_awaiting_checkpoints_does_not_double_pick_same_node_for_simultaneous_exclusions(
    tmp_path,
):
    """Two nodes crossing the crash-loop threshold in the same tick must not
    both claim the same idle replacement. _do_awaiting_checkpoints calls
    _replenish_pool once per newly-excluded node in `excluded`; without
    _live_roster, both calls would independently see the same stale
    run_config.nodes and the same idle candidate, and both would pick and
    sbatch node007. With _live_roster, the second call reads the roster the
    first call already grew and correctly finds node007 already claimed."""
    mesh_shared = tmp_path / "mesh_shared"
    pending_dir = mesh_shared / "checkpoints" / "pending"
    pending_dir.mkdir(parents=True)
    old_mtime = 1000.0
    for node in ("node001", "node002"):
        path = pending_dir / f"RUN1_{node}.pt"
        path.write_text("x")
        os.utime(path, (old_mtime, old_mtime))

    nodes = ["node001", "node002", "node003", "node004"]
    run_config = _run_config(nodes=nodes, n_shards=len(nodes))

    run_configs_path = tmp_path / "mesh_run_configs.json"
    run_configs_path.write_text(
        json.dumps(
            [
                {
                    "run_id": "RUN1",
                    "experiment": run_config.experiment,
                    "nodes": list(nodes),
                    "n_shards": run_config.n_shards,
                    "outer_lr": 0.7,
                    "outer_momentum": 0.9,
                    "weighting": "data",
                    "rounds": 10,
                    "merge": True,
                }
            ]
        )
    )

    state = WatcherState()
    state.run("RUN1").status = "AWAITING_CHECKPOINTS"
    state.run("RUN1").target_pool_size = 4
    # Both node003 and node004 already have one miss recorded -- this
    # tick's timeout pushes both over the threshold=2 in the same call.
    state.run("RUN1").missed_rounds["node003"] = 1
    state.run("RUN1").missed_rounds["node004"] = 1

    idle_sinfo = MagicMock(returncode=0, stdout="node007 idle\n")  # only 1 idle
    sbatch_result = MagicMock(returncode=0, stdout="Submitted batch job 999\n")

    def _fake_run(cmd, *args, **kwargs):
        return idle_sinfo if cmd[0] == "sinfo" else sbatch_result

    now = old_mtime + 4000.0  # past ROUND_TIMEOUT_S
    with patch(
        "video_encoding.watcher.mesh_tick.subprocess.run", side_effect=_fake_run
    ) as mock_run:
        advance_one_run(
            run_config,
            state=state,
            repo_root=tmp_path,
            mesh_shared=mesh_shared,
            mesh_root=tmp_path,
            now=now,
            run_configs_path=run_configs_path,
        )

    assert "node003" in state.excluded_nodes
    assert "node004" in state.excluded_nodes

    node007_sbatch_calls = [
        c
        for c in mock_run.call_args_list
        if c[0][0][0] == "sbatch" and "--nodelist=node007" in c[0][0]
    ]
    assert len(node007_sbatch_calls) == 1
