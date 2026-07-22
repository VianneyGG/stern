from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import torch

from video_encoding.mesh.shard_map import Holder, save_shard_map
from video_encoding.watcher.mesh_tick import (
    MeshRunConfig,
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

    fake = MagicMock(returncode=1, stdout="", stderr="boom")
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
    assert remaining == [f"RUN1_{node}.pt" for node in sorted(_NODES)]
    assert state.run("RUN1").status == "MERGING"


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
