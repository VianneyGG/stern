from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from video_encoding.watcher.epoch_advance import exclude_node
from video_encoding.mesh.outer_step import BOOTSTRAP_MASTER_VERSION
from video_encoding.watcher.mesh_tick import (
    MeshRunConfig,
    _clean_stale_run_artifacts,
    _do_awaiting_checkpoints,
    _recover_pool,
    _relaunch_chunk_if_finished,
    _replenish_pool,
    _submit_worker_job,
    advance_all_runs,
    advance_one_run,
    effective_quorum_k,
    load_run_configs,
    redistribute_shards,
    resubmit_with_override,
    sweep_orphaned_pending,
)
from video_encoding.watcher.state import WatcherState


@pytest.fixture(autouse=True)
def _local_merge_root(tmp_path_factory, monkeypatch):
    """_do_merging writes the momentum/residual buffers to watcher-local
    disk, which defaults to /Data -- unwritable here (and not somewhere a
    test should be writing anyway)."""
    monkeypatch.setenv("MESH_LOCAL_ROOT", str(tmp_path_factory.mktemp("mesh_local")))


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


def test_meshrunconfig_broadcast_dtype_defaults_to_fp16():
    """3.1: halves the 349 MiB/round x 16 nodes master broadcast by default
    -- the unbiased error-feedback path already exists, this only flips
    which dtype a run gets when it doesn't ask."""
    assert _run_config().broadcast_dtype == "fp16"


def test_meshrunconfig_extra_overrides_defaults_to_empty_list():
    assert _run_config().extra_overrides == []


def test_meshrunconfig_extra_overrides_round_trips_through_load_run_configs(tmp_path):
    path = tmp_path / "mesh_run_configs.json"
    path.write_text(
        json.dumps(
            [
                {
                    "run_id": "RUN1",
                    "experiment": "videomae_mesh_ssl",
                    "nodes": ["node001"],
                    "n_shards": 9,
                    "outer_lr": 0.7,
                    "outer_momentum": 0.9,
                    "weighting": "data",
                    "rounds": 10,
                    "merge": True,
                    "extra_overrides": ["train.accum_steps=128", "train.lr_mult=1.5"],
                }
            ]
        )
    )
    configs = load_run_configs(path)
    assert configs[0].extra_overrides == ["train.accum_steps=128", "train.lr_mult=1.5"]


def test_effective_quorum_k_unchanged_when_pool_healthy():
    state = WatcherState()
    run_config = _run_config(nodes=["a", "b", "c"], n_shards=3, quorum_k=3)
    run = state.run("RUN1")
    run.target_pool_size = 3
    assert effective_quorum_k(run_config, run, state, roster=["a", "b", "c"]) == 3


def test_effective_quorum_k_relaxes_for_unreplaceable_missing_nodes():
    state = WatcherState()
    run_config = _run_config(nodes=["a", "b", "c"], n_shards=3, quorum_k=3)
    run = state.run("RUN1")
    run.target_pool_size = 3
    result = effective_quorum_k(
        run_config, run, state, roster=["a", "b", "c"], missing_unreplaceable={"c"}
    )
    assert result == 2


def test_effective_quorum_k_clamps_at_60_percent_floor():
    state = WatcherState()
    run_config = _run_config(nodes=["a", "b", "c", "d", "e"], n_shards=5, quorum_k=5)
    run = state.run("RUN1")
    run.target_pool_size = 5
    result = effective_quorum_k(
        run_config,
        run,
        state,
        roster=["a", "b", "c", "d", "e"],
        missing_unreplaceable={"b", "c", "d", "e"},
    )
    assert result == 3  # ceil(0.6 * 5) = 3, not 1


def test_effective_quorum_k_no_floor_configured_returns_roster_size():
    state = WatcherState()
    run_config = _run_config(nodes=["a", "b", "c"], n_shards=3)  # quorum_k=None
    run = state.run("RUN1")
    assert effective_quorum_k(run_config, run, state, roster=["a", "b"]) == 2


def _seed_shard_map(mesh_shared) -> None:
    """9 shards spread 3-per-node across the roster.

    Assignment is now static -- a node is only ever given shards it verifiably
    holds -- so the fixture records real per-node holdings rather than piling
    everything on node001 and relying on a remainder-fill phase to spread it.
    shard_map.json is still written for the paths that have not migrated off
    it yet (fetch, reassign_failed).
    """
    shards = [f"shard-{i:06d}.tar" for i in range(9)]
    holdings = {node: shards[i::3] for i, node in enumerate(_NODES)}
    _seed_placement(mesh_shared, holdings, shards)


def _seed_placement(mesh_shared, holdings, shards=None) -> None:
    """Write verified_holdings.json (+ manifest) -- the only shard-location truth.

    Both `_do_assigning` and the `reassign_failed` handoff read this now;
    shard_map.json is no longer consulted by any watcher path.
    """
    if shards is None:
        shards = sorted({s for ss in holdings.values() for s in ss})
    metadata = mesh_shared / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / "verified_holdings.json").write_text(json.dumps({"fish": holdings}))
    (metadata / "shard_manifest.json").write_text(
        json.dumps({"train": {name: 100 for name in shards}, "val": {}})
    )


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
    assert state.run("RUN1").round_started_at == 1000.0

    # +run_tag must disambiguate the result-JSON filename
    # (tb_logger.result_json_filename) between mesh_tick-driven runs that
    # share the same `experiment` but a different `run_id` -- without it
    # they'd silently overwrite each other's result file.
    submitted_cmd = mock_run.call_args_list[0][0][0]
    assert "+run_tag=RUN1" in submitted_cmd

    assert state.run("RUN1").status == "RUNNING"
    assert "RUNNING" in action


def test_clean_stale_run_artifacts_removes_matching_flags_and_pending(tmp_path):
    mesh_shared = tmp_path / "mesh_shared"
    flags_dir = mesh_shared / "checkpoints" / "flags"
    pending_dir = mesh_shared / "checkpoints" / "pending"
    flags_dir.mkdir(parents=True)
    pending_dir.mkdir(parents=True)
    (flags_dir / "sync_complete_pool-fish-5r_0.flag").touch()
    (flags_dir / "sync_complete_pool-fish-5r_1.flag").touch()
    (pending_dir / "pool-fish-5r_node001.pt").touch()

    deleted = _clean_stale_run_artifacts("pool-fish-5r", mesh_shared)

    assert not (flags_dir / "sync_complete_pool-fish-5r_0.flag").exists()
    assert not (flags_dir / "sync_complete_pool-fish-5r_1.flag").exists()
    assert not (pending_dir / "pool-fish-5r_node001.pt").exists()
    assert len(deleted) == 3


def test_clean_stale_run_artifacts_does_not_touch_other_run_ids(tmp_path):
    mesh_shared = tmp_path / "mesh_shared"
    flags_dir = mesh_shared / "checkpoints" / "flags"
    flags_dir.mkdir(parents=True)
    # Hyphen-prefix collision case: cleaning "pool-fish" must not delete
    # "pool-fish-5r"'s flag -- this is the exact shape of names this repo uses.
    (flags_dir / "sync_complete_pool-fish-5r_0.flag").touch()
    (flags_dir / "sync_complete_pool-fish_0.flag").touch()

    deleted = _clean_stale_run_artifacts("pool-fish", mesh_shared)

    assert (flags_dir / "sync_complete_pool-fish-5r_0.flag").exists()
    assert not (flags_dir / "sync_complete_pool-fish_0.flag").exists()
    assert deleted == [flags_dir / "sync_complete_pool-fish_0.flag"]


def test_clean_stale_run_artifacts_does_not_touch_underscore_suffixed_sibling(
    tmp_path,
):
    mesh_shared = tmp_path / "mesh_shared"
    flags_dir = mesh_shared / "checkpoints" / "flags"
    flags_dir.mkdir(parents=True)
    # This is what the digit character class in the glob actually guards
    # against: a bare trailing `*` would match "pool_extra"'s flag too,
    # since "extra_0" satisfies `*` right after the literal "pool_"
    # delimiter. The hyphen-suffix pair above doesn't exercise this --
    # the literal "_" there already rejects the "-5r" sibling regardless
    # of what follows it in the pattern.
    (flags_dir / "sync_complete_pool_0.flag").touch()
    (flags_dir / "sync_complete_pool_extra_0.flag").touch()

    deleted = _clean_stale_run_artifacts("pool", mesh_shared)

    assert not (flags_dir / "sync_complete_pool_0.flag").exists()
    assert (flags_dir / "sync_complete_pool_extra_0.flag").exists()
    assert deleted == [flags_dir / "sync_complete_pool_0.flag"]


def test_clean_stale_run_artifacts_is_noop_when_nothing_stale(tmp_path):
    mesh_shared = tmp_path / "mesh_shared"  # flags/ and pending/ don't exist at all

    deleted = _clean_stale_run_artifacts("brand-new-run", mesh_shared)

    assert deleted == []


def test_running_transitions_on_first_pending_checkpoint(tmp_path):
    mesh_shared = tmp_path / "mesh_shared"
    pending_dir = mesh_shared / "checkpoints" / "pending"
    pending_dir.mkdir(parents=True)
    (pending_dir / "RUN1_0_node001.pt").write_text("x")

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
        (pending_dir / f"RUN1_0_{node}.pt").write_text("x")

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
        path = pending_dir / f"RUN1_0_{node}.pt"
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


def test_running_is_a_noop_with_zero_checkpoints_before_timeout(tmp_path):
    """No pending files at all yet is not itself a timeout condition for
    RUNNING -- it just means no one has finished the round. That's the
    _do_awaiting_checkpoints timeout's job once the round enters it."""
    mesh_shared = tmp_path / "mesh_shared"
    (mesh_shared / "checkpoints" / "pending").mkdir(parents=True)

    run_config = _run_config()
    state = WatcherState()
    run = state.run("RUN1")
    run.status = "RUNNING"
    run.round_started_at = 1000.0

    action = advance_one_run(
        run_config,
        state=state,
        repo_root=tmp_path,
        mesh_shared=mesh_shared,
        mesh_root=tmp_path,
        now=1500.0,  # 500s in, well under ROUND_TIMEOUT_S
    )

    assert state.run("RUN1").status == "RUNNING"
    assert action is None


def test_awaiting_checkpoints_times_out_when_zero_checkpoints_ever_arrive(tmp_path):
    """Regression test: a pool that dies before a single node checks in for
    the round used to make _round_elapsed_s return 0.0 forever (no pending
    files -> no mtimes to measure from), so ROUND_TIMEOUT_S could never
    trip and this state was a permanent stall. round_started_at gives it a
    real elapsed time to measure against instead."""
    mesh_shared = tmp_path / "mesh_shared"
    (mesh_shared / "checkpoints" / "pending").mkdir(
        parents=True
    )  # empty -- no one checked in

    run_config = _run_config()
    state = WatcherState()
    run = state.run("RUN1")
    run.status = "AWAITING_CHECKPOINTS"
    run.round_started_at = 1000.0

    action = advance_one_run(
        run_config,
        state=state,
        repo_root=tmp_path,
        mesh_shared=mesh_shared,
        mesh_root=tmp_path,
        now=1000.0 + 901.0,  # just past ROUND_TIMEOUT_S (900s)
    )

    assert state.run("RUN1").status == "MERGING"
    assert state.run("RUN1").timeout_forced is True
    for node in _NODES:
        assert state.run("RUN1").missed_rounds.get(node) == 1
    assert action is not None


def test_awaiting_checkpoints_ignores_stale_straggler_from_prior_round(tmp_path):
    """Regression: a timeout-forced merge clears pending/ (_do_merging) but a
    straggler still finishing the OLD round's training can push its
    checkpoint into pending/ AFTER the new round has already started -- it's
    a separate SLURM process, still mid-training, so its arrival is
    necessarily in real wall-clock time AFTER round_started_at, never
    before (_do_merging clears pending/ before stamping round_started_at
    for the new round, so an mtime-before-round-start straggler is not a
    scenario that can occur in production). mtime alone can't tell this
    straggler's round-0 checkpoint apart from a genuine round-1 arrival;
    only the round number carried in the filename (mesh/pending_files.py)
    can -- without it, the straggler's node name would silently complete
    the illusion of full round-1 arrival one node early, merging a stale
    checkpoint into what should be a fresh round's average."""
    mesh_shared = tmp_path / "mesh_shared"
    pending_dir = mesh_shared / "checkpoints" / "pending"
    pending_dir.mkdir(parents=True)
    round_started_at = 1740.0

    # node001's stale round-0 checkpoint, pushed shortly after round 1
    # officially started.
    stale = pending_dir / "RUN1_0_node001.pt"
    stale.write_text("x")
    os.utime(stale, (round_started_at + 10.0, round_started_at + 10.0))

    # node002 and node003 have genuinely submitted round 1; node001 hasn't.
    for node in ("node002", "node003"):
        path = pending_dir / f"RUN1_1_{node}.pt"
        path.write_text("x")
        os.utime(path, (round_started_at + 50.0, round_started_at + 50.0))

    run_config = _run_config()  # quorum_k=None: plain ROUND_TIMEOUT_S path
    state = WatcherState()
    state.run("RUN1").status = "AWAITING_CHECKPOINTS"
    state.run("RUN1").round = 1
    state.run("RUN1").round_started_at = round_started_at

    action = advance_one_run(
        run_config,
        state=state,
        repo_root=tmp_path,
        mesh_shared=mesh_shared,
        mesh_root=tmp_path,
        now=round_started_at + 150.0,  # nowhere near ROUND_TIMEOUT_S (900s)
    )

    # Only 2 of 3 nodes have genuinely checked in for round 1 -- node001's
    # stale round-0 leftover must not count, so this must not merge yet.
    assert state.run("RUN1").status == "AWAITING_CHECKPOINTS"
    assert action is None


def test_awaiting_checkpoints_merges_early_when_quorum_reached_before_full_arrival(
    tmp_path,
):
    """2.1: a run configured with quorum_k merges as soon as quorum_k nodes
    check in, without waiting for the rest of the pool or ROUND_TIMEOUT_S --
    the whole point of killing the barrier. The still-missing node must not
    be excluded or counted as a missed round: it just hasn't finished yet,
    not crashed."""
    mesh_shared = tmp_path / "mesh_shared"
    pending_dir = mesh_shared / "checkpoints" / "pending"
    pending_dir.mkdir(parents=True)
    for node in ("node001", "node002"):  # 2 of 3 -- node003 still training
        (pending_dir / f"RUN1_0_{node}.pt").write_text("x")

    run_config = _run_config(quorum_k=2, grace_s=900.0)
    state = WatcherState()
    state.run("RUN1").status = "AWAITING_CHECKPOINTS"
    state.run("RUN1").round_started_at = 1000.0

    action = advance_one_run(
        run_config,
        state=state,
        repo_root=tmp_path,
        mesh_shared=mesh_shared,
        mesh_root=tmp_path,
        now=1010.0,  # only 10s in -- nowhere near grace_s or ROUND_TIMEOUT_S
    )

    assert state.run("RUN1").status == "MERGING"
    assert "node003" not in state.run("RUN1").missed_rounds
    assert "node003" not in state.excluded_nodes
    assert action is not None
    assert "quorum" in action


def test_awaiting_checkpoints_quarantined_stale_checkpoint_does_not_satisfy_quorum(
    tmp_path,
):
    """Task 11 (design §5.1b): a quarantined node can still be training
    under self-requeue's enforcement window and drop a stale checkpoint --
    that file must not count toward quorum_k."""
    mesh_shared = tmp_path / "mesh_shared"
    pending_dir = mesh_shared / "checkpoints" / "pending"
    pending_dir.mkdir(parents=True)
    (pending_dir / "RUN1_0_node001.pt").write_text("x")  # 1 genuinely eligible
    (pending_dir / "RUN1_0_node003.pt").write_text("x")  # quarantined, stale

    run_config = _run_config(quorum_k=2, grace_s=900.0)
    state = WatcherState()
    state.run("RUN1").status = "AWAITING_CHECKPOINTS"
    state.run("RUN1").round_started_at = 1000.0
    exclude_node(state, "node003", "crash-loop: missed 2 consecutive rounds of RUN1")

    action = advance_one_run(
        run_config,
        state=state,
        repo_root=tmp_path,
        mesh_shared=mesh_shared,
        mesh_root=tmp_path,
        now=1010.0,  # nowhere near grace_s
    )

    # Without the eligible-only fix, {"node001", "node003"} would meet
    # quorum_k=2 on a naive count of pending files.
    assert state.run("RUN1").status == "AWAITING_CHECKPOINTS"
    assert action is None


def test_awaiting_checkpoints_quorum_configured_still_waits_below_quorum(tmp_path):
    """Below quorum_k, a quorum-configured run waits exactly like the
    default path does below ROUND_TIMEOUT_S -- grace_s hasn't expired yet
    either, so there's nothing to merge with."""
    mesh_shared = tmp_path / "mesh_shared"
    pending_dir = mesh_shared / "checkpoints" / "pending"
    pending_dir.mkdir(parents=True)
    (pending_dir / "RUN1_0_node001.pt").write_text("x")  # only 1 of 3

    run_config = _run_config(quorum_k=2, grace_s=900.0)
    state = WatcherState()
    state.run("RUN1").status = "AWAITING_CHECKPOINTS"
    state.run("RUN1").round_started_at = 1000.0

    action = advance_one_run(
        run_config,
        state=state,
        repo_root=tmp_path,
        mesh_shared=mesh_shared,
        mesh_root=tmp_path,
        now=1010.0,
    )

    assert state.run("RUN1").status == "AWAITING_CHECKPOINTS"
    assert action is None


def test_awaiting_checkpoints_quorum_configured_falls_back_to_exclusion_after_grace(
    tmp_path,
):
    """Quorum never reached and grace_s has fully elapsed: runs the exact same
    exclusion/replenish bookkeeping the plain timeout uses -- grace_s plays
    ROUND_TIMEOUT_S's old role once quorum is configured -- and THEN halts.

    Only one node of the two quorum_k demands ever checked in, so merging
    here would average below the configured floor. Bookkeeping still runs
    first (missed_rounds is recorded, a replacement is submitted) so the log
    and state show a human what happened; the run just does not advance."""
    mesh_shared = tmp_path / "mesh_shared"
    pending_dir = mesh_shared / "checkpoints" / "pending"
    pending_dir.mkdir(parents=True)
    old_mtime = 1000.0
    path = pending_dir / "RUN1_0_node001.pt"
    path.write_text("x")
    os.utime(path, (old_mtime, old_mtime))

    run_config = _run_config(quorum_k=2, grace_s=900.0)
    state = WatcherState()
    state.run("RUN1").status = "AWAITING_CHECKPOINTS"
    state.run("RUN1").round_started_at = old_mtime

    action = advance_one_run(
        run_config,
        state=state,
        repo_root=tmp_path,
        mesh_shared=mesh_shared,
        mesh_root=tmp_path,
        now=old_mtime + 901.0,  # past grace_s (900s)
    )

    assert state.run("RUN1").status == "DEGRADED"
    for node in ("node002", "node003"):
        assert state.run("RUN1").missed_rounds.get(node) == 1
    assert action is not None and "below quorum_k=2" in action


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
    # uv must be an ABSOLUTE path, not the bare name: the watcher ticks from
    # cron, whose PATH (/usr/bin:/bin) excludes ~/.local/bin, so a bare "uv"
    # raises FileNotFoundError on every tick and stalls the merge forever.
    assert Path(cmd[0]).is_absolute()
    assert Path(cmd[0]).name == "uv"
    # --no-sync: this cron runs once a minute and `uv sync` reinstalls into
    # the 9.4 GB NFS-hosted venv unconditionally, even on an up-to-date lock.
    assert cmd[1:6] == [
        "run",
        "--no-sync",
        "python",
        "-m",
        "video_encoding.scripts.run_outer_step",
    ]
    # pending/ is shared by every concurrent run -- an unscoped merge averages
    # sibling runs and finished runs' leftovers into this run's master.
    assert cmd[cmd.index("--run-id") + 1] == "RUN1"
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


def test_merging_passes_eligible_nodes_excluding_quarantined(tmp_path):
    """Task 11 (design §5.1b): a quarantined node's stale checkpoint must
    not be averaged into the merge -- run_outer_step.py drops anything not
    named in --eligible-nodes."""
    mesh_shared = tmp_path / "mesh_shared"
    (mesh_shared / "checkpoints" / "pending").mkdir(parents=True)

    run_config = _run_config()
    state = WatcherState()
    state.run("RUN1").status = "MERGING"
    exclude_node(state, "node003", "crash-loop: missed 2 consecutive rounds of RUN1")

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
    idx = cmd.index("--eligible-nodes")
    assert cmd[idx + 1] == "node001,node002"
    assert "node003" not in cmd[idx + 1]


def test_merging_does_not_pass_tb_dir(tmp_path):
    """The merge subprocess must not write TensorBoard scalars.

    run_outer_step's _write_tb_scalars rescans every node's entire metrics
    JSONL on every merge, so its cost grows with round count against a fixed
    300s subprocess timeout. On 2026-08-28 it overran that budget ~90s after
    the merge had already durably completed on two runs; the resulting
    SIGKILL is not catchable by its own except clause, and the watcher threw
    away 722 finished merges. flush_final_tb_rounds at DONE still writes the
    scalars once per run (see the mock_flush test below)."""
    mesh_shared = tmp_path / "mesh_shared"
    (mesh_shared / "checkpoints" / "pending").mkdir(parents=True)

    run_config = _run_config(outer_lr=0.7, outer_momentum=0.9, weighting="data")
    state = WatcherState()
    state.run("RUN1").status = "MERGING"

    fake = MagicMock(returncode=0, stdout="", stderr="")
    with patch(
        "video_encoding.watcher.mesh_tick.subprocess.run", return_value=fake
    ) as m:
        advance_one_run(
            run_config,
            state=state,
            repo_root=tmp_path,
            mesh_shared=mesh_shared,
            mesh_root=tmp_path,
            now=1000.0,
        )

    cmd = m.call_args[0][0]
    assert "--tb-dir" not in cmd


def test_merging_passes_metrics_out_pointing_at_results_mesh_run_id(tmp_path):
    """0c: without --metrics-out, per-round merge data (contributor set,
    alpha weights) exists only in this subprocess's stdout -- unworkable
    across an 8-run sweep."""
    mesh_shared = tmp_path / "mesh_shared"
    (mesh_shared / "checkpoints" / "pending").mkdir(parents=True)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    run_config = _run_config()
    state = WatcherState()
    state.run("RUN1").status = "MERGING"

    fake = MagicMock(returncode=0, stdout="", stderr="")
    with patch(
        "video_encoding.watcher.mesh_tick.subprocess.run", return_value=fake
    ) as mock_run:
        advance_one_run(
            run_config,
            state=state,
            repo_root=repo_root,
            mesh_shared=mesh_shared,
            mesh_root=tmp_path,
            now=1000.0,
        )

    cmd = mock_run.call_args[0][0]
    assert "--metrics-out" in cmd
    metrics_out_idx = cmd.index("--metrics-out")
    assert cmd[metrics_out_idx + 1] == str(
        repo_root / "runs" / "results" / "mesh" / "RUN1_rounds.jsonl"
    )


def test_merging_passes_timeout_forced_when_run_state_flag_is_set(tmp_path):
    mesh_shared = tmp_path / "mesh_shared"
    (mesh_shared / "checkpoints" / "pending").mkdir(parents=True)

    run_config = _run_config()
    state = WatcherState()
    state.run("RUN1").status = "MERGING"
    state.run("RUN1").timeout_forced = True

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

    assert "--timeout-forced" in mock_run.call_args[0][0]


def test_merging_passes_observability_counters(tmp_path):
    """Task 15 (design §8): _do_merging reads the counters
    _do_awaiting_checkpoints stashed on RunState for this round."""
    mesh_shared = tmp_path / "mesh_shared"
    (mesh_shared / "checkpoints" / "pending").mkdir(parents=True)

    run_config = _run_config()
    state = WatcherState()
    run = state.run("RUN1")
    run.status = "MERGING"
    run.round_misses_charged = 2
    run.round_exclusions_suspended = 1
    run.round_effective_quorum_k = 2

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
    assert cmd[cmd.index("--misses-charged") + 1] == "2"
    assert cmd[cmd.index("--exclusions-suspended-no-replacement") + 1] == "1"
    assert cmd[cmd.index("--effective-quorum-k") + 1] == "2"


def test_merging_omits_effective_quorum_k_when_none(tmp_path):
    mesh_shared = tmp_path / "mesh_shared"
    (mesh_shared / "checkpoints" / "pending").mkdir(parents=True)

    run_config = _run_config()  # quorum_k not configured for this run
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

    assert "--effective-quorum-k" not in mock_run.call_args[0][0]


def test_merging_omits_timeout_forced_on_the_normal_full_arrival_path(tmp_path):
    mesh_shared = tmp_path / "mesh_shared"
    (mesh_shared / "checkpoints" / "pending").mkdir(parents=True)

    run_config = _run_config()
    state = WatcherState()
    state.run("RUN1").status = "MERGING"
    assert state.run("RUN1").timeout_forced is False  # dataclass default

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

    assert "--timeout-forced" not in mock_run.call_args[0][0]


def test_awaiting_checkpoints_timeout_branch_sets_timeout_forced_flag(tmp_path):
    """The forced-merge branch (partial quorum, ROUND_TIMEOUT_S elapsed) must
    mark the round so _do_merging can tell run_outer_step.py this merge is
    not a full check-in."""
    mesh_shared = tmp_path / "mesh_shared"
    pending_dir = mesh_shared / "checkpoints" / "pending"
    pending_dir.mkdir(parents=True)
    old_mtime = 1000.0
    for node in ("node001", "node002"):
        path = pending_dir / f"RUN1_0_{node}.pt"
        path.write_text("x")
        os.utime(path, (old_mtime, old_mtime))

    run_config = _run_config()
    state = WatcherState()
    state.run("RUN1").status = "AWAITING_CHECKPOINTS"

    advance_one_run(
        run_config,
        state=state,
        repo_root=tmp_path,
        mesh_shared=mesh_shared,
        mesh_root=tmp_path,
        now=old_mtime + 4000.0,  # well past ROUND_TIMEOUT_S (3600s)
    )

    assert state.run("RUN1").status == "MERGING"
    assert state.run("RUN1").timeout_forced is True


def test_awaiting_checkpoints_full_arrival_clears_a_stale_timeout_forced_flag(
    tmp_path,
):
    """A run that timeout-forced round N must not leak that flag into round
    N+1's merge if round N+1 arrives in full."""
    mesh_shared = tmp_path / "mesh_shared"
    pending_dir = mesh_shared / "checkpoints" / "pending"
    pending_dir.mkdir(parents=True)
    for node in _NODES:
        (pending_dir / f"RUN1_0_{node}.pt").write_text("x")

    run_config = _run_config()
    state = WatcherState()
    state.run("RUN1").status = "AWAITING_CHECKPOINTS"
    state.run("RUN1").timeout_forced = True  # left over from a prior round

    advance_one_run(
        run_config,
        state=state,
        repo_root=tmp_path,
        mesh_shared=mesh_shared,
        mesh_root=tmp_path,
        now=1000.0,
    )

    assert state.run("RUN1").status == "MERGING"
    assert state.run("RUN1").timeout_forced is False


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
        _write_pending_checkpoint(pending_dir / f"RUN1_0_{node}.pt")

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
    # Phase 3.2: BOOTSTRAP_MASTER_VERSION predates every real round_num
    # merge_round ever stamps, so every worker's first real diff (against a
    # genuine merge) is never rejected as stale by comparison to a bootstrap
    # master -- and being negative it cannot be mistaken for round 0's own
    # completed merge by merge_round's already-merged recovery.
    assert payload["master_version"] == BOOTSTRAP_MASTER_VERSION
    assert BOOTSTRAP_MASTER_VERSION < 0


def test_merging_bootstraps_master_upcasts_fp16_pending_to_fp32(tmp_path):
    """Pending checkpoints arrive in fp16 (mesh_round.py's
    push_checkpoint_and_wait casts model_state_dict before the NFS push) --
    _bootstrap_master must upcast before seeding master.pt, or every later
    merge step (run_outer_step.py) accumulates the outer-optimizer momentum
    at half precision forever."""
    mesh_shared = tmp_path / "mesh_shared"
    pending_dir = mesh_shared / "checkpoints" / "pending"
    pending_dir.mkdir(parents=True)
    torch.save(
        {"model_state_dict": {"w": torch.ones(2).half()}, "n_samples": 100},
        pending_dir / "RUN1_0_node001.pt",
    )

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

    payload = torch.load(master_path, map_location="cpu", weights_only=False)
    assert payload["model_state_dict"]["w"].dtype == torch.float32


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


def test_syncing_done_warns_when_any_round_relaxed_quorum(tmp_path):
    """Task 12 (design §5.2): a run that merged under a relaxed quorum even
    once must not read as a clean DONE."""
    mesh_shared = tmp_path / "mesh_shared"
    run_config = _run_config(rounds=3)
    state = WatcherState()
    state.run("RUN1").status = "SYNCING"
    state.run("RUN1").round = 2
    state.run("RUN1").any_round_quorum_relaxed = True

    action = advance_one_run(
        run_config,
        state=state,
        repo_root=tmp_path,
        mesh_shared=mesh_shared,
        mesh_root=tmp_path,
        now=1000.0,
    )

    assert state.run("RUN1").status == "DONE"
    assert "WARNING" in action
    assert "relaxed quorum" in action


def test_syncing_flushes_final_tb_rounds_on_done(tmp_path):
    """No merge cycle follows DONE, so the round(s) write_mesh_round_scalars's
    straggler holdback left pending on the live run must be flushed here --
    otherwise a finished run's TB curve is permanently short its last round(s)."""
    mesh_shared = tmp_path / "mesh_shared"
    run_config = _run_config(rounds=3)
    state = WatcherState()
    state.run("RUN1").status = "SYNCING"
    state.run("RUN1").round = 2  # entering the final round's sync

    with patch("video_encoding.watcher.mesh_tick.flush_final_tb_rounds") as mock_flush:
        advance_one_run(
            run_config,
            state=state,
            repo_root=tmp_path,
            mesh_shared=mesh_shared,
            mesh_root=tmp_path,
            now=1000.0,
        )

    expected_tb_dir = Path(os.environ["MESH_LOCAL_ROOT"]) / "RUN1" / "tb"
    mock_flush.assert_called_once_with(expected_tb_dir, mesh_shared, "RUN1")
    assert state.run("RUN1").status == "DONE"


def test_syncing_transfers_results_to_node089_on_done(tmp_path):
    mesh_shared = tmp_path / "mesh_shared"
    run_config = _run_config(rounds=3)
    local_dir = mesh_shared / "checkpoints" / "RUN1"
    local_dir.mkdir(parents=True)
    (local_dir / "master.pt").write_bytes(b"fake checkpoint")

    state = WatcherState()
    state.run("RUN1").status = "SYNCING"
    state.run("RUN1").round = 2  # entering the final round's sync

    ok = MagicMock(returncode=0, stdout="", stderr="")

    with patch(
        "video_encoding.watcher.mesh_tick.subprocess.run",
        side_effect=lambda *a, **k: ok,
    ) as mock_run:
        action = advance_one_run(
            run_config,
            state=state,
            repo_root=tmp_path,
            mesh_shared=mesh_shared,
            mesh_root=tmp_path,
            now=1000.0,
        )

    assert not local_dir.exists()
    assert "transferred" in action
    assert state.run("RUN1").status == "DONE"

    commands = [c[0][0] for c in mock_run.call_args_list]
    assert [
        "ssh",
        "node089",
        "mkdir",
        "-p",
        "/Data/$USER/stern_mesh_results/RUN1",
    ] in commands
    assert [
        "rsync",
        "-az",
        "--checksum",
        f"{local_dir}/",
        "node089:/Data/$USER/stern_mesh_results/RUN1/",
    ] in commands
    assert [
        "ssh",
        "node089",
        "test",
        "-s",
        "/Data/$USER/stern_mesh_results/RUN1/master.pt",
    ] in commands


def test_syncing_done_keeps_local_checkpoints_if_transfer_fails(tmp_path):
    mesh_shared = tmp_path / "mesh_shared"
    run_config = _run_config(rounds=3)
    local_dir = mesh_shared / "checkpoints" / "RUN1"
    local_dir.mkdir(parents=True)
    (local_dir / "master.pt").write_bytes(b"fake checkpoint")

    state = WatcherState()
    state.run("RUN1").status = "SYNCING"
    state.run("RUN1").round = 2

    ok = MagicMock(returncode=0, stdout="", stderr="")
    rsync_fail = MagicMock(returncode=1, stdout="", stderr="disk full")

    def _fake_run(cmd, *args, **kwargs):
        return rsync_fail if cmd[0] == "rsync" else ok

    with patch(
        "video_encoding.watcher.mesh_tick.subprocess.run", side_effect=_fake_run
    ):
        action = advance_one_run(
            run_config,
            state=state,
            repo_root=tmp_path,
            mesh_shared=mesh_shared,
            mesh_root=tmp_path,
            now=1000.0,
        )

    assert local_dir.exists()
    assert "FAILED" in action
    assert state.run("RUN1").status == "DONE"


def test_syncing_done_keeps_local_checkpoints_if_post_transfer_check_fails(tmp_path):
    mesh_shared = tmp_path / "mesh_shared"
    run_config = _run_config(rounds=3)
    local_dir = mesh_shared / "checkpoints" / "RUN1"
    local_dir.mkdir(parents=True)
    (local_dir / "master.pt").write_bytes(b"fake checkpoint")

    state = WatcherState()
    state.run("RUN1").status = "SYNCING"
    state.run("RUN1").round = 2

    ok = MagicMock(returncode=0, stdout="", stderr="")
    check_fail = MagicMock(returncode=1, stdout="", stderr="")

    def _fake_run(cmd, *args, **kwargs):
        return check_fail if cmd[0] == "ssh" and cmd[2] == "test" else ok

    with patch(
        "video_encoding.watcher.mesh_tick.subprocess.run", side_effect=_fake_run
    ):
        action = advance_one_run(
            run_config,
            state=state,
            repo_root=tmp_path,
            mesh_shared=mesh_shared,
            mesh_root=tmp_path,
            now=1000.0,
        )

    assert local_dir.exists()
    assert "FAILED" in action


def test_syncing_transfer_skipped_when_no_local_checkpoints(tmp_path):
    mesh_shared = tmp_path / "mesh_shared"
    run_config = _run_config(rounds=3)

    state = WatcherState()
    state.run("RUN1").status = "SYNCING"
    state.run("RUN1").round = 2

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
    assert "skipped" in action
    assert state.run("RUN1").status == "DONE"


def test_advance_all_runs_skips_done_runs(tmp_path):
    mesh_shared = tmp_path / "mesh_shared"
    pending_dir = mesh_shared / "checkpoints" / "pending"
    pending_dir.mkdir(parents=True)
    (pending_dir / "RUN2_0_node001.pt").write_text("x")

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


def test_advance_all_runs_cascades_through_multiple_ready_transitions(tmp_path):
    """Cron fires once/minute; without cascading, a run whose next several
    transitions are all already satisfied (sync flag ready to write,
    next round's checkpoints already landed) would still burn one tick per
    transition -- ~60-70s of pure dispatch latency each, observed in
    production watcher.log. Seeds a run one flag-write away from DONE with
    the (unrealistic in production, but valid here) next round's
    checkpoints already sitting in pending/, so a single advance_all_runs
    call must walk SYNCING -> RUNNING -> AWAITING_CHECKPOINTS -> MERGING ->
    SYNCING -> DONE without waiting for five separate ticks.
    """
    mesh_shared = tmp_path / "mesh_shared"
    pending_dir = mesh_shared / "checkpoints" / "pending"
    pending_dir.mkdir(parents=True)
    for node in _NODES:
        _write_pending_checkpoint(pending_dir / f"RUN1_1_{node}.pt")

    run_config = _run_config(rounds=2)
    state = WatcherState()
    state.run("RUN1").status = "SYNCING"
    state.run("RUN1").round = 0
    state.run("RUN1").target_pool_size = 3

    fake = MagicMock(returncode=0, stdout="", stderr="")
    with patch("video_encoding.watcher.mesh_tick.subprocess.run", return_value=fake):
        actions = advance_all_runs(
            [run_config],
            state=state,
            repo_root=tmp_path,
            mesh_shared=mesh_shared,
            mesh_root=tmp_path,
            now=1000.0,
        )

    assert state.run("RUN1").status == "DONE"
    assert len(actions) >= 5


def test_advance_all_runs_stops_at_first_blocking_state(tmp_path):
    mesh_shared = tmp_path / "mesh_shared"
    pending_dir = mesh_shared / "checkpoints" / "pending"
    pending_dir.mkdir(parents=True)
    for node in ("node001", "node002"):  # node003 never checked in
        _write_pending_checkpoint(pending_dir / f"RUN1_0_{node}.pt")

    run_config = _run_config()
    state = WatcherState()
    state.run("RUN1").status = "AWAITING_CHECKPOINTS"
    state.run("RUN1").target_pool_size = 3

    with patch("video_encoding.watcher.mesh_tick.subprocess.run") as mock_run:
        actions = advance_all_runs(
            [run_config],
            state=state,
            repo_root=tmp_path,
            mesh_shared=mesh_shared,
            mesh_root=tmp_path,
            now=1000.0,  # freshly-written files, nowhere near ROUND_TIMEOUT_S
        )

    mock_run.assert_not_called()
    assert actions == []
    assert state.run("RUN1").status == "AWAITING_CHECKPOINTS"


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


def test_submit_worker_job_overrides_compose_against_real_experiment_config(tmp_path):
    """Regression: production mesh workers crashed on launch with Hydra's
    "Could not append to config. An item is already at 'mesh.merge'" --
    _submit_worker_job used '+mesh.merge=...' ('+' = add a NEW key) for
    mesh.merge/outer_lr/outer_momentum/rounds, all of which
    conf/experiment/videomae_mesh_ssl.yaml already defines. Compose the
    real config with the exact override strings this function emits so this
    class of bug is caught before it reaches sbatch."""
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    run_config = _run_config()
    registry_path = tmp_path / "jobs_registry.json"
    fake = MagicMock(returncode=0, stdout="Submitted batch job 555\n")
    with patch(
        "video_encoding.watcher.mesh_tick.subprocess.run", return_value=fake
    ) as mock_run:
        _submit_worker_job(
            run_config, "node001", tmp_path, registry_path, "2026-01-01T00:00:00Z"
        )
    submitted_cmd = mock_run.call_args[0][0]
    hydra_overrides = [a for a in submitted_cmd if a.startswith("+")]

    conf_dir = Path(__file__).resolve().parents[1] / "conf"
    with initialize_config_dir(config_dir=str(conf_dir), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[f"experiment={run_config.experiment}", *hydra_overrides],
        )
    assert cfg.mesh.merge is True
    assert cfg.mesh.start_round == 0


def test_submit_worker_job_appends_extra_overrides(tmp_path):
    run_config = _run_config(extra_overrides=["train.accum_steps=128"])
    registry_path = tmp_path / "jobs_registry.json"
    fake = MagicMock(returncode=0, stdout="Submitted batch job 557\n")
    with patch(
        "video_encoding.watcher.mesh_tick.subprocess.run", return_value=fake
    ) as mock_run:
        _submit_worker_job(
            run_config, "node001", tmp_path, registry_path, "2026-01-01T00:00:00Z"
        )
    submitted_cmd = mock_run.call_args[0][0]
    assert "train.accum_steps=128" in submitted_cmd


def test_submit_worker_job_omits_extra_overrides_when_unset(tmp_path):
    run_config = _run_config()  # extra_overrides defaults to []
    registry_path = tmp_path / "jobs_registry.json"
    fake = MagicMock(returncode=0, stdout="Submitted batch job 558\n")
    with patch(
        "video_encoding.watcher.mesh_tick.subprocess.run", return_value=fake
    ) as mock_run:
        _submit_worker_job(
            run_config, "node001", tmp_path, registry_path, "2026-01-01T00:00:00Z"
        )
    submitted_cmd = mock_run.call_args[0][0]
    assert submitted_cmd[-1] == "+mesh.start_round=0"


def test_submit_worker_job_extra_override_composes_against_real_experiment_config(
    tmp_path,
):
    """Same regression class as
    test_submit_worker_job_overrides_compose_against_real_experiment_config
    (production crash: '+' on an already-existing key), but for the new
    extra_overrides path -- train.accum_steps has no '+' prefix in the
    override string, matching conf/train/ssl.yaml already defining it."""
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    run_config = _run_config(extra_overrides=["train.accum_steps=128"])
    registry_path = tmp_path / "jobs_registry.json"
    fake = MagicMock(returncode=0, stdout="Submitted batch job 559\n")
    with patch(
        "video_encoding.watcher.mesh_tick.subprocess.run", return_value=fake
    ) as mock_run:
        _submit_worker_job(
            run_config, "node001", tmp_path, registry_path, "2026-01-01T00:00:00Z"
        )
    submitted_cmd = mock_run.call_args[0][0]
    overrides = [
        a for a in submitted_cmd if a.startswith("+") or a.startswith("train.")
    ]

    conf_dir = Path(__file__).resolve().parents[1] / "conf"
    with initialize_config_dir(config_dir=str(conf_dir), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[f"experiment={run_config.experiment}", *overrides],
        )
    assert cfg.train.accum_steps == 128


def test_submit_worker_job_appends_call_site_extra_overrides(tmp_path):
    """The extra_overrides kwarg (Task 8: a per-call override, e.g. a
    halved-LR NaN retry) is separate from run_config.extra_overrides
    (a per-run override set at submission time) -- both must be appended."""
    run_config = _run_config()
    registry_path = tmp_path / "jobs_registry.json"
    fake = MagicMock(returncode=0, stdout="Submitted batch job 42\n")
    with patch(
        "video_encoding.watcher.mesh_tick.subprocess.run", return_value=fake
    ) as mock_run:
        job_id = _submit_worker_job(
            run_config,
            "node001",
            tmp_path,
            registry_path,
            "2026-01-01T00:00:00Z",
            extra_overrides=["train.lr_mult=0.5"],
        )
    assert job_id == "42"
    assert "train.lr_mult=0.5" in mock_run.call_args[0][0]


def test_resubmit_with_override_delegates_to_submit_worker_job(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "video_encoding.watcher.mesh_tick._submit_worker_job",
        lambda *a, **kw: "99",
    )
    run_config = _run_config()
    result = resubmit_with_override(
        run_config,
        "node001",
        tmp_path,
        tmp_path / "registry.json",
        "2026-01-01T00:00:00Z",
        start_round=3,
        override="train.lr_mult=0.5",
    )
    assert result == "99"


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

    # node004 (the idle replacement) also holds s3 -- a replacement can only
    # inherit shards it physically has, so it is the least-loaded real holder.
    _seed_placement(
        mesh_shared,
        {
            "node001": ["s1", "s3"],
            "node002": ["s2", "s3"],
            "node003": ["s3"],
            "node004": ["s3"],
        },
    )
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
    # ssh is the node-health probe (_replenish_pool -> query_node_health):
    # free GPU MiB then seeded shard count. A candidate must look genuinely
    # healthy here or the health gate correctly refuses to submit it.
    healthy_probe = MagicMock(returncode=0, stdout="24423\n26\n")

    def _fake_run(cmd, *args, **kwargs):
        if cmd[0] == "sinfo":
            return idle_sinfo
        if cmd[0] == "ssh":
            return healthy_probe
        return sbatch_result

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
        (tmp_path / "runs" / "state" / "jobs_registry.json").read_text()
    )
    assert any(r["nodelist"] == "node004" for r in registry)


def test_awaiting_checkpoints_merges_below_quorum_k_when_relaxed(tmp_path):
    """Task 12 (design §5.2): a genuinely unreplaceable missing node relaxes
    the quorum floor below quorum_k rather than halting the run DEGRADED --
    clamped no lower than 60% of target_pool_size."""
    mesh_shared = tmp_path / "mesh_shared"
    pending_dir = mesh_shared / "checkpoints" / "pending"
    pending_dir.mkdir(parents=True)
    old_mtime = 1000.0
    for node in ("node001", "node002", "node004"):
        path = pending_dir / f"RUN1_0_{node}.pt"
        path.write_text("x")
        os.utime(path, (old_mtime, old_mtime))

    nodes = ["node001", "node002", "node003", "node004", "node005"]
    run_config = _run_config(nodes=nodes, n_shards=len(nodes), quorum_k=4, grace_s=0.0)
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
                    "quorum_k": 4,
                    "grace_s": 0.0,
                }
            ]
        )
    )

    state = WatcherState()
    run = state.run("RUN1")
    run.status = "AWAITING_CHECKPOINTS"
    run.round_started_at = 1000.0
    run.target_pool_size = 5
    exclude_node(state, "node005", "crash-loop: missed 2 consecutive rounds of RUN1")
    run.missed_rounds["node003"] = 1  # this tick's miss is the 2nd

    no_idle_sinfo = MagicMock(returncode=0, stdout="")  # nothing idle anywhere

    with patch(
        "video_encoding.watcher.mesh_tick.subprocess.run", return_value=no_idle_sinfo
    ):
        action = advance_one_run(
            run_config,
            state=state,
            repo_root=tmp_path,
            mesh_shared=mesh_shared,
            mesh_root=tmp_path,
            now=1010.0,
            run_configs_path=run_configs_path,
        )

    assert state.run("RUN1").status == "MERGING"
    assert run.quorum_relaxed is True
    assert run.any_round_quorum_relaxed is True
    assert "node003" not in state.excluded_nodes  # suspended, not excluded
    assert action is not None

    on_disk = json.loads(run_configs_path.read_text())
    assert on_disk[0]["quorum_k"] == 4  # never persisted (design §5.2)


def test_redistribute_shards_runs_even_with_no_replacement(tmp_path):
    """Task 10: shard redistribution is unconditional -- a suspended
    exclusion (no idle replacement) must not leave a dead node's shards
    orphaned in shard_assignments."""
    mesh_shared = tmp_path / "mesh_shared"
    run_config = _run_config()
    _seed_placement(
        mesh_shared,
        {"node001": ["s1"], "node002": ["s2", "s3"], "node003": ["s3"]},
    )
    assignments_path = mesh_shared / "metadata" / "shard_assignments_RUN1.json"
    assignments_path.write_text(
        json.dumps({"node001": ["s1"], "node002": ["s2"], "node003": ["s3"]})
    )

    note = redistribute_shards(run_config, mesh_shared, "node003", picked=[])

    assert note is not None and "reassigned" in note
    assignments = json.loads(assignments_path.read_text())
    assert "node003" not in assignments or not assignments["node003"]


def test_exclusion_suspended_when_no_idle_replacement(tmp_path):
    """Task 10 (design §5.1): a tier with zero slack must not exclude a
    node that missed its round -- there is nowhere to send its shards, so
    exclusion is suspended (the miss still counts, so a later miss excludes
    it once a replacement exists).

    Slack is already exhausted here (node004 was excluded earlier and
    target_pool_size=4 with 3 nodes left eligible, deficit=1) -- a fresh
    exclusion of node003 on top of that would need a replacement, and
    sinfo reports nothing idle anywhere.
    """
    mesh_shared = tmp_path / "mesh_shared"
    pending_dir = mesh_shared / "checkpoints" / "pending"
    pending_dir.mkdir(parents=True)
    old_mtime = 1000.0
    for node in ("node001", "node002"):
        path = pending_dir / f"RUN1_0_{node}.pt"
        path.write_text("x")
        os.utime(path, (old_mtime, old_mtime))

    nodes = [*_NODES, "node004"]
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
    run = state.run("RUN1")
    run.status = "AWAITING_CHECKPOINTS"
    run.target_pool_size = 4
    exclude_node(state, "node004", "crash-loop: missed 2 consecutive rounds of RUN1")
    run.missed_rounds["node003"] = 1  # this tick's miss is the 2nd

    no_idle_sinfo = MagicMock(returncode=0, stdout="")  # nothing idle anywhere
    now = old_mtime + 4000.0  # past ROUND_TIMEOUT_S

    with patch(
        "video_encoding.watcher.mesh_tick.subprocess.run", return_value=no_idle_sinfo
    ):
        action = _do_awaiting_checkpoints(
            run_config, run, state, mesh_shared, now, tmp_path, run_configs_path
        )

    assert "node003" not in state.excluded_nodes  # suspended, not excluded
    assert run.missed_rounds["node003"] == 2  # counter still bumped
    assert "MERGING" in action


def test_replenish_pool_does_not_submit_a_replacement_with_no_shards_to_inherit(
    tmp_path,
):
    """0.4 (redefined): a picked replacement only ever receives shards via
    excluded_node's own reassign_failed handoff. If excluded_node has no
    entry in shard_assignments at all, submitting a replacement anyway
    leaves it with nothing to train on -- it crashes immediately on
    read_my_shards, gets excluded itself, and _replenish_pool repeats with
    the next idle node. Reproduced live on login-node 2026-08-05
    (pool-cars-2r, node066/node051 both submitted with zero shard_assignments
    entries)."""
    mesh_shared = tmp_path / "mesh_shared"
    run_config = _run_config()  # node001, node002, node003 -- fish tier
    state = WatcherState()
    run = state.run("RUN1")
    run.target_pool_size = 3
    exclude_node(state, "node003", "crash-loop")

    _seed_placement(mesh_shared, {"node001": ["s1"], "node002": ["s2"]})
    assignments_path = mesh_shared / "metadata" / "shard_assignments_RUN1.json"
    # node003 has NO entry at all -- e.g. it was itself a prior replacement
    # that fell into this same gap, or assign_locality_first never gave it
    # any shards.
    assignments_path.write_text(json.dumps({"node001": ["s1"], "node002": ["s2"]}))

    run_configs_path = tmp_path / "mesh_run_configs.json"
    run_configs_path.write_text(
        json.dumps(
            [
                {
                    "run_id": "RUN1",
                    "experiment": "videomae_mesh_ssl",
                    "nodes": list(_NODES),
                    "n_shards": 2,
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
    # node004 must clear the node-health gate, otherwise this test would pass
    # for the wrong reason -- an unhealthy candidate is dropped before the
    # no-shards-to-inherit guard under test is ever reached.
    healthy_probe = MagicMock(returncode=0, stdout="24423\n26\n")

    def _fake_run(cmd, *args, **kwargs):
        return healthy_probe if cmd[0] == "ssh" else idle_sinfo

    with patch(
        "video_encoding.watcher.mesh_tick.subprocess.run", side_effect=_fake_run
    ) as mock_run:
        result = _replenish_pool(
            run_config,
            run,
            state,
            mesh_shared,
            tmp_path,
            run_configs_path,
            "node003",
        )

    # sinfo (idle query) is allowed; sbatch (job submission) must never fire.
    assert not any(call.args[0][0] == "sbatch" for call in mock_run.call_args_list)
    grown = json.loads(run_configs_path.read_text())
    assert "node004" not in grown[0]["nodes"]
    assert not (tmp_path / "runs" / "state" / "jobs_registry.json").exists()
    assignments = json.loads(assignments_path.read_text())
    assert "node004" not in assignments
    assert result is not None  # still reports the skip, not silence


def test_replenish_pool_does_not_submit_a_replacement_for_an_empty_shard_list(
    tmp_path,
):
    """Same root cause, the other shape: excluded_node has an entry but it's
    already empty (a prior reassignment took its shards, or it started with
    none). read_my_shards treats an empty list as fatal too, so this must be
    gated the same as a missing key."""
    mesh_shared = tmp_path / "mesh_shared"
    run_config = _run_config()
    state = WatcherState()
    run = state.run("RUN1")
    run.target_pool_size = 3
    exclude_node(state, "node003", "crash-loop")

    _seed_placement(mesh_shared, {"node001": ["s1"], "node002": ["s2"]})
    assignments_path = mesh_shared / "metadata" / "shard_assignments_RUN1.json"
    assignments_path.write_text(
        json.dumps({"node001": ["s1"], "node002": ["s2"], "node003": []})
    )

    run_configs_path = tmp_path / "mesh_run_configs.json"
    run_configs_path.write_text(
        json.dumps(
            [
                {
                    "run_id": "RUN1",
                    "experiment": "videomae_mesh_ssl",
                    "nodes": list(_NODES),
                    "n_shards": 2,
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
    # node004 must clear the node-health gate, otherwise this test would pass
    # for the wrong reason -- an unhealthy candidate is dropped before the
    # no-shards-to-inherit guard under test is ever reached.
    healthy_probe = MagicMock(returncode=0, stdout="24423\n26\n")

    def _fake_run(cmd, *args, **kwargs):
        return healthy_probe if cmd[0] == "ssh" else idle_sinfo

    with patch(
        "video_encoding.watcher.mesh_tick.subprocess.run", side_effect=_fake_run
    ) as mock_run:
        _replenish_pool(
            run_config,
            run,
            state,
            mesh_shared,
            tmp_path,
            run_configs_path,
            "node003",
        )

    assert not any(call.args[0][0] == "sbatch" for call in mock_run.call_args_list)
    grown = json.loads(run_configs_path.read_text())
    assert "node004" not in grown[0]["nodes"]


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
    # ssh is the node-health probe (_replenish_pool -> query_node_health):
    # free GPU MiB then seeded shard count. A candidate must look genuinely
    # healthy here or the health gate correctly refuses to submit it.
    healthy_probe = MagicMock(returncode=0, stdout="24423\n26\n")

    def _fake_run(cmd, *args, **kwargs):
        if cmd[0] == "sinfo":
            return idle_sinfo
        if cmd[0] == "ssh":
            return healthy_probe
        return sbatch_result

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


def test_replenish_pool_reassigns_shards_when_no_idle_nodes_available(tmp_path):
    """Regression: assign_locality_first only runs once per run (at
    ASSIGNING), so shard_assignments_RUN1.json is never regenerated later.
    Previously, when select_idle_nodes found zero idle replacements,
    _replenish_pool returned early *before* reaching the reassign_failed
    call -- excluded_node's shards stayed assigned to a node that will
    never train again, silently shrinking shard coverage for the rest of
    the run. Reassignment to survivors must happen regardless of whether a
    replacement node was found.
    """
    mesh_shared = tmp_path / "mesh_shared"
    run_config = _run_config()  # node001, node002, node003 -- fish tier
    state = WatcherState()
    run = state.run("RUN1")
    run.target_pool_size = 3
    exclude_node(state, "node003", "crash-loop")

    # node001 holds a replica of s3, so the handoff has a real destination.
    _seed_placement(
        mesh_shared,
        {"node001": ["s1", "s3"], "node002": ["s2"], "node003": ["s3"]},
    )
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

    no_idle_sinfo = MagicMock(returncode=0, stdout="")  # nothing idle

    with patch(
        "video_encoding.watcher.mesh_tick.subprocess.run", return_value=no_idle_sinfo
    ):
        _replenish_pool(
            run_config,
            run,
            state,
            mesh_shared,
            tmp_path,
            run_configs_path,
            "node003",
        )

    assignments = json.loads(assignments_path.read_text())
    assert "node003" not in assignments
    assert "s3" in assignments["node001"] + assignments["node002"]


def test_replenish_pool_reassigns_shards_when_deficit_zero(tmp_path):
    """Same root cause as the sibling test above, via the other early-return
    path: when the roster already meets target_pool_size even after this
    exclusion (e.g. a prior replenishment already grew it), _replenish_pool
    returned None before ever reaching reassign_failed -- orphaning
    excluded_node's shards despite there being no actual pool deficit to
    excuse it.
    """
    mesh_shared = tmp_path / "mesh_shared"
    run_config = _run_config(nodes=_NODES + ["node004"])  # 4 nodes
    state = WatcherState()
    run = state.run("RUN1")
    run.target_pool_size = 3  # 4 - 1 excluded = 3, meets target -> deficit 0
    exclude_node(state, "node003", "crash-loop")

    # node001 holds a replica of s3, so the handoff has a real destination.
    _seed_placement(
        mesh_shared,
        {"node001": ["s1", "s3"], "node002": ["s2"], "node003": ["s3"]},
    )
    assignments_path = mesh_shared / "metadata" / "shard_assignments_RUN1.json"
    assignments_path.write_text(
        json.dumps(
            {"node001": ["s1"], "node002": ["s2"], "node003": ["s3"], "node004": []}
        )
    )

    run_configs_path = tmp_path / "mesh_run_configs.json"
    run_configs_path.write_text(
        json.dumps(
            [
                {
                    "run_id": "RUN1",
                    "experiment": "videomae_mesh_ssl",
                    "nodes": _NODES + ["node004"],
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

    _replenish_pool(
        run_config,
        run,
        state,
        mesh_shared,
        tmp_path,
        run_configs_path,
        "node003",
    )

    assignments = json.loads(assignments_path.read_text())
    assert "node003" not in assignments
    assert (
        "s3" in assignments["node001"] + assignments["node002"] + assignments["node004"]
    )


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
        path = pending_dir / f"RUN1_0_{node}.pt"
        path.write_text("x")
        os.utime(path, (old_mtime, old_mtime))

    _seed_placement(
        mesh_shared, {"node001": ["s1", "s2"], "node002": [], "node003": ["s2"]}
    )
    # node003 must have a real shard to hand off -- an empty list would
    # correctly block replenishment (0.4's fix), which is not what this
    # test is exercising.
    (mesh_shared / "metadata" / "shard_assignments_RUN1.json").write_text(
        json.dumps({"node001": ["s1"], "node002": [], "node003": ["s2"]})
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
    # ssh is the node-health probe (_replenish_pool -> query_node_health):
    # free GPU MiB then seeded shard count. A candidate must look genuinely
    # healthy here or the health gate correctly refuses to submit it.
    healthy_probe = MagicMock(returncode=0, stdout="24423\n26\n")

    def _fake_run(cmd, *args, **kwargs):
        if cmd[0] == "sinfo":
            return idle_sinfo
        if cmd[0] == "ssh":
            return healthy_probe
        return sbatch_result

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
        path = pending_dir / f"RUN1_0_{node}.pt"
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

    # ssh is the node-health probe (_replenish_pool -> query_node_health):
    # free GPU MiB then seeded shard count. A candidate must look genuinely
    # healthy here or the health gate correctly refuses to submit it.
    healthy_probe = MagicMock(returncode=0, stdout="24423\n26\n")

    def _fake_run(cmd, *args, **kwargs):
        if cmd[0] == "sinfo":
            return idle_sinfo
        if cmd[0] == "ssh":
            return healthy_probe
        return sbatch_result

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


def test_momentum_and_residual_live_off_nfs(tmp_path):
    """Both buffers are read/written only by the watcher, so keeping them on
    NFS cost 698 MiB of round-trip per round per pool for data no worker
    ever opens."""
    mesh_shared = tmp_path / "mesh_shared"
    (mesh_shared / "checkpoints" / "pending").mkdir(parents=True)

    run_config = _run_config(outer_lr=0.7, outer_momentum=0.9, weighting="data")
    state = WatcherState()
    state.run("RUN1").status = "MERGING"

    fake = MagicMock(returncode=0, stdout="", stderr="")
    with patch(
        "video_encoding.watcher.mesh_tick.subprocess.run", return_value=fake
    ) as m:
        advance_one_run(
            run_config,
            state=state,
            repo_root=tmp_path,
            mesh_shared=mesh_shared,
            mesh_root=tmp_path,
            now=1000.0,
        )

    cmd = m.call_args[0][0]
    momentum = Path(cmd[cmd.index("--momentum") + 1])
    residual = Path(cmd[cmd.index("--residual") + 1])
    assert mesh_shared not in momentum.parents
    assert mesh_shared not in residual.parents
    assert momentum.parent == residual.parent
    # And the merge is told which round it is, so a vanished momentum buffer
    # past round 0 is fatal rather than a silent degrade to plain averaging.
    assert "--round" in cmd


def test_done_run_sweeps_its_leftover_pending_checkpoints(tmp_path):
    """A run whose final round timed out leaves late arrivals in pending/,
    and nothing else collects them: _clean_stale_run_artifacts only fires on
    run_id reuse. Observed in production as 15 stale checkpoints (2.6 GiB)
    sitting where the next run's merge would have read them."""
    mesh_shared = tmp_path / "mesh_shared"
    pending_dir = mesh_shared / "checkpoints" / "pending"
    pending_dir.mkdir(parents=True)
    (mesh_shared / "checkpoints" / "RUN1").mkdir(parents=True)

    for node in _NODES:
        _write_pending_checkpoint(pending_dir / f"RUN1_{node}.pt")
    _write_pending_checkpoint(pending_dir / "OTHER_a.pt")

    run_config = _run_config(
        outer_lr=0.7, outer_momentum=0.9, weighting="data", rounds=1
    )
    state = WatcherState()
    state.run("RUN1").status = "SYNCING"
    state.run("RUN1").round = 0

    with patch(
        "video_encoding.watcher.mesh_tick._transfer_results_to_node089",
        return_value="transfer skipped",
    ):
        advance_one_run(
            run_config,
            state=state,
            repo_root=tmp_path,
            mesh_shared=mesh_shared,
            mesh_root=tmp_path,
            now=1000.0,
        )

    assert state.run("RUN1").status == "DONE"
    assert list(pending_dir.glob("RUN1_*.pt")) == []
    # A sibling run's checkpoints must survive the sweep.
    assert (pending_dir / "OTHER_a.pt").exists()


def test_done_run_clears_only_its_own_crash_loop_exclusions(tmp_path):
    """0.5: SYNCING -> DONE must wipe this run's own crash-loop exclusions
    (the 'dead list') so a node that misbehaved on a finished run isn't
    permanently locked out of every future run -- but an exclusion tagged
    for a different run, or flagged by hand, must survive."""
    mesh_shared = tmp_path / "mesh_shared"
    (mesh_shared / "checkpoints" / "RUN1").mkdir(parents=True)

    run_config = _run_config(
        outer_lr=0.7, outer_momentum=0.9, weighting="data", rounds=1
    )
    state = WatcherState()
    state.run("RUN1").status = "SYNCING"
    state.run("RUN1").round = 0
    state.excluded_nodes["node003"] = "crash-loop: missed 2 consecutive rounds of RUN1"
    state.excluded_nodes["node004"] = "crash-loop: missed 2 consecutive rounds of RUN2"
    state.excluded_nodes["cigogne"] = "manually flagged: broken GPU"

    with patch(
        "video_encoding.watcher.mesh_tick._transfer_results_to_node089",
        return_value="transfer skipped",
    ):
        advance_one_run(
            run_config,
            state=state,
            repo_root=tmp_path,
            mesh_shared=mesh_shared,
            mesh_root=tmp_path,
            now=1000.0,
        )

    assert state.run("RUN1").status == "DONE"
    assert "node003" not in state.excluded_nodes
    assert "node004" in state.excluded_nodes
    assert "cigogne" in state.excluded_nodes
    # state.runs["RUN1"] must survive -- it's the only record protecting
    # this run's worker jobs from remediate_dead_job going forward.
    assert "RUN1" in state.runs


def test_done_run_stages_local_momentum_back_for_the_node089_archive(tmp_path):
    """The buffer moved off NFS for the per-round budget, but it was in every
    previous archive and is part of a reproducible result -- so it comes back
    exactly once, at DONE, before the transfer runs."""
    mesh_shared = tmp_path / "mesh_shared"
    (mesh_shared / "checkpoints" / "pending").mkdir(parents=True)
    local_root = Path(os.environ["MESH_LOCAL_ROOT"]) / "RUN1"
    local_root.mkdir(parents=True, exist_ok=True)
    torch.save({"momentum_buf": {"w": torch.ones(2)}}, local_root / "outer_momentum.pt")

    run_config = _run_config(
        outer_lr=0.7, outer_momentum=0.9, weighting="data", rounds=1
    )
    state = WatcherState()
    state.run("RUN1").status = "SYNCING"
    state.run("RUN1").round = 0

    seen: dict[str, bool] = {}

    def _fake_transfer(rc, ms):
        seen["momentum_present"] = (
            ms / "checkpoints" / rc.run_id / "outer_momentum.pt"
        ).exists()
        return "transfer skipped"

    with patch(
        "video_encoding.watcher.mesh_tick._transfer_results_to_node089", _fake_transfer
    ):
        advance_one_run(
            run_config,
            state=state,
            repo_root=tmp_path,
            mesh_shared=mesh_shared,
            mesh_root=tmp_path,
            now=1000.0,
        )

    assert seen["momentum_present"] is True


def test_watcher_cron_entry_uses_no_sync():
    """This cron fires once a minute; `uv sync --dry-run` reports an
    unconditional reinstall into the NFS-hosted venv even against an
    up-to-date lockfile, so a bare `uv run` here is ~1440 rewrites/day."""
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "launch_mesh_pool.sh"
    ).read_text()
    cron_lines = [
        ln for ln in script.splitlines() if "watch_training" in ln and "* * * * *" in ln
    ]
    assert cron_lines, "no watcher cron line found in launch_mesh_pool.sh"
    for line in cron_lines:
        assert "uv run --no-sync" in line


def test_sweep_orphaned_pending_deletes_nothing_with_empty_active_run_ids(tmp_path):
    """load_run_configs returns [] when its file is missing (e.g. a
    transient read failure) -- an empty active_run_ids must be treated as
    "unknown", not "nothing is active", or this would wipe every live
    checkpoint in pending/ on a bad tick."""
    pending_dir = tmp_path / "pending"
    pending_dir.mkdir()
    _write_pending_checkpoint(pending_dir / "RUN1_node001.pt")

    swept = sweep_orphaned_pending(pending_dir, set(), max_age_s=0.0)

    assert swept == 0
    assert (pending_dir / "RUN1_node001.pt").exists()


def test_sweep_orphaned_pending_deletes_old_files_from_unknown_runs(tmp_path):
    pending_dir = tmp_path / "pending"
    pending_dir.mkdir()
    orphan = pending_dir / "DEAD_node001.pt"
    _write_pending_checkpoint(orphan)
    old_time = os.path.getmtime(orphan) - 7200
    os.utime(orphan, (old_time, old_time))

    swept = sweep_orphaned_pending(pending_dir, {"RUN1"}, max_age_s=3600.0)

    assert swept == 1
    assert not orphan.exists()


def test_sweep_orphaned_pending_keeps_recent_files_from_unknown_runs(tmp_path):
    pending_dir = tmp_path / "pending"
    pending_dir.mkdir()
    _write_pending_checkpoint(pending_dir / "DEAD_node001.pt")

    swept = sweep_orphaned_pending(pending_dir, {"RUN1"}, max_age_s=3600.0)

    assert swept == 0
    assert (pending_dir / "DEAD_node001.pt").exists()


def test_sweep_orphaned_pending_never_touches_an_active_runs_files(tmp_path):
    pending_dir = tmp_path / "pending"
    pending_dir.mkdir()
    active = pending_dir / "RUN1_node001.pt"
    _write_pending_checkpoint(active)
    old_time = os.path.getmtime(active) - 7200
    os.utime(active, (old_time, old_time))

    swept = sweep_orphaned_pending(pending_dir, {"RUN1"}, max_age_s=3600.0)

    assert swept == 0
    assert active.exists()


def _seed_holdings(mesh_shared, holdings, shards=None) -> None:
    shards = shards or sorted({s for v in holdings.values() for s in v})
    metadata = mesh_shared / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / "verified_holdings.json").write_text(json.dumps({"fish": holdings}))
    (metadata / "shard_manifest.json").write_text(
        json.dumps({"train": {name: 100 for name in shards}, "val": {}})
    )


def test_assigning_never_gives_a_node_a_shard_it_does_not_hold(tmp_path):
    """The 2026-08-20 regression test at the coordinator level: node055 was
    assigned 19 shards it held none of, fetched nothing, and trained zero
    samples while reporting a warning."""
    mesh_shared = tmp_path / "mesh_shared"
    holdings = {
        "node001": ["shard-000000.tar", "shard-000001.tar"],
        "node002": ["shard-000002.tar"],
        "node003": ["shard-000003.tar"],
    }
    _seed_holdings(mesh_shared, holdings)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    fake = MagicMock(returncode=0, stdout="Submitted batch job 101\n")
    with patch("video_encoding.watcher.mesh_tick.subprocess.run", return_value=fake):
        advance_one_run(
            _run_config(),
            state=WatcherState(),
            repo_root=repo_root,
            mesh_shared=mesh_shared,
            mesh_root=tmp_path / "mesh_root",
            now=1000.0,
        )

    assignments = json.loads(
        (mesh_shared / "metadata" / "shard_assignments_RUN1.json").read_text()
    )
    for node, shards in assignments.items():
        assert set(shards) <= set(holdings[node])


def test_assigning_skips_a_node_that_holds_no_shards(tmp_path):
    """node017, node069 and five birds nodes hold nothing. Submitting a worker
    there burns a GPU at 0 samp/s and stalls the round waiting for a push
    that never comes."""
    mesh_shared = tmp_path / "mesh_shared"
    _seed_holdings(
        mesh_shared,
        {
            "node001": ["shard-000000.tar", "shard-000001.tar"],
            "node002": ["shard-000002.tar"],
            "node003": [],
        },
    )
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    fake = MagicMock(returncode=0, stdout="Submitted batch job 101\n")
    with patch(
        "video_encoding.watcher.mesh_tick.subprocess.run", return_value=fake
    ) as mock_run:
        action = advance_one_run(
            _run_config(),
            state=WatcherState(),
            repo_root=repo_root,
            mesh_shared=mesh_shared,
            mesh_root=tmp_path / "mesh_root",
            now=1000.0,
        )

    assert mock_run.call_count == 2, "a shardless node must not get a worker job"
    assignments = json.loads(
        (mesh_shared / "metadata" / "shard_assignments_RUN1.json").read_text()
    )
    assert "node003" not in assignments
    assert "node003" in action and "holding no shards" in action


def test_assigning_refuses_to_launch_onto_an_unvalidated_pool(tmp_path):
    """No verified_holdings.json means nobody has checked what is on disk.
    Inferring assignment from a placement plan is what went stale."""
    mesh_shared = tmp_path / "mesh_shared"
    (mesh_shared / "metadata").mkdir(parents=True)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    with pytest.raises(FileNotFoundError, match="validate_seeding"):
        advance_one_run(
            _run_config(),
            state=WatcherState(),
            repo_root=repo_root,
            mesh_shared=mesh_shared,
            mesh_root=tmp_path / "mesh_root",
            now=1000.0,
        )


def test_advance_all_runs_terminates_when_a_run_cannot_progress(tmp_path):
    """A handler that reports an action without advancing must not spin.

    advance_all_runs drives one run_id until a handler returns None or the
    run reaches DONE. _do_assigning with every node excluded returns an
    action string ("no eligible nodes") and deliberately leaves the status
    at ASSIGNING, so neither exit condition is ever met: on 2026-08-21 this
    span forever, appending one string per iteration, and grew the watcher
    to 61 GB. That exhausted node089' memory and left the host unreachable
    over ssh for ~40 minutes.
    """
    mesh_shared = tmp_path / "mesh_shared"
    (mesh_shared / "checkpoints" / "pending").mkdir(parents=True)

    config = _run_config(run_id="RUN1")
    state = WatcherState()
    # Every node in the roster is excluded -> eligible_nodes() returns [].
    for node in config.nodes:
        state.excluded_nodes[node] = "crash-loop: missed 2 consecutive rounds of OLD"
    state.run("RUN1").status = "ASSIGNING"

    actions = advance_all_runs(
        [config],
        state=state,
        repo_root=tmp_path,
        mesh_shared=mesh_shared,
        mesh_root=tmp_path,
        now=1000.0,
    )

    # Reported exactly once, then the loop gives up instead of spinning.
    assert len(actions) == 1, f"expected one action, got {len(actions)}"
    assert "no eligible nodes" in actions[0]
    assert state.run("RUN1").status == "ASSIGNING"  # still needs a human


def _probe_stdout(gpu_free_mib: int, shards: int) -> str:
    return f"{gpu_free_mib}\n{shards}\n"


def _replenish_fixture(tmp_path):
    """Shared setup for the node-health replenishment tests: RUN1 is one node
    short and node004 is the only same-tier candidate, holding s3."""
    mesh_shared = tmp_path / "mesh_shared"
    run_config = _run_config()
    state = WatcherState()
    run = state.run("RUN1")
    run.target_pool_size = 3
    run.round = 4
    exclude_node(state, "node003", "crash-loop: missed 2 consecutive rounds of RUN1")
    _seed_placement(
        mesh_shared,
        {
            "node001": ["s1", "s3"],
            "node002": ["s2", "s3"],
            "node003": ["s3"],
            "node004": ["s3"],
        },
    )
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
    return run_config, run, state, mesh_shared, run_configs_path


def _run_replenish(tmp_path, fixture, probe_stdout):
    run_config, run, state, mesh_shared, run_configs_path = fixture
    responses = {
        "sinfo": MagicMock(returncode=0, stdout="node004 idle\n"),
        "ssh": MagicMock(returncode=0, stdout=probe_stdout),
        "sbatch": MagicMock(returncode=0, stdout="Submitted batch job 999\n"),
    }

    def _fake_run(cmd, *args, **kwargs):
        return responses.get(cmd[0], responses["sbatch"])

    with patch(
        "video_encoding.watcher.mesh_tick.subprocess.run", side_effect=_fake_run
    ) as mock_run:
        result = _replenish_pool(
            run_config, run, state, mesh_shared, tmp_path, run_configs_path, "node003"
        )
    submitted = [c for c in mock_run.call_args_list if c[0][0][0] == "sbatch"]
    return result, submitted


def test_replenish_pool_skips_slurm_idle_node_whose_gpu_is_held(tmp_path):
    """The 2026-08-21 birds regression: sinfo said idle, nvidia-smi said a
    foreign process held 21.7 of 24.5 GiB, and the submitted worker died on
    torch.OutOfMemoryError before round 1."""
    fixture = _replenish_fixture(tmp_path)
    result, submitted = _run_replenish(tmp_path, fixture, _probe_stdout(2914, 26))

    assert submitted == []
    assert (
        "node004"
        not in json.loads((tmp_path / "mesh_run_configs.json").read_text())[0]["nodes"]
    )
    assert result is not None and "node004" not in result


def test_replenish_pool_skips_unseeded_node(tmp_path):
    """The other half of the same regression: a free GPU on a node that was
    never seeded, which died on FileNotFoundError for manifest.json."""
    fixture = _replenish_fixture(tmp_path)
    _, submitted = _run_replenish(tmp_path, fixture, _probe_stdout(24426, 0))
    assert submitted == []


def test_replenish_pool_skips_unreachable_node(tmp_path):
    fixture = _replenish_fixture(tmp_path)
    _, submitted = _run_replenish(tmp_path, fixture, "")
    assert submitted == []


def test_replenish_pool_submits_a_genuinely_healthy_node(tmp_path):
    fixture = _replenish_fixture(tmp_path)
    result, submitted = _run_replenish(tmp_path, fixture, _probe_stdout(24423, 26))

    assert len(submitted) == 1
    assert "node004" in result
    assert (
        "node004"
        in json.loads((tmp_path / "mesh_run_configs.json").read_text())[0]["nodes"]
    )


def test_replenish_pool_still_reassigns_shards_when_health_gate_rejects(tmp_path):
    """Rejecting a candidate must not regress the coverage guarantee: the
    excluded node's shards still move to a survivor that holds them."""
    fixture = _replenish_fixture(tmp_path)
    _, _ = _run_replenish(tmp_path, fixture, _probe_stdout(2914, 26))

    assignments = json.loads(
        (
            tmp_path / "mesh_shared" / "metadata" / "shard_assignments_RUN1.json"
        ).read_text()
    )
    assert "node003" not in assignments
    assert "s3" in assignments["node001"] + assignments["node002"]


def _orphan_fixture(tmp_path):
    """RUN1 loses node003, whose shard s3 NO survivor and no replacement
    holds -- the 1x-replication case that dropped 46 of birds' 260 shards."""
    mesh_shared = tmp_path / "mesh_shared"
    run_config = _run_config()
    state = WatcherState()
    run = state.run("RUN1")
    run.target_pool_size = 3
    run.round = 4
    exclude_node(state, "node003", "crash-loop: missed 2 consecutive rounds of RUN1")
    _seed_placement(
        mesh_shared,
        {
            "node001": ["s1"],
            "node002": ["s2"],
            "node003": ["s3"],  # node021 holder of s3
            "node004": [],  # healthy replacement, holds nothing
        },
    )
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
    return run_config, run, state, mesh_shared, assignments_path, run_configs_path


def test_replenish_pool_copies_orphaned_shards_to_the_replacement(tmp_path):
    (
        run_config,
        run,
        state,
        mesh_shared,
        assignments_path,
        run_configs_path,
    ) = _orphan_fixture(tmp_path)

    responses = {
        "sinfo": MagicMock(returncode=0, stdout="node004 idle\n"),
        "ssh": MagicMock(returncode=0, stdout="24423\n26\n"),
        "sbatch": MagicMock(returncode=0, stdout="Submitted batch job 999\n"),
    }

    def _fake_run(cmd, *args, **kwargs):
        return responses.get(cmd[0], responses["sbatch"])

    with (
        patch("video_encoding.watcher.mesh_tick.subprocess.run", side_effect=_fake_run),
        patch(
            "video_encoding.watcher.mesh_tick.copy_shards", return_value=["s3"]
        ) as mock_copy,
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

    mock_copy.assert_called_once_with("node003", "node004", ["s3"])
    assignments = json.loads(assignments_path.read_text())
    assert assignments["node004"] == ["s3"]
    assert "node003" not in assignments
    assert "recovered 1 shard(s)" in result
    assert "dropped" not in result


def test_replenish_pool_reports_a_drop_when_the_copy_fails(tmp_path):
    """A failed rsync must not leave node004 named for a shard it lacks --
    read_my_shards would crash the worker and exclude it two rounds later."""
    (
        run_config,
        run,
        state,
        mesh_shared,
        assignments_path,
        run_configs_path,
    ) = _orphan_fixture(tmp_path)

    responses = {
        "sinfo": MagicMock(returncode=0, stdout="node004 idle\n"),
        "ssh": MagicMock(returncode=0, stdout="24423\n26\n"),
        "sbatch": MagicMock(returncode=0, stdout="Submitted batch job 999\n"),
    }

    def _fake_run(cmd, *args, **kwargs):
        return responses.get(cmd[0], responses["sbatch"])

    with (
        patch("video_encoding.watcher.mesh_tick.subprocess.run", side_effect=_fake_run),
        patch("video_encoding.watcher.mesh_tick.copy_shards", return_value=[]),
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

    assignments = json.loads(assignments_path.read_text())
    assert assignments.get("node004") == []
    assert "dropped 1 shard(s)" in result


def test_replenish_pool_readmits_a_recovered_roster_node_and_submits_it(tmp_path):
    """node003 is excluded and no fresh tier node is idle, but node003's own
    GPU has freed up. It still holds its shards, so recovering it is strictly
    better than leaving the pool short."""
    mesh_shared = tmp_path / "mesh_shared"
    run_config = _run_config()
    state = WatcherState()
    run = state.run("RUN1")
    run.target_pool_size = 3
    run.round = 4
    run.missed_rounds["node003"] = 2
    exclude_node(state, "node003", "crash-loop: missed 2 consecutive rounds of RUN1")
    _seed_placement(
        mesh_shared,
        {"node001": ["s1"], "node002": ["s2"], "node003": ["s3"]},
    )
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

    # Only node003 is idle -- no fresh replacement exists on the tier.
    responses = {
        "sinfo": MagicMock(returncode=0, stdout="node003 idle\n"),
        "ssh": MagicMock(returncode=0, stdout="24423\n26\n"),
        "sbatch": MagicMock(returncode=0, stdout="Submitted batch job 999\n"),
    }

    def _fake_run(cmd, *args, **kwargs):
        return responses.get(cmd[0], responses["sbatch"])

    with patch(
        "video_encoding.watcher.mesh_tick.subprocess.run", side_effect=_fake_run
    ) as mock_run:
        _replenish_pool(
            run_config,
            run,
            state,
            mesh_shared,
            tmp_path,
            run_configs_path,
            "node003",
        )

    assert "node003" not in state.excluded_nodes
    assert state.run("RUN1").readmitted["node003"] == 1
    submitted = [c for c in mock_run.call_args_list if c[0][0][0] == "sbatch"]
    assert len(submitted) == 1


def test_replenish_pool_leaves_an_unhealthy_excluded_node_excluded(tmp_path):
    """The counterpart: node003 is idle to SLURM but its GPU is still held,
    so it must stay out rather than be re-admitted into `eligible` with no
    job running -- which would stall every subsequent round."""
    mesh_shared = tmp_path / "mesh_shared"
    run_config = _run_config()
    state = WatcherState()
    run = state.run("RUN1")
    run.target_pool_size = 3
    run.round = 4
    run.missed_rounds["node003"] = 2
    exclude_node(state, "node003", "crash-loop: missed 2 consecutive rounds of RUN1")
    _seed_placement(mesh_shared, {"node001": ["s1"], "node002": ["s2"]})
    (mesh_shared / "metadata" / "shard_assignments_RUN1.json").write_text(
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

    responses = {
        "sinfo": MagicMock(returncode=0, stdout="node003 idle\n"),
        # GPU held by a foreign process: 2443 MiB free of 24.5 GiB.
        "ssh": MagicMock(returncode=0, stdout="2443\n26\n"),
        "sbatch": MagicMock(returncode=0, stdout="Submitted batch job 999\n"),
    }

    def _fake_run(cmd, *args, **kwargs):
        return responses.get(cmd[0], responses["sbatch"])

    with patch(
        "video_encoding.watcher.mesh_tick.subprocess.run", side_effect=_fake_run
    ) as mock_run:
        _replenish_pool(
            run_config,
            run,
            state,
            mesh_shared,
            tmp_path,
            run_configs_path,
            "node003",
        )

    assert "node003" in state.excluded_nodes
    assert state.run("RUN1").readmitted == {}
    assert not any(c[0][0][0] == "sbatch" for c in mock_run.call_args_list)


# --- periodic pool recovery ---------------------------------------------


def _recovery_fixture(tmp_path, *, holdings=None, assignments=None):
    """RUN1 has settled into a degraded steady state: node003 was excluded
    long ago, nothing new is failing, so _replenish_pool is never called
    again. node003 is healthy and still holds s3, which nobody trains."""
    mesh_shared = tmp_path / "mesh_shared"
    run_config = _run_config()
    state = WatcherState()
    run = state.run("RUN1")
    run.target_pool_size = 3
    run.round = 7
    run.missed_rounds["node003"] = 2
    exclude_node(state, "node003", "crash-loop: missed 2 consecutive rounds of RUN1")
    _seed_placement(
        mesh_shared,
        holdings
        if holdings is not None
        else {"node001": ["s1"], "node002": ["s2"], "node003": ["s3"]},
    )
    assignments_path = mesh_shared / "metadata" / "shard_assignments_RUN1.json"
    assignments_path.write_text(
        json.dumps(
            assignments
            if assignments is not None
            else {"node001": ["s1"], "node002": ["s2"]}
        )
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
    return run_config, run, state, mesh_shared, assignments_path, run_configs_path


def _recovery_responses(probe_stdout="24423\n26\n"):
    responses = {
        "sinfo": MagicMock(returncode=0, stdout="node003 idle\n"),
        "ssh": MagicMock(returncode=0, stdout=probe_stdout),
        "sbatch": MagicMock(returncode=0, stdout="Submitted batch job 777\n"),
    }

    def _fake_run(cmd, *args, **kwargs):
        return responses.get(cmd[0], responses["sbatch"])

    return _fake_run


def test_recover_pool_readmits_and_resubmits_a_healthy_excluded_node(tmp_path):
    """The gap that let birds finish 4 rounds at 10 workers with 12 idle
    machines beside it: nothing new was failing, so no recovery ever ran."""
    (
        run_config,
        run,
        state,
        mesh_shared,
        assignments_path,
        run_configs_path,
    ) = _recovery_fixture(tmp_path)

    with patch(
        "video_encoding.watcher.mesh_tick.subprocess.run",
        side_effect=_recovery_responses(),
    ) as mock_run:
        note = _recover_pool(
            run_config,
            run,
            state,
            mesh_shared,
            tmp_path,
            run_configs_path,
            now=10_000.0,
        )

    assert "node003" not in state.excluded_nodes
    assert json.loads(assignments_path.read_text())["node003"] == ["s3"]
    submitted = [c for c in mock_run.call_args_list if c[0][0][0] == "sbatch"]
    assert len(submitted) == 1
    assert "+mesh.start_round=8" in submitted[0][0][0]  # run.round(7) + 1
    assert note is not None and "node003" in note


def test_recover_pool_leaves_an_unhealthy_node_excluded(tmp_path):
    fixture = _recovery_fixture(tmp_path)
    run_config, run, state, mesh_shared, assignments_path, run_configs_path = fixture

    with patch(
        "video_encoding.watcher.mesh_tick.subprocess.run",
        # GPU still held by a foreign process
        side_effect=_recovery_responses("2443\n26\n"),
    ) as mock_run:
        note = _recover_pool(
            run_config,
            run,
            state,
            mesh_shared,
            tmp_path,
            run_configs_path,
            now=10_000.0,
        )

    assert "node003" in state.excluded_nodes
    assert note is None
    assert not any(c[0][0][0] == "sbatch" for c in mock_run.call_args_list)


def test_recover_pool_does_not_resubmit_a_node_with_no_shards_to_train(tmp_path):
    """node003 is healthy but every shard it holds is already being trained
    by a survivor. Submitting it would crash on read_my_shards' empty-list
    guard and re-exclude it."""
    fixture = _recovery_fixture(
        tmp_path,
        holdings={"node001": ["s1", "s3"], "node002": ["s2"], "node003": ["s3"]},
        assignments={"node001": ["s1", "s3"], "node002": ["s2"]},
    )
    run_config, run, state, mesh_shared, assignments_path, run_configs_path = fixture

    with patch(
        "video_encoding.watcher.mesh_tick.subprocess.run",
        side_effect=_recovery_responses(),
    ) as mock_run:
        note = _recover_pool(
            run_config,
            run,
            state,
            mesh_shared,
            tmp_path,
            run_configs_path,
            now=10_000.0,
        )

    assert note is None
    assert "node003" in state.excluded_nodes
    assert not any(c[0][0][0] == "sbatch" for c in mock_run.call_args_list)


def test_recover_pool_is_throttled_between_probes(tmp_path):
    fixture = _recovery_fixture(tmp_path)
    run_config, run, state, mesh_shared, assignments_path, run_configs_path = fixture
    run.last_recovery_at = 9_900.0

    with patch(
        "video_encoding.watcher.mesh_tick.subprocess.run",
        side_effect=_recovery_responses(),
    ) as mock_run:
        note = _recover_pool(
            run_config,
            run,
            state,
            mesh_shared,
            tmp_path,
            run_configs_path,
            now=10_000.0,
        )

    assert note is None
    assert mock_run.call_count == 0  # not even an sinfo probe
    assert "node003" in state.excluded_nodes


def test_recover_pool_noop_when_the_pool_is_already_full(tmp_path):
    fixture = _recovery_fixture(tmp_path)
    run_config, run, state, mesh_shared, assignments_path, run_configs_path = fixture
    run.target_pool_size = 2  # node001 + node002 already meet it

    with patch(
        "video_encoding.watcher.mesh_tick.subprocess.run",
        side_effect=_recovery_responses(),
    ) as mock_run:
        note = _recover_pool(
            run_config,
            run,
            state,
            mesh_shared,
            tmp_path,
            run_configs_path,
            now=10_000.0,
        )

    assert note is None
    assert mock_run.call_count == 0


# --- quorum floor --------------------------------------------------------


def _quorum_fixture(tmp_path, *, quorum_k, checked_in, excluded=()):
    """RUN1 with a configured quorum floor; `checked_in` nodes have pushed a
    checkpoint for round 0 and `excluded` are permanently out."""
    mesh_shared = tmp_path / "mesh_shared"
    pending_dir = mesh_shared / "checkpoints" / "pending"
    pending_dir.mkdir(parents=True)
    for node in checked_in:
        _write_pending_checkpoint(pending_dir / f"RUN1_0_{node}.pt")
    run_config = _run_config(quorum_k=quorum_k)
    state = WatcherState()
    run = state.run("RUN1")
    run.status = "AWAITING_CHECKPOINTS"
    run.round = 0
    run.round_started_at = 0.0
    for node in excluded:
        exclude_node(state, node, "crash-loop: missed 2 consecutive rounds of RUN1")
    return run_config, run, state, mesh_shared


def test_awaiting_checkpoints_halts_when_the_roster_falls_below_quorum(tmp_path):
    """The bypass that let factorial-tube-per_frame merge 10-wide against
    quorum_k=12 for its last four rounds: `checked_in >= eligible` is
    trivially true once exclusions shrink eligible below the floor, and it
    fires before quorum is ever consulted."""
    run_config, run, state, mesh_shared = _quorum_fixture(
        tmp_path,
        quorum_k=3,
        checked_in=["node001", "node002"],
        excluded=["node003"],
    )

    note = _do_awaiting_checkpoints(
        run_config, run, state, mesh_shared, 1.0, tmp_path, None
    )

    assert run.status == "DEGRADED"
    assert "quorum_k=3" in note
    assert "2 checked in" in note


def test_awaiting_checkpoints_merges_normally_when_quorum_is_met(tmp_path):
    run_config, run, state, mesh_shared = _quorum_fixture(
        tmp_path, quorum_k=3, checked_in=["node001", "node002", "node003"]
    )

    note = _do_awaiting_checkpoints(
        run_config, run, state, mesh_shared, 1.0, tmp_path, None
    )

    assert run.status == "MERGING"
    assert "DEGRADED" not in note


def test_awaiting_checkpoints_without_quorum_k_keeps_the_old_behaviour(tmp_path):
    """Every run configured before quorum_k existed has None for it and must
    keep merging on the all-checked-in path."""
    run_config, run, state, mesh_shared = _quorum_fixture(
        tmp_path,
        quorum_k=None,
        checked_in=["node001", "node002"],
        excluded=["node003"],
    )

    _do_awaiting_checkpoints(run_config, run, state, mesh_shared, 1.0, tmp_path, None)

    assert run.status == "MERGING"


def test_degraded_run_is_not_advanced_further(tmp_path):
    run_config, run, state, mesh_shared = _quorum_fixture(
        tmp_path, quorum_k=3, checked_in=["node001"]
    )
    run.status = "DEGRADED"

    note = advance_one_run(
        run_config,
        state=state,
        repo_root=tmp_path,
        mesh_shared=mesh_shared,
        mesh_root=tmp_path,
        now=1.0,
    )

    assert note is None
    assert run.status == "DEGRADED"


def test_degraded_run_resumes_once_quorum_is_recovered(tmp_path):
    """A DEGRADED run whose replenished/relaunched roster later meets
    quorum_k on the current round must not need a manual status write --
    2026-08-28's e2b-b-birds recurrence, where checkpoints had fully landed
    and only a hand edit to watcher_state.json ever moved the run forward."""
    run_config, run, state, mesh_shared = _quorum_fixture(
        tmp_path, quorum_k=3, checked_in=["node001", "node002", "node003"]
    )
    run.status = "DEGRADED"

    note = advance_one_run(
        run_config,
        state=state,
        repo_root=tmp_path,
        mesh_shared=mesh_shared,
        mesh_root=tmp_path,
        now=1.0,
    )

    assert run.status == "MERGING"
    assert "DEGRADED -> MERGING" in note


def test_awaiting_checkpoints_halts_on_the_timeout_path_below_quorum(tmp_path):
    """The grace-expired fall-through merges "whatever arrived", which is the
    other way a sub-quorum merge slips through."""
    run_config, run, state, mesh_shared = _quorum_fixture(
        tmp_path, quorum_k=3, checked_in=[]
    )

    note = _do_awaiting_checkpoints(
        run_config, run, state, mesh_shared, 1000.0, tmp_path, None
    )

    assert run.status == "DEGRADED"
    assert "below quorum_k=3" in note


def test_awaiting_checkpoints_timeout_still_merges_when_quorum_holds(tmp_path):
    """A straggler short of the full roster but at or above the floor is
    exactly what quorum merge is for -- it must still merge."""
    run_config, run, state, mesh_shared = _quorum_fixture(
        tmp_path, quorum_k=2, checked_in=["node001", "node002"]
    )

    note = _do_awaiting_checkpoints(
        run_config, run, state, mesh_shared, 1000.0, tmp_path, None
    )

    assert run.status == "MERGING"
    assert "DEGRADED" not in note


def test_assigning_reports_nodes_whose_sbatch_failed(tmp_path):
    """A failed submission must not be reported as a running worker.

    The return value of `_submit_worker_job` used to be discarded and the
    transition counted `eligible`, so a pool logged "16 nodes" while 15 jobs
    existed. With quorum_k == pool size the missing worker cannot be waited
    out: every round burns the full grace_s and then merges timeout-forced.
    """
    mesh_shared = tmp_path / "mesh_shared"
    _seed_shard_map(mesh_shared)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    outcomes = [
        MagicMock(returncode=0, stdout="Submitted batch job 101\n"),
        MagicMock(returncode=1, stdout="", stderr="sbatch: error: node down"),
        MagicMock(returncode=0, stdout="Submitted batch job 103\n"),
    ]
    with patch("video_encoding.watcher.mesh_tick.subprocess.run", side_effect=outcomes):
        action = advance_one_run(
            _run_config(),
            state=WatcherState(),
            repo_root=repo_root,
            mesh_shared=mesh_shared,
            mesh_root=tmp_path / "mesh_root",
            now=1000.0,
        )

    assert "2/3 nodes submitted" in action
    assert "sbatch FAILED for" in action
    # Exactly the node whose submission failed -- the other two are running.
    failed = action.split("sbatch FAILED for ")[1].split(";")[0]
    assert failed == _NODES[1]

    # ...and only the successful submissions reach the registry, so the
    # relaunch detector never waits on a job that was never created.
    registry = json.loads(
        (repo_root / "runs" / "state" / "jobs_registry.json").read_text()
    )
    assert [r["slurm_job_id"] for r in registry] == ["101", "103"]
    assert [r["nodelist"] for r in registry] == [_NODES[0], _NODES[2]]


def _write_straggler_state(
    tmp_path, *, job_id="709", node="node067", state_name="RUNNING"
):
    """A registry record plus a squeue snapshot describing one live worker."""
    state_dir = tmp_path / "runs" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "jobs_registry.json").write_text(
        json.dumps(
            [
                {
                    "slurm_job_id": job_id,
                    "experiment": "videomae_mesh_ssl",
                    "run_tag": "RUN1",
                    "submitted_at": "2026-08-23T09:00:00+00:00",
                    "nodelist": node,
                }
            ]
        )
    )
    (state_dir / "squeue.json").write_text(
        json.dumps(
            {
                "jobs": [
                    {"job_id": int(job_id), "job_state": [state_name], "nodes": node}
                ]
            }
        )
    )


def test_done_cancels_straggling_workers(tmp_path):
    """A run reaches DONE on quorum, so a node behind the pool's cadence is
    still training rounds nobody will merge. Observed 2026-08-23: node067 was
    on round 39 of a run that had reported DONE at 40."""
    mesh_shared = tmp_path / "mesh_shared"
    (mesh_shared / "checkpoints" / "pending").mkdir(parents=True)
    _write_straggler_state(tmp_path)

    run_config = _run_config(rounds=10)
    state = WatcherState()
    state.run("RUN1").status = "SYNCING"
    state.run("RUN1").round = 9  # one flag-write from DONE

    with patch("video_encoding.watcher.mesh_tick.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""
        action = advance_one_run(
            run_config,
            state=state,
            repo_root=tmp_path,
            mesh_shared=mesh_shared,
            mesh_root=tmp_path,
            now=1000.0,
        )

    scancels = [
        c.args[0]
        for c in mock_run.call_args_list
        if c.args and c.args[0][0] == "scancel"
    ]
    assert scancels == [["scancel", "709"]]
    assert "cancelled 1 straggling worker(s): 709" in action


def test_done_does_not_cancel_a_recycled_job_id_on_another_node(tmp_path):
    """Job ids are not unique over time -- the 2026-08-22 rebuild reset the
    counter -- so a stale record must never cancel a stranger's job."""
    mesh_shared = tmp_path / "mesh_shared"
    (mesh_shared / "checkpoints" / "pending").mkdir(parents=True)
    _write_straggler_state(tmp_path, node="node067")
    state_dir = tmp_path / "runs" / "state"
    (state_dir / "squeue.json").write_text(
        json.dumps(
            {"jobs": [{"job_id": 709, "job_state": ["RUNNING"], "nodes": "not-ours"}]}
        )
    )

    run_config = _run_config(rounds=10)
    state = WatcherState()
    state.run("RUN1").status = "SYNCING"
    state.run("RUN1").round = 9

    with patch("video_encoding.watcher.mesh_tick.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""
        action = advance_one_run(
            run_config,
            state=state,
            repo_root=tmp_path,
            mesh_shared=mesh_shared,
            mesh_root=tmp_path,
            now=1000.0,
        )

    assert not [
        c for c in mock_run.call_args_list if c.args and c.args[0][0] == "scancel"
    ]
    assert "no stragglers to cancel" in action


def test_done_still_completes_when_scancel_fails(tmp_path):
    """A failed cancel must not stop a run reporting DONE or releasing state."""
    mesh_shared = tmp_path / "mesh_shared"
    (mesh_shared / "checkpoints" / "pending").mkdir(parents=True)
    _write_straggler_state(tmp_path)

    run_config = _run_config(rounds=10)
    state = WatcherState()
    state.run("RUN1").status = "SYNCING"
    state.run("RUN1").round = 9

    def _fake_run(cmd, *args, **kwargs):
        if cmd[0] == "scancel":
            raise OSError("scancel: command not found")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with patch("video_encoding.watcher.mesh_tick.subprocess.run", _fake_run):
        action = advance_one_run(
            run_config,
            state=state,
            repo_root=tmp_path,
            mesh_shared=mesh_shared,
            mesh_root=tmp_path,
            now=1000.0,
        )

    assert state.run("RUN1").status == "DONE"
    assert "failed to run" in action


@pytest.mark.parametrize(
    "experiment",
    [
        p.stem
        for p in sorted(
            (Path(__file__).resolve().parents[1] / "conf" / "experiment").glob(
                "videomae_mesh_*.yaml"
            )
        )
    ],
)
def test_submit_worker_job_composes_with_chunk_rounds_for_every_mesh_experiment(
    tmp_path, experiment
):
    """Same '+'-on-an-existing-key regression as
    test_submit_worker_job_overrides_compose_against_real_experiment_config,
    but for mesh.chunk_rounds, which that test never exercises: its
    _run_config() leaves chunk_rounds None, so the override is omitted
    entirely. chunk_rounds is not optional in practice -- the 2h partition
    walltime cap makes an unchunked run unschedulable -- and the
    paper-recipe experiment configs declare it, so '+' aborts every worker
    at Hydra composition before it reaches a GPU. Parametrized over every
    mesh experiment config so a config added later cannot silently
    reintroduce it."""
    from hydra import compose, initialize_config_dir

    run_config = _run_config(experiment=experiment, chunk_rounds=4)
    registry_path = tmp_path / "jobs_registry.json"
    fake = MagicMock(returncode=0, stdout="Submitted batch job 556\n")
    with patch(
        "video_encoding.watcher.mesh_tick.subprocess.run", return_value=fake
    ) as mock_run:
        _submit_worker_job(
            run_config, "node001", tmp_path, registry_path, "2026-01-01T00:00:00Z"
        )
    submitted_cmd = mock_run.call_args[0][0]
    hydra_overrides = [a for a in submitted_cmd if a.startswith("+")]
    assert any("chunk_rounds" in o for o in hydra_overrides)

    conf_dir = Path(__file__).resolve().parents[1] / "conf"
    with initialize_config_dir(config_dir=str(conf_dir), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[f"experiment={experiment}", *hydra_overrides],
        )
    assert cfg.mesh.chunk_rounds == 4


def test_merge_timeout_leaves_headroom_over_the_measured_durable_phase(tmp_path):
    """The merge's durable phase measured 210s on a 16-node round (2026-08-28,
    e2b-c-cars). A budget close to that is what let a finished merge get
    SIGKILLed and discarded, so pin that the timeout keeps real headroom and
    that _do_merging actually passes it."""
    from video_encoding.watcher.mesh_tick import MERGE_TIMEOUT_S

    assert MERGE_TIMEOUT_S >= 3 * 210

    mesh_shared = tmp_path / "mesh_shared"
    (mesh_shared / "checkpoints" / "pending").mkdir(parents=True)
    run_config = _run_config()
    state = WatcherState()
    state.run("RUN1").status = "MERGING"

    fake = MagicMock(returncode=0, stdout="", stderr="")
    with patch(
        "video_encoding.watcher.mesh_tick.subprocess.run", return_value=fake
    ) as m:
        advance_one_run(
            run_config,
            state=state,
            repo_root=tmp_path,
            mesh_shared=mesh_shared,
            mesh_root=tmp_path,
            now=1000.0,
        )

    assert m.call_args.kwargs["timeout"] == MERGE_TIMEOUT_S


def _shard_assignments_fixture(
    tmp_path, assignments: dict, holdings: dict | None = None
):
    """mesh_shared with a live assignments file, optionally verified holdings."""
    mesh_shared = tmp_path / "mesh_shared"
    metadata = mesh_shared / "metadata"
    metadata.mkdir(parents=True)
    (metadata / "shard_assignments_RUN1.json").write_text(json.dumps(assignments))
    if holdings is not None:
        (metadata / "verified_holdings.json").write_text(json.dumps({"fish": holdings}))
    return mesh_shared


def _relaunch(run_config, run, state, tmp_path, mesh_shared):
    """Drive _relaunch_chunk_if_finished with sbatch stubbed out."""
    fake = MagicMock(returncode=0, stdout="Submitted batch job 999\n")
    with (
        patch(
            "video_encoding.watcher.mesh_tick.needs_worker_relaunch",
            return_value=True,
        ),
        patch(
            "video_encoding.watcher.mesh_tick.subprocess.run", return_value=fake
        ) as mock_run,
    ):
        note = _relaunch_chunk_if_finished(
            run_config, run, state, tmp_path, mesh_shared, now=1000.0
        )
    submitted = {
        arg.removeprefix("--nodelist=")
        for call in mock_run.call_args_list
        for arg in call.args[0]
        if arg.startswith("--nodelist=")
    }
    return note, submitted, mock_run


def test_relaunch_chunk_repairs_node_missing_from_shard_assignments(tmp_path):
    """Regression for the shard_assignments race (2026-08-29 birds handoff):
    a node can still be in run_config.nodes (only excluded_nodes is checked)
    while redistribute_shards has already deleted its entry. Submitting it
    anyway crashes it on read_my_shards; skipping it silently leaves it in
    eligible_nodes, stalling every round on a checkpoint no job will make.
    It still physically holds its shards, so hand back the untrained ones."""
    mesh_shared = _shard_assignments_fixture(
        tmp_path,
        # "node003" deleted by a concurrent redistribute_shards
        {"node001": ["s0", "s1"], "node002": ["s2", "s3"]},
        holdings={"node003": ["s4", "s5"], "node001": ["s0"]},
    )
    run_config = _run_config(chunk_rounds=5)
    state = WatcherState()
    run = state.run("RUN1")
    run.round = 5

    note, submitted, _ = _relaunch(run_config, run, state, tmp_path, mesh_shared)

    assert submitted == {"node001", "node002", "node003"}
    assert "re-assigned shards to node003" in note
    assert "node003" not in state.excluded_nodes
    written = json.loads(
        (mesh_shared / "metadata" / "shard_assignments_RUN1.json").read_text()
    )
    assert written["node003"] == ["s4", "s5"]


def test_relaunch_chunk_never_reassigns_a_shard_someone_else_trains(tmp_path):
    """Repair must not double-assign: node003 holds s0, but node001 is
    already training it, so only its untrained holdings come back."""
    mesh_shared = _shard_assignments_fixture(
        tmp_path,
        {"node001": ["s0", "s1"], "node002": ["s2"]},
        holdings={"node003": ["s0", "s5"]},
    )
    run_config = _run_config(chunk_rounds=5)
    state = WatcherState()
    run = state.run("RUN1")

    _relaunch(run_config, run, state, tmp_path, mesh_shared)

    written = json.loads(
        (mesh_shared / "metadata" / "shard_assignments_RUN1.json").read_text()
    )
    assert written["node003"] == ["s5"]
    assert written["node001"] == ["s0", "s1"]


def test_relaunch_chunk_excludes_a_node_with_no_shards_to_give(tmp_path):
    """A node that holds nothing untrained cannot be repaired. Leaving it
    unexcluded keeps it in eligible_nodes, so every round waits out the
    grace period for a checkpoint no job exists to produce -- exclude it
    instead, with the "of {run_id}" suffix clear_run_state/_recover_pool
    both key off."""
    mesh_shared = _shard_assignments_fixture(
        tmp_path,
        {"node001": ["s0"], "node002": ["s1"]},
        holdings={"node003": ["s0"]},  # already trained by node001
    )
    run_config = _run_config(chunk_rounds=5)
    state = WatcherState()
    run = state.run("RUN1")

    note, submitted, _ = _relaunch(run_config, run, state, tmp_path, mesh_shared)

    assert submitted == {"node001", "node002"}
    assert state.excluded_nodes["node003"].endswith("of RUN1")
    assert "excluded node003" in note


def test_relaunch_chunk_stalls_once_when_no_node_has_shards(tmp_path):
    """The all-unassigned case cannot clear itself, and a non-None return
    short-circuits the whole state machine. Report it once, then fall
    through so the merge/timeout handlers still get their tick (and the
    backoff is not pinned at FAST_INTERVAL_S forever)."""
    mesh_shared = _shard_assignments_fixture(tmp_path, {}, holdings={})
    run_config = _run_config(chunk_rounds=5)
    state = WatcherState()
    run = state.run("RUN1")

    first, submitted, _ = _relaunch(run_config, run, state, tmp_path, mesh_shared)
    # Every node was excluded by the first pass, so the second reports the
    # *other* terminal condition once; from the third on it is silent.
    second, _, _ = _relaunch(run_config, run, state, tmp_path, mesh_shared)
    third, _, _ = _relaunch(run_config, run, state, tmp_path, mesh_shared)

    assert submitted == set()
    assert first is not None and "no node has shards to train" in first
    assert second is not None and "every node is excluded" in second
    assert third is None


def test_relaunch_chunk_without_assignments_file_submits_every_node(tmp_path):
    """No shard-tracking started for this run yet: a missing file gates
    nothing, matching _recover_pool's own missing-file semantics."""
    mesh_shared = tmp_path / "mesh_shared"
    (mesh_shared / "metadata").mkdir(parents=True)
    run_config = _run_config(chunk_rounds=5)
    state = WatcherState()
    run = state.run("RUN1")

    note, submitted, _ = _relaunch(run_config, run, state, tmp_path, mesh_shared)

    assert submitted == set(_NODES)
    assert "relaunched 3/3" in note


def test_relaunch_chunk_skips_unassigned_when_holdings_unknown(tmp_path):
    """verified_holdings.json missing: we can neither hand these nodes
    shards nor honestly call them unusable, and excluding a whole roster
    on a missing file needs a human to undo. Skip and say why."""
    mesh_shared = _shard_assignments_fixture(
        tmp_path, {"node001": ["s0"], "node002": ["s1"]}
    )
    run_config = _run_config(chunk_rounds=5)
    state = WatcherState()
    run = state.run("RUN1")

    note, submitted, _ = _relaunch(run_config, run, state, tmp_path, mesh_shared)

    assert submitted == {"node001", "node002"}
    assert state.excluded_nodes == {}
    assert "verified_holdings.json missing" in note


def test_relaunch_chunk_freezes_each_job_shard_list_at_submission(tmp_path):
    """The live assignments file is rewritten on every exclusion, but a
    worker reads it once, at process start -- minutes to hours after
    sbatch, since the job sits PENDING. Point each job at an immutable
    per-node snapshot so a later redistribute_shards cannot invalidate a
    submission already in flight (job 2708, `node045`, 2026-08-29)."""
    mesh_shared = _shard_assignments_fixture(
        tmp_path, {"node001": ["s0"], "node002": ["s1"], "node003": ["s2"]}
    )
    run_config = _run_config(chunk_rounds=5)
    state = WatcherState()
    run = state.run("RUN1")

    _, _, mock_run = _relaunch(run_config, run, state, tmp_path, mesh_shared)

    snapshot = mesh_shared / "metadata" / "shard_assignments_RUN1__node001.json"
    assert json.loads(snapshot.read_text()) == {"node001": ["s0"]}
    exports = [
        arg
        for call in mock_run.call_args_list
        for arg in call.args[0]
        if arg.startswith("--export=")
    ]
    assert f"--export=ALL,MESH_ASSIGNMENTS={snapshot}" in exports
    # The live file is what the snapshot was taken from; mutating it now
    # must not change what an already-submitted job will read.
    (mesh_shared / "metadata" / "shard_assignments_RUN1.json").write_text("{}")
    assert json.loads(snapshot.read_text()) == {"node001": ["s0"]}
