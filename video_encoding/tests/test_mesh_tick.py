from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from video_encoding.mesh.shard_map import Holder, save_shard_map
from video_encoding.watcher.epoch_advance import exclude_node
from video_encoding.watcher.mesh_tick import (
    MeshRunConfig,
    _clean_stale_run_artifacts,
    _replenish_pool,
    _submit_worker_job,
    advance_all_runs,
    advance_one_run,
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
        (pending_dir / f"RUN1_{node}.pt").write_text("x")

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


def test_awaiting_checkpoints_quorum_configured_still_waits_below_quorum(tmp_path):
    """Below quorum_k, a quorum-configured run waits exactly like the
    default path does below ROUND_TIMEOUT_S -- grace_s hasn't expired yet
    either, so there's nothing to merge with."""
    mesh_shared = tmp_path / "mesh_shared"
    pending_dir = mesh_shared / "checkpoints" / "pending"
    pending_dir.mkdir(parents=True)
    (pending_dir / "RUN1_node001.pt").write_text("x")  # only 1 of 3

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
    """Quorum never reached and grace_s has fully elapsed: falls back to the
    exact same exclusion/replenish path the plain timeout uses -- grace_s
    plays ROUND_TIMEOUT_S's old role once quorum is configured."""
    mesh_shared = tmp_path / "mesh_shared"
    pending_dir = mesh_shared / "checkpoints" / "pending"
    pending_dir.mkdir(parents=True)
    old_mtime = 1000.0
    path = pending_dir / "RUN1_node001.pt"
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

    assert state.run("RUN1").status == "MERGING"
    assert state.run("RUN1").timeout_forced is True
    for node in ("node002", "node003"):
        assert state.run("RUN1").missed_rounds.get(node) == 1
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
        path = pending_dir / f"RUN1_{node}.pt"
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
        (pending_dir / f"RUN1_{node}.pt").write_text("x")

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
    # Phase 3.2: 0 predates every real round_num merge_round ever stamps, so
    # every worker's first real diff (against a genuine merge) is never
    # rejected as stale by comparison to a bootstrap master.
    assert payload["master_version"] == 0


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
        pending_dir / "RUN1_node001.pt",
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
        _write_pending_checkpoint(pending_dir / f"RUN1_{node}.pt")

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
        _write_pending_checkpoint(pending_dir / f"RUN1_{node}.pt")

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
        (tmp_path / "runs" / "state" / "jobs_registry.json").read_text()
    )
    assert any(r["nodelist"] == "node004" for r in registry)


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

    shard_map = {
        "s1": [Holder(host="node001", subnet="/254")],
        "s2": [Holder(host="node002", subnet="/254")],
    }
    save_shard_map(mesh_shared / "metadata" / "shard_map.json", shard_map)
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

    with patch(
        "video_encoding.watcher.mesh_tick.subprocess.run", return_value=idle_sinfo
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
    assert all(call.args[0][0] == "sinfo" for call in mock_run.call_args_list)
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

    shard_map = {
        "s1": [Holder(host="node001", subnet="/254")],
        "s2": [Holder(host="node002", subnet="/254")],
    }
    save_shard_map(mesh_shared / "metadata" / "shard_map.json", shard_map)
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

    with patch(
        "video_encoding.watcher.mesh_tick.subprocess.run", return_value=idle_sinfo
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

    assert all(call.args[0][0] == "sinfo" for call in mock_run.call_args_list)
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

    shard_map = {
        "s1": [Holder(host="node001", subnet="/254")],
        "s2": [Holder(host="node002", subnet="/254")],
        "s3": [Holder(host="node003", subnet="/254")],
    }
    save_shard_map(mesh_shared / "metadata" / "shard_map.json", shard_map)
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
        path = pending_dir / f"RUN1_{node}.pt"
        path.write_text("x")
        os.utime(path, (old_mtime, old_mtime))

    save_shard_map(
        mesh_shared / "metadata" / "shard_map.json",
        {
            "s1": [Holder(host="node001", subnet="/254")],
            "s2": [Holder(host="node003", subnet="/254")],
        },
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
