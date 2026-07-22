"""video_encoding/watcher/mesh_tick.py

Round-state driver for the decentralized P2P training mesh (design spec §9).
Called every cron tick (unconditionally -- see watch_training.py's wiring),
this module owns the ASSIGNING -> RUNNING -> AWAITING_CHECKPOINTS -> MERGING
-> SYNCING -> DONE state machine for each configured mesh run: it submits
one sbatch job per node, invokes the existing run_outer_step.py DiLoCo merge
once every eligible node has checked in (or a round times out), and writes
the sync flags Task 2's worker-side round loop
(video_encoding/training/mesh_round.py) polls for.

Job health (dead/stalled detection, auto-resubmit, escalation) is entirely
delegated to the existing, unmodified video_encoding/watcher/tick.py --
mesh jobs are appended to the SAME jobs_registry.json so that machinery
covers them for free; this module only tracks round progress.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from video_encoding.mesh.assignment import assign_locality_first
from video_encoding.mesh.scoring import (
    NodeInfo,
    compute_effective_weights,
    compute_shard_counts,
)
from video_encoding.mesh.shard_map import Holder, load_shard_map
from video_encoding.scripts.seed_mesh import NODE_GROUPS, NODE_SUBNETS
from video_encoding.training import checkpoint as ckpt_io
from video_encoding.watcher.epoch_advance import (
    eligible_nodes,
    pending_nodes,
    record_missed_round,
)
from video_encoding.watcher.registry import JobRecord, append_job_record
from video_encoding.watcher.state import RunState, WatcherState

# Twice the mesh's default per-round sync timeout (videomae_mesh_ssl.yaml's
# mesh.sync_timeout_s: 1800.0) -- gives checkpoints twice as long to arrive
# as a single worker's own poll-and-exit(1) timeout before the watcher gives
# up on a straggler and merges a partial round (design spec §9: "this round
# merges whatever arrived (partial average)").
ROUND_TIMEOUT_S = 3600.0


@dataclass(frozen=True)
class MeshRunConfig:
    run_id: str
    experiment: str
    nodes: list[str]  # fixed roster for this run -- same list mesh_ab.sh submitted to
    n_shards: int
    outer_lr: float
    outer_momentum: float
    weighting: str  # "data" | "inverse_loss"
    rounds: int
    merge: bool = True  # False for the ab-single_node reference arm


def load_run_configs(path: Path) -> list[MeshRunConfig]:
    """Read the small JSON list written once per run at submit time.

    Missing file means no mesh run is currently configured -- return []
    rather than raising, so the cron wiring can unconditionally call this
    every tick without a mesh run ever having been registered.
    """
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    return [MeshRunConfig(**entry) for entry in payload]


def advance_all_runs(
    run_configs: list[MeshRunConfig],
    *,
    state: WatcherState,
    repo_root: Path,
    mesh_shared: Path,
    mesh_root: Path,
    now: float,
) -> list[str]:
    """Advance every non-DONE run by one state-machine step.

    Called every cron tick (unconditional -- not gated by run_full_check's
    health-check backoff timer): worker jobs push checkpoints and wait on
    sync flags on their own schedule, independent of that cadence.
    """
    actions: list[str] = []
    for run_config in run_configs:
        if state.run(run_config.run_id).status == "DONE":
            continue
        action = advance_one_run(
            run_config,
            state=state,
            repo_root=repo_root,
            mesh_shared=mesh_shared,
            mesh_root=mesh_root,
            now=now,
        )
        if action is not None:
            actions.append(action)
    return actions


def advance_one_run(
    run_config: MeshRunConfig,
    *,
    state: WatcherState,
    repo_root: Path,
    mesh_shared: Path,
    mesh_root: Path,
    now: float,
) -> str | None:
    """One state transition (or no-op) for a single run_id.

    Dispatches on state.run(run_config.run_id).status. `mesh_root` is
    accepted for interface parity with the worker side (video_encoding/
    training/mesh_round.py reads MESH_ROOT from its own environment) but is
    not consumed by any state transition here -- every path this module
    touches (shard_map.json, shard_assignments, checkpoints, sync flags)
    lives under mesh_shared, the watcher's NFS-shared metadata/checkpoint
    tree, not the per-node local mesh_root.
    """
    del mesh_root  # unused -- see docstring
    run = state.run(run_config.run_id)
    if run.status == "ASSIGNING":
        return _do_assigning(run_config, run, state, repo_root, mesh_shared)
    if run.status == "RUNNING":
        return _do_running(run_config, run, mesh_shared)
    if run.status == "AWAITING_CHECKPOINTS":
        return _do_awaiting_checkpoints(run_config, run, state, mesh_shared, now)
    if run.status == "MERGING":
        return _do_merging(run_config, run, repo_root, mesh_shared)
    if run.status == "SYNCING":
        return _do_syncing(run_config, run, mesh_shared)
    return None  # DONE, or an unrecognized status -- nothing to do


# --- ASSIGNING -------------------------------------------------------------


def _invert_placement(shard_map: dict[str, list[Holder]]) -> dict[str, list[str]]:
    """`shard -> [Holder, ...]` (shard_map.py's schema) to `host -> [shard, ...]`
    (assign_locality_first's `placement` schema), in shard_map iteration order.
    """
    placement: dict[str, list[str]] = {}
    for shard, holders in shard_map.items():
        for holder in holders:
            placement.setdefault(holder.host, []).append(shard)
    return placement


def _parse_submitted_job_id(stdout: str) -> str | None:
    """Mirrors remediate_dead_job's "Submitted batch job N" parsing."""
    for line in stdout.splitlines():
        if "Submitted batch job" in line:
            tail = line.strip().split()[-1]
            return tail if tail.isdigit() else None
    return None


def _do_assigning(
    run_config: MeshRunConfig,
    run: RunState,
    state: WatcherState,
    repo_root: Path,
    mesh_shared: Path,
) -> str | None:
    eligible = eligible_nodes(state, run_config.nodes)
    if not eligible:
        # Every node in this run's roster has been excluded (crash-loop) --
        # nothing to assign to. Leave status as ASSIGNING; a human clearing
        # excluded_nodes is the only way forward (epoch_advance.exclude_node
        # is never auto-reversed).
        return f"{run_config.run_id}: ASSIGNING skipped -- no eligible nodes"

    shard_map = load_shard_map(mesh_shared / "metadata" / "shard_map.json")
    placement = _invert_placement(shard_map)
    all_shards = list(shard_map)

    nodes_info = [
        NodeInfo(
            hostname=host,
            group=NODE_GROUPS[host],
            subnet=NODE_SUBNETS[host],
            free_gb=0.0,  # not consumed by effective_weight -- mesh/scoring.py
        )
        for host in eligible
    ]
    weights = compute_effective_weights(nodes_info)
    shard_counts = compute_shard_counts(weights, run_config.n_shards)
    assignments = assign_locality_first(shard_counts, placement, all_shards)

    assignments_path = (
        mesh_shared / "metadata" / f"shard_assignments_{run_config.run_id}.json"
    )
    _write_json_atomic(assignments_path, assignments)

    registry_path = repo_root / ".claude" / "state" / "jobs_registry.json"
    submitted_at = datetime.now(timezone.utc).isoformat()
    for node in eligible:
        cmd = [
            "sbatch",
            "--nodes=1",
            "--ntasks=1",
            f"--nodelist={node}",
            "video_encoding/slurm/mesh_train.sbatch",
            run_config.experiment,
            run_config.run_id,
            f"+run_tag={run_config.run_id}",
            f"+mesh.merge={'true' if run_config.merge else 'false'}",
            f"+mesh.outer_lr={run_config.outer_lr}",
            f"+mesh.outer_momentum={run_config.outer_momentum}",
            f"+mesh.rounds={run_config.rounds}",
        ]
        try:
            result = subprocess.run(
                cmd, cwd=repo_root, capture_output=True, text=True, timeout=300
            )
        except (OSError, subprocess.SubprocessError):
            continue
        job_id = (
            _parse_submitted_job_id(result.stdout) if result.returncode == 0 else None
        )
        if job_id is not None:
            append_job_record(
                registry_path,
                JobRecord(
                    slurm_job_id=job_id,
                    experiment=run_config.experiment,
                    run_tag=run_config.run_id,
                    submitted_at=submitted_at,
                    nodelist=node,
                ),
            )

    run.status = "RUNNING"
    return f"{run_config.run_id}: ASSIGNING -> RUNNING ({len(eligible)} nodes)"


# --- RUNNING -----------------------------------------------------------


def _do_running(
    run_config: MeshRunConfig, run: RunState, mesh_shared: Path
) -> str | None:
    if not run_config.merge:
        # ab-single_node arm: Task 2's worker trains every round inside one
        # job, pushing checkpoints nobody merges and never waiting on a sync
        # flag -- no watcher-side round coordination applies once the job is
        # submitted, so this run is done as far as the state machine cares.
        run.status = "DONE"
        return f"{run_config.run_id}: RUNNING -> DONE (merge=False, single-node arm)"

    pending_dir = mesh_shared / "checkpoints" / "pending"
    if pending_nodes(pending_dir, run_config.run_id):
        run.status = "AWAITING_CHECKPOINTS"
        return f"{run_config.run_id}: RUNNING -> AWAITING_CHECKPOINTS"
    return None  # no-op -- job health is tick.py's job, not this module's


# --- AWAITING_CHECKPOINTS ---------------------------------------------------


def _round_elapsed_s(pending_dir: Path, run_id: str, now: float) -> float:
    """Seconds since this round's earliest checkpoint arrived.

    pending/{run_id}_{node}.pt files are overwritten every round (see
    epoch_advance.pending_nodes' docstring), so their mtimes always reflect
    only the current round's check-ins -- no separate round-start timestamp
    needs to be persisted in WatcherState for this.
    """
    files = list(pending_dir.glob(f"{run_id}_*.pt"))
    if not files:
        return 0.0
    return now - min(f.stat().st_mtime for f in files)


def _do_awaiting_checkpoints(
    run_config: MeshRunConfig,
    run: RunState,
    state: WatcherState,
    mesh_shared: Path,
    now: float,
) -> str | None:
    pending_dir = mesh_shared / "checkpoints" / "pending"
    checked_in = pending_nodes(pending_dir, run_config.run_id)
    eligible = set(eligible_nodes(state, run_config.nodes))

    if checked_in >= eligible:
        run.status = "MERGING"
        return (
            f"{run_config.run_id}: AWAITING_CHECKPOINTS -> MERGING "
            f"({len(checked_in)} nodes)"
        )

    if _round_elapsed_s(pending_dir, run_config.run_id, now) < ROUND_TIMEOUT_S:
        return None  # still waiting, not timed out yet

    missing = sorted(eligible - checked_in)
    # excluded here permanently loses its assigned shards for the rest of
    # this run: ASSIGNING (and therefore assign_locality_first) only runs
    # once per run now (see SYNCING -> RUNNING, not ASSIGNING, above), so
    # there is no later round that redistributes an excluded node's shards
    # to a survivor -- shard coverage silently shrinks for the remainder.
    excluded = [
        node for node in missing if record_missed_round(state, run_config.run_id, node)
    ]
    run.status = "MERGING"
    excluded_note = f", permanently excluded {excluded}" if excluded else ""
    return (
        f"{run_config.run_id}: AWAITING_CHECKPOINTS -> MERGING "
        f"(timeout, missing {missing}{excluded_note})"
    )


# --- MERGING -----------------------------------------------------------


def _bootstrap_master(pending_dir: Path, run_id: str, master_path: Path) -> None:
    """Seed master.pt before the first-ever merge of a run.

    merge_round (run_outer_step.py) unconditionally torch.loads master_path
    to diff worker deltas against -- it has no cold-start fallback (only
    outer_momentum.pt gets a missing-is-OK branch), and this plan's Global
    Constraints forbid modifying that file. No earlier round writes
    master.pt either, so without this, every run's first MERGING would
    crash on FileNotFoundError and retry forever against the same missing
    file. Promote one already-arrived worker checkpoint (deterministic:
    lexicographically first pending filename) to the initial master --
    mirrors mesh_round.py's own cold-start contract, where a worker with no
    master.pt on disk trains from a freshly-initialized model.
    """
    candidates = sorted(pending_dir.glob(f"{run_id}_*.pt"))
    if not candidates:
        return
    payload = torch.load(candidates[0], map_location="cpu", weights_only=False)
    master_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_io.save(master_path, {"model_state_dict": payload["model_state_dict"]})


def _do_merging(
    run_config: MeshRunConfig, run: RunState, repo_root: Path, mesh_shared: Path
) -> str | None:
    pending_dir = mesh_shared / "checkpoints" / "pending"
    master_path = mesh_shared / "checkpoints" / run_config.run_id / "master.pt"
    momentum_path = (
        mesh_shared / "checkpoints" / run_config.run_id / "outer_momentum.pt"
    )
    if not master_path.exists():
        _bootstrap_master(pending_dir, run_config.run_id, master_path)
    cmd = [
        "uv",
        "run",
        "python",
        "video_encoding/scripts/run_outer_step.py",
        "--pending-dir",
        str(pending_dir),
        "--master",
        str(master_path),
        "--momentum",
        str(momentum_path),
        "--outer-lr",
        str(run_config.outer_lr),
        "--outer-momentum",
        str(run_config.outer_momentum),
        "--weighting",
        run_config.weighting,
    ]
    try:
        result = subprocess.run(
            cmd, cwd=repo_root, capture_output=True, text=True, timeout=300
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"{run_config.run_id}: MERGING failed ({exc}) -- retrying next tick"

    if result.returncode != 0:
        return (
            f"{run_config.run_id}: MERGING failed (exit {result.returncode}) "
            "-- retrying next tick"
        )

    for pending_file in pending_dir.glob(f"{run_config.run_id}_*.pt"):
        pending_file.unlink()

    run.status = "SYNCING"
    return f"{run_config.run_id}: MERGING -> SYNCING"


# --- SYNCING -------------------------------------------------------------


def _do_syncing(run_config: MeshRunConfig, run: RunState, mesh_shared: Path) -> str:
    flag_path = (
        mesh_shared
        / "checkpoints"
        / "flags"
        / f"sync_complete_{run_config.run_id}_{run.round}.flag"
    )
    _atomic_touch(flag_path)
    run.round += 1

    if run.round >= run_config.rounds:
        run.status = "DONE"
        return (
            f"{run_config.run_id}: SYNCING -> DONE "
            f"(round {run.round}/{run_config.rounds})"
        )
    # Not ASSIGNING: Task 2's worker job is a single long-lived SLURM
    # allocation that loops over every round internally, polling this exact
    # flag between rounds -- it never exits and needs no resubmission. Shard
    # assignment is also static for this plan (per-round throughput
    # reweighting is explicitly out of scope), so there is nothing to
    # recompute either. Re-entering ASSIGNING here would unconditionally
    # sbatch a second job per node on top of the still-alive original.
    # RUNNING already does exactly what's needed next: wait for the
    # following round's checkpoints to land in pending/ (cleared by MERGING
    # on success) and re-enter AWAITING_CHECKPOINTS once they do.
    run.status = "RUNNING"
    return (
        f"{run_config.run_id}: SYNCING -> RUNNING "
        f"(round {run.round}/{run_config.rounds})"
    )


# --- shared atomic-write helpers (tmp+rename, matching state.py/shard_map.py) --


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def _atomic_touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("")
    tmp.replace(path)
