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
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from video_encoding.mesh.assignment import assign_locality_first, reassign_failed
from video_encoding.mesh.scoring import (
    NodeInfo,
    compute_effective_weights,
    compute_shard_counts,
)
from video_encoding.mesh.shard_map import Holder, load_shard_map
from video_encoding.scripts.seed_mesh import NODE_GROUPS, NODE_SUBNETS
from video_encoding.training import checkpoint as ckpt_io
from video_encoding.watcher.cluster import query_idle_nodes
from video_encoding.watcher.epoch_advance import (
    candidate_replacements,
    eligible_nodes,
    pending_nodes,
    pool_deficit,
    record_missed_round,
    select_idle_nodes,
)
from video_encoding.watcher.registry import JobRecord, append_job_record
from video_encoding.watcher.state import RunState, WatcherState

# Must stay well below the mesh's default per-round sync timeout
# (videomae_mesh_ssl.yaml's mesh.sync_timeout_s: 1800.0) -- that's how long a
# worker that already pushed its own checkpoint will wait at the barrier
# before giving up and exiting. If this threshold were >= sync_timeout_s (it
# was previously 3600.0, twice as long), every already-finished worker exits
# before the watcher ever attempts the force-merge meant to save it. 900s
# leaves a 900s margin under the worker's 1800s patience for tick cadence and
# merge-operation overhead (design spec §9: "this round merges whatever
# arrived (partial average)").
ROUND_TIMEOUT_S = 900.0


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
    run_configs_path: Path | None = None,
) -> list[str]:
    """Advance every non-DONE run as far as it will go this tick.

    Called every cron tick (unconditional -- not gated by run_full_check's
    health-check backoff timer): worker jobs push checkpoints and wait on
    sync flags on their own schedule, independent of that cadence.

    Cron itself only fires once/minute, so without the inner while-loop a
    run sitting in a state whose condition is already satisfied (all
    checkpoints already landed, sync flag already written) would still wait
    for the *next* tick to notice -- pure dispatch latency, not real work
    (observed: ~60-70s per such transition in production watcher.log).
    advance_one_run's own per-status handlers are unchanged; this just
    keeps calling the same run_id until one returns None (genuinely
    blocked, e.g. still waiting on checkpoints) or the run reaches DONE.
    """
    actions: list[str] = []
    for run_config in run_configs:
        while state.run(run_config.run_id).status != "DONE":
            action = advance_one_run(
                run_config,
                state=state,
                repo_root=repo_root,
                mesh_shared=mesh_shared,
                mesh_root=mesh_root,
                now=now,
                run_configs_path=run_configs_path,
            )
            if action is None:
                break
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
    run_configs_path: Path | None = None,
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
    if run.target_pool_size == 0:
        run.target_pool_size = len(run_config.nodes)
    if run.status == "ASSIGNING":
        return _do_assigning(run_config, run, state, repo_root, mesh_shared)
    if run.status == "RUNNING":
        return _do_running(run_config, run, mesh_shared)
    if run.status == "AWAITING_CHECKPOINTS":
        return _do_awaiting_checkpoints(
            run_config, run, state, mesh_shared, now, repo_root, run_configs_path
        )
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


def _submit_worker_job(
    run_config: MeshRunConfig,
    node: str,
    repo_root: Path,
    registry_path: Path,
    submitted_at: str,
    *,
    start_round: int = 0,
) -> str | None:
    """Submits one sbatch job for `node`, appends its record to
    registry_path on success. Returns the parsed job id, or None if
    sbatch failed to run or its stdout didn't contain a job id --
    mirrors the existing per-node loop's own best-effort handling
    (continue past a single node's submission failure)."""
    cmd = [
        "sbatch",
        "--nodes=1",
        "--ntasks=1",
        f"--nodelist={node}",
        "video_encoding/slurm/mesh_train.sbatch",
        run_config.experiment,
        run_config.run_id,
        f"+run_tag={run_config.run_id}",
        # No '+' prefix on these four: conf/experiment/videomae_mesh_ssl.yaml
        # already defines mesh.merge/outer_lr/outer_momentum/rounds, and '+'
        # means "add a NEW key" -- Hydra raises ConfigCompositionException
        # ("Could not append to config") if the key already exists. Only
        # mesh.start_round is genuinely new, so it keeps the '+'.
        f"mesh.merge={'true' if run_config.merge else 'false'}",
        f"mesh.outer_lr={run_config.outer_lr}",
        f"mesh.outer_momentum={run_config.outer_momentum}",
        f"mesh.rounds={run_config.rounds}",
        f"+mesh.start_round={start_round}",
    ]
    try:
        result = subprocess.run(
            cmd, cwd=repo_root, capture_output=True, text=True, timeout=300
        )
    except (OSError, subprocess.SubprocessError):
        return None
    job_id = _parse_submitted_job_id(result.stdout) if result.returncode == 0 else None
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
    return job_id


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

    deleted = _clean_stale_run_artifacts(run_config.run_id, mesh_shared)

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
        _submit_worker_job(run_config, node, repo_root, registry_path, submitted_at)

    run.status = "RUNNING"
    action = f"{run_config.run_id}: ASSIGNING -> RUNNING ({len(eligible)} nodes)"
    if deleted:
        action += f"; cleaned {len(deleted)} stale artifact(s) from a prior attempt"
    return action


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
    repo_root: Path,
    run_configs_path: Path | None,
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
    replenish_notes = [
        note
        for node in excluded
        if (
            note := _replenish_pool(
                run_config, run, state, mesh_shared, repo_root, run_configs_path, node
            )
        )
        is not None
    ]
    run.status = "MERGING"
    excluded_note = f", permanently excluded {excluded}" if excluded else ""
    result = (
        f"{run_config.run_id}: AWAITING_CHECKPOINTS -> MERGING "
        f"(timeout, missing {missing}{excluded_note})"
    )
    if replenish_notes:
        result += "; " + "; ".join(replenish_notes)
    return result


# --- pool replenishment ------------------------------------------------


def _append_nodes_to_run_config(path: Path, run_id: str, new_nodes: list[str]) -> None:
    """Rewrites the run-configs JSON `load_run_configs` reads: finds the
    entry matching run_id and extends its "nodes" list with any of
    new_nodes not already present (idempotency guard -- a re-run of
    this function with the same picked nodes is a no-op on the second
    call), atomic tmp+rename."""
    payload = json.loads(path.read_text())
    for entry in payload:
        if entry["run_id"] == run_id:
            for node in new_nodes:
                if node not in entry["nodes"]:
                    entry["nodes"].append(node)
            break
    _write_json_atomic(path, payload)


def _live_roster(run_configs_path: Path, run_id: str, fallback: list[str]) -> list[str]:
    """Current on-disk roster for run_id, reflecting any nodes a prior
    _replenish_pool call already appended earlier in this same tick.

    `run_config.nodes` is a snapshot frozen when advance_one_run's caller
    loaded it -- it never updates mid-tick even after
    _append_nodes_to_run_config writes a grown roster to disk. If two
    nodes cross the crash-loop exclusion threshold in the same tick (e.g.
    a shared NFS blip timing out several checkpoints at once),
    _do_awaiting_checkpoints calls _replenish_pool once per newly-excluded
    node -- reading the live roster here (instead of the stale
    run_config.nodes) is what stops the second call from re-picking and
    re-submitting the same idle node the first call already claimed.
    Falls back to `fallback` if the file or this run's entry doesn't
    exist yet.
    """
    if not run_configs_path.exists():
        return fallback
    payload = json.loads(run_configs_path.read_text())
    for entry in payload:
        if entry["run_id"] == run_id:
            return entry["nodes"]
    return fallback


def _replenish_pool(
    run_config: MeshRunConfig,
    run: RunState,
    state: WatcherState,
    mesh_shared: Path,
    repo_root: Path,
    run_configs_path: Path | None,
    excluded_node: str,
) -> str | None:
    """Called right after `excluded_node` is freshly excluded (a NEW
    exclusion this tick, not a pre-existing one from before this
    feature existed -- see Task 5 call site). Tops the pool back up
    toward run.target_pool_size with idle same-tier nodes and hands
    them excluded_node's orphaned shards. Returns a status string
    describing what happened, or None if there was nothing to do
    (no deficit) or nothing that could be done (no run_configs_path,
    no idle candidates)."""
    if run_configs_path is None:
        return None

    roster = _live_roster(run_configs_path, run_config.run_id, run_config.nodes)
    deficit = pool_deficit(run, roster, state)
    if deficit <= 0:
        return None

    tier = NODE_GROUPS[roster[0]]
    tier_roster = [h for h, g in NODE_GROUPS.items() if g == tier]
    idle = query_idle_nodes()
    candidates = candidate_replacements(
        tier_roster, in_use=set(roster), excluded=set(state.excluded_nodes)
    )
    picked = select_idle_nodes(candidates, deficit, idle)
    if not picked:
        return (
            f"{run_config.run_id}: replenish skipped -- no idle {tier} nodes "
            f"available (deficit {deficit})"
        )

    _append_nodes_to_run_config(run_configs_path, run_config.run_id, picked)

    registry_path = repo_root / ".claude" / "state" / "jobs_registry.json"
    submitted_at = datetime.now(timezone.utc).isoformat()
    for node in picked:
        _submit_worker_job(
            run_config,
            node,
            repo_root,
            registry_path,
            submitted_at,
            start_round=run.round + 1,
        )

    shard_map_path = mesh_shared / "metadata" / "shard_map.json"
    assignments_path = (
        mesh_shared / "metadata" / f"shard_assignments_{run_config.run_id}.json"
    )
    if assignments_path.exists():
        assignments = json.loads(assignments_path.read_text())
        if excluded_node in assignments:
            placement = _invert_placement(load_shard_map(shard_map_path))
            for node in picked:
                assignments.setdefault(node, [])
            assignments = reassign_failed(excluded_node, assignments, placement)
            _write_json_atomic(assignments_path, assignments)

    return f"{run_config.run_id}: replenished with {picked} (deficit was {deficit})"


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
    # Pending checkpoints are pushed in fp16 (mesh_round.py's
    # push_checkpoint_and_wait, to halve NFS traffic) -- master.pt must
    # start in fp32 or every later merge step (run_outer_step.py) silently
    # accumulates the outer-optimizer momentum at half precision forever.
    model_state_dict = {
        k: (v.float() if torch.is_floating_point(v) else v)
        for k, v in payload["model_state_dict"].items()
    }
    master_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_io.save(master_path, {"model_state_dict": model_state_dict})


def _uv_bin() -> str:
    """Absolute path to uv, for subprocesses launched from the watcher.

    The watcher's own cron entry invokes ${HOME}/.local/bin/uv by absolute
    path, but `uv run` only prepends the venv's bin dir to PATH -- it does
    not add ~/.local/bin. A bare "uv" here therefore resolves fine from an
    interactive shell and fails with FileNotFoundError on every tick that
    runs from cron, stalling the merge indefinitely (observed 2026-07-27:
    both mesh pools retried MERGING every tick until this was fixed).
    run_mesh_worker.sh guards the same gap by exporting PATH; the watcher
    resolves the binary instead of trusting its inherited environment.
    """
    return shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")


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
        _uv_bin(),
        "run",
        "python",
        "-m",
        "video_encoding.scripts.run_outer_step",
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
        stderr_lines = [line for line in result.stderr.splitlines() if line.strip()]
        last_line = stderr_lines[-1] if stderr_lines else "(no stderr)"
        return (
            f"{run_config.run_id}: MERGING failed (exit {result.returncode}: "
            f"{last_line}) -- retrying next tick"
        )

    for pending_file in pending_dir.glob(f"{run_config.run_id}_*.pt"):
        pending_file.unlink()

    run.status = "SYNCING"
    return f"{run_config.run_id}: MERGING -> SYNCING"


# --- SYNCING -------------------------------------------------------------

node089_HOST = "node089"
node089_RESULTS_ROOT = "/Data/$USER/stern_mesh_results"


def _transfer_results_to_node089(run_config: MeshRunConfig, mesh_shared: Path) -> str:
    """Best-effort: rsync this run's checkpoint dir to node089's local NVMe,
    then delete the NFS copy on success to free quota.

    Never raises -- a failed transfer just leaves the run's results on NFS
    (safe fallback: DONE runs are never re-advanced by advance_all_runs, so
    there is no automatic retry, and losing the only surviving copy of a
    finished run is far worse than leaving it on NFS a while longer).
    """
    local_dir = mesh_shared / "checkpoints" / run_config.run_id
    if not local_dir.exists() or not any(local_dir.iterdir()):
        return f"{run_config.run_id}: transfer skipped -- no local checkpoint dir"

    remote_dir = f"{node089_RESULTS_ROOT}/{run_config.run_id}"
    try:
        mkdir_result = subprocess.run(
            ["ssh", node089_HOST, "mkdir", "-p", remote_dir],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if mkdir_result.returncode != 0:
            return (
                f"{run_config.run_id}: transfer to node089 FAILED "
                f"(mkdir: {mkdir_result.stderr.strip()}) -- results kept on NFS"
            )

        rsync_result = subprocess.run(
            [
                "rsync",
                "-az",
                "--checksum",
                f"{local_dir}/",
                f"{node089_HOST}:{remote_dir}/",
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if rsync_result.returncode != 0:
            return (
                f"{run_config.run_id}: transfer to node089 FAILED "
                f"(rsync exit {rsync_result.returncode}) -- results kept on NFS"
            )

        # rsync exiting 0 doesn't guarantee the destination is intact (e.g.
        # node089 ran out of disk mid-transfer) -- the next step deletes the
        # only other copy, so verify before committing to that.
        check_result = subprocess.run(
            ["ssh", node089_HOST, "test", "-s", f"{remote_dir}/master.pt"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if check_result.returncode != 0:
            return (
                f"{run_config.run_id}: transfer to node089 FAILED "
                f"(post-transfer check) -- results kept on NFS"
            )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"{run_config.run_id}: transfer to node089 FAILED ({exc}) -- results kept on NFS"

    shutil.rmtree(local_dir)
    return (
        f"{run_config.run_id}: results transferred to node089:{remote_dir}, freed NFS"
    )


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
        transfer_note = _transfer_results_to_node089(run_config, mesh_shared)
        return (
            f"{run_config.run_id}: SYNCING -> DONE "
            f"(round {run.round}/{run_config.rounds}); {transfer_note}"
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


def _clean_stale_run_artifacts(run_id: str, mesh_shared: Path) -> list[Path]:
    """Delete sync flags and pending checkpoints left over from a prior
    attempt at this run_id (e.g. after an aborted relaunch).

    Left in place, these silently short-circuit the round-sync logic --
    workers see a stale sync_complete flag and skip loading the merged
    master checkpoint. Hyphen-suffixed collisions (cleaning "pool-fish"
    must not touch "pool-fish-5r"'s flags) are already prevented by the
    glob's literal "_" delimiter, which has no counterpart at that
    position in the sibling run's filename. The flags glob additionally
    uses a digit character class (not a bare trailing `*`) so
    underscore-suffixed collisions are also rejected -- cleaning "pool"
    must not delete "pool_extra"'s flags, which a bare `*` would wrongly
    match. Missing flags/ or pending/ directories are not an error --
    Path.glob on a non-existent directory returns no matches.
    """
    flags_dir = mesh_shared / "checkpoints" / "flags"
    pending_dir = mesh_shared / "checkpoints" / "pending"
    deleted: list[Path] = []
    for path in flags_dir.glob(f"sync_complete_{run_id}_[0-9]*.flag"):
        path.unlink()
        deleted.append(path)
    for path in pending_dir.glob(f"{run_id}_*.pt"):
        path.unlink()
        deleted.append(path)
    return deleted
