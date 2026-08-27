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
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import torch

from video_encoding.mesh.assignment import (
    assign_static,
    reassign_failed,
)
from video_encoding.mesh.pending_files import (
    pending_files_for_round,
)
from video_encoding.mesh.quorum import ready_for_merge
from video_encoding.mesh.topology import NODE_GROUPS
from video_encoding.training import checkpoint as ckpt_io
from video_encoding.training.tb_logger import flush_final_tb_rounds
from video_encoding.watcher.cluster import query_idle_nodes
from video_encoding.watcher.node_health import query_node_health, truly_idle
from video_encoding.watcher.epoch_advance import (
    candidate_replacements,
    eligible_nodes,
    pending_nodes,
    pool_deficit,
    readmit_healthy_nodes,
    record_missed_round,
    select_idle_nodes,
)
from video_encoding.watcher.chunk_relaunch import (
    cancellable_job_ids,
    needs_worker_relaunch,
)
from video_encoding.watcher.registry import (
    JobRecord,
    append_job_record,
    load_registry,
)
from video_encoding.watcher.preemption import starving_nodes
from video_encoding.watcher.squeue import load_squeue_snapshot
from video_encoding.watcher.shard_transfer import copy_shards
from video_encoding.watcher.state import RunState, WatcherState, clear_run_state

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

# Watcher-local scratch for merge state nobody but the watcher reads: the
# Nesterov momentum buffer and the fp16 broadcast's error-feedback residual.
# Both were on NFS, costing 698 MiB of read+write per round per pool for data
# no worker ever opens. The watcher is pinned to one host (login-node's crontab),
# so node-local disk is safe here -- and 26x faster (measured: 3.0 GB/s on
# /Data vs 114 MB/s to the NFS home).
#
# The momentum buffer is still copied back to the run's NFS dir at DONE so
# _transfer_results_to_node089 keeps archiving it; the residual is a pure
# optimization and is deliberately not preserved.
MESH_LOCAL_ROOT_DEFAULT = Path("/Data/$USER/stern_mesh_local")


def _mesh_local_root() -> Path:
    """Overridable via MESH_LOCAL_ROOT so tests and a relocated watcher don't
    have to write to a hardcoded /Data path."""
    return Path(os.environ.get("MESH_LOCAL_ROOT", MESH_LOCAL_ROOT_DEFAULT))


@dataclass(frozen=True)
class MeshRunConfig:
    run_id: str
    experiment: str
    nodes: list[str]  # fixed roster for this run -- same list mesh_ab.sh submitted to
    n_shards: int
    outer_lr: float
    outer_momentum: float
    weighting: str  # "data" | "inverse_loss" | "token"
    rounds: int
    merge: bool = True  # False for the ab-single_node reference arm
    # "fp32" or "fp16" (default) for the master.pt every worker reads back
    # each round. fp16 halves the dominant per-round NFS term (349 MiB/round
    # x 16 nodes) via the unbiased error-feedback path in
    # run_outer_step._quantize_broadcast_with_error_feedback -- correlated
    # rounding across rounds does not accumulate. See
    # run_outer_step.BROADCAST_DTYPE_CHOICES.
    broadcast_dtype: str = "fp16"
    # Quorum merge (spec 2.1): None keeps the old all-or-ROUND_TIMEOUT_S wait
    # -- required for backward compatibility, since every run currently
    # configured on login-node predates this field and has no value for it in
    # its on-disk JSON. Set quorum_k to merge once that many nodes check in,
    # with grace_s as the below-quorum fallback deadline (replaces
    # ROUND_TIMEOUT_S's role for quorum-configured runs).
    quorum_k: int | None = None
    grace_s: float = ROUND_TIMEOUT_S
    # Raw Hydra override strings (no '+'/'mesh.' prefixing assumed --
    # e.g. "train.accum_steps=128") appended verbatim to every
    # _submit_worker_job sbatch call for this run, on both initial
    # assignment and later replenishment. Empty by default since every
    # run registered before this field existed has no value for it in
    # its on-disk JSON. list (not tuple): matches what json.loads produces,
    # so a value round-tripped through mesh_run_configs.json and one
    # constructed directly in Python have the same type.
    extra_overrides: list[str] = field(default_factory=list)
    # Rounds per worker JOB, vs `rounds` for the whole run. None keeps one job
    # per run -- correct for every run registered before the 2026-08-22 cluster
    # rebuild capped SallesInfo at a 2h walltime, and the reason this defaults
    # to None rather than a number: those entries have no value for it on disk.
    # Set it and the watcher relaunches successive chunks at the round boundary
    # (see needs_worker_relaunch), each resuming through master.pt.
    chunk_rounds: int | None = None


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
        # Loop guard: a handler may report an action *without* advancing the
        # state machine -- _do_assigning does exactly that when every node is
        # excluded, returning "no eligible nodes" and deliberately leaving the
        # status at ASSIGNING for a human to clear. Neither exit condition
        # below is then ever reached, so on 2026-08-21 this spun forever,
        # appending one string per iteration until the watcher had grown to
        # 61 GB and made node089 unreachable. Revisiting a (status, round) pair
        # means the last call changed nothing, so stop: the action is still
        # reported once, but the run waits for the next tick instead of
        # burning this one.
        seen: set[tuple[str, int]] = set()
        while state.run(run_config.run_id).status != "DONE":
            run = state.run(run_config.run_id)
            key = (run.status, run.round)
            if key in seen:
                break
            seen.add(key)
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


def _relaunch_chunk_if_finished(
    run_config: MeshRunConfig,
    run: RunState,
    state: WatcherState,
    repo_root: Path,
    now: float,
) -> str | None:
    """Submit the next chunk once every worker job for this run has exited.

    Only for chunk_rounds runs: without it a worker spans the whole run and
    its disappearance means a real failure, which tick.py's dead-job
    detection already owns. Here disappearance is the expected end of a
    chunk, and the round to resume from is simply the one the run is on --
    master.pt already carries every merge up to it.

    Excluded nodes are left out rather than given a fresh job: exclusion is
    deliberately not auto-reversed (a human clears excluded_nodes), and
    _recover_pool is the one path allowed to readmit.
    """
    if run_config.chunk_rounds is None:
        return None

    state_dir = repo_root / "runs" / "state"
    snapshot_path = state_dir / "squeue.json"
    registry_path = state_dir / "jobs_registry.json"
    if not needs_worker_relaunch(
        run_id=run_config.run_id,
        status=run.status,
        registry=load_registry(registry_path),
        snapshot=load_squeue_snapshot(snapshot_path),
        snapshot_available=snapshot_path.exists(),
        now=now,
    ):
        return None

    nodes = [n for n in run_config.nodes if n not in state.excluded_nodes]
    if not nodes:
        return f"{run_config.run_id}: chunk finished but every node is excluded"

    submitted_at = datetime.now(timezone.utc).isoformat()
    submitted = [
        node
        for node in nodes
        if _submit_worker_job(
            run_config,
            node,
            repo_root,
            registry_path,
            submitted_at,
            start_round=run.round,
        )
        is not None
    ]
    failed = [n for n in nodes if n not in submitted]
    note = (
        f"{run_config.run_id}: chunk done -- relaunched {len(submitted)}/"
        f"{len(nodes)} worker(s) from round {run.round}"
    )
    if failed:
        note += f"; sbatch FAILED for {', '.join(failed)}"
    return note


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
    # Before dispatching: a chunked run whose workers have all exited needs
    # its next chunk submitted, and that is true in RUNNING and in
    # AWAITING_CHECKPOINTS alike, so it does not belong in either handler.
    chunk_note = _relaunch_chunk_if_finished(run_config, run, state, repo_root, now)
    if chunk_note is not None:
        return chunk_note
    if run.status == "ASSIGNING":
        return _do_assigning(run_config, run, state, repo_root, mesh_shared, now)
    if run.status == "RUNNING":
        # Self-heal before checking on the round. RUNNING is where a run
        # spends nearly all its wall-clock, and _recover_pool throttles
        # itself, so this is the cheapest place to give excluded-but-
        # recovered nodes a way back in. Its note is returned only when
        # _do_running has nothing to say -- advance_all_runs' (status,
        # round) loop guard then stops after reporting it once.
        recovery_note = _recover_pool(
            run_config, run, state, mesh_shared, repo_root, run_configs_path, now=now
        )
        running_note = _do_running(run_config, run, mesh_shared)
        if running_note and recovery_note:
            return f"{recovery_note}; {running_note}"
        return running_note or recovery_note
    if run.status == "AWAITING_CHECKPOINTS":
        return _do_awaiting_checkpoints(
            run_config, run, state, mesh_shared, now, repo_root, run_configs_path
        )
    if run.status == "MERGING":
        return _do_merging(run_config, run, repo_root, mesh_shared)
    if run.status == "SYNCING":
        return _do_syncing(run_config, run, state, mesh_shared, repo_root, now)
    return None  # DONE, or an unrecognized status -- nothing to do


# --- ASSIGNING -------------------------------------------------------------


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
    # '++' (add-or-override), not '+': videomae_mesh_ssl.yaml defines no
    # chunk_rounds key, but the paper-recipe experiment configs do, and '+'
    # aborts composition on an existing key -- which would kill every worker
    # of those runs at launch. Omitted entirely when unset so the worker's
    # own None default applies.
    if run_config.chunk_rounds is not None:
        cmd.append(f"++mesh.chunk_rounds={run_config.chunk_rounds}")
    cmd.extend(run_config.extra_overrides)
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


def _verified_holdings(mesh_shared: Path) -> dict[str, list[str]]:
    """`{node: [shard, ...]}` of what each node *physically holds*.

    Read from verified_holdings.json, which validate_seeding.py writes from
    a live `ls` of every node. This is the mesh's only source of truth about
    shard location. The old shard_map.json was a placement *plan* -- what
    seeding intended -- and was never reconciled with what seeding achieved;
    by 2026-08-21 it was 1062/1113 false, which is what let a node be handed
    89 shards it did not have.

    Raises FileNotFoundError if the pool has never been validated.
    """
    holdings_path = mesh_shared / "metadata" / "verified_holdings.json"
    if not holdings_path.exists():
        raise FileNotFoundError(
            f"{holdings_path} missing -- run validate_seeding.py before launching. "
            "Shard location is never inferred from a placement plan."
        )
    # Keyed by pool; a run's roster may span only part of one, so flatten.
    by_node: dict[str, list[str]] = {}
    for pool_holdings in json.loads(holdings_path.read_text()).values():
        for node, shards in pool_holdings.items():
            by_node[node] = list(shards)
    return by_node


def _static_assignments(
    nodes: list[str], mesh_shared: Path
) -> tuple[dict[str, list[str]], str]:
    """Partition the shards `nodes` actually hold, from verified_holdings.json.

    The expected set is the union of this roster's own verified holdings
    intersected with the golden manifest -- not the full 260. A roster that
    is a subset of its tier trains what that subset collectively has, and the
    resulting coverage is reported so a degraded round is visible rather than
    silent. Asking for shards nobody in the roster holds is what produced the
    2026-08-20 incident, so it is not expressible here.

    Raises FileNotFoundError if the pool has never been validated -- an
    unvalidated pool must not launch.
    """
    manifest_path = mesh_shared / "metadata" / "shard_manifest.json"
    by_node = _verified_holdings(mesh_shared)
    roster = {node: by_node.get(node, []) for node in nodes}

    golden: set[str] | None = None
    if manifest_path.exists():
        golden = set(json.loads(manifest_path.read_text()).get("train", {}))

    expected: set[str] = set()
    for shards in roster.values():
        expected |= set(shards)
    if golden is not None:
        expected &= golden

    assignments = assign_static(roster, expected)
    total = len(golden) if golden else len(expected)
    note = f"coverage {len(expected)}/{total}" if total else "coverage 0/0"
    return assignments, note


def _do_assigning(
    run_config: MeshRunConfig,
    run: RunState,
    state: WatcherState,
    repo_root: Path,
    mesh_shared: Path,
    now: float,
) -> str | None:
    eligible = eligible_nodes(state, run_config.nodes)
    if not eligible:
        # Every node in this run's roster has been excluded (crash-loop) --
        # nothing to assign to. Leave status as ASSIGNING; a human clearing
        # excluded_nodes is the only way forward (epoch_advance.exclude_node
        # is never auto-reversed).
        return f"{run_config.run_id}: ASSIGNING skipped -- no eligible nodes"

    deleted = _clean_stale_run_artifacts(run_config.run_id, mesh_shared)

    assignments, coverage_note = _static_assignments(eligible, mesh_shared)

    # A node holding nothing would train zero samples and stall the round --
    # exactly the 2026-08-20 failure. Drop it from this round instead.
    starved = [node for node in eligible if not assignments.get(node)]
    if starved:
        eligible = [node for node in eligible if assignments.get(node)]
        assignments = {node: assignments[node] for node in eligible}
    if not eligible:
        return (
            f"{run_config.run_id}: ASSIGNING skipped -- no eligible node holds "
            "any shard (pool is unseeded)"
        )

    assignments_path = (
        mesh_shared / "metadata" / f"shard_assignments_{run_config.run_id}.json"
    )
    _write_json_atomic(assignments_path, assignments)

    registry_path = repo_root / "runs" / "state" / "jobs_registry.json"
    submitted_at = datetime.now(timezone.utc).isoformat()
    # Report what sbatch ACCEPTED, not what we intended to submit. A single
    # failed submission used to vanish here -- the return value was discarded
    # and the note below counted `eligible` -- so on 2026-08-22 a pool logged
    # "16 nodes" while 15 jobs existed. With quorum_k == pool size that one
    # missing worker cannot be waited out: every round runs to the full
    # grace_s and then merges timeout-forced, at 30 minutes a round.
    failed_submit = [
        node
        for node in eligible
        if _submit_worker_job(run_config, node, repo_root, registry_path, submitted_at)
        is None
    ]

    run.status = "RUNNING"
    run.round_started_at = now
    action = (
        f"{run_config.run_id}: ASSIGNING -> RUNNING "
        f"({len(eligible) - len(failed_submit)}/{len(eligible)} nodes submitted)"
    )
    if failed_submit:
        action += f"; sbatch FAILED for {', '.join(failed_submit)}"
    action += f"; {coverage_note}"
    if starved:
        action += f"; skipped {len(starved)} node(s) holding no shards: " + ", ".join(
            starved
        )
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
    if pending_nodes(pending_dir, run_config.run_id, round_num=run.round):
        run.status = "AWAITING_CHECKPOINTS"
        note = f"{run_config.run_id}: RUNNING -> AWAITING_CHECKPOINTS"
        # Once per round, not once per tick: this transition fires exactly
        # once a round, and a starving node is a standing condition that
        # would otherwise repeat every 60s for hours.
        starving = starving_nodes(mesh_shared / "metrics", run_config.run_id)
        if starving:
            detail = ", ".join(f"{n} ({c} rounds)" for n, c in sorted(starving.items()))
            note += (
                f"; merged around without contributing: {detail}"
                " -- holding a node for nothing, check its CPUs"
            )
        return note
    return None  # no-op -- job health is tick.py's job, not this module's


# --- AWAITING_CHECKPOINTS ---------------------------------------------------


def _round_elapsed_s(
    pending_dir: Path, run_id: str, round_num: int, now: float, round_started_at: float
) -> float:
    """Seconds since this round's earliest checkpoint arrived, or since the
    round started if none have arrived yet.

    Scoped to `round_num` (mesh/pending_files.pending_files_for_round) so a
    straggler finishing the PRIOR round -- who can write into pending/ after
    a timeout-forced merge clears it (mesh_tick._do_merging), before this
    round's real checkpoints land -- can't anchor this clock to its own
    early arrival. mtime can't make that distinction on its own: the
    straggler necessarily writes in real wall-clock time AFTER this round
    already started. If the whole pool dies before a single current-round
    node checks in, there are no files to read mtimes from at all --
    falling back to `now - round_started_at` is what lets the
    AWAITING_CHECKPOINTS timeout still fire in that case too, instead of
    reading a permanent 0.0.
    """
    files = pending_files_for_round(pending_dir, run_id, round_num)
    if not files:
        return now - round_started_at
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
    checked_in = pending_nodes(pending_dir, run_config.run_id, round_num=run.round)
    eligible = set(eligible_nodes(state, run_config.nodes))

    if checked_in >= eligible:
        # quorum_k is a floor on how many nodes may contribute to a merge, but
        # this fast path was written for the happy case and skipped it: once
        # permanent exclusions shrink `eligible` below quorum_k, "everyone
        # checked in" is trivially true and the merge proceeds anyway.
        # factorial-tube-per_frame merged its last four rounds 10-wide against
        # quorum_k=12, logged as a clean "MERGING (10 nodes)", and still
        # reported DONE 40/40. Halt instead: a run that reaches DONE has to
        # mean its configured floor held every round.
        if run_config.quorum_k is not None and len(checked_in) < run_config.quorum_k:
            run.status = "DEGRADED"
            return (
                f"{run_config.run_id}: AWAITING_CHECKPOINTS -> DEGRADED "
                f"(roster collapsed to {len(eligible)} eligible, "
                f"{len(checked_in)} checked in, below quorum_k="
                f"{run_config.quorum_k}) -- halted rather than merge below "
                "the configured floor; clear excluded_nodes to resume"
            )
        run.status = "MERGING"
        run.timeout_forced = False
        return (
            f"{run_config.run_id}: AWAITING_CHECKPOINTS -> MERGING "
            f"({len(checked_in)} nodes)"
        )

    elapsed = _round_elapsed_s(
        pending_dir, run_config.run_id, run.round, now, run.round_started_at
    )

    if run_config.quorum_k is not None:
        if not ready_for_merge(
            checked_in,
            quorum_k=run_config.quorum_k,
            grace_s=run_config.grace_s,
            now=elapsed,
        ):
            return None  # still below quorum, grace not yet expired
        if len(checked_in) >= run_config.quorum_k:
            # Quorum reached before grace expired: merge now with whoever's
            # here. No exclusion, no missed_rounds bump -- the stragglers
            # haven't crashed, they just haven't finished yet, and get
            # another chance next round. Still a partial merge from
            # run_outer_step's point of view, so timeout_forced stays True
            # (2.2: run_config.weighting="token" -- run_outer_step.py's
            # _token_weights -- corrects the bias this used to carry via
            # plain renormalization; run quorum-configured pools with it).
            run.status = "MERGING"
            run.timeout_forced = True
            return (
                f"{run_config.run_id}: AWAITING_CHECKPOINTS -> MERGING "
                f"(quorum {run_config.quorum_k}/{len(eligible)} reached, "
                f"{len(checked_in)} nodes)"
            )
        # Grace expired without reaching quorum -- fall through to the same
        # exclusion/replenish path the plain timeout uses below.
    elif elapsed < ROUND_TIMEOUT_S:
        return None  # still waiting, not timed out yet

    missing = sorted(eligible - checked_in)
    # ASSIGNING (and therefore assign_locality_first) only runs once per run
    # now (see SYNCING -> RUNNING, not ASSIGNING, above), so _replenish_pool
    # is the only place that redistributes an excluded node's shards to a
    # survivor afterward -- it always does so, even when no replacement
    # node is found, so shard coverage does not silently shrink.
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
    excluded_note = f", permanently excluded {excluded}" if excluded else ""
    # Same floor as the all-checked-in path above, applied after the
    # exclusion/replenish bookkeeping so the log still records what went wrong
    # and a replacement still gets submitted for a human to pick up from.
    if run_config.quorum_k is not None and len(checked_in) < run_config.quorum_k:
        run.status = "DEGRADED"
        degraded = (
            f"{run_config.run_id}: AWAITING_CHECKPOINTS -> DEGRADED "
            f"(grace expired with {len(checked_in)} checked in, below "
            f"quorum_k={run_config.quorum_k}, missing {missing}{excluded_note})"
            " -- halted rather than merge below the configured floor"
        )
        if replenish_notes:
            degraded += "; " + "; ".join(replenish_notes)
        return degraded
    run.status = "MERGING"
    run.timeout_forced = True
    result = (
        f"{run_config.run_id}: AWAITING_CHECKPOINTS -> MERGING "
        f"(timeout, missing {missing}{excluded_note})"
    )
    if replenish_notes:
        result += "; " + "; ".join(replenish_notes)
    return result


# --- periodic pool recovery ---------------------------------------------

# _recover_pool ssh-probes every excluded roster node, so it must not run on
# every one-minute cron tick. A node whose GPU a foreign process grabbed frees
# up on that process's schedule, not ours; five minutes is far below any round
# length and cheap enough to repeat for hours.
RECOVERY_INTERVAL_S = 300.0


def _recover_pool(
    run_config: MeshRunConfig,
    run: RunState,
    state: WatcherState,
    mesh_shared: Path,
    repo_root: Path,
    run_configs_path: Path | None,
    *,
    now: float,
) -> str | None:
    """Put recovered nodes back to work, independent of any fresh failure.

    Nothing else in the system does this. tick.py filters mesh jobs out of
    detect_dead before remediate_dead_job sees them (it hardcodes
    train.sbatch, which cannot run a mesh worker), on the stated grounds that
    mesh_tick owns recovery -- but mesh_tick's _replenish_pool fires only from
    the branch where `record_missed_round` returns True, which happens exactly
    once per node, on the tick it crosses the exclusion threshold. Once every
    node that is going to fail has failed, no new exclusion occurs, nothing
    calls _replenish_pool, and the pool is frozen at its degraded size for the
    rest of the run. factorial-tube-per_frame finished its last four rounds
    with 10 of 22 workers while twelve healthy machines sat idle beside it.

    So the pool could only ever shrink. This is the restoring term: every
    RECOVERY_INTERVAL_S, re-probe this run's own excluded nodes and put back
    the ones that are healthy again.

    A recovered node is only resubmitted if it can be handed shards nobody is
    training -- normally the ones dropped when it was excluded, which it still
    physically holds. Submitting it with an empty assignment would raise in
    read_my_shards and re-exclude it, so an under-filled pool stays the safe
    failure mode here exactly as it is in _replenish_pool.
    """
    if run_configs_path is None:
        return None
    if now - run.last_recovery_at < RECOVERY_INTERVAL_S:
        return None
    run.last_recovery_at = now

    roster = _live_roster(run_configs_path, run_config.run_id, run_config.nodes)
    if pool_deficit(run, roster, state) <= 0:
        return None
    recoverable = [
        node
        for node in roster
        if state.excluded_nodes.get(node, "").endswith(f"of {run_config.run_id}")
    ]
    if not recoverable:
        return None

    assignments_path = (
        mesh_shared / "metadata" / f"shard_assignments_{run_config.run_id}.json"
    )
    if not assignments_path.exists():
        return None

    idle = query_idle_nodes()
    healthy = truly_idle(
        query_node_health([n for n in recoverable if n in idle], idle=idle)
    )
    if not healthy:
        return None

    assignments = json.loads(assignments_path.read_text())
    trained = {shard for shards in assignments.values() for shard in shards}
    placement = _verified_holdings(mesh_shared)

    registry_path = repo_root / "runs" / "state" / "jobs_registry.json"
    submitted_at = datetime.now(timezone.utc).isoformat()
    restored: list[str] = []
    regained = 0
    for node in sorted(healthy):
        own = [s for s in placement.get(node, []) if s not in trained]
        if not own:
            continue  # nothing to give it -- see docstring
        if not readmit_healthy_nodes(state, run_config.run_id, {node}):
            continue  # already used up its one re-admission
        assignments[node] = own
        trained.update(own)
        _submit_worker_job(
            run_config,
            node,
            repo_root,
            registry_path,
            submitted_at,
            start_round=run.round + 1,
        )
        restored.append(node)
        regained += len(own)

    if not restored:
        return None
    _write_json_atomic(assignments_path, assignments)
    return (
        f"{run_config.run_id}: recovered {restored} back into the pool "
        f"({regained} shard(s) training again)"
    )


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
    describing what happened, or None if there was nothing to do at all
    (no run_configs_path and no shard reassignment to make either).

    Shard reassignment (below) always runs when possible, independent of
    whether a replacement node is found: assign_locality_first only runs
    once per run (at ASSIGNING), so this is the only place that ever
    updates shard_assignments_<run_id>.json again. Returning early before
    reaching it -- as this function previously did whenever deficit <= 0
    or no idle node was available -- left excluded_node's shards assigned
    to a node that will never train again, silently shrinking shard
    coverage for the rest of the run.

    A picked replacement only ever receives shards via excluded_node's own
    handoff below -- if excluded_node has no entry (or an empty one) in
    shard_assignments, a submitted replacement would start with nothing to
    train on and crash immediately on read_my_shards, which then excludes
    IT too and repeats the pick with the next idle node. Observed live on
    login-node 2026-08-05 (pool-cars-2r, node066/node051 both submitted with
    zero shard_assignments entries), so replenishment is skipped rather
    than submitted in that case -- an under-filled pool is a safe failure
    mode, a guaranteed-crash job is not. A missing assignments file
    entirely (no shard-tracking started for this run yet) does not gate
    replenishment; only a present file where excluded_node's own entry is
    absent or empty does.
    """
    picked: list[str] = []
    replenish_note: str | None = None

    assignments_path = (
        mesh_shared / "metadata" / f"shard_assignments_{run_config.run_id}.json"
    )
    assignments = (
        json.loads(assignments_path.read_text()) if assignments_path.exists() else None
    )
    blocks_replenish = assignments is not None and not assignments.get(excluded_node)

    if run_configs_path is not None:
        roster = _live_roster(run_configs_path, run_config.run_id, run_config.nodes)
        deficit = pool_deficit(run, roster, state)
        if deficit > 0:
            tier = NODE_GROUPS[roster[0]]
            tier_roster = [h for h, g in NODE_GROUPS.items() if g == tier]
            idle = query_idle_nodes()
            candidates = candidate_replacements(
                tier_roster, in_use=set(roster), excluded=set(state.excluded_nodes)
            )
            # sinfo alone is not readiness: on 2026-08-21 it called all 12
            # excluded birds nodes "idle" while 7 had ~21.7 GiB of their GPU
            # held by a foreign process and 3 had never been seeded. All 6
            # picked from that set crashed on submit (CUDA OOM /
            # FileNotFoundError on manifest.json) and were themselves excluded
            # two rounds later. Probe the GPU and the seed before submitting;
            # only SLURM-idle candidates are worth an ssh round-trip, since
            # truly_idle requires slurm_idle anyway.
            # This run's own previously-excluded roster members are the best
            # replacements available: unlike a fresh tier node they still hold
            # their shards, so recovering one costs no transfer at all. Probe
            # them alongside the fresh candidates.
            recoverable = [
                node
                for node in roster
                if state.excluded_nodes.get(node, "").endswith(
                    f"of {run_config.run_id}"
                )
            ]
            healthy = truly_idle(
                query_node_health(
                    [
                        n
                        for n in dict.fromkeys([*candidates, *recoverable])
                        if n in idle
                    ],
                    idle=idle,
                )
            )
            picked = select_idle_nodes(candidates, deficit, healthy)
            if len(picked) < deficit:
                # Re-admit only the ones we are about to submit a job for.
                # Clearing an exclusion without submitting would put the node
                # back into _do_awaiting_checkpoints' `eligible` set with
                # nothing running on it, so every round would wait out the
                # grace period for a checkpoint that cannot arrive and then
                # re-exclude it -- strictly worse than leaving it excluded.
                wanted = set(
                    select_idle_nodes(recoverable, deficit - len(picked), healthy)
                )
                readmitted = readmit_healthy_nodes(state, run_config.run_id, wanted)
                picked = [*picked, *readmitted]
            if picked and blocks_replenish:
                replenish_note = (
                    f"{run_config.run_id}: replenish skipped for {excluded_node} "
                    f"-- it had no shards to hand off a replacement "
                    f"({picked} not submitted)"
                )
                picked = []
            elif picked:
                _append_nodes_to_run_config(run_configs_path, run_config.run_id, picked)

                registry_path = repo_root / "runs" / "state" / "jobs_registry.json"
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
                replenish_note = (
                    f"{run_config.run_id}: replenished with {picked} "
                    f"(deficit was {deficit})"
                )
            else:
                replenish_note = (
                    f"{run_config.run_id}: replenish skipped -- no idle {tier} nodes "
                    f"available (deficit {deficit})"
                )

    reassign_note = None
    if assignments is not None and excluded_node in assignments:
        placement = _verified_holdings(mesh_shared)
        for node in picked:
            assignments.setdefault(node, [])
        assignments, dropped = reassign_failed(excluded_node, assignments, placement)
        reassign_note = (
            f"{run_config.run_id}: reassigned {excluded_node}'s shards to survivors"
        )
        if dropped and picked:
            # No survivor holds these, but excluded_node still does -- it was
            # excluded because it could not RUN a worker (foreign process on
            # its GPU, or never seeded), not because it went away. It stays up
            # and reachable, so copying the files to the replacement turns a
            # permanent coverage loss back into a transfer. Replication is 1x
            # for 40-68% of the corpus on every tier but fish, so this is the
            # common path, not an edge case: it is what dropped 46 of birds'
            # 260 shards on 2026-08-21.
            recovered = copy_shards(excluded_node, picked[0], dropped)
            if recovered:
                assignments[picked[0]] = assignments.get(picked[0], []) + recovered
                dropped = [s for s in dropped if s not in recovered]
                reassign_note += (
                    f"; recovered {len(recovered)} shard(s) to {picked[0]} "
                    f"({', '.join(recovered[:5])}"
                    f"{' ...' if len(recovered) > 5 else ''})"
                )
        if dropped:
            # Coverage genuinely shrank: no survivor holds these and the copy
            # did not land them. Say so -- the alternative is handing them to
            # a node that cannot open them, which raises inside the worker and
            # gets IT excluded two rounds later.
            reassign_note += (
                f"; dropped {len(dropped)} shard(s) no survivor holds "
                f"({', '.join(dropped[:5])}{' ...' if len(dropped) > 5 else ''})"
            )
        _write_json_atomic(assignments_path, assignments)

    notes = [n for n in (replenish_note, reassign_note) if n]
    return "; ".join(notes) if notes else None


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

    Scoped to round 0 (mesh/pending_files.pending_files_for_round): nothing
    else leaves master.pt missing, so the very first merge is always round
    0's -- and scoping here for the same reason every other pending/ reader
    now does (see _round_elapsed_s) rather than trusting an unscoped glob.
    """
    candidates = sorted(pending_files_for_round(pending_dir, run_id, 0))
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
    # master_version=0 predates every real round_num merge_round stamps
    # (Phase 3.2), so a worker's first real diff is never rejected as stale
    # by comparison to this bootstrap master.
    ckpt_io.save(
        master_path, {"model_state_dict": model_state_dict, "master_version": 0}
    )


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
    local_dir = _mesh_local_root() / run_config.run_id
    momentum_path = local_dir / "outer_momentum.pt"
    residual_path = local_dir / "broadcast_residual.pt"
    tb_dir = local_dir / "tb"
    local_dir.mkdir(parents=True, exist_ok=True)
    if not master_path.exists():
        _bootstrap_master(pending_dir, run_config.run_id, master_path)
    cmd = [
        _uv_bin(),
        "run",
        # Without this, every merge re-syncs the 9.4 GB NFS-hosted venv
        # before running -- `uv sync --dry-run` reports an unconditional
        # reinstall even against an up-to-date lockfile, and this cron runs
        # once a minute.
        "--no-sync",
        "python",
        "-m",
        "video_encoding.scripts.run_outer_step",
        "--pending-dir",
        str(pending_dir),
        # pending/ is one flat dir shared by every concurrent run; without
        # this the merge averages sibling and already-finished runs in.
        "--run-id",
        run_config.run_id,
        "--master",
        str(master_path),
        "--momentum",
        str(momentum_path),
        "--residual",
        str(residual_path),
        "--broadcast-dtype",
        run_config.broadcast_dtype,
        "--round",
        str(run.round),
        "--outer-lr",
        str(run_config.outer_lr),
        "--outer-momentum",
        str(run_config.outer_momentum),
        "--weighting",
        run_config.weighting,
        "--metrics-out",
        str(
            repo_root
            / "runs"
            / "results"
            / "mesh"
            / f"{run_config.run_id}_rounds.jsonl"
        ),
        "--tb-dir",
        str(tb_dir),
    ]
    if run.timeout_forced:
        cmd.append("--timeout-forced")
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


def _cancel_run_workers(run_id: str, repo_root: Path) -> str:
    """scancel any worker still running for a run that has reached DONE.

    The state machine has never cancelled anything -- a worker exits when its
    own round loop ends, which for a straggler is well after the run stopped
    needing it. Every round it still computes is merged by nobody and swept
    by _sweep_pending_checkpoints, so this is pure waste, and on a cluster
    where a whole node is allocated per worker it is waste other people's
    jobs are queued behind.

    Best-effort by design: a failed scancel must not stop a run from
    reporting DONE, transferring its results, or releasing its state. The
    reason lands in the returned note either way -- a silent cancel is how
    we would end up believing nodes are free when they are not.
    """
    state_dir = repo_root / "runs" / "state"
    job_ids = cancellable_job_ids(
        run_id=run_id,
        registry=load_registry(state_dir / "jobs_registry.json"),
        snapshot=load_squeue_snapshot(state_dir / "squeue.json"),
    )
    if not job_ids:
        return "no stragglers to cancel"
    try:
        result = subprocess.run(
            ["scancel", *job_ids],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"scancel of {len(job_ids)} straggler(s) failed to run ({exc})"
    if result.returncode != 0:
        return (
            f"scancel of {len(job_ids)} straggler(s) exited "
            f"{result.returncode}: {result.stderr.strip()}"
        )
    return f"cancelled {len(job_ids)} straggling worker(s): {', '.join(job_ids)}"


def _sweep_pending_checkpoints(run_id: str, mesh_shared: Path) -> int:
    """Delete this run's pending checkpoints once it reaches DONE.

    MERGING already clears them after each successful merge, but a run that
    finishes on a *timed-out* round leaves behind whatever arrived late, and
    nothing else ever collects it: _clean_stale_run_artifacts only fires on
    run_id *reuse*, so a new run_id never sweeps a dead one's leftovers.
    Observed in production: 15 pool-birds-5r checkpoints (2.6 GiB against a
    30 GiB quota) still in pending/ hours after that run went DONE, where
    the next run's merge would have averaged them in.
    """
    pending_dir = mesh_shared / "checkpoints" / "pending"
    swept = 0
    for path in pending_dir.glob(f"{run_id}_*.pt"):
        path.unlink(missing_ok=True)
        swept += 1
    return swept


def sweep_orphaned_pending(
    pending_dir: Path, active_run_ids: set[str], *, max_age_s: float
) -> int:
    """Delete pending/ checkpoints whose run_id is not among the currently
    configured runs, once they are older than max_age_s.

    Catches what _sweep_pending_checkpoints cannot: checkpoints from a run
    that never reached SYNCING/DONE at all (killed mid-run, or dropped from
    mesh_run_configs.json by hand) -- observed in production as 9.3 GiB of
    orphans from five dead runs, against a 30 GiB NFS quota.

    Critical guard: load_run_configs returns [] when its file is missing, so
    an empty active_run_ids means "unknown", not "nothing is active" -- this
    must no-op rather than delete every live checkpoint in pending/.
    """
    if not active_run_ids or not pending_dir.exists():
        return 0
    now = time.time()
    swept = 0
    for path in pending_dir.glob("*.pt"):
        if any(path.name.startswith(f"{run_id}_") for run_id in active_run_ids):
            continue
        if now - path.stat().st_mtime < max_age_s:
            continue
        path.unlink(missing_ok=True)
        swept += 1
    return swept


def _stage_local_merge_state_for_archive(
    run_config: MeshRunConfig, mesh_shared: Path
) -> None:
    """Copy the watcher-local momentum buffer into the run's NFS dir so the
    existing node089 transfer still archives it.

    The buffer moved to local disk to keep it off the per-round NFS budget,
    but it is part of a run's reproducible result (it was in every previous
    archive) -- so it comes back exactly once, at DONE. The error-feedback
    residual deliberately does not: it is a transient compression aid, not a
    result.
    """
    momentum_path = _mesh_local_root() / run_config.run_id / "outer_momentum.pt"
    if not momentum_path.exists():
        return
    dest_dir = mesh_shared / "checkpoints" / run_config.run_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(momentum_path, dest_dir / "outer_momentum.pt")


def _do_syncing(
    run_config: MeshRunConfig,
    run: RunState,
    state: WatcherState,
    mesh_shared: Path,
    repo_root: Path,
    now: float,
) -> str:
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
        # No merge cycle is coming next to flush the round(s) that
        # write_mesh_round_scalars's straggler holdback always leaves
        # pending on a live run -- final=True catches those up now.
        flush_final_tb_rounds(
            _mesh_local_root() / run_config.run_id / "tb",
            mesh_shared,
            run_config.run_id,
        )
        _stage_local_merge_state_for_archive(run_config, mesh_shared)
        transfer_note = _transfer_results_to_node089(run_config, mesh_shared)
        # Before the sweep: a straggler cancelled here cannot push one more
        # checkpoint into pending/ after _sweep_pending_checkpoints has run,
        # which would leak it until sweep_orphaned_pending's age cutoff.
        cancel_note = _cancel_run_workers(run_config.run_id, repo_root)
        swept = _sweep_pending_checkpoints(run_config.run_id, mesh_shared)
        clear_run_state(state, run_config.run_id)
        return (
            f"{run_config.run_id}: SYNCING -> DONE "
            f"(round {run.round}/{run_config.rounds}); {transfer_note}; "
            f"{cancel_note}; swept {swept} pending checkpoint(s)"
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
    run.round_started_at = now
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
