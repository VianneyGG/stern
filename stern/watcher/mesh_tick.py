"""Round-state driver for the decentralized P2P training mesh.

Runs on every cron tick. Owns the ASSIGNING -> RUNNING -> AWAITING_CHECKPOINTS
-> MERGING -> SYNCING -> DONE state machine for each configured mesh run: it
submits one sbatch job per node, runs the DiLoCo outer-step merge once every
eligible node has checked in (or the round times out), and writes the sync
flags the worker-side round loop polls for.

Job health (dead/stalled detection, resubmission, escalation) belongs to
stern/watcher/tick.py. Mesh jobs are appended to the same jobs_registry.json
so that machinery covers them; this module only tracks round progress.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Collection
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import torch

from stern.cluster_env import data_root, setting
from stern.mesh.outer_step import BOOTSTRAP_MASTER_VERSION
from stern.mesh.assignment import (
    assign_static,
    reassign_failed,
)
from stern.mesh.pending_files import (
    pending_files_for_round,
)
from stern.mesh.quorum import ready_for_merge
from stern.mesh.topology import NODE_GROUPS
from stern.watcher import class_hours
from stern.training import checkpoint as ckpt_io
from stern.training.tb_logger import flush_final_tb_rounds
from stern.watcher.cluster import (
    query_idle_nodes,
    query_job_states,
    query_node_states,
)
from stern.watcher.detectors import detect_drained_nodes
from stern.watcher.notify import notify
from stern.watcher.node_health import query_node_health, truly_idle
from stern.watcher.epoch_advance import (
    MISSED_ROUND_THRESHOLD,
    candidate_replacements,
    clear_missed_rounds,
    eligible_nodes,
    exclude_node,
    missed_round_count,
    pending_nodes,
    pool_deficit,
    readmit_healthy_nodes,
    select_idle_nodes,
)
from stern.watcher.chunk_relaunch import (
    cancellable_job_ids,
    needs_worker_relaunch,
    stale_jobs_for_nodes,
)
from stern.watcher.registry import (
    JobRecord,
    append_job_record,
    load_registry,
)
from stern.watcher.preemption import starving_nodes
from stern.watcher.squeue import load_squeue_snapshot
from stern.watcher.shard_transfer import copy_shards
from stern.watcher.state import RunState, WatcherState, clear_run_state

# Must stay well below the workers' per-round sync timeout (mesh.sync_timeout_s,
# 1800s). A worker that already pushed its checkpoint waits that long at the
# barrier; the watcher's force-merge has to fire before it gives up. The gap
# leaves room for tick cadence and merge overhead. A partial round merges
# whatever arrived.
ROUND_TIMEOUT_S = 900.0

# Watcher-local scratch for merge state only the watcher reads: the Nesterov
# momentum buffer and the fp16 broadcast's error-feedback residual. Keeping
# them on NFS cost ~700 MiB of I/O per round per pool. The watcher is pinned
# to one host, so node-local disk is safe there.
#
# The momentum buffer is copied back to the run's NFS dir at DONE so it is
# still archived. The residual is a pure optimization and is not preserved.
MESH_LOCAL_ROOT_DEFAULT = data_root() / "stern_mesh_local"


def _mesh_local_root() -> Path:
    """Local scratch root. Overridable via MESH_LOCAL_ROOT."""
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
    # each round. fp16 uses error feedback, so rounding errors do not
    # accumulate across rounds.
    broadcast_dtype: str = "fp16"
    # None keeps the all-or-ROUND_TIMEOUT_S wait. Set quorum_k to merge once
    # that many nodes check in; grace_s is the fallback deadline below quorum.
    quorum_k: int | None = None
    grace_s: float = ROUND_TIMEOUT_S
    # Raw Hydra overrides (e.g. "train.accum_steps=128"), appended verbatim to
    # every worker sbatch call for this run. A list, to match json.loads output.
    extra_overrides: list[str] = field(default_factory=list)
    # Rounds per worker job. None runs one job for the whole run. Otherwise the
    # watcher relaunches the next chunk at each round boundary (see
    # needs_worker_relaunch), and each chunk resumes from master.pt.
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


def _exclude_drained_nodes(
    run_configs: list[MeshRunConfig],
    state: WatcherState,
    node_states: dict[str, str],
) -> list[str]:
    """Exclude roster nodes that SLURM has drained.

    Otherwise sbatch keeps failing against them on every relaunch. node_states
    is one batched sinfo call made by the caller. Already-excluded nodes are
    skipped so notify() does not refire each tick, and DONE runs are skipped.
    """
    actions: list[str] = []
    for run_config in run_configs:
        if state.run(run_config.run_id).status == "DONE":
            continue
        roster = [n for n in run_config.nodes if n not in state.excluded_nodes]
        for flag in detect_drained_nodes(node_states, roster=roster):
            node = flag.subject.split(":", 1)[1]
            exclude_node(
                state, node, f"slurm drained ({flag.detail}), of {run_config.run_id}"
            )
            notify(flag.subject, flag.detail)
            actions.append(
                f"{run_config.run_id}: excluded {node} (SLURM drained) -- {flag.detail}"
            )
    return actions


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

    Called every tick, not gated by run_full_check's health-check backoff:
    worker jobs push checkpoints and wait on sync flags on their own schedule.

    Cron fires once a minute, so the inner loop keeps advancing a run whose
    condition is already met instead of waiting for the next tick. It stops
    when a handler returns None (blocked) or the run reaches DONE.
    """
    actions: list[str] = _exclude_drained_nodes(run_configs, state, query_node_states())
    for run_config in run_configs:
        # Loop guard: a handler can report an action without changing state
        # (e.g. _do_assigning with every node excluded). Revisiting a
        # (status, round) pair means nothing changed, so stop. The action is
        # still reported once, and the run waits for the next tick.
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


def _stall(run: RunState, note: str) -> str | None:
    """Return a terminal chunk-relaunch note once, then None until it changes.

    A non-None return short-circuits the state machine. A condition that cannot
    clear itself would otherwise re-report every tick and starve the
    merge/timeout handlers, which are the only paths that can make progress.
    """
    if run.last_stall_note == note:
        return None
    run.last_stall_note = note
    return note


def _relaunch_chunk_if_finished(
    run_config: MeshRunConfig,
    run: RunState,
    state: WatcherState,
    repo_root: Path,
    mesh_shared: Path,
    now: float,
) -> str | None:
    """Submit the next chunk once every worker job for this run has exited.

    Only for chunk_rounds runs. Without chunking, a worker vanishing means a
    real failure, which tick.py's dead-job detection owns. Here it is the
    expected end of a chunk; the run resumes from its current round, since
    master.pt already holds every merge up to it.

    Excluded nodes are not given a new job. Exclusion is not auto-reversed;
    only _recover_pool can readmit a node.
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
        return _stall(
            run, f"{run_config.run_id}: chunk finished but every node is excluded"
        )

    # A node can be in run_config.nodes while its shard_assignments entry is
    # gone (a concurrent redistribute_shards). Submitting it crashes the
    # worker on read_my_shards, and skipping it leaves every round waiting out
    # the grace period. So repair the entry, as _recover_pool does: hand back
    # any shards the node still holds that nobody else trains. Only a node
    # with no shards left is excluded, with the "of {run_id}" suffix so
    # clear_run_state and _recover_pool can handle it later.
    assignments_path = (
        mesh_shared / "metadata" / f"shard_assignments_{run_config.run_id}.json"
    )
    assignments = (
        json.loads(assignments_path.read_text()) if assignments_path.exists() else None
    )
    repaired: list[str] = []
    starved: list[str] = []
    unrepairable: list[str] = []
    if assignments is not None:
        unassigned = [n for n in nodes if not assignments.get(n)]
        if unassigned:
            try:
                placement = _verified_holdings(mesh_shared)
            except FileNotFoundError:
                # The pool has never been validated (or the file vanished).
                # We cannot tell what these nodes hold, so we can neither
                # hand them shards nor honestly call them unusable --
                # excluding a whole roster on a missing file is not
                # recoverable without a human. Leave them and say so.
                placement = None
            if placement is None:
                unrepairable = unassigned
                nodes = [n for n in nodes if n not in unassigned]
                if not nodes:
                    return _stall(
                        run,
                        f"{run_config.run_id}: chunk finished but no node has a "
                        f"shard_assignments entry and verified_holdings.json is "
                        f"missing -- cannot repair ({', '.join(unassigned)})",
                    )
            else:
                trained = {s for shards in assignments.values() for s in shards}
                for node in unassigned:
                    own = [s for s in placement.get(node, []) if s not in trained]
                    if own:
                        assignments[node] = own
                        trained.update(own)
                        repaired.append(node)
                    else:
                        starved.append(node)
                        exclude_node(
                            state,
                            node,
                            f"no shards left to assign, of {run_config.run_id}",
                        )
                if repaired:
                    _write_json_atomic(assignments_path, assignments)
                nodes = [n for n in nodes if n not in starved]
                if not nodes:
                    return _stall(
                        run,
                        f"{run_config.run_id}: chunk finished but no node has "
                        f"shards to train -- excluded {', '.join(starved)}",
                    )

    # A worker that hit its walltime self-requeues under the same job id and
    # still holds its --exclusive node. Cancel it before resubmitting, or the
    # new job sits pending. Best-effort: a failed scancel must not block the
    # relaunch.
    stale_note = ""
    stale_ids = stale_jobs_for_nodes(
        run_id=run_config.run_id,
        nodes=nodes,
        registry=load_registry(registry_path),
        snapshot=load_squeue_snapshot(snapshot_path),
    )
    if stale_ids:
        try:
            result = subprocess.run(
                ["scancel", *stale_ids],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            stale_note = f"; cancel of {len(stale_ids)} self-requeued job(s) failed to run ({exc})"
        else:
            if result.returncode != 0:
                stale_note = (
                    f"; scancel of {len(stale_ids)} self-requeued job(s) exited "
                    f"{result.returncode}: {result.stderr.strip()}"
                )
            else:
                stale_note = (
                    f"; cancelled {len(stale_ids)} self-requeued job(s): "
                    f"{', '.join(stale_ids)}"
                )

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
            mesh_shared=mesh_shared,
            shards=None if assignments is None else assignments.get(node),
        )
        is not None
    ]
    failed = [n for n in nodes if n not in submitted]
    note = (
        f"{run_config.run_id}: chunk done -- relaunched {len(submitted)}/"
        f"{len(nodes)} worker(s) from round {run.round}"
    )
    note += stale_note
    if failed:
        note += f"; sbatch FAILED for {', '.join(failed)}"
    if repaired:
        note += f"; re-assigned shards to {', '.join(repaired)}"
    if starved:
        note += f"; excluded {', '.join(starved)} (no shards left to assign)"
    if unrepairable:
        note += (
            f"; skipped unassigned node(s) {', '.join(unrepairable)} "
            "-- verified_holdings.json missing, cannot repair"
        )
    run.last_stall_note = ""
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
    accepted for interface parity with the worker side (stern/
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
    # No job may run Mon-Fri 08:00-20:00 (class_hours.py), so while a fresh
    # 2h job would overlap that window, skip every handler that submits one:
    # chunk relaunch, ASSIGNING, _recover_pool. The run just waits in place.
    # Merging and syncing still proceed -- a job launched before 06:00 may
    # legitimately still be training -- and none of the held paths runs a
    # timer, so nothing is charged against the nodes for the wait.
    hold = class_hours.blocked(now)
    if hold is not None:
        if run.status == "ASSIGNING":
            return _stall(run, f"{run_config.run_id}: launch held -- {hold}")
        if run.status == "RUNNING":
            return _do_running(run_config, run, mesh_shared)
    # Before dispatching: a chunked run whose workers have all exited needs
    # its next chunk submitted, and that is true in RUNNING and in
    # AWAITING_CHECKPOINTS alike, so it does not belong in either handler.
    chunk_note = (
        None
        if hold is not None
        else _relaunch_chunk_if_finished(
            run_config, run, state, repo_root, mesh_shared, now
        )
    )
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
        return _do_merging(run_config, run, state, repo_root, mesh_shared)
    if run.status == "SYNCING":
        return _do_syncing(run_config, run, state, mesh_shared, repo_root, now)
    if run.status == "DEGRADED":
        return _do_degraded(run_config, run, state, mesh_shared)
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
    extra_overrides: list[str] = (),
    mesh_shared: Path | None = None,
    shards: list[str] | None = None,
) -> str | None:
    """Submit one sbatch job for `node` and record it in registry_path.

    Returns the job id, or None if sbatch failed or printed no id. A single
    node's failure does not stop the caller's loop.

    With `mesh_shared` and `shards`, the shard list is frozen into a per-node
    snapshot that MESH_ASSIGNMENTS points at. The live shard_assignments file
    is rewritten on every exclusion, and the worker reads it only at start,
    which can be hours after sbatch. Callers that do not know the list yet
    pass neither argument and use the live file. _replenish_pool relies on
    that, since it submits before shards are redistributed.

    Returns None without calling sbatch during class hours.
    """
    held = class_hours.blocked(time.time())
    if held is not None:
        print(
            f"_submit_worker_job: {node} ({run_config.run_id}) not submitted -- {held}",
            file=sys.stderr,
        )
        return None
    # Name ALL explicitly so the submitter's environment still propagates.
    # Only the variables listed below are overridden.
    exports: list[str] = []
    if mesh_shared is not None and shards is not None:
        snapshot_path = (
            mesh_shared
            / "metadata"
            / f"shard_assignments_{run_config.run_id}__{node}.json"
        )
        _write_json_atomic(snapshot_path, {node: list(shards)})
        # run_mesh_worker.sh defaults MESH_ASSIGNMENTS with `:-`, so this wins.
        exports.append(f"MESH_ASSIGNMENTS={snapshot_path}")
    if run_config.chunk_rounds is not None:
        # Stops the worker's self-requeue at the chunk boundary. Otherwise the
        # same job id keeps requeuing while the watcher submits a new job for
        # the same node, and two chains race for one node. SLURM_RESTART_COUNT
        # is 0 on the first submission, so the stop trips at chunk_rounds - 1.
        exports.append(f"MESH_MAX_REQUEUES={max(run_config.chunk_rounds - 1, 0)}")
    export = "ALL," + ",".join(exports) if exports else "ALL"
    cmd = [
        "sbatch",
        "--nodes=1",
        "--ntasks=1",
        f"--export={export}",
        f"--nodelist={node}",
        "cluster/slurm/mesh_train.sbatch",
        run_config.experiment,
        run_config.run_id,
        f"+run_tag={run_config.run_id}",
        # No '+' on these four: the experiment config already defines them, and
        # Hydra's '+' fails on an existing key. mesh.start_round is new, so it
        # keeps the '+'.
        f"mesh.merge={'true' if run_config.merge else 'false'}",
        f"mesh.outer_lr={run_config.outer_lr}",
        f"mesh.outer_momentum={run_config.outer_momentum}",
        f"mesh.rounds={run_config.rounds}",
        f"+mesh.start_round={start_round}",
    ]
    # '++' (add-or-override), not '+': some experiment configs already define
    # chunk_rounds, and '+' would abort composition. Omitted when unset.
    if run_config.chunk_rounds is not None:
        cmd.append(f"++mesh.chunk_rounds={run_config.chunk_rounds}")
    cmd.extend(run_config.extra_overrides)
    cmd.extend(extra_overrides)
    # Retry once: a single transient scheduler failure would otherwise lead to
    # grace expiry, exclusion, and shard loss. Failures are logged on the last
    # attempt because no accounting database exists to diagnose them later.
    job_id = None
    for attempt in range(2):
        try:
            result = subprocess.run(
                cmd, cwd=repo_root, capture_output=True, text=True, timeout=300
            )
        except (OSError, subprocess.SubprocessError) as exc:
            if attempt == 1:
                print(
                    f"_submit_worker_job: sbatch for {node} ({run_config.run_id}) "
                    f"raised on attempt {attempt + 1}/2: {exc}",
                    file=sys.stderr,
                )
            continue
        job_id = (
            _parse_submitted_job_id(result.stdout) if result.returncode == 0 else None
        )
        if job_id is not None:
            break
        if attempt == 1:
            print(
                f"_submit_worker_job: sbatch for {node} ({run_config.run_id}) "
                f"failed on attempt {attempt + 1}/2: returncode={result.returncode} "
                f"stderr={result.stderr.strip()!r}",
                file=sys.stderr,
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
    return job_id


def resubmit_with_override(
    run_config: MeshRunConfig,
    node: str,
    repo_root: Path,
    registry_path: Path,
    submitted_at: str,
    *,
    start_round: int,
    override: str,
) -> str | None:
    """Public entrypoint for escalate.py's scripted NaN retry: submits
    `node` for `run_config` with one extra Hydra override appended (e.g. a
    halved inner LR), reusing _submit_worker_job's sbatch command building.
    Kept separate from _submit_worker_job's own extra_overrides plumbing
    since escalate.py must not import MeshRunConfig's private submission
    internals directly."""
    return _submit_worker_job(
        run_config,
        node,
        repo_root,
        registry_path,
        submitted_at,
        start_round=start_round,
        extra_overrides=[override],
    )


def _verified_holdings(mesh_shared: Path) -> dict[str, list[str]]:
    """`{node: [shard, ...]}` of the shards each node physically holds.

    Read from verified_holdings.json, which is written from a live listing of
    every node. This is the only source of truth for shard location. A plan
    of intended placement is not enough.

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

    The expected set is the union of this roster's verified holdings,
    intersected with the golden manifest. A roster that is a subset of its tier
    trains what it collectively holds. Coverage is reported so a degraded round
    is visible. Shards that nobody in the roster holds are never requested.

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

    # A node with no shards would train on nothing and stall the round. Drop it.
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
    # The note reports submissions sbatch accepted, not the intended count. A
    # failed submission must not be counted as running: with quorum_k equal to
    # the pool size, one missing worker forces every round to time out.
    submitted_job_ids = {
        node: _submit_worker_job(
            run_config,
            node,
            repo_root,
            registry_path,
            submitted_at,
            start_round=run.round,
            mesh_shared=mesh_shared,
            shards=assignments[node],
        )
        for node in eligible
    }
    failed_submit = [node for node, jid in submitted_job_ids.items() if jid is None]

    # sbatch can accept a job that SLURM fails moments later (another launcher
    # took the node). One batched recheck right after submission catches this
    # instead of waiting out the round timeout.
    accepted_ids = {jid: node for node, jid in submitted_job_ids.items() if jid}
    if accepted_ids:
        bad_states = {"NODE_FAIL", "BOOT_FAIL", "FAILED", "CANCELLED", "PREEMPTED"}
        job_states = query_job_states(list(accepted_ids))
        failed_submit.extend(
            accepted_ids[jid]
            for jid, state in job_states.items()
            if state in bad_states
        )

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
        # Single-node reference arm: the worker trains every round inside one
        # job and never waits on a sync flag, so there is nothing to coordinate.
        run.status = "DONE"
        return f"{run_config.run_id}: RUNNING -> DONE (merge=False, single-node arm)"

    pending_dir = mesh_shared / "checkpoints" / "pending"
    if pending_nodes(pending_dir, run_config.run_id, round_num=run.round):
        run.status = "AWAITING_CHECKPOINTS"
        note = f"{run_config.run_id}: RUNNING -> AWAITING_CHECKPOINTS"
        # Reported once per round, not per tick: a starving node is a standing
        # condition and would otherwise repeat every minute.
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
    """Seconds since this round's earliest checkpoint, or since the round started.

    Only files for `round_num` count. A straggler from the prior round can
    still write into pending/ after a forced merge, and its mtime must not
    start this round's clock. With no current-round files, fall back to the
    round start so the timeout still fires.
    """
    files = pending_files_for_round(pending_dir, run_id, round_num)
    if not files:
        return now - round_started_at
    return now - min(f.stat().st_mtime for f in files)


def _has_replacement_available(
    run_config: MeshRunConfig,
    run: RunState,
    state: WatcherState,
    run_configs_path: Path | None,
) -> bool:
    """Whether excluding a node now has a replacement available.

    Checked tier-wide, since every missing node draws on the same deficit and
    candidate pool."""
    if run_configs_path is None:
        return True  # no way to check -- matches the pre-existing behaviour
    roster = _live_roster(run_configs_path, run_config.run_id, run_config.nodes)
    if pool_deficit(run, roster, state) <= 0:
        return True  # pool isn't short -- exclusion doesn't need a replacement
    tier = NODE_GROUPS[roster[0]]
    tier_roster = [h for h, g in NODE_GROUPS.items() if g == tier]
    idle = query_idle_nodes()
    candidates = candidate_replacements(
        tier_roster, in_use=set(roster), excluded=set(state.excluded_nodes)
    )
    healthy = truly_idle(
        query_node_health([n for n in candidates if n in idle], idle=idle)
    )
    return bool(healthy)


def effective_quorum_k(
    run_config: MeshRunConfig,
    run: RunState,
    state: WatcherState,
    *,
    roster: list[str],
    missing_unreplaceable: Collection[str] = (),
) -> int:
    """quorum_k minus unreplaceable-missing nodes, floored at 60% of the target pool.

    Recomputed every round and never written back to the run config, so the
    quorum rises again when nodes return.
    """
    del state  # unused -- kept for interface parity with the design doc
    if run_config.quorum_k is None:
        return len(roster)  # no floor configured -- nothing to relax
    floor = math.ceil(0.6 * run.target_pool_size) if run.target_pool_size else 0
    relaxed = run_config.quorum_k - len(missing_unreplaceable)
    return max(relaxed, floor)


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
    # A quarantined node can still drop checkpoints for a few rounds. Its stale
    # file must not count toward quorum. pending_nodes() parses filenames only.
    checked_in &= eligible

    roster = (
        _live_roster(run_configs_path, run_config.run_id, run_config.nodes)
        if run_configs_path is not None
        else run_config.nodes
    )

    if checked_in >= eligible:
        clear_missed_rounds(run, checked_in)
        # quorum_k is a floor on merge contributors, and this fast path must
        # respect it too. Once exclusions shrink `eligible` below quorum_k,
        # "everyone checked in" is trivially true, so halt rather than merge
        # below the floor. The floor is the effective quorum, so a tier with
        # no idle replacement can still finish.
        eff_quorum_k = effective_quorum_k(run_config, run, state, roster=roster)
        quorum_relaxed = (
            run_config.quorum_k is not None and eff_quorum_k < run_config.quorum_k
        )
        if run_config.quorum_k is not None and len(checked_in) < eff_quorum_k:
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
        run.quorum_relaxed = quorum_relaxed
        run.any_round_quorum_relaxed = run.any_round_quorum_relaxed or quorum_relaxed
        run.round_misses_charged = 0
        run.round_exclusions_suspended = 0
        run.round_effective_quorum_k = (
            eff_quorum_k if run_config.quorum_k is not None else None
        )
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
            # Quorum reached before grace expired: merge with whoever checked in.
            # Stragglers are not excluded and get another chance next round.
            # This is still a partial merge, so timeout_forced stays True. Use
            # weighting="token" for quorum-configured runs to avoid renormalization bias.
            clear_missed_rounds(run, checked_in)
            run.status = "MERGING"
            run.timeout_forced = True
            run.quorum_relaxed = False  # reached the configured floor as-is
            run.round_misses_charged = 0
            run.round_exclusions_suspended = 0
            run.round_effective_quorum_k = run_config.quorum_k
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
    # _replenish_pool is the only path that moves an excluded node's shards to
    # a survivor, and it runs even when no replacement is found, so shard
    # coverage does not silently shrink.
    #
    # Exclusion requires a replacement to exist, so a tier with no slack stops
    # losing nodes it cannot replace. The miss counter still increments, so a
    # node skipped today can be excluded on a later miss.
    excluded = []
    unreplaceable = set()
    for node in missing:
        count = missed_round_count(state, run_config.run_id, node)
        if count < MISSED_ROUND_THRESHOLD:
            continue
        if not _has_replacement_available(run_config, run, state, run_configs_path):
            unreplaceable.add(node)
            continue
        exclude_node(
            state,
            node,
            f"crash-loop: missed {count} consecutive rounds of {run_config.run_id}",
        )
        excluded.append(node)
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
    # Same floor as the all-checked-in path, applied after exclusion so the log
    # still records what went wrong.
    eff_quorum_k = effective_quorum_k(
        run_config, run, state, roster=roster, missing_unreplaceable=unreplaceable
    )
    quorum_relaxed = (
        run_config.quorum_k is not None and eff_quorum_k < run_config.quorum_k
    )
    if run_config.quorum_k is not None and len(checked_in) < eff_quorum_k:
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
    run.quorum_relaxed = quorum_relaxed
    run.any_round_quorum_relaxed = run.any_round_quorum_relaxed or quorum_relaxed
    run.round_misses_charged = len(missing)
    run.round_exclusions_suspended = len(unreplaceable)
    run.round_effective_quorum_k = (
        eff_quorum_k if run_config.quorum_k is not None else None
    )
    result = (
        f"{run_config.run_id}: AWAITING_CHECKPOINTS -> MERGING "
        f"(timeout, missing {missing}{excluded_note})"
    )
    if replenish_notes:
        result += "; " + "; ".join(replenish_notes)
    return result


def _do_degraded(
    run_config: MeshRunConfig,
    run: RunState,
    state: WatcherState,
    mesh_shared: Path,
) -> str | None:
    """Re-check a DEGRADED run's round for quorum, so it can recover.

    Replenishment and chunk relaunch can bring checkpoints back after a halt,
    so the run must be re-checked each tick rather than left halted.
    """
    pending_dir = mesh_shared / "checkpoints" / "pending"
    checked_in = pending_nodes(pending_dir, run_config.run_id, round_num=run.round)
    eligible = set(eligible_nodes(state, run_config.nodes))
    checked_in &= eligible  # a quarantined node cannot satisfy quorum either
    if run_config.quorum_k is None or len(checked_in) < run_config.quorum_k:
        return None
    run.status = "MERGING"
    run.timeout_forced = True
    return (
        f"{run_config.run_id}: DEGRADED -> MERGING "
        f"(quorum {run_config.quorum_k}/{len(eligible)} recovered, "
        f"{len(checked_in)} nodes)"
    )


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
    """Return this run's healthy excluded nodes to the pool.

    _replenish_pool only fires on the tick a node crosses the exclusion
    threshold, so without this the pool would only ever shrink. Every
    RECOVERY_INTERVAL_S this re-probes the run's excluded nodes.

    A node is resubmitted only if it can receive shards nobody is training,
    normally the ones it still holds from before exclusion. An empty assignment
    would fail on read_my_shards, so an under-filled pool is the safe outcome.
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
            mesh_shared=mesh_shared,
            shards=own,
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
    """Add `new_nodes` to a run's roster in the run-configs JSON.

    Idempotent: nodes already listed are skipped. The write is atomic.
    """
    payload = json.loads(path.read_text())
    for entry in payload:
        if entry["run_id"] == run_id:
            for node in new_nodes:
                if node not in entry["nodes"]:
                    entry["nodes"].append(node)
            break
    _write_json_atomic(path, payload)


def _live_roster(run_configs_path: Path, run_id: str, fallback: list[str]) -> list[str]:
    """Current on-disk roster for run_id, including nodes added earlier this tick.

    run_config.nodes is a snapshot and does not see roster writes made during
    the tick. Reading the file prevents two replenishments from picking the
    same idle node. Falls back to `fallback` if the file or entry is missing.
    """
    if not run_configs_path.exists():
        return fallback
    payload = json.loads(run_configs_path.read_text())
    for entry in payload:
        if entry["run_id"] == run_id:
            return entry["nodes"]
    return fallback


def submit_replacement(
    run_config: MeshRunConfig,
    run: RunState,
    state: WatcherState,
    mesh_shared: Path,
    repo_root: Path,
    run_configs_path: Path | None,
    excluded_node: str,
) -> tuple[list[str], str | None]:
    """Top the pool back up toward target size after `excluded_node` is excluded.

    Returns (picked_nodes, status_note). The caller hands shards to the picks
    via redistribute_shards, since choosing a replacement and moving shards are
    separate decisions.

    A replacement only gets shards from excluded_node's handoff. If that entry
    is absent or empty, the replacement would crash on read_my_shards, so
    replenishment is skipped. An under-filled pool is the safe failure. A
    missing assignments file does not block replenishment.
    """
    held = class_hours.blocked(time.time())
    if held is not None:
        return [], f"{run_config.run_id}: replacement for {excluded_node} held -- {held}"
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
            # SLURM "idle" is not readiness: a node can have its GPU held by
            # another process or lack its seed. Probe the GPU and seed before
            # submitting. Only SLURM-idle candidates need the ssh probe.
            # This run's own excluded nodes still hold their shards, so they
            # are the cheapest replacements and are probed alongside the rest.
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

    return picked, replenish_note


def redistribute_shards(
    run_config: MeshRunConfig,
    mesh_shared: Path,
    excluded_node: str,
    picked: list[str],
) -> str | None:
    """Move an excluded node's shards to survivors, with or without a replacement.

    This is the only path that updates shard_assignments after ASSIGNING, so it
    must not return early. Otherwise the shards stay on a node that will never
    train again, and coverage silently shrinks.
    """
    assignments_path = (
        mesh_shared / "metadata" / f"shard_assignments_{run_config.run_id}.json"
    )
    if not assignments_path.exists():
        return None
    assignments = json.loads(assignments_path.read_text())
    if excluded_node not in assignments:
        return None
    placement = _verified_holdings(mesh_shared)
    for node in picked:
        assignments.setdefault(node, [])
    assignments, dropped = reassign_failed(excluded_node, assignments, placement)
    note = f"{run_config.run_id}: reassigned {excluded_node}'s shards to survivors"
    if dropped and picked:
        # No survivor holds these, but excluded_node still does. It was excluded
        # because it could not run a worker, not because it went away, so copy
        # the shards to the replacement. Replication is 1x for much of the
        # corpus, so this is a common path.
        recovered = copy_shards(excluded_node, picked[0], dropped)
        if recovered:
            assignments[picked[0]] = assignments.get(picked[0], []) + recovered
            dropped = [s for s in dropped if s not in recovered]
            note += (
                f"; recovered {len(recovered)} shard(s) to {picked[0]} "
                f"({', '.join(recovered[:5])}"
                f"{' ...' if len(recovered) > 5 else ''})"
            )
    if dropped:
        # Coverage shrank: no survivor holds these and the copy failed. Do not
        # hand them to a node that cannot open them.
        note += (
            f"; dropped {len(dropped)} shard(s) no survivor holds "
            f"({', '.join(dropped[:5])}{' ...' if len(dropped) > 5 else ''})"
        )
    _write_json_atomic(assignments_path, assignments)
    return note


def _replenish_pool(
    run_config: MeshRunConfig,
    run: RunState,
    state: WatcherState,
    mesh_shared: Path,
    repo_root: Path,
    run_configs_path: Path | None,
    excluded_node: str,
) -> str | None:
    picked, replenish_note = submit_replacement(
        run_config, run, state, mesh_shared, repo_root, run_configs_path, excluded_node
    )
    reassign_note = redistribute_shards(run_config, mesh_shared, excluded_node, picked)
    notes = [n for n in (replenish_note, reassign_note) if n]
    return "; ".join(notes) if notes else None


# --- MERGING -----------------------------------------------------------


def _bootstrap_master(pending_dir: Path, run_id: str, master_path: Path) -> None:
    """Seed master.pt from round 0's first pending checkpoint.

    The merge loads master.pt unconditionally and has no cold-start fallback,
    so the first merge of a run would otherwise fail. The seed is the
    lexicographically first pending file, matching a worker's cold start.
    """
    candidates = sorted(pending_files_for_round(pending_dir, run_id, 0))
    if not candidates:
        return
    payload = torch.load(candidates[0], map_location="cpu", weights_only=False)
    # Pending checkpoints are fp16. master.pt must start in fp32, or the
    # outer-optimizer momentum accumulates at half precision on every merge.
    model_state_dict = {
        k: (v.float() if torch.is_floating_point(v) else v)
        for k, v in payload["model_state_dict"].items()
    }
    master_path.parent.mkdir(parents=True, exist_ok=True)
    # BOOTSTRAP_MASTER_VERSION is negative, so it is older than every real
    # round and never collides with round 0's already-merged check.
    ckpt_io.save(
        master_path,
        {
            "model_state_dict": model_state_dict,
            "master_version": BOOTSTRAP_MASTER_VERSION,
            "init_seed": payload.get("init_seed"),
        },
    )


def _uv_bin() -> str:
    """Absolute path to uv for merge subprocesses.

    Cron's PATH does not include ~/.local/bin, so a bare "uv" fails under cron
    while working in a shell. Resolve the binary explicitly.
    """
    return shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")


# Wall-clock budget for one merge subprocess. The durable merge phase measured
# ~210s, so this leaves about 3x headroom. A timeout SIGKILLs a merge that may
# already have rewritten master.pt, so keep the budget well above the measured
# cost. Ticks do not pile up: the watcher's flock makes overlapping ticks exit.
MERGE_TIMEOUT_S = 900


def _do_merging(
    run_config: MeshRunConfig,
    run: RunState,
    state: WatcherState,
    repo_root: Path,
    mesh_shared: Path,
) -> str | None:
    pending_dir = mesh_shared / "checkpoints" / "pending"
    master_path = mesh_shared / "checkpoints" / run_config.run_id / "master.pt"
    local_dir = _mesh_local_root() / run_config.run_id
    momentum_path = local_dir / "outer_momentum.pt"
    residual_path = local_dir / "broadcast_residual.pt"
    local_dir.mkdir(parents=True, exist_ok=True)
    if not master_path.exists():
        _bootstrap_master(pending_dir, run_config.run_id, master_path)
    cmd = [
        _uv_bin(),
        "run",
        # Without this, every merge re-syncs the large NFS-hosted venv, since
        # uv reports a reinstall even when the lockfile is current.
        "--no-sync",
        "python",
        "-m",
        "stern.mesh.run_outer_step",
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
        "--eligible-nodes",
        ",".join(sorted(eligible_nodes(state, run_config.nodes))),
        # No --tb-dir: TensorBoard scalars are not written inside the merge. Their
        # cost grows with round count, and a SIGKILL during that tail would
        # discard a finished merge. Diagnostics must not spend the merge's time
        # budget. Scalars are written once at DONE by flush_final_tb_rounds.
    ]
    if run.timeout_forced:
        cmd.append("--timeout-forced")
    if run.quorum_relaxed:
        cmd.append("--quorum-relaxed")
    cmd.extend(["--misses-charged", str(run.round_misses_charged)])
    cmd.extend(
        ["--exclusions-suspended-no-replacement", str(run.round_exclusions_suspended)]
    )
    if run.round_effective_quorum_k is not None:
        cmd.extend(["--effective-quorum-k", str(run.round_effective_quorum_k)])
    # misses_excused_job_absent, misses_correlated and requeues_seen are left at
    # their defaults of 0. Every miss is counted by missed_round_count, and
    # requeues are counted in the worker's own journal.
    try:
        result = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=MERGE_TIMEOUT_S,
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

RESULTS_HOST = setting("STERN_RESULTS_HOST")
RESULTS_ROOT = str(data_root() / "stern_mesh_results")


def _transfer_results_to_node089(run_config: MeshRunConfig, mesh_shared: Path) -> str:
    """Best-effort: rsync the run's checkpoint dir to the results host, then
    delete the NFS copy.

    Never raises. A failed transfer leaves the results on NFS. DONE runs are not
    retried, and losing the only copy of a finished run is worse than keeping
    it on NFS longer.
    """
    local_dir = mesh_shared / "checkpoints" / run_config.run_id
    if not local_dir.exists() or not any(local_dir.iterdir()):
        return f"{run_config.run_id}: transfer skipped -- no local checkpoint dir"

    remote_dir = f"{RESULTS_ROOT}/{run_config.run_id}"
    try:
        mkdir_result = subprocess.run(
            ["ssh", RESULTS_HOST, "mkdir", "-p", remote_dir],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if mkdir_result.returncode != 0:
            return (
                f"{run_config.run_id}: transfer to {RESULTS_HOST} FAILED "
                f"(mkdir: {mkdir_result.stderr.strip()}) -- results kept on NFS"
            )

        rsync_result = subprocess.run(
            [
                "rsync",
                "-az",
                "--checksum",
                f"{local_dir}/",
                f"{RESULTS_HOST}:{remote_dir}/",
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if rsync_result.returncode != 0:
            return (
                f"{run_config.run_id}: transfer to {RESULTS_HOST} FAILED "
                f"(rsync exit {rsync_result.returncode}) -- results kept on NFS"
            )

        # rsync exiting 0 does not prove the copy is intact (the disk may have
        # filled). Verify before deleting the only other copy.
        check_result = subprocess.run(
            ["ssh", RESULTS_HOST, "test", "-s", f"{remote_dir}/master.pt"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if check_result.returncode != 0:
            return (
                f"{run_config.run_id}: transfer to {RESULTS_HOST} FAILED "
                f"(post-transfer check) -- results kept on NFS"
            )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"{run_config.run_id}: transfer to {RESULTS_HOST} FAILED ({exc}) -- results kept on NFS"

    shutil.rmtree(local_dir)
    return (
        f"{run_config.run_id}: results transferred to {RESULTS_HOST}:{remote_dir}, freed NFS"
    )


def _cancel_run_workers(run_id: str, repo_root: Path) -> str:
    """Cancel any worker still running for a run that has reached DONE.

    Stragglers keep computing rounds that nothing merges, and each holds a
    whole node. Best-effort: a failed scancel must not block DONE, the result
    transfer, or state cleanup. The outcome is always returned in the note.
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
    """Delete this run's leftover pending checkpoints once it reaches DONE.

    A run that ends on a timed-out round can leave late checkpoints behind.
    Nothing else collects them, and a later merge would average them in.
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
    """Delete pending/ checkpoints from runs that are no longer configured.

    Only files older than max_age_s are removed. This covers runs that never
    reached DONE. An empty active_run_ids means the config is unknown (the file
    is missing), so this must no-op rather than delete live checkpoints.
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
    """Copy the local momentum buffer into the run's NFS dir for archiving.

    The momentum is part of the run's result, so it is copied once, at DONE.
    The error-feedback residual is not copied because it is not a result.
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
        # No merge is coming to flush the TB rounds held back for stragglers,
        # so flush them now.
        flush_final_tb_rounds(
            _mesh_local_root() / run_config.run_id / "tb",
            mesh_shared,
            run_config.run_id,
        )
        _stage_local_merge_state_for_archive(run_config, mesh_shared)
        transfer_note = _transfer_results_to_node089(run_config, mesh_shared)
        # Cancel before sweeping, so no straggler writes into pending/ after the
        # sweep.
        cancel_note = _cancel_run_workers(run_config.run_id, repo_root)
        swept = _sweep_pending_checkpoints(run_config.run_id, mesh_shared)
        clear_run_state(state, run_config.run_id)
        done_note = (
            f"{run_config.run_id}: SYNCING -> DONE "
            f"(round {run.round}/{run_config.rounds}); {transfer_note}; "
            f"{cancel_note}; swept {swept} pending checkpoint(s)"
        )
        if run.any_round_quorum_relaxed:
            done_note += (
                " -- WARNING: one or more rounds merged under a relaxed "
                "quorum, not a clean DONE"
            )
        return done_note
    # Do not re-enter ASSIGNING here. The worker is one long-lived job that
    # loops over rounds and polls this flag itself, and shard assignment is
    # static. Re-entering would submit a second job per node. RUNNING waits for
    # the next round's checkpoints.
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
    """Delete sync flags and pending checkpoints left by a prior attempt at run_id.

    Stale flags would make workers skip the merged master. The globs must not
    match sibling runs: the "_" delimiter and the digit class in the flag glob
    keep "pool" from matching "pool-fish-5r" or "pool_extra". Missing
    directories simply yield no matches.
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
