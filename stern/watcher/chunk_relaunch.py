"""Detect a finished worker chunk so the next one can be submitted.

SallesInfo caps walltime at 2h, so a worker can no longer span a run (a
40-round DiLoCo run is ~11h). `mesh.chunk_rounds` makes each job exit cleanly
after a few rounds, and the watcher submits the next chunk from the round the
last one reached, resuming through master.pt.

The trigger has to be "the workers are gone", not "a checkpoint is late".
Leaning on the existing exclusion path instead would be actively harmful: at
a chunk boundary all 16 workers exit at the same instant, so every one of
them misses the same round, and `record_missed_round` would permanently
exclude the entire roster rather than relaunch it.

The failure this module must not have is the mirror image: relaunching while
workers are still alive double-submits 16 `--exclusive` jobs onto nodes
already running them. Both guards below exist for that, and both treat
"I cannot tell" as "do not relaunch".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from stern.watcher.registry import JobRecord

# A job is submitted during one tick and only becomes visible to `squeue`
# moments later, while the snapshot this reads was refreshed at the START of
# the tick -- so a just-submitted chunk legitimately appears nowhere. Treat
# recently-submitted jobs as alive regardless of the snapshot; 300s is far
# longer than sbatch->squeue visibility and far shorter than a chunk.
SUBMIT_VISIBILITY_S = 300.0

# squeue lists only active jobs, so presence is essentially liveness. These
# are listed anyway because a job caught mid-teardown can still appear with a
# terminal state, and relaunching is correct then.
_TERMINAL = frozenset(
    {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}
)

# Statuses where the workers are expected to be running. Deliberately not
# MERGING/SYNCING: those are watcher-side steps the workers wait through, and
# a run sitting in them briefly has no bearing on whether a chunk ended.
#
# DEGRADED is included even though it's a halt state: a chunk boundary exits
# every worker at once regardless of quorum, so a run can land in DEGRADED
# with its whole roster already gone. Without relaunching here, that roster
# never comes back and the run can only progress via a manual status write
# Without this, a DEGRADED run only progresses after a manual status write.
RELAUNCHABLE = frozenset({"RUNNING", "AWAITING_CHECKPOINTS", "DEGRADED"})


def _submitted_at_epoch(record: JobRecord) -> float | None:
    try:
        return datetime.fromisoformat(record.submitted_at).timestamp()
    except (ValueError, TypeError):
        return None


def needs_worker_relaunch(
    *,
    run_id: str,
    status: str,
    registry: Sequence[JobRecord],
    snapshot: Mapping[str, Mapping[str, Any]],
    snapshot_available: bool,
    now: float,
    visibility_s: float = SUBMIT_VISIBILITY_S,
) -> bool:
    """True when this run has registered worker jobs and none are still alive.

    `snapshot_available` is separate from an empty `snapshot` on purpose:
    `load_squeue_snapshot` returns {} both for "the queue is empty" and for
    "the file isn't there yet", and those must not mean the same thing here.
    A missing snapshot is the single most dangerous input -- it makes every
    run look finished at once -- so it returns False.

    A run with no registered jobs also returns False: it has never been
    submitted, which is _do_assigning's job, not a relaunch.
    """
    if not snapshot_available:
        return False
    if status not in RELAUNCHABLE:
        return False

    jobs = [r for r in registry if r.run_tag == run_id]
    if not jobs:
        return False

    for record in jobs:
        submitted = _submitted_at_epoch(record)
        if submitted is not None and now - submitted < visibility_s:
            return False
        entry = snapshot.get(record.slurm_job_id)
        # Job ids are not unique over time: a cluster rebuild reset the
        # counter, so a registry record can collide with a live id that now
        # belongs to someone else's job. The nodelist tells a genuine
        # still-running worker apart from a stranger's job wearing a recycled
        # id (same guard as `cancellable_job_ids` below).
        if (
            entry is not None
            and entry.get("state") not in _TERMINAL
            and entry.get("nodelist") == record.nodelist
        ):
            return False
    return True


def cancellable_job_ids(
    *,
    run_id: str,
    registry: Sequence[JobRecord],
    snapshot: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Job ids for `run_id` that are still alive and verifiably still ours.

    A run reaches DONE the moment the final merge lands, and that merge only
    ever needed `quorum_k` of the roster -- so a node that fell behind the
    pool's cadence is still training rounds whose merges are already over.
    Nothing reaps it: the worker loop counts its own rounds
    (`range(start_round, stop_round)`), never consults the coordinator, and
    the sync flag it waits on was written long ago, so it runs to the end of
    its chunk burning a GPU for output that will be swept unread.

    Two guards, because `scancel` is irreversible and this runs unattended:

    * Only jobs the squeue snapshot still reports in a non-terminal state.
      A job id absent from the snapshot has already exited; asking SLURM to
      cancel it again is noise at best.
    * Only jobs whose snapshot nodelist still matches the node we recorded
      at submission. Job ids are not unique over time: a cluster rebuild
      reset the counter, so old registry records collide with live ids
      belonging to other people's jobs. The nodelist tells a genuine
      straggler apart from a stranger's job wearing a recycled id.

    Returns ids, not a cancellation: the caller owns the side effect, and a
    pure selection is what makes the collision guard testable.
    """
    ids: list[str] = []
    for record in registry:
        if record.run_tag != run_id:
            continue
        entry = snapshot.get(record.slurm_job_id)
        if entry is None or entry.get("state") in _TERMINAL:
            continue
        if entry.get("nodelist") != record.nodelist:
            continue
        ids.append(record.slurm_job_id)
    return ids


def stale_jobs_for_nodes(
    *,
    run_id: str,
    nodes: Sequence[str],
    registry: Sequence[JobRecord],
    snapshot: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Job ids for `run_id` still occupying a node this chunk is about to resubmit.

    mesh_train.sbatch carries `--requeue`: a worker that hits its 2h SLURM
    walltime instead of exiting cleanly at the chunk boundary goes back to
    PENDING under the *same* job id, still requesting the *same*
    `--exclusive` node. squeue reports no `nodes` for a pending job, so
    `cancellable_job_ids`'s live-nodelist match can't see it -- SLURM's own
    requeue and this module's chunk relaunch then both try to claim the
    node, and the requeued job loses and sits PD forever.

    The record's OWN nodelist -- fixed at submission by `_submit_worker_job`,
    unlike the live snapshot -- is what disambiguates here: a self-requeued
    job is still filed under the node it was ever submitted for, so match on
    that instead of squeue's current (possibly blank) nodes field. Only jobs
    for nodes this call is about to resubmit are candidates, and only those
    still present in the snapshot in a non-terminal state -- a record absent
    from squeue has already exited and needs no scancel.
    """
    node_set = set(nodes)
    ids: list[str] = []
    for record in registry:
        if record.run_tag != run_id or record.nodelist not in node_set:
            continue
        entry = snapshot.get(record.slurm_job_id)
        if entry is None or entry.get("state") in _TERMINAL:
            continue
        ids.append(record.slurm_job_id)
    return ids
