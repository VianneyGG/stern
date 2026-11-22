"""Deterministic cleanup on quota pressure (spec: quota remediation).

Two disks, three artifact classes, all scoped by the same idea -- a file is
safe to delete once the run_id it belongs to will never be read again:

- "home" (the 30G NFS per-user quota, watcher/quota.py::home_quota_pct) --
  two things:
  - `<run_id>_*.pt.tmp` orphans in mesh_shared/checkpoints/pending/. Both
    existing sweeps (mesh_tick.py::_sweep_pending_checkpoints,
    sweep_orphaned_pending) glob `*.pt` only; a quota-exceeded mid-write is
    exactly what leaves a `.pt.tmp` behind, so that is the one gap that
    matters under quota pressure. active_run_ids uses the same "empty means
    unknown, no-op" guard as sweep_orphaned_pending, for the same reason.
  - `mesh_shared/checkpoints/<run_id>/outer_momentum.pt` for DONE runs.
    mesh_tick.py::_stage_local_merge_state_for_archive copies this here at
    DONE (366M/run) so the node089 transfer would pick it up as part of the
    reproducible result -- but node089 is currently off (2026-08-14), so
    nothing drains these and they accumulate with no offsetting sink. Kept
    once node089 is back on; deleted here in the meantime since a run whose
    status is DONE never resumes and a live viewer can always re-derive the
    momentum buffer for a *new* run. `master.pt` in the same directory is
    the actual trained-model result and is never touched.
- "node:<name>" (a worker's node-local /Data, watcher/quota.py::
  node_data_usage_pct) -- STERN_CHECKPOINT_ROOT (train.sbatch:55,
  run_mesh_worker.sh:65) holds one <run_id>.pt / <run_id>_push_residual.pt /
  <run_id>.pt.tmp per mesh run per worker, plus a `tb/` directory orphaned
  by mesh-tensorboard-centralize (commit 8d5b567 removed the worker-side TB
  writer; nothing writes there now). None of these are the coordinator's
  own artifacts (_mesh_local_root()'s outer_momentum.pt / tb before DONE, or
  the merged master.pt under mesh_shared/checkpoints/<run_id>/) -- those
  live on different paths entirely and this module never touches them.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

DEAD_SUBDIRS = ("tb",)
_CHECKPOINT_SUFFIXES = ("_push_residual.pt", ".pt.tmp", ".pt")
DEFAULT_PENDING_TMP_MAX_AGE_S = 3600.0


def orphaned_tmp_checkpoints(
    pending_dir: Path,
    active_run_ids: set[str],
    *,
    max_age_s: float = DEFAULT_PENDING_TMP_MAX_AGE_S,
    now: float,
) -> list[Path]:
    """`<run_id>_*.pt.tmp` in pending_dir whose run_id is not active and is
    older than max_age_s. Empty active_run_ids means "unknown, not
    verified" (load_run_configs returns [] when its file is missing) and
    must no-op rather than treat every run as inactive."""
    if not active_run_ids or not pending_dir.exists():
        return []
    stale: list[Path] = []
    for path in pending_dir.glob("*.pt.tmp"):
        if any(path.name.startswith(f"{run_id}_") for run_id in active_run_ids):
            continue
        if now - path.stat().st_mtime < max_age_s:
            continue
        stale.append(path)
    return stale


def stale_outer_momentum_checkpoints(
    mesh_shared: Path, done_run_ids: set[str]
) -> list[Path]:
    """`outer_momentum.pt` under mesh_shared/checkpoints/<run_id>/ for each
    DONE run_id, while node089 (its only consumer) is off. `master.pt` in
    the same directory is never included -- it is the trained-model result,
    not archive-staged state."""
    checkpoints_dir = mesh_shared / "checkpoints"
    stale: list[Path] = []
    for run_id in done_run_ids:
        path = checkpoints_dir / run_id / "outer_momentum.pt"
        if path.exists():
            stale.append(path)
    return stale


def safe_node_checkpoints(filenames: list[str], done_run_ids: set[str]) -> list[str]:
    """One node's STERN_CHECKPOINT_ROOT listing -> entries safe to delete:
    DEAD_SUBDIRS unconditionally, plus <run_id>.pt / .pt.tmp /
    _push_residual.pt where run_id is DONE (state.runs[run_id].status)."""
    safe: list[str] = []
    for name in filenames:
        if name in DEAD_SUBDIRS:
            safe.append(name)
            continue
        for suffix in _CHECKPOINT_SUFFIXES:
            if name.endswith(suffix):
                run_id = name[: -len(suffix)]
                if run_id in done_run_ids:
                    safe.append(name)
                break
    return safe


def _cleanup_home(
    mesh_shared: Path, active_run_ids: set[str], done_run_ids: set[str], *, now: float
) -> str | None:
    pending_dir = mesh_shared / "checkpoints" / "pending"
    stale_tmp = orphaned_tmp_checkpoints(pending_dir, active_run_ids, now=now)
    stale_momentum = stale_outer_momentum_checkpoints(mesh_shared, done_run_ids)
    stale = stale_tmp + stale_momentum
    if not stale:
        return None
    for path in stale:
        path.unlink(missing_ok=True)
    parts = []
    if stale_tmp:
        names = ", ".join(p.name for p in stale_tmp)
        parts.append(f"{len(stale_tmp)} orphaned pending .tmp file(s): {names}")
    if stale_momentum:
        names = ", ".join(f"{p.parent.name}/{p.name}" for p in stale_momentum)
        parts.append(f"{len(stale_momentum)} stale outer_momentum.pt: {names}")
    return "deleted " + "; ".join(parts)


_CHECKPOINT_ROOT_EXPR = '"${STERN_CHECKPOINT_ROOT:-/Data/$USER/stern_checkpoints}"'


def _cleanup_node(node: str, done_run_ids: set[str]) -> str | None:
    # The remote command is built as a SINGLE string, not split across argv
    # elements ("sh", "-c", ...) -- ssh joins every argument after the
    # hostname with a plain space and hands the result to ONE remote shell,
    # so any quoting spread across separate Python list elements is
    # destroyed before it's parsed (verified live: the old ["ssh", node,
    # "sh", "-c", "ls -1 \"$STERN...\""] form actually ran a bare `ls` of
    # $HOME, since `-c`'s own [command_name [args...]] form swallowed the
    # real path as $0). Same single-string-after-host convention as
    # seed_mesh.py::_run_ssh.
    list_cmd = ["ssh", node, f"ls -1 {_CHECKPOINT_ROOT_EXPR}"]
    try:
        result = subprocess.run(list_cmd, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    filenames = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    to_delete = safe_node_checkpoints(filenames, done_run_ids)
    if not to_delete:
        return None

    # Filenames are ls-derived from a path this process controls, but are
    # still shlex.quote'd individually before being joined into the one
    # remote command string above -- defense in depth, not a trust boundary.
    quoted_names = " ".join(shlex.quote(name) for name in to_delete)
    delete_cmd = [
        "ssh",
        node,
        f"cd {_CHECKPOINT_ROOT_EXPR} && rm -r -f -- {quoted_names}",
    ]
    try:
        result = subprocess.run(delete_cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    names = ", ".join(to_delete)
    return f"deleted {len(to_delete)} item(s) on {node}: {names}"


def remediate_quota_pressure(
    location: str,
    *,
    mesh_shared: Path,
    done_run_ids: set[str],
    active_run_ids: set[str],
    now: float,
) -> str | None:
    """Dispatches on a quota Flag's location ("home" or "node:<name>",
    detectors.py::detect_quota's Flag.subject minus its "quota:" prefix).
    Returns a one-line action-log summary, or None if nothing was safe to
    delete or a subprocess failed -- the caller then falls through to the
    existing claude -p escalation."""
    if location == "home":
        return _cleanup_home(mesh_shared, active_run_ids, done_run_ids, now=now)
    if location.startswith("node:"):
        return _cleanup_node(location.removeprefix("node:"), done_run_ids)
    return None
