"""video_encoding/scripts/watch_training.py

Single cron entrypoint for the SLURM training watcher
(docs/superpowers/specs/2026-07-15-slurm-training-watcher-design.md), replacing
the never-installed scripts/cron/poll_squeue.sh + aggregate_results.sh pair.

Every tick: refresh squeue.json (flock-guarded, cheap), then either exit (fast
path, most ticks) or run a full check (detectors -> remediate/escalate ->
backoff update -> one log line) once the adaptive backoff timer says it's due.

Install: crontab -e on login-node ->
  * * * * *  cd ~/stern && uv run --no-sync python -m video_encoding.scripts.watch_training

--no-sync is load-bearing, not tidiness: the project venv is on the NFS home
and `uv sync --dry-run` reports an unconditional reinstall even against an
up-to-date lockfile, so without it this cron rewrites a shared 9.4 GB venv
1440 times a day.
"""

from __future__ import annotations

import fcntl
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from video_encoding.watcher.detectors import detect_nan
from video_encoding.watcher.escalate import remediate_nan_node
from video_encoding.watcher.mesh_tick import (
    advance_all_runs,
    load_run_configs,
    sweep_orphaned_pending,
)
from video_encoding.watcher.squeue import refresh_squeue_snapshot
from video_encoding.watcher.state import load_state, save_state
from video_encoding.watcher.tick import run_full_check

REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = REPO_ROOT / "runs" / "state"
RESULTS_DIR = REPO_ROOT / "runs" / "results"
LOG_PATH = STATE_DIR / "watcher.log"
LOCK_PATH = STATE_DIR / "watch_training.lock"
HOST_PATH = STATE_DIR / "watcher_host"
# Short name: systemd/SLURM report login-node, socket.gethostname() reports
# login-node.polytechnique.fr. The designation file records the short form.
HOSTNAME = socket.gethostname().split(".")[0]
# Applies only to pending/ checkpoints whose run_id is not currently
# configured -- an active run's own checkpoints are never touched
# regardless of age. Generous relative to sync_timeout_s (1800s default) so
# a slow-but-live round is never mistaken for an orphan.
_ORPHANED_PENDING_MAX_AGE_S = 3600.0


def is_designated_watcher(host_file: Path, hostname: str) -> bool:
    """True when `hostname` is the host `host_file` designates as the watcher.

    Both sides are reduced to their short name so a hand-edited file holding an
    FQDN still matches. Raises OSError when the file is absent or unreadable --
    the caller must fail closed rather than assume it is the designated host.
    """
    designated = host_file.read_text().strip().split(".")[0].lower()
    return bool(designated) and designated == hostname.split(".")[0].lower()


def main() -> int:
    # Ahead of every side effect below -- STATE_DIR.mkdir and the open(LOCK_PATH,
    # "w") that truncates a file shared over NFS both happen before flock is even
    # attempted, so a guard placed any later would still let a decommissioned
    # host clobber shared state once a minute.
    try:
        designated = is_designated_watcher(HOST_PATH, HOSTNAME)
    except OSError as exc:
        print(f"{HOST_PATH} unreadable, refusing to tick: {exc}", file=sys.stderr)
        return 1
    if not designated:
        # stderr, not watcher.log: journald keeps it, and a minute-cadence line
        # in the shared log would drown the records used to audit a migration.
        print(f"{HOSTNAME} is not the designated watcher host", file=sys.stderr)
        return 0

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    lock_fh = open(LOCK_PATH, "w")
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return (
            0  # previous tick still running -- skip silently (poll_squeue.sh pattern)
        )

    now = time.time()
    try:
        refresh_squeue_snapshot(
            STATE_DIR / "squeue.json", user=os.environ.get("USER", "")
        )
    except Exception as exc:
        _log(f"squeue refresh FAILED: {exc}")
        return 0

    state_path = STATE_DIR / "watcher_state.json"
    state = load_state(state_path)

    # Mesh round advancement is unconditional -- not gated by the health-check
    # backoff timer below, since worker jobs push checkpoints and wait on sync
    # flags on their own schedule, independent of that cadence.
    mesh_shared = Path(os.environ.get("MESH_SHARED", Path.home() / "stern_mesh"))
    run_configs = load_run_configs(STATE_DIR / "mesh_run_configs.json")
    if run_configs:
        try:
            mesh_actions = advance_all_runs(
                run_configs,
                state=state,
                repo_root=REPO_ROOT,
                mesh_shared=mesh_shared,
                mesh_root=Path("/Data/$USER/mesh"),
                now=now,
                run_configs_path=STATE_DIR / "mesh_run_configs.json",
            )
        except Exception as exc:
            # A single bad mesh run config (e.g. n_shards mismatched against
            # shard_map.json, or an unknown hostname) must not take down
            # dead/stalled-job detection for the rest of the fleet below.
            _log(f"mesh advancement FAILED: {exc}")
            mesh_actions = []
        save_state(state_path, state)  # advance_all_runs may have mutated state
        for action in mesh_actions:
            _log(f"mesh: {action}")

        # Unconditional like advance_all_runs above -- catches checkpoints
        # from runs that died before ever reaching SYNCING/DONE (the only
        # other sweep site), independent of the full-check backoff timer.
        swept = sweep_orphaned_pending(
            mesh_shared / "checkpoints" / "pending",
            {c.run_id for c in run_configs},
            max_age_s=_ORPHANED_PENDING_MAX_AGE_S,
        )
        if swept:
            _log(f"swept {swept} orphaned pending checkpoint(s)")

        # Also unconditional: a NaN'd node must be excluded before the next
        # round, not after a stalled/dead-job backoff timer says it's due.
        # Defaults to dry-run (report only, no exclude/spawn) until the
        # first production deploy has been observed behaving correctly.
        nan_dry_run = (
            os.environ.get("NAN_REMEDIATION_DRY_RUN", "true").strip().lower() != "false"
        )
        nan_acted = False
        for run_config in run_configs:
            for flag in detect_nan(run_config.run_id, mesh_shared, state):
                node = flag.subject.split(":", 2)[2]
                result = remediate_nan_node(
                    flag,
                    run_id=run_config.run_id,
                    node=node,
                    round_num=state.run(run_config.run_id).round,
                    state=state,
                    repo_root=REPO_ROOT,
                    transcript_dir=STATE_DIR / "escalations",
                    now=now,
                    now_str=datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
                    dry_run=nan_dry_run,
                )
                nan_acted = True
                _log(f"nan: {flag.subject} -> {result}")
        if nan_acted:
            save_state(state_path, state)  # remediate_nan_node mutated state

    if now < state.next_full_check_at:
        return 0  # fast path -- not due yet

    # Exempt every mesh run the watcher has EVER tracked, not just the
    # currently-configured ones. state.runs is written only by mesh_tick, so
    # its keys are exactly the set of mesh run_ids seen so far. A run that has
    # finished (or was dropped from mesh_run_configs.json by hand) still leaves
    # dead worker jobs in jobs_registry.json; scoping the exemption to the live
    # config let those fall through to remediate_dead_job, which hardcodes
    # train.sbatch and resubmitted 39 junk jobs on 2026-07-27. Those jobs then
    # queue AHEAD of the mesh worker submissions _replenish_pool needs, so the
    # leak also stalls recovery of the runs that are still going.
    result = run_full_check(
        repo_root=REPO_ROOT,
        state_dir=STATE_DIR,
        results_root=RESULTS_DIR,
        now=now,
        state=state,
        mesh_run_ids=frozenset(c.run_id for c in run_configs) | frozenset(state.runs),
        mesh_shared=mesh_shared,
        active_run_ids=frozenset(c.run_id for c in run_configs),
    )
    save_state(state_path, state)
    _log(
        f"flags={result.num_flags} actions={result.actions or 'none'} next_interval_s={state.current_interval_s:.0f}"
    )
    return 0


def _log(line: str) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as fh:
        fh.write(f"{ts} [{HOSTNAME}] {line}\n")


if __name__ == "__main__":
    sys.exit(main())
