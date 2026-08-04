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
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from video_encoding.watcher.mesh_tick import advance_all_runs, load_run_configs
from video_encoding.watcher.squeue import refresh_squeue_snapshot
from video_encoding.watcher.state import load_state, save_state
from video_encoding.watcher.tick import run_full_check

REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = REPO_ROOT / ".claude" / "state"
RESULTS_DIR = REPO_ROOT / ".claude" / "results"
LOG_PATH = STATE_DIR / "watcher.log"
LOCK_PATH = STATE_DIR / "watch_training.lock"


def main() -> int:
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
    run_configs = load_run_configs(STATE_DIR / "mesh_run_configs.json")
    if run_configs:
        try:
            mesh_actions = advance_all_runs(
                run_configs,
                state=state,
                repo_root=REPO_ROOT,
                mesh_shared=Path(
                    os.environ.get("MESH_SHARED", Path.home() / "stern_mesh")
                ),
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
        fh.write(f"{ts} {line}\n")


if __name__ == "__main__":
    sys.exit(main())
