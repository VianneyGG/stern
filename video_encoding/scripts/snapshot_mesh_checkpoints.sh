#!/usr/bin/env bash
# video_encoding/scripts/snapshot_mesh_checkpoints.sh
# Preserve a run's merged master.pt every N rounds, so the ViT-internals
# tooling (export_vit_internals.py -> plot_vit_internals.py) can compare
# mid-training checkpoints afterwards. The mesh overwrites master.pt on every
# merge, so without this nothing survives except the final round.
#
# Usage: snapshot_mesh_checkpoints.sh <run_id> <every_n> <max_round> [poll_interval_s]
#
# Run under nohup on login-node alongside the watcher; exits 0 on its own once
# the max_round snapshot is taken. Deliberately external to watcher/mesh_tick.py
# -- copying a file needs no coupling to the coordinator state machine.
#
# Round semantics: mesh_tick's _do_syncing increments run.round *after* round
# R's merge and sync complete, so watcher_state's `round == N` means rounds
# 0..N-1 are done and master.pt holds round N-1's merge -- i.e. exactly N
# completed rounds. master.pt is written via checkpoint.save's tmp+rename, so
# a plain cp always reads a complete file; no locking needed.
set -euo pipefail

USAGE="usage: snapshot_mesh_checkpoints.sh <run_id> <every_n> <max_round> [poll_interval_s]"
RUN_ID="${1:?${USAGE}}"
EVERY_N="${2:?${USAGE}}"
MAX_ROUND="${3:?${USAGE}}"
POLL_S="${4:-300}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SNAPSHOT_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
MESH_SHARED="${SNAPSHOT_MESH_SHARED:-${HOME}/stern_mesh}"

WATCHER_STATE="${REPO_ROOT}/runs/state/watcher_state.json"
CKPT_DIR="${MESH_SHARED}/checkpoints/${RUN_ID}"
MASTER="${CKPT_DIR}/master.pt"

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) snapshot[${RUN_ID}] $*"; }

# "<status> <round>", or "MISSING 0" -- same idiom as sweep_queue.sh's run_status.
run_status() {
  python3 - "${WATCHER_STATE}" "${RUN_ID}" <<'PYEOF'
import json
import sys
from pathlib import Path

path, run_id = Path(sys.argv[1]), sys.argv[2]
try:
    runs = json.loads(path.read_text()).get("runs", {})
except (OSError, ValueError):
    runs = {}
run = runs.get(run_id)
print(run["status"], run.get("round", 0)) if run else print("MISSING", 0)
PYEOF
}

final_snapshot="${CKPT_DIR}/master_round${MAX_ROUND}.pt"
log "watching for every ${EVERY_N} rounds up to ${MAX_ROUND} (poll ${POLL_S}s)"

while true; do
  read -r status round <<<"$(run_status)"

  # Only capture the round master.pt CURRENTLY holds (round % EVERY_N == 0,
  # or round == MAX_ROUND for the final snapshot even if MAX_ROUND isn't a
  # multiple of EVERY_N) -- never backfill earlier boundaries the loop
  # didn't poll at. master.pt is overwritten on every merge, so if the round
  # counter advanced past several boundaries between two polls (poll
  # interval coarser than the round cadence, or the script started mid-run),
  # today's master.pt is NOT those earlier rounds' weights; copying it under
  # an earlier round's filename would silently mislabel a checkpoint (bug
  # caught in review, 2026-08-17). A boundary missed this way is simply
  # never captured -- logged as a gap, not faked.
  dest="${CKPT_DIR}/master_round${round}.pt"
  if (( round > 0 )) && (( round % EVERY_N == 0 || round == MAX_ROUND )) \
      && [[ ! -f "${dest}" ]]; then
    if [[ -f "${MASTER}" ]]; then
      cp "${MASTER}" "${dest}"
      log "captured round ${round} -> ${dest}"
    else
      log "WARNING round ${round} reached but ${MASTER} is missing -- skipping"
    fi
  fi

  if [[ -f "${final_snapshot}" ]]; then
    log "done -- round ${MAX_ROUND} snapshot captured"
    exit 0
  fi

  if [[ "${status}" == "DONE" || "${status}" == "MISSING" ]] && (( round < MAX_ROUND )); then
    log "ERROR run ended at round ${round} (status ${status}) before ${MAX_ROUND} -- giving up"
    exit 1
  fi

  sleep "${POLL_S}"
done
