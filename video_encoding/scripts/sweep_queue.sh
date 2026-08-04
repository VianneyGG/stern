#!/usr/bin/env bash
# video_encoding/scripts/sweep_queue.sh
# Drain one tier's DiLoCo sweep queue unattended: as soon as the watcher marks
# the in-flight run DONE, launch the next queued config (within one poll).
# Run under nohup on login-node, one instance per tier.
#
# Usage: sweep_queue.sh <fish|birds>
#
#   queue  .claude/state/sweep_queue_<tier>.txt -- "run_id rounds outer_lr
#          outer_momentum" per line ('#' comments allowed), read ONCE at start
#          and drained top-down.
#   stop   touch .claude/state/sweep_queue_<tier>.stop to exit cleanly at the
#          next poll, leaving the in-flight run alone.
set -euo pipefail

TIER="${1:?usage: sweep_queue.sh <fish|birds>}"
case "${TIER}" in
  fish|birds) ;;
  *) echo "ERROR: unknown tier '${TIER}' -- expected fish|birds" >&2; exit 1 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SWEEP_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
MESH_SHARED="${SWEEP_MESH_SHARED:-${HOME}/stern_mesh}"
POLL_S="${SWEEP_POLL_S:-120}"
LAUNCH_CMD="${SWEEP_LAUNCH_CMD:-${REPO_ROOT}/video_encoding/scripts/launch_mesh_pool.sh}"
REGISTER_CMD="${SWEEP_REGISTER_CMD:-${REPO_ROOT}/video_encoding/scripts/register_mesh_run.sh}"
SINFO="${SWEEP_SINFO:-sinfo}"
SQUEUE="${SWEEP_SQUEUE:-squeue}"
STALL_S="${SWEEP_STALL_S:-1800}"

# launch_mesh_pool.sh writes .claude/state/* through RELATIVE paths, so the
# fallback path is only correct with cwd at the repo root -- which a nohup'd
# daemon started from anywhere else would not have.
cd "${REPO_ROOT}"

STATE_DIR="${REPO_ROOT}/.claude/state"
QUEUE_FILE="${STATE_DIR}/sweep_queue_${TIER}.txt"
LOG_FILE="${STATE_DIR}/sweep_queue_${TIER}.log"
LOCK_FILE="${STATE_DIR}/sweep_queue_${TIER}.lock"
STOP_FILE="${STATE_DIR}/sweep_queue_${TIER}.stop"
READY_FILE="${STATE_DIR}/mesh_pool_ready_${TIER}.txt"
ACTIVE_FILE="${STATE_DIR}/mesh_pool_active_${TIER}.txt"
WATCHER_STATE="${STATE_DIR}/watcher_state.json"
RUN_CONFIGS="${STATE_DIR}/mesh_run_configs.json"
QUORUM=13  # of 16 pool nodes: both the reseed threshold and the abort-round quorum

[[ -f "${QUEUE_FILE}" ]] || { echo "ERROR: no queue file ${QUEUE_FILE}" >&2; exit 1; }

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "ERROR: another sweep_queue holds ${LOCK_FILE} -- exiting" >&2
  exit 1
fi

log() {
  printf '%s sweep_queue[%s] %s\n' "$(date -u +%FT%TZ)" "${TIER}" "$*" | tee -a "${LOG_FILE}"
}

active_nodes() { tr ',' '\n' < "${ACTIVE_FILE}" | awk 'NF'; }

# Prints the available subset of the pool, one node per line. "Available"
# spans idle nodes AND nodes still running our own jobs: the previous run's
# workers may still be tearing down, and those nodes are ours. Returns 1 if a
# probe failed, so an empty list from a broken sinfo/squeue is never logged as
# a genuine availability finding.
available_nodes() {
  local idle mine ours node
  if ! idle=$("${SINFO}" -N -h -o "%N %T"); then return 1; fi
  if ! mine=$("${SQUEUE}" --me -h -o "%N"); then return 1; fi
  ours=$(
    { awk '$2 ~ /^idle/ {print $1}' <<< "${idle}"
      tr ',' '\n' <<< "${mine}"; } | awk 'NF' | sort -u
  )
  for node in $(active_nodes); do
    if grep -qxF "${node}" <<< "${ours}"; then printf '%s\n' "${node}"; fi
  done
}

run_state() {
  python3 - "${WATCHER_STATE}" "$1" <<'PYEOF'
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

check_abort() {
  python3 - "${MESH_SHARED}/metrics/$1" "${QUORUM}" <<'PYEOF'
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

quorum = int(sys.argv[2])

per_round = defaultdict(list)
for path in Path(sys.argv[1]).glob("*.jsonl"):
    for line in path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            per_round[int(rec["round"])].append(float(rec["val_loss"]))


def finite(values):
    return [v for v in values if math.isfinite(v)]


verdict = "OK"
# Nodes write round R the instant they see the sync flag for R-1, so the
# newest round is routinely represented by a handful of nodes whose pre-merge
# spread is widest -- judge only a round the pool has actually finished.
quorate = [r for r in per_round if len(per_round[r]) >= quorum]
if quorate:
    last = max(quorate)
    reports = per_round[last]
    healthy = finite(reports)
    prior = [
        sum(finite(v)) / len(finite(v))
        for r, v in per_round.items()
        if r < last and finite(v)
    ]
    best = min(prior) if prior else float("nan")
    # A lone NaN is one node whose val subsample came out empty (thin shard
    # assignment after replenishment), not a diverged run.
    if len(healthy) * 2 < len(reports):
        verdict = f"ABORT {last} nan {best}"
    else:
        cur = sum(healthy) / len(healthy)
        if prior and cur > 1.5 * best:
            verdict = f"ABORT {last} {cur} {best}"
print(verdict)
PYEOF
}

# Abort by capping rounds, never by scancel: the watcher's _do_syncing flips
# the run to DONE once round >= rounds, which runs the real DONE path
# (momentum copy-back, pending-checkpoint cleanup, stale sync-flag clearing).
cap_rounds() {
  python3 - "${RUN_CONFIGS}" "$1" "$2" <<'PYEOF'
import json
import sys
from pathlib import Path

path, run_id, rounds = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
cfgs = json.loads(path.read_text())
if not any(cfg["run_id"] == run_id for cfg in cfgs):
    sys.exit(f"cap_rounds: {run_id} not in {path}")
for cfg in cfgs:
    if cfg["run_id"] == run_id:
        cfg["rounds"] = rounds
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(cfgs, indent=2))
tmp.replace(path)
PYEOF
}

launch_item() {
  local run_id="$1" rounds="$2" outer_lr="$3" outer_momentum="$4" avail n_avail n_pool
  local params="rounds=${rounds} outer_lr=${outer_lr} outer_momentum=${outer_momentum}"
  if [[ -f "${READY_FILE}" && -f "${ACTIVE_FILE}" ]]; then
    n_pool=$(active_nodes | wc -l)
    if ! avail=$(available_nodes); then
      log "full launch ${run_id} (${params}) -- sinfo/squeue error, availability unknown"
    else
      n_avail=$(awk 'NF' <<< "${avail}" | wc -l)
      # Reseeding costs 30-60 min and a pool that just finished a run is
      # already seeded; below 13/16 the lost shard coverage is worse than the
      # reseed. Register only the available subset: an unavailable node never
      # checks in and burns a full 900s round timeout before being excluded.
      if [[ "${n_avail}" -ge "${QUORUM}" ]]; then
        log "register-only ${run_id} (${params}) -- ${n_avail}/${n_pool} pool nodes available"
        "${REGISTER_CMD}" "${run_id}" "$(paste -sd, <<< "${avail}")" \
          "${rounds}" "${outer_lr}" "${outer_momentum}"
        return 0
      fi
      log "full launch ${run_id} (${params}) -- only ${n_avail}/${n_pool} pool nodes available"
    fi
  else
    log "full launch ${run_id} (${params}) -- no seeded pool recorded for ${TIER}"
  fi
  "${LAUNCH_CMD}" "${TIER}" "${run_id}" "${rounds}" "${outer_lr}" "${outer_momentum}"
}

# 0 = run reached DONE, 1 = stop file appeared.
wait_for_done() {
  local run_id="$1" aborted=0 seen_active=0 last="" changed_at stale note
  local status round verdict abort_round cur best
  changed_at=$(date +%s)
  while true; do
    if [[ -e "${STOP_FILE}" ]]; then return 1; fi
    read -r status round < <(run_state "${run_id}")
    if [[ "${status}" != "DONE" && "${status}" != "MISSING" ]]; then seen_active=1; fi
    note=""
    # watcher_state.json is never pruned, so a reused run_id can read DONE from
    # a previous run before the watcher has even picked this one up.
    if [[ "${status}" == "DONE" && "${seen_active}" -eq 0 ]]; then
      note=" -- stale, watcher has not picked the run up yet"
    fi
    if [[ "${status} ${round}" != "${last}" ]]; then
      log "${run_id} ${status} round=${round}${note}"
      last="${status} ${round}"
      changed_at=$(date +%s)
    else
      stale=$(( $(date +%s) - changed_at ))
      if [[ "${stale}" -ge "${STALL_S}" ]]; then
        log "STALL ${run_id}: ${status} round=${round} unchanged for ${stale}s -- watcher cron may be dead"
      fi
    fi
    if [[ "${status}" == "DONE" && "${seen_active}" -eq 1 ]]; then return 0; fi
    if [[ "${aborted}" -eq 0 && "${round}" -ge 3 ]]; then
      verdict=$(check_abort "${run_id}")
      if [[ "${verdict}" == ABORT* ]]; then
        read -r _ abort_round cur best <<< "${verdict}"
        if cap_rounds "${run_id}" "${round}"; then
          log "ABORT ${run_id}: round ${abort_round} mean val_loss=${cur} vs best prior=${best} -- capped rounds at ${round}"
          aborted=1
        else
          log "ERROR ${run_id}: could not cap rounds at ${round} -- retrying next poll"
        fi
      fi
    fi
    sleep "${POLL_S}"
  done
}

mapfile -t ITEMS < <(awk '{sub(/#.*/, "")} NF' "${QUEUE_FILE}")
log "start -- ${#ITEMS[@]} queued item(s) from ${QUEUE_FILE}"

for item in "${ITEMS[@]}"; do
  read -r run_id rounds outer_lr outer_momentum <<< "${item}"
  if [[ -e "${STOP_FILE}" ]]; then
    log "stop file present -- exiting before ${run_id}"
    exit 0
  fi
  launch_item "${run_id}" "${rounds}" "${outer_lr}" "${outer_momentum}"
  if ! wait_for_done "${run_id}"; then
    log "stop file present -- exiting, ${run_id} left running"
    exit 0
  fi
done

log "queue exhausted -- exiting"
