#!/usr/bin/env bash
# cluster/mesh_continue_allowed.sh
# Layers 1/2/4 of the self-requeue termination guarantee (design §1.2).
# Layer 3 (futile-requeue counter) is added by Task 3 -- this file's
# structure leaves a clearly marked insertion point for it.
#
# Usage: mesh_continue_allowed.sh <run_id> <node>
# Exit 0 = continue (requeue), exit 1 = stop.
set -uo pipefail  # not -e: every check below must run even if an earlier
                   # one fails to read (fail-open policy lives in the checks
                   # themselves, not in early script termination)

RUN_ID="${1:?usage: mesh_continue_allowed.sh <run_id> <node>}"
NODE="${2:?usage: mesh_continue_allowed.sh <run_id> <node>}"
MESH_SHARED="${MESH_SHARED:?MESH_SHARED must be set}"
STATE_DIR="${MESH_SHARED}/state"

# --- Layer 1: absolute deadline -- reads only the worker's own env ---------
if [[ -n "${MESH_DEADLINE_EPOCH:-}" ]]; then
  now=$(date +%s)
  if (( now >= MESH_DEADLINE_EPOCH )); then
    echo "mesh_continue_allowed: stopping -- past MESH_DEADLINE_EPOCH" >&2
    exit 1
  fi
fi

# --- Layer 2: total requeue budget -- SLURM sets both sides ----------------
if [[ -n "${MESH_MAX_REQUEUES:-}" ]]; then
  restart_count="${SLURM_RESTART_COUNT:-0}"
  if (( restart_count >= MESH_MAX_REQUEUES )); then
    echo "mesh_continue_allowed: stopping -- SLURM_RESTART_COUNT=${restart_count} >= MESH_MAX_REQUEUES=${MESH_MAX_REQUEUES}" >&2
    exit 1
  fi
fi

# --- Watcher staleness (fifth, cheaper signal; design §3b) -----------------
MESH_WATCHER_STALE_S="${MESH_WATCHER_STALE_S:-600}"
WATCHER_STATE="${STATE_DIR}/watcher_state.json"
if [[ -f "${WATCHER_STATE}" ]]; then
  last_tick_at=$(python3 -c "import json; print(json.load(open('${WATCHER_STATE}')).get('last_tick_at', 0))" 2>/dev/null || echo "")
  if [[ -n "${last_tick_at}" ]]; then
    now=$(date +%s)
    staleness=$(python3 -c "print(int(${now} - ${last_tick_at}))" 2>/dev/null || echo 0)
    if (( staleness > MESH_WATCHER_STALE_S )); then
      echo "mesh_continue_allowed: stopping -- watcher stale (${staleness}s since last tick)" >&2
      exit 1
    fi
  fi
fi
# Missing/unreadable watcher_state.json: best-effort, same fail-open policy
# as layer 4 -- this check is skipped, layers 1-3 still bound the loop.

# --- Layer 3: futile-requeue counter ----------------------------------------
REQUEUE_DIR="${STATE_DIR}/requeue"
mkdir -p "${REQUEUE_DIR}"
JOURNAL="${REQUEUE_DIR}/${RUN_ID}-${NODE}.json"
CURRENT_ROUND_MARKER="${REQUEUE_DIR}/${RUN_ID}-${NODE}-current_round"
MESH_MAX_FUTILE_REQUEUES="${MESH_MAX_FUTILE_REQUEUES:-2}"

current_round="$(cat "${CURRENT_ROUND_MARKER}" 2>/dev/null || echo -1)"
prev_round=-1
prev_futile=0
if [[ -f "${JOURNAL}" ]]; then
  prev_round=$(python3 -c "import json,sys; print(json.load(open('${JOURNAL}')).get('round', -1))" 2>/dev/null || echo -1)
  prev_futile=$(python3 -c "import json,sys; print(json.load(open('${JOURNAL}')).get('consecutive_futile', 0))" 2>/dev/null || echo 0)
fi

if [[ "${current_round}" -gt "${prev_round}" ]]; then
  new_futile=0
else
  new_futile=$((prev_futile + 1))
fi

python3 -c "import json; json.dump({'round': ${current_round}, 'consecutive_futile': ${new_futile}}, open('${JOURNAL}', 'w'))"

if (( new_futile >= MESH_MAX_FUTILE_REQUEUES )); then
  ts=$(date -u +%Y%m%d-%H%M%S)
  escalation_dir="${STATE_DIR}/../../runs/state/escalations"
  mkdir -p "${escalation_dir}"
  detail="worker ${NODE} made ${new_futile} requeues with no round advancement on ${RUN_ID} (stuck at round ${current_round}) -- watcher likely down or wedged"
  escalation_file="${escalation_dir}/watcher_stalled-${RUN_ID}-${NODE}-${ts}.md"
  echo "${detail}" > "${escalation_file}"
  # Dependency-free delivery (design §3 worker-side half): a bare curl, not a
  # re-entry into the watcher's own Python/NFS-venv stack, since that stack
  # may be exactly what's down.
  if [[ -n "${MESH_NOTIFY_WEBHOOK_URL:-}" ]]; then
    curl -sf -m 5 -X POST "${MESH_NOTIFY_WEBHOOK_URL}" \
      -H "Content-Type: application/json" \
      -d "{\"subject\": \"watcher_stalled\", \"detail\": $(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "${detail}"), \"timestamp\": ${current_round}}" \
      || echo "${ts} watcher_stalled ${RUN_ID} ${NODE}: ${detail}" >> "${STATE_DIR}/../../runs/state/escalations/NEEDS_HUMAN.log"
  else
    echo "${ts} watcher_stalled ${RUN_ID} ${NODE}: ${detail}" >> "${STATE_DIR}/../../runs/state/escalations/NEEDS_HUMAN.log"
  fi
  echo "mesh_continue_allowed: stopping -- futile-requeue bound reached (${new_futile} >= ${MESH_MAX_FUTILE_REQUEUES})" >&2
  exit 1
fi

# --- Layer 4: the routine, watcher-written path -----------------------------
if [[ -f "${STATE_DIR}/HALT" || -f "${STATE_DIR}/HALT-${RUN_ID}" ]]; then
  echo "mesh_continue_allowed: stopping -- HALT file present" >&2
  exit 1
fi
if [[ -f "${STATE_DIR}/quarantine/${RUN_ID}/${NODE}" ]]; then
  echo "mesh_continue_allowed: stopping -- ${NODE} quarantined for ${RUN_ID}" >&2
  exit 1
fi

exit 0
