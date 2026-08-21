#!/usr/bin/env bash
# video_encoding/scripts/launch_node_scaling_sweep.sh
# Register one wave of the node-count scaling sweep onto disjoint node sets
# drawn from the fish pool roster, and record the partition.
#
# Usage: launch_node_scaling_sweep.sh <a|b>
#
# PRECONDITIONS -- both are the operator's job, not checked here:
#   * The fish sweep queue must be STOPPED (touch runs/state/sweep_queue_fish.stop
#     and confirm it exited). It would otherwise launch a queued run onto the
#     very nodes these cells occupy.
#   * The pool must ALREADY BE SEEDED. Registration alone starts a run: the
#     watcher's _do_assigning derives shard assignments from the run's node
#     list plus the on-disk placement in shard_map.json.
#
# Nodes are allocated LARGEST CELL FIRST, in roster order. Filling the
# 16-node cell last from a roster already picked over by smaller cells is
# what fails; largest-first makes the allocation deterministic and total.
# A roster too small for the whole wave exits 1 BEFORE registering anything:
# a partially registered wave trains cells on node sets the analysis cannot
# interpret.
#
# Env overrides (tests set all three; production uses the defaults):
#   NSS_READY_FILE   roster, one hostname per line
#   NSS_STATE_DIR    where the partition file is written
#   NSS_REGISTER_CMD the register_mesh_run.sh to call
set -euo pipefail

USAGE="usage: launch_node_scaling_sweep.sh <a|b>"
WAVE="${1:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
READY_FILE="${NSS_READY_FILE:-${REPO_ROOT}/runs/state/mesh_pool_ready_fish.txt}"
STATE_DIR="${NSS_STATE_DIR:-${REPO_ROOT}/runs/state}"
REGISTER_CMD="${NSS_REGISTER_CMD:-${SCRIPT_DIR}/register_mesh_run.sh}"

# Held fixed across every cell so worker count is the only variable.
ROUNDS=24
OUTER_LR=0.7
OUTER_MOMENTUM=0.9

# "<run_id> <n_nodes> <quorum_k> <weighting>"
case "${WAVE}" in
  a)
    CELLS=(
      "nsl-n01a 1 1 data"
      "nsl-n02 2 2 data"
      "nsl-n04 4 3 token"
      "nsl-n08 8 6 token"
    )
    ;;
  b)
    # nsl-n01b repeats the single-node cell in the quieter wave: without a
    # cell common to both waves, wave A's four-way subnet contention is a
    # silent bias rather than a measured effect.
    CELLS=(
      "nsl-n16 16 13 token"
      "nsl-n01b 1 1 data"
    )
    ;;
  *)
    echo "${USAGE}" >&2
    exit 1
    ;;
esac

if [[ ! -f "${READY_FILE}" ]]; then
  echo "launch_node_scaling_sweep: no roster at ${READY_FILE}" >&2
  exit 1
fi
mapfile -t ROSTER < <(grep -v '^[[:space:]]*$' "${READY_FILE}")

TOTAL=0
for cell in "${CELLS[@]}"; do
  set -- ${cell}
  TOTAL=$((TOTAL + $2))
done
if (( ${#ROSTER[@]} < TOTAL )); then
  echo "launch_node_scaling_sweep: wave ${WAVE} needs ${TOTAL} nodes, roster has ${#ROSTER[@]} -- registering nothing" >&2
  exit 1
fi

declare -A ALLOC
NEXT=0
while read -r count idx; do
  set -- ${CELLS[${idx}]}
  nodes=("${ROSTER[@]:NEXT:count}")
  NEXT=$((NEXT + count))
  ALLOC["$1"]="$(IFS=,; printf '%s' "${nodes[*]}")"
done < <(
  for i in "${!CELLS[@]}"; do
    set -- ${CELLS[${i}]}
    echo "$2 ${i}"
  done | sort -k1,1nr -k2,2n
)

mkdir -p "${STATE_DIR}"
PARTITION="${STATE_DIR}/node_scaling_wave_${WAVE}.txt"
: > "${PARTITION}"

for cell in "${CELLS[@]}"; do
  set -- ${cell}
  run_id="$1"
  quorum_k="$3"
  weighting="$4"
  csv="${ALLOC[${run_id}]}"
  "${REGISTER_CMD}" "${run_id}" "${csv}" "${ROUNDS}" "${OUTER_LR}" "${OUTER_MOMENTUM}" "${quorum_k}" "${weighting}"
  printf '%s %s\n' "${run_id}" "${csv}" >> "${PARTITION}"
done

echo "launch_node_scaling_sweep: wave ${WAVE} registered ${#CELLS[@]} cells over ${TOTAL} nodes; partition at ${PARTITION}"
