#!/usr/bin/env bash
# video_encoding/scripts/pick_cross_tier_pools.sh
# Deal idle cluster nodes into N disjoint pools that each draw equally from
# every tier. Prints one comma-separated roster per line on stdout, N lines;
# progress and diagnostics go to stderr so stdout can be read directly:
#
#   mapfile -t POOLS < <(pick_cross_tier_pools.sh 4 4)
#   POOL_NODES="${POOLS[0]}" launch_mesh_pool.sh arm-a run-a ...
#
# Usage: pick_cross_tier_pools.sh <n_pools> <per_tier> [tier ...]
#   <n_pools>   how many pools to deal (one per experiment arm)
#   <per_tier>  nodes each pool takes from each tier
#   [tier ...]  tiers to draw from; default: fish birds cars countries
#
# Why this exists: launch_mesh_pool.sh polls a single tier, so one arm per tier
# was the only shape available. That makes "arm" and "GPU tier" the same
# variable -- the 2026-08-21 factorial ran mask_type on cars and tube on fish,
# and no difference between them was attributable to mask_type. Dealing every
# arm the same number of nodes from every tier makes GPU heterogeneity a
# constant across arms instead of a confound. It costs wall-clock: with
# mesh.round_budget_s=null every node runs the same step count, so each round
# ends at the slowest tier's pace and faster nodes idle at the barrier.
#
# Blocks until every tier has n_pools*per_tier idle nodes, polling every 60s,
# because a partially-idle cluster would otherwise yield short rosters and
# silently reintroduce the imbalance this script exists to remove.
set -euo pipefail

N_POOLS="${1:?usage: pick_cross_tier_pools.sh <n_pools> <per_tier> [tier ...]}"
PER_TIER="${2:?usage: pick_cross_tier_pools.sh <n_pools> <per_tier> [tier ...]}"
shift 2
TIERS=("$@")
[[ "${#TIERS[@]}" -eq 0 ]] && TIERS=(fish birds cars countries)

source "$(dirname "${BASH_SOURCE[0]}")/tiers_lib.sh"

NEED_PER_TIER=$(( N_POOLS * PER_TIER ))

# Fail before waiting on something that can never happen: a tier smaller than
# the demand would spin the poll loop forever.
for tier in "${TIERS[@]}"; do
  total=$(tier_nodes "${tier}" | tr ',' '\n' | grep -c .)
  if [[ "${total}" -lt "${NEED_PER_TIER}" ]]; then
    echo "ERROR: tier '${tier}' has ${total} node(s), need ${NEED_PER_TIER}" \
         "(${N_POOLS} pools x ${PER_TIER})" >&2
    exit 1
  fi
done

# picked[tier] = space-separated idle nodes chosen from that tier
declare -A PICKED
while true; do
  IDLE=$(sinfo -N -o "%N %T" --state=idle 2>/dev/null | awk '$2=="idle"{print $1}')
  satisfied=1
  summary=""
  for tier in "${TIERS[@]}"; do
    chosen=()
    while IFS= read -r node; do
      [[ -z "${node}" ]] && continue
      if echo "${IDLE}" | grep -qx "${node}"; then
        chosen+=("${node}")
        [[ "${#chosen[@]}" -ge "${NEED_PER_TIER}" ]] && break
      fi
    done < <(tier_nodes "${tier}" | tr ',' '\n')
    PICKED["${tier}"]="${chosen[*]:-}"
    summary+=" ${tier}=${#chosen[@]}/${NEED_PER_TIER}"
    [[ "${#chosen[@]}" -lt "${NEED_PER_TIER}" ]] && satisfied=0
  done
  echo "pick_cross_tier_pools: idle${summary} ($(date -u +%FT%TZ))" >&2
  [[ "${satisfied}" -eq 1 ]] && break
  sleep 60
done

# Contiguous blocks, not round-robin: nodes within a tier are the same GPU, so
# which block a pool gets cannot bias it, and blocks keep the mapping obvious
# in the log when a node has to be traced back to its pool.
for (( i = 0; i < N_POOLS; i++ )); do
  roster=()
  for tier in "${TIERS[@]}"; do
    read -r -a tier_picked <<< "${PICKED[${tier}]}"
    roster+=("${tier_picked[@]:$(( i * PER_TIER )):${PER_TIER}}")
  done
  (IFS=,; echo "${roster[*]}")
done
