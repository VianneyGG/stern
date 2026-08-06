#!/usr/bin/env bash
# video_encoding/scripts/launch_mesh_pool.sh
# Bring up one tier's 16-node mesh pool (+6 idle backup nodes left for
# auto-replenishment) once enough of the tier is idle. Run ON login-node
# (requires sinfo, and seed_mesh.py's --shard-map output must land on the
# shared NFS home the workers actually read from).
#
# Usage: launch_mesh_pool.sh <fish|birds|cars> <run_id> [rounds] [outer_lr] [outer_momentum]
#   [rounds]          number of DiLoCo merge rounds for the pool; default 25
#   [outer_lr]         DiLoCo outer-optimizer LR; default 0.7 (literature)
#   [outer_momentum]   DiLoCo outer-optimizer momentum; default 0.9 (literature)
#
# Does NOT rebuild the generic shard_map.json workers read by default
# (run_mesh_worker.sh's MESH_SHARD_MAP fallback) -- run
# recover_shard_map.py once, across every tier's active node list
# combined, after all pools you're launching this round have registered.
set -euo pipefail

TIER="${1:?usage: launch_mesh_pool.sh <fish|birds|cars> <run_id> [rounds] [outer_lr] [outer_momentum]}"
RUN_ID="${2:?usage: launch_mesh_pool.sh <fish|birds|cars> <run_id> [rounds] [outer_lr] [outer_momentum]}"
ROUNDS="${3:-25}"
OUTER_LR="${4:-0.7}"
OUTER_MOMENTUM="${5:-0.9}"

source "$(dirname "${BASH_SOURCE[0]}")/tiers_lib.sh"
NODELIST=$(tier_nodes "${TIER}")

echo "launch_mesh_pool: checking watcher status..."
if [[ -z "$(squeue --me -h 2>/dev/null)" ]]; then
  if ! crontab -l 2>/dev/null | grep -qF "watch_training"; then
    # --no-sync: this fires once a minute and `uv sync` reinstalls into the
    # NFS-hosted venv on every invocation (uv sync --dry-run reports an
    # uninstall+install even against an up-to-date lockfile) -- ~1440
    # redundant rewrites of a shared 9.4 GB venv per day without it.
    if (crontab -l 2>/dev/null; echo "* * * * * cd ${HOME}/stern && ${HOME}/.local/bin/uv run --no-sync python -m video_encoding.scripts.watch_training") | crontab -; then
      echo "launch_mesh_pool: installed watcher cron entry (squeue --me was empty)"
    else
      echo "launch_mesh_pool: WARNING -- failed to install watcher cron entry, continuing anyway" >&2
    fi
  else
    echo "launch_mesh_pool: watcher cron entry already present"
  fi
else
  echo "launch_mesh_pool: squeue --me non-empty -- assuming watcher already active, skipping cron install"
fi

NEED=22  # 16 active + 6 backup

echo "launch_mesh_pool: polling ${TIER} for ${NEED} idle nodes (60s interval)..."
IFS=',' read -r -a TIER_NODES <<< "${NODELIST}"
while true; do
  IDLE=$(sinfo -N -o "%N %T" --state=idle 2>/dev/null | awk '$2=="idle"{print $1}')
  READY=()
  for node in "${TIER_NODES[@]}"; do
    if echo "${IDLE}" | grep -qx "${node}"; then
      READY+=("${node}")
      [[ "${#READY[@]}" -ge "${NEED}" ]] && break
    fi
  done
  echo "launch_mesh_pool: ${TIER} idle=${#READY[@]}/${NEED} ($(date -u +%FT%TZ))"
  [[ "${#READY[@]}" -ge "${NEED}" ]] && break
  sleep 60
done

ACTIVE=("${READY[@]:0:16}")
BACKUP=("${READY[@]:16:6}")
ACTIVE_CSV=$(IFS=,; echo "${ACTIVE[*]}")
READY_CSV=$(IFS=,; echo "${READY[*]}")
echo "launch_mesh_pool: ${TIER} active=[${ACTIVE_CSV}]"
echo "launch_mesh_pool: ${TIER} backup (seeded, not active)=[$(IFS=,; echo "${BACKUP[*]}")]"

# All 22 (active + backup), one hostname per line -- lets the watcher
# replenish a dead active node from a backup that already has a seeded
# /Data/$USER/mesh payload, instead of the replacement dying on
# read_my_shards' KeyError with an empty local shard dir.
printf '%s\n' "${READY[@]}" > "runs/state/mesh_pool_ready_${TIER}.txt"

SOURCE_NODE="${ACTIVE[0]}"
echo "launch_mesh_pool: reseeding ${SOURCE_NODE} from node089..."
ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=3 "${SOURCE_NODE}" \
  "mkdir -p /Data/$USER/voxceleb2_shards && \
   rsync -az node089:/Data/$USER/voxceleb2_shards/ /Data/$USER/voxceleb2_shards/"

echo "launch_mesh_pool: seeding mesh across ${TIER} pool (22 = 16 active + 6 backup)..."
uv run python -m video_encoding.scripts.seed_mesh \
  --nodes "${READY_CSV}" \
  --active-nodes "${ACTIVE_CSV}" \
  --source-node "${SOURCE_NODE}" \
  --shards-dir /Data/$USER/voxceleb2_shards \
  --shard-map "${HOME}/stern_mesh/metadata/shard_map_${TIER}16.json" \
  --dataset-gb 223 \
  --replication 2

echo "launch_mesh_pool: registering run config ${RUN_ID} (outer_lr=${OUTER_LR} outer_momentum=${OUTER_MOMENTUM})..."
"$(dirname "${BASH_SOURCE[0]}")/register_mesh_run.sh" \
  "${RUN_ID}" "${ACTIVE_CSV}" "${ROUNDS}" "${OUTER_LR}" "${OUTER_MOMENTUM}"

echo "${ACTIVE_CSV}" > "runs/state/mesh_pool_active_${TIER}.txt"
echo "launch_mesh_pool: ${TIER} pool launched -- watcher cron will pick up ${RUN_ID} within a minute"
