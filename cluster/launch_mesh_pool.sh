#!/usr/bin/env bash
# cluster/launch_mesh_pool.sh
# Bring up one tier's 16-node mesh pool (+6 idle backup nodes left for
# auto-replenishment) once enough of the tier is idle. Run on the login node
# (needs sinfo; the --shard-map output must land on the shared NFS home that
# the workers read from).
#
# Usage: launch_mesh_pool.sh <fish|birds|cars> <run_id> [rounds] [outer_lr] [outer_momentum]
#   [rounds]          number of DiLoCo merge rounds for the pool; default 25
#   [outer_lr]         DiLoCo outer-optimizer LR; default 0.7 (literature)
#   [outer_momentum]   DiLoCo outer-optimizer momentum; default 0.9 (literature)
#
# Environment overrides: NEED (idle nodes to wait for), DATASET_HOLDER,
# EXPECTED_SHARDS, SHARDS_DIR (default: the 8fps shard dir on the dataset
# source node, STERN_SOURCE_NODE; point it at a smaller subset dir, e.g.
# symlinked shards plus val/, for a lean test run), DATASET_GB (must match
# SHARDS_DIR's actual size for plan_placement's capacity math; default 841),
# QUORUM_K, WEIGHTING, SKIP_SEED_VALIDATION, CHUNK_ROUNDS,
# EXPERIMENT (configs/experiment/<name>.yaml the workers compose; default
# videomae_mesh_ssl), GRACE_S, REPLICATION (seed_nodes' --replication;
# default 1: a dead node is replenished on demand by watcher/shard_transfer.py,
# not served from a pre-replicated peer), inherited by register_mesh_run.sh
# and not read here. Raise GRACE_S above the 900s default for a roster whose
# slowest tier cannot finish a round in time. EXTRA_OVERRIDES is space-separated
# Hydra tokens, e.g. "model.mask_type=mixed model.target_mode=per_frame",
# forwarded verbatim to register_mesh_run.sh so every worker sbatch carries them.
#
# POOL_NODES  explicit comma-separated roster, bypassing the tier idle poll.
#   The first positional then names the POOL LABEL, not a tier. The label keys
#   verified_holdings.json and runs/state/mesh_pool_*_<label>.txt, and need not
#   exist in the tiers file. Use it for a cross-tier pool (pick_cross_tier_pools.sh):
#   with one tier per arm, "arm" and "tier" are the same variable, so no
#   comparison between arms can be attributed to the arm.
#   Every listed node is active, with no seeded backups. plan_placement spreads
#   over the whole roster, and planning over a superset of the active nodes
#   leaves shards unplaced. A node lost mid-run is replenished on demand by
#   watcher/shard_transfer.py rather than from a pre-seeded spare.
# POOL_SOURCE_NODE  node seed_nodes distributes from; default is the first of
#   the roster. Point it at a node that already holds the full shard set to
#   skip the pre-stage copy.
set -euo pipefail

TIER="${1:?usage: launch_mesh_pool.sh <fish|birds|cars> <run_id> [rounds] [outer_lr] [outer_momentum]}"
RUN_ID="${2:?usage: launch_mesh_pool.sh <fish|birds|cars> <run_id> [rounds] [outer_lr] [outer_momentum]}"
ROUNDS="${3:-25}"
OUTER_LR="${4:-0.7}"
OUTER_MOMENTUM="${5:-0.9}"

source "$(dirname "${BASH_SOURCE[0]}")/tiers_lib.sh"
# With an explicit roster, TIER is only a label; tier_nodes would abort on it.
if [[ -z "${POOL_NODES:-}" ]]; then
  NODELIST=$(tier_nodes "${TIER}")
fi

echo "launch_mesh_pool: checking watcher status..."
if [[ -z "$(squeue --me -h 2>/dev/null)" ]]; then
  if ! crontab -l 2>/dev/null | grep -qF "watch_training"; then
    # --no-sync: this fires once a minute and `uv sync` reinstalls into the
    # NFS-hosted venv on every invocation (uv sync --dry-run reports an
    # uninstall+install even against an up-to-date lockfile) -- ~1440
    # redundant rewrites of a shared 9.4 GB venv per day without it.
    if (crontab -l 2>/dev/null; echo "* * * * * cd ${HOME}/stern && ${HOME}/.local/bin/uv run --no-sync python -m stern.watcher.watch_training") | crontab -; then
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

if [[ -n "${POOL_NODES:-}" ]]; then
  IFS=',' read -r -a READY <<< "${POOL_NODES}"
  ACTIVE=("${READY[@]}")
  echo "launch_mesh_pool: ${TIER} explicit roster, ${#ACTIVE[@]} node(s), no idle poll"
else
  # 16 active + 6 backup. Overridable because not every tier is that big --
  # some tiers have fewer nodes than that, so the default can never be met and
  # the poll loop would spin forever.
  NEED="${NEED:-22}"

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
fi
BACKUP=("${READY[@]:${#ACTIVE[@]}}")
ACTIVE_CSV=$(IFS=,; echo "${ACTIVE[*]}")
READY_CSV=$(IFS=,; echo "${READY[*]}")
echo "launch_mesh_pool: ${TIER} active=[${ACTIVE_CSV}]"
echo "launch_mesh_pool: ${TIER} backup (seeded, not active)=[$(IFS=,; echo "${BACKUP[*]}")]"

# Active + backup, one hostname per line. The watcher replenishes a dead active
# node from a backup that already has a seeded mesh payload; a replacement with
# an empty local shard dir would die on read_my_shards' KeyError. With POOL_NODES
# there are no backups, and replenishment goes through shard_transfer.py instead.
printf '%s\n' "${READY[@]}" > "runs/state/mesh_pool_ready_${TIER}.txt"

# The 8fps corpus (800 shards) is the default. It lives only on the dataset
# source node, which serves it and is outside the pool.
SOURCE_NODE="${POOL_SOURCE_NODE:-${STERN_SOURCE_NODE}}"
# Node holding the full 260-shard dataset that seed_nodes distributes from.
# Overridable because the holder is a property of the cluster today, not of
# any tier.
DATASET_HOLDER="${DATASET_HOLDER:-${STERN_SOURCE_NODE}}"
# Ask the node what it actually has rather than trusting a hostname: several
# tiers' first nodes already carry the full set, and reseeding one needlessly
# drags 223 GB across a 1GbE link for nothing.
EXPECTED_SHARDS="${EXPECTED_SHARDS:-800}"
HAVE=$(ssh -o ConnectTimeout=15 "${SOURCE_NODE}" \
  "ls ${STERN_DATA_ROOT}/voxceleb2_shards/*.tar 2>/dev/null | wc -l" || echo 0)
if [[ "${SOURCE_NODE}" == "${DATASET_HOLDER}" || "${HAVE}" -ge "${EXPECTED_SHARDS}" ]]; then
  echo "launch_mesh_pool: ${SOURCE_NODE} already holds ${HAVE} shard(s) -- no reseed needed"
else
  echo "launch_mesh_pool: reseeding ${SOURCE_NODE} from ${DATASET_HOLDER}..."
  ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=3 "${SOURCE_NODE}" \
    "mkdir -p ${STERN_DATA_ROOT}/voxceleb2_shards && \
     rsync -az ${DATASET_HOLDER}:${STERN_DATA_ROOT}/voxceleb2_shards/ ${STERN_DATA_ROOT}/voxceleb2_shards/"
fi

echo "launch_mesh_pool: seeding mesh across ${TIER} pool (${#READY[@]} = ${#ACTIVE[@]} active + ${#BACKUP[@]} backup)..."
uv run python -m stern.mesh.seed_nodes \
  --nodes "${READY_CSV}" \
  --active-nodes "${ACTIVE_CSV}" \
  --source-node "${SOURCE_NODE}" \
  --shards-dir "${SHARDS_DIR:-/Data/voxceleb2_shards_8fps}" \
  --shard-map "${HOME}/stern_mesh/metadata/shard_map_${TIER}16.json" \
  --dataset-gb "${DATASET_GB:-841}" \
  --replication "${REPLICATION:-1}"

# Verify what actually landed on every node before anything trains against
# it. seed_nodes reports success per rsync, not per shard, so placement metadata
# can disagree with the disks without any error. validate_seeding writes
# verified_holdings.json, which _do_assigning then requires.
if [[ -n "${SKIP_SEED_VALIDATION:-}" ]]; then
  echo "launch_mesh_pool: WARNING -- SKIP_SEED_VALIDATION set, registering ${RUN_ID} without verifying seeded shards" >&2
else
  echo "launch_mesh_pool: validating seeded shards across ${TIER} pool..."
  # set -e aborts here on a non-zero exit, before the run is registered.
  # Pass --nodes, not the whole tier: the gate must cover exactly the roster
  # being launched. Unused nodes in the tier could fail the gate for a run that
  # never touches them.
  uv run python -m stern.mesh.validate_seeding \
    --pool "${TIER}" \
    --nodes "${READY_CSV}" \
    --metadata-dir "${HOME}/stern_mesh/metadata"
fi

# Merge once three quarters of the pool has checked in rather than waiting
# all-or-ROUND_TIMEOUT_S: one dead node would otherwise stall every round.
# Pair it with token weighting, which corrects a partial merge's sample-count
# bias instead of plain renormalization.
# Rounds per worker job, against the partition's 2h walltime. At the measured
# throughput (~28 samples/s, 9600 steps) a round is ~340s of training plus
# ~271s of merge, broadcast and barrier, about 611s. Eight rounds is 4888s,
# 68% of the 7200s cap.
#
# Four rounds was the old setting, from when a round took 1230s. The bottleneck
# was read_shard_frames_at re-scanning every shard's tar headers per window
# (CPU). With the indexed reader every tier is GPU-bound, so the tier spread
# that used to set the barrier is gone.
CHUNK_ROUNDS="${CHUNK_ROUNDS:-8}"
export CHUNK_ROUNDS

N_ACTIVE=$(awk -F, '{print NF}' <<< "${ACTIVE_CSV}")
QUORUM_K="${QUORUM_K:-$(( N_ACTIVE * 3 / 4 ))}"
[[ "${QUORUM_K}" -lt 1 ]] && QUORUM_K=1
WEIGHTING="${WEIGHTING:-token}"

echo "launch_mesh_pool: registering run config ${RUN_ID} (outer_lr=${OUTER_LR} outer_momentum=${OUTER_MOMENTUM} quorum_k=${QUORUM_K}/${N_ACTIVE} weighting=${WEIGHTING} chunk_rounds=${CHUNK_ROUNDS} grace_s=${GRACE_S:-default} extra_overrides=[${EXTRA_OVERRIDES:-}])..."
"$(dirname "${BASH_SOURCE[0]}")/register_mesh_run.sh" \
  "${RUN_ID}" "${ACTIVE_CSV}" "${ROUNDS}" "${OUTER_LR}" "${OUTER_MOMENTUM}" \
  "${QUORUM_K}" "${WEIGHTING}" "${EXTRA_OVERRIDES:-}"

echo "${ACTIVE_CSV}" > "runs/state/mesh_pool_active_${TIER}.txt"
echo "launch_mesh_pool: ${TIER} pool launched -- watcher cron will pick up ${RUN_ID} within a minute"
