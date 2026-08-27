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
# Environment overrides: NEED (idle nodes to wait for), DATASET_HOLDER,
# EXPECTED_SHARDS, QUORUM_K, WEIGHTING, SKIP_SEED_VALIDATION, CHUNK_ROUNDS,
# EXPERIMENT (conf/experiment/<name>.yaml the workers compose; default
# videomae_mesh_ssl), GRACE_S
# (inherited by register_mesh_run.sh, not read here -- raise it above the
# 900s default for a roster whose slowest tier cannot finish a round in time),
# EXTRA_OVERRIDES (space-separated Hydra tokens, e.g.
# "model.mask_type=mixed model.target_mode=per_frame", forwarded verbatim to
# register_mesh_run.sh so every worker sbatch for this run carries them).
#
# POOL_NODES  explicit comma-separated roster, bypassing the tier idle poll.
#   The first positional then names the POOL LABEL, not a tier -- it keys
#   verified_holdings.json and runs/state/mesh_pool_*_<label>.txt, and need not
#   exist in tiers.tsv. Use it for a cross-tier pool (pick_cross_tier_pools.sh),
#   which is how an experiment arm is decoupled from a single GPU tier: with one
#   tier per arm, "arm" and "tier" are the same variable and no comparison
#   between arms is attributable to the arm.
#   Every listed node is active -- there are no seeded backups, because the
#   whole roster is what plan_placement spreads over, and planning over a
#   superset of the active nodes is what left the 2026-08-21 factorial arms
#   holding 192-250 of 260 shards. A node lost mid-run is replenished on demand
#   by watcher/shard_transfer.py rather than from a pre-seeded spare.
# POOL_SOURCE_NODE  node seed_mesh distributes from; default is the first of
#   the roster. Point it at a node that already holds the full shard set to
#   skip a 223 GB pre-stage.
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

if [[ -n "${POOL_NODES:-}" ]]; then
  IFS=',' read -r -a READY <<< "${POOL_NODES}"
  ACTIVE=("${READY[@]}")
  echo "launch_mesh_pool: ${TIER} explicit roster, ${#ACTIVE[@]} node(s), no idle poll"
else
  # 16 active + 6 backup. Overridable because not every tier is that big --
  # countries has only 19 nodes total, so the default can never be satisfied
  # and the poll loop spins forever.
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

# Active + backup, one hostname per line -- lets the watcher replenish a dead
# active node from a backup that already has a seeded
# /Data/$USER/mesh payload, instead of the replacement dying on
# read_my_shards' KeyError with an empty local shard dir. With POOL_NODES there
# are no backups here and replenishment goes through shard_transfer.py instead.
printf '%s\n' "${READY[@]}" > "runs/state/mesh_pool_ready_${TIER}.txt"

SOURCE_NODE="${POOL_SOURCE_NODE:-${ACTIVE[0]}}"
# Node holding the full 260-shard dataset that seed_mesh distributes from.
# This was node089 until the 2026-08-17 reboot wiped node-local /Data there;
# node001 is the node whose copy survived. Overridable because the holder is
# a fact about the cluster today, not a property of any tier.
DATASET_HOLDER="${DATASET_HOLDER:-node001}"
# Ask the node what it actually has rather than trusting a hostname: several
# tiers' first nodes already carry the full set, and reseeding one needlessly
# drags 223 GB across a 1GbE link for nothing.
EXPECTED_SHARDS="${EXPECTED_SHARDS:-260}"
HAVE=$(ssh -o ConnectTimeout=15 "${SOURCE_NODE}" \
  "ls /Data/$USER/voxceleb2_shards/*.tar 2>/dev/null | wc -l" || echo 0)
if [[ "${SOURCE_NODE}" == "${DATASET_HOLDER}" || "${HAVE}" -ge "${EXPECTED_SHARDS}" ]]; then
  echo "launch_mesh_pool: ${SOURCE_NODE} already holds ${HAVE} shard(s) -- no reseed needed"
else
  echo "launch_mesh_pool: reseeding ${SOURCE_NODE} from ${DATASET_HOLDER}..."
  ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=3 "${SOURCE_NODE}" \
    "mkdir -p /Data/$USER/voxceleb2_shards && \
     rsync -az ${DATASET_HOLDER}:/Data/$USER/voxceleb2_shards/ /Data/$USER/voxceleb2_shards/"
fi

echo "launch_mesh_pool: seeding mesh across ${TIER} pool (${#READY[@]} = ${#ACTIVE[@]} active + ${#BACKUP[@]} backup)..."
uv run python -m video_encoding.scripts.seed_mesh \
  --nodes "${READY_CSV}" \
  --active-nodes "${ACTIVE_CSV}" \
  --source-node "${SOURCE_NODE}" \
  --shards-dir /Data/$USER/voxceleb2_shards \
  --shard-map "${HOME}/stern_mesh/metadata/shard_map_${TIER}16.json" \
  --dataset-gb 223 \
  --replication 2

# Verify what actually landed on every node before anything trains against
# it. seed_mesh reports success per rsync, not per shard, and on 2026-08-20 a
# pool whose placement metadata disagreed with its disks trained zero samples
# for eleven hours without a single error. validate_seeding writes
# verified_holdings.json, which _do_assigning then requires.
if [[ -n "${SKIP_SEED_VALIDATION:-}" ]]; then
  echo "launch_mesh_pool: WARNING -- SKIP_SEED_VALIDATION set, registering ${RUN_ID} without verifying seeded shards" >&2
else
  echo "launch_mesh_pool: validating seeded shards across ${TIER} pool..."
  # set -e aborts here on a non-zero exit, before the run is registered.
  # --nodes, not the whole tier: the gate must cover exactly the roster being
  # launched. A cross-tier pool has no tier to look up, and for a tier pool the
  # unused two thirds of the tier could fail the gate for a run that never
  # touches them.
  uv run python -m video_encoding.scripts.validate_seeding \
    --pool "${TIER}" \
    --nodes "${READY_CSV}" \
    --metadata-dir "${HOME}/stern_mesh/metadata"
fi

# Merge once three quarters of the pool has checked in rather than waiting
# all-or-ROUND_TIMEOUT_S: one dead node otherwise costs every round a
# 30-minute stall. Pair it with token weighting, which corrects a partial
# merge's sample-count bias instead of plain renormalization.
# Rounds per worker job, against the partition's 2h walltime. Measured
# 2026-08-23 on idle node055 (cars) and node035 (birds), running the real
# dataset + model: 28.0 and 28.3 samples/s, so 9600 steps is ~340s of
# training on either tier, plus ~271s of merge/broadcast/barrier = ~611s a
# round. 8 rounds is 4888s, 68% of the 7200s cap.
#
# It was 4 when a round took 1230s. The tier spread that used to set the
# barrier is gone: the bottleneck was read_shard_frames_at re-scanning every
# shard's tar headers per window (CPU), and with the indexed reader every
# tier is GPU-bound on the same RTX 4000 Ada -- birds went 10.0 -> 28.3
# samp/s (x2.8) against cars' x1.8, and the two now land within 1% of each
# other.
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
