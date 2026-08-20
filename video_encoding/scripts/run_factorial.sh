#!/usr/bin/env bash
# video_encoding/scripts/run_factorial.sh
#
# Launches the mask_type x target_mode factorial: four solo SSL runs that test
# why the mesh runs learned so little (train_loss 0.308 -> 0.256 over 110
# rounds on pool-cars-best-120r).
#
#   H1  the reconstruction target is time-averaged (mae.py's .mean(dim=3)),
#       so motion is erased before the loss ever sees it   -> target_mode
#   H2  tube masking guarantees every masked spatial position is masked in
#       EVERY frame, so there is no visible temporal twin to copy from and no
#       pressure to model motion                            -> mask_type
#
# Arms are <mask_type>-<target_mode> over {tube,mixed} x {time_avg,per_frame}.
#
# Three things this deliberately does NOT do:
#
#   1. It never calls sync_login-node.sh and never writes to ~/stern. Code is
#      deployed to a separate ~/stern_factorial checkout, so the live sweep and
#      its watcher keep running against the code they were launched with. The
#      local tree cannot load the cluster's checkpoints anyway (the ViT
#      refactor renamed 128 parameters), which is exactly why overwriting
#      ~/stern would break the running jobs.
#   2. It never writes to NFS home. MESH_ROOT and MESH_SHARED both point into
#      node-local /Data, so the ~7.7 GB of remaining home quota is untouched --
#      4 arms x 12 rounds x 183 MB of checkpoint pushes would otherwise blow
#      straight through it.
#   3. It does not merge. Each arm is a solo node with mesh.merge=false (the
#      ab-single_node reference mode), so there is no quorum, no barrier and
#      no watcher involvement.
#
# Why the mesh entrypoint for a non-mesh experiment: it is the only training
# path that can reach the data. pretrain.py reads voxmini's manifest.csv,
# which exists on no node; the WebDataset shards staged on the fish tier are
# readable only through MeshShardIterableDataset.
#
# Every arm trains on a byte-identical shard subset with the same seed, so the
# only differences between the four are the two knobs under test.
#
# Usage:
#   run_factorial.sh --dry-run          # print every remote command, do nothing
#   run_factorial.sh stage              # deploy code + stage shards
#   run_factorial.sh submit             # sbatch the four arms
#
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-login-node}"
REMOTE_DIR="${REMOTE_DIR:-\$HOME/stern_factorial}"
# node001 holds the full 260-shard VoxCeleb2 set plus manifest.json and val/;
# the other three hold only their own mesh subsets, which is why the subset is
# copied from node001 rather than reusing whatever each node happens to have.
SOURCE_NODE="${SOURCE_NODE:-node001}"
SOURCE_DATA="/Data/$USER/voxceleb2_shards"
NODES="${NODES:-node001 node002 node004 node007}"
DATA_ROOT="/Data/$USER/factorial"
SHARED_ROOT="/Data/$USER/factorial_shared"
EXPERIMENT="${EXPERIMENT:-videomae_mesh_diloco}"

# 12 shards ~= 50k windows per pass, comfortably more than the 32000
# micro-batches a round consumes, so no round ends early on an exhausted
# dataset (which would silently shorten one arm).
N_SHARDS="${N_SHARDS:-12}"
# rounds x steps_per_round / accum_steps(128) = 12 x 32000 / 128 = 3000
# optimizer steps at the tuned eff_batch of 128. At the ~11 samp/s the cluster
# actually sustains on this path, that is ~10 h per arm, all four in parallel.
ROUNDS="${ROUNDS:-12}"
SEED="${SEED:-0}"

DRY_RUN=0
case "${1:-}" in
  --dry-run) DRY_RUN=1; ACTION="all" ;;
  stage)     ACTION="stage" ;;
  submit)    ACTION="submit" ;;
  *) echo "usage: run_factorial.sh [--dry-run|stage|submit]" >&2; exit 1 ;;
esac

run() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '  %s\n' "$*"
  else
    "$@"
  fi
}

# Shard names are derived here, once, and reused for every node so all four
# arms see the same data. Deterministic (a fixed prefix of the sorted list),
# not sampled, so a re-run stages the identical subset.
shard_list() {
  local i
  for ((i = 0; i < N_SHARDS; i++)); do
    printf 'shard-%06d.tar\n' "$i"
  done
}
SHARDS="$(shard_list | tr '\n' ' ')"

echo "run_factorial: experiment=${EXPERIMENT} nodes=${NODES}"
echo "run_factorial: ${N_SHARDS} shards x ${ROUNDS} rounds, seed=${SEED}"

if [[ "${ACTION}" == "stage" || "${ACTION}" == "all" ]]; then
  echo
  echo "--- 1. deploy code to ${REMOTE_HOST}:${REMOTE_DIR} (never ~/stern) ---"
  run rsync -az --delete \
    --exclude '.venv' --exclude '.git' --exclude 'runs' \
    --exclude '__pycache__' --exclude 'video_encoding/slurm/logs' \
    --exclude 'graphify-out' --exclude '*.pt' \
    ./ "${REMOTE_HOST}:stern_factorial/"
  # The 9.4 GB venv is on NFS and must not be duplicated; --no-sync means uv
  # never tries to rebuild it, so a symlink is enough.
  run ssh "${REMOTE_HOST}" \
    "ln -sfn \$HOME/stern/.venv \$HOME/stern_factorial/.venv && mkdir -p \$HOME/stern_factorial/video_encoding/slurm/logs"

  echo
  echo "--- 2. stage an identical shard subset on each node ---"
  for node in ${NODES}; do
    echo "  [${node}]"
    # Copies straight from node001 node-to-node rather than through NFS home.
    run ssh "${REMOTE_HOST}" "timeout 3600 ssh -o StrictHostKeyChecking=no ${node} '
      set -e
      mkdir -p ${DATA_ROOT} ${SHARED_ROOT}/metadata
      if [ \"${node}\" = \"${SOURCE_NODE}\" ]; then SRC=\"\"; else SRC=\"${SOURCE_NODE}:\"; fi
      rsync -a \${SRC}${SOURCE_DATA}/manifest.json ${DATA_ROOT}/
      for s in ${SHARDS}; do
        [ -f ${DATA_ROOT}/\$s ] || rsync -a \${SRC}${SOURCE_DATA}/\$s ${DATA_ROOT}/
      done
      # val/ is replicated in full to every node, matching seed_mesh.py, so
      # val_loss is computed on identical data across all four arms.
      mkdir -p ${DATA_ROOT}/val
      rsync -a \${SRC}${SOURCE_DATA}/val/ ${DATA_ROOT}/val/
      # {hostname: [shard, ...]} -- every node gets the SAME list.
      python3 -c \"
import json, socket
shards = \\\"${SHARDS}\\\".split()
host = socket.gethostname().split(\\\".\\\", 1)[0]
json.dump({host: shards}, open(\\\"${SHARED_ROOT}/metadata/shard_assignments_factorial.json\\\", \\\"w\\\"))
\"
      # Empty shard map: every shard is local, so nothing is ever fetched P2P.
      echo {} > ${SHARED_ROOT}/metadata/shard_map.json
    '"
  done
fi

if [[ "${ACTION}" == "submit" || "${ACTION}" == "all" ]]; then
  echo
  echo "--- 3. submit the four arms ---"
  set -- ${NODES}
  for mask_type in tube mixed; do
    for target_mode in time_avg per_frame; do
      arm="${mask_type}-${target_mode}"
      node="$1"; shift
      run_id="factorial-${arm}"
      echo "  [${arm}] -> ${node}"
      run ssh "${REMOTE_HOST}" "cd \$HOME/stern_factorial && \
        MESH_ROOT=${DATA_ROOT} \
        MESH_SHARED=${SHARED_ROOT} \
        MESH_ASSIGNMENTS=${SHARED_ROOT}/metadata/shard_assignments_factorial.json \
        MESH_SHARD_MAP=${SHARED_ROOT}/metadata/shard_map.json \
        STERN_CHECKPOINT_ROOT=${DATA_ROOT}/checkpoints/${run_id} \
        sbatch --nodes=1 --ntasks=1 --nodelist=${node} \
          --job-name=stern-factorial-${arm} \
          video_encoding/slurm/mesh_train.sbatch \
          ${EXPERIMENT} ${run_id} \
          +run_tag=${arm} \
          model.mask_type=${mask_type} \
          model.target_mode=${target_mode} \
          train.seed=${SEED} \
          mesh.merge=false \
          mesh.rounds=${ROUNDS}"
    done
  done
fi

cat <<'EOF'

run_factorial: how to read the result.

  Do NOT compare train_loss across target_mode. time_avg and per_frame
  optimize different targets on different scales, so a lower number there
  means nothing. Compare arms on a target-agnostic readout:

    probe_motion_decodability.py  -- ridge R^2 predicting frame-diff motion
                                     from frozen features. This is the
                                     headline number.

  H1 is supported if the per_frame arms decode motion better than the
  time_avg arms. H2 is supported if the mixed arms beat the tube arms. If
  neither separates, the bad training has a different cause and the next
  suspect is H3 (sampling stride: at 25 fps a tubelet_size of 2 spans 80 ms,
  so consecutive windows may simply contain no motion to learn).
EOF
