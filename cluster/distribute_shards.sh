#!/usr/bin/env bash
# cluster/distribute_shards.sh
# Copy the WDS shard set (build_wds_shards.py output) from the node it was
# built on to the other three VoxCeleb2 data-staging fish nodes, so every
# node a training job might land on (per FIXED_NODELIST in submit_login-node.sh)
# is self-sufficient -- no cross-node reads during training on this 1GbE
# cluster. Data movement is local to the cluster's internal network, distinct
# from submit_login-node.sh's external SSH-only channel to login-node.
#
# Usage: distribute_shards.sh [source-node] [shard-dir-name]
#   source-node     node the shards were built on (default: node001)
#   shard-dir-name  directory name under /Data/$USER (default: voxceleb2_shards)
set -euo pipefail

SOURCE_NODE="${1:-node001}"
SHARD_DIR="${2:-voxceleb2_shards}"

ALL_NODES=(node001 node002 node003 node004)
TARGET_NODES=()
for node in "${ALL_NODES[@]}"; do
  [[ "${node}" != "${SOURCE_NODE}" ]] && TARGET_NODES+=("${node}")
done

for target in "${TARGET_NODES[@]}"; do
  echo "=== ${SOURCE_NODE} -> ${target} ==="
  ssh "${target}" \
    "rsync -avz -e ssh ${SOURCE_NODE}:/Data/\${USER}/${SHARD_DIR}/ /Data/\${USER}/${SHARD_DIR}/"
done

echo "Distribution complete: ${SOURCE_NODE} -> ${TARGET_NODES[*]}"
