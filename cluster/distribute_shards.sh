#!/usr/bin/env bash
# cluster/distribute_shards.sh
# Copy the WDS shard set (build_wds_shards.py output) from the node it was
# built on to every other node in the fixed training roster (FIXED_NODELIST
# in submit_to_cluster.sh), so each node is self-sufficient: there are no
# cross-node reads during training on this 1GbE cluster. The transfer runs
# over the cluster's internal network, not the login node's SSH-only channel.
#
# Usage: distribute_shards.sh [source-node] [shard-dir-name]
#   source-node     node the shards were built on (default: STERN_SOURCE_NODE)
#   shard-dir-name  directory name under /Data/$USER (default: voxceleb2_shards)
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/cluster_env.sh"

SOURCE_NODE="${1:-${STERN_SOURCE_NODE}}"
SHARD_DIR="${2:-voxceleb2_shards}"

read -ra ALL_NODES <<< "${ALL_NODES:-${STERN_FACTORIAL_NODES}}"
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
