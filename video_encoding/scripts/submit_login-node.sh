#!/usr/bin/env bash
# video_encoding/scripts/submit_login-node.sh
# Full round trip: pull results, push code, pick idle nodes, submit sbatch.
#
# Usage: scripts/submit_login-node.sh <experiment-name> <N> [extra hydra overrides...]
#   <experiment-name>  name of a file in conf/experiment/ (without .yaml),
#                      e.g. videomae_lora_ssl_voxmini
#   <N>                number of nodes (recommend 8 for LoRA/SSL runs on fish per
#                      T1.7/T3 findings)
#
# Node selection: pick_nodes.sh runs on login-node and selects N idle nodes from
# the best available tier (fish → birds → cars → countries → orchids).
set -euo pipefail

EXPERIMENT="${1:?usage: submit_login-node.sh <experiment-name> <N> [hydra overrides...]}"
N="${2:?usage: submit_login-node.sh <experiment-name> <N> [hydra overrides...]}"
shift 2

REMOTE_HOST="login-node"
REMOTE_DIR="~/stern"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. Pull results from cluster, then push updated source.
"${SCRIPT_DIR}/sync_login-node.sh"

# 2. On login-node: pick idle nodes → refresh env → submit.
OVERRIDES="$(printf '%q ' "$@")"
ssh "${REMOTE_HOST}" bash -s <<REMOTE
set -euo pipefail
cd ${REMOTE_DIR}

NODELIST=\$(bash video_encoding/scripts/pick_nodes.sh ${N}) || {
  echo "ERROR: no ${N} idle nodes found — aborting submission" >&2; exit 1
}
echo "Submitting experiment=${EXPERIMENT} N=${N} nodelist=\${NODELIST}"

uv sync --quiet

sbatch \
  --nodes=${N} --ntasks=${N} \
  --nodelist="\${NODELIST}" \
  video_encoding/slurm/train.sbatch ${EXPERIMENT} ${OVERRIDES}
REMOTE
