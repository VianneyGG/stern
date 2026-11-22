#!/usr/bin/env bash
# cluster/submit_to_cluster.sh
# Full round trip: pull results, push code, pick idle nodes, submit sbatch.
#
# Usage: cluster/submit_to_cluster.sh <experiment-name> <N> [extra hydra overrides...]
#   <experiment-name>  name of a file in configs/experiment/ (without .yaml),
#                      e.g. videomae_ssl_pretrain_voxmini
#   <N>                number of nodes (8 is the recommended size for LoRA/SSL
#                      runs on the fish tier)
#
# Node selection: pick_nodes.sh runs on the login node and selects N idle nodes from
# the best available tier (fish → birds → cars → countries → orchids).
set -euo pipefail

# Class-hours gate (class_hours.py): refuse to launch anything that would
# run Mon-Fri 08:00-20:00 Europe/Paris. No override.
python3 "$(dirname "${BASH_SOURCE[0]}")/../stern/watcher/class_hours.py" || exit 1

EXPERIMENT="${1:?usage: submit_to_cluster.sh <experiment-name> <N> [hydra overrides...]}"
N="${2:?usage: submit_to_cluster.sh <experiment-name> <N> [hydra overrides...]}"
shift 2

source "$(dirname "${BASH_SOURCE[0]}")/cluster_env.sh"

REMOTE_HOST="${REMOTE_HOST:-${STERN_LOGIN_HOST}}"
REMOTE_DIR="~/stern"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. Pull results from cluster, then push updated source.
"${SCRIPT_DIR}/sync_to_cluster.sh"

# 2. On the login node: pick idle nodes → refresh env → submit.
#    FIXED_NODELIST (env) pins to specific nodes and skips pick_nodes — required
#    when data is staged node-locally (e.g. VoxCeleb2 on a fixed set of nodes).
# Extract run_tag from Hydra overrides (e.g. "+run_tag=p2-eb256"), if present,
# so the jobs registry entry can be joined back to sweep_ssl.sh's per-cell
# tagging by stern/watcher/results.py.
RUN_TAG=""
for arg in "$@"; do
  case "$arg" in
    +run_tag=*|run_tag=*) RUN_TAG="${arg#*run_tag=}" ;;
  esac
done

OVERRIDES="$(printf '%q ' "$@")"
FIXED_NODELIST="${FIXED_NODELIST:-}"
ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=3 "${REMOTE_HOST}" bash -s <<REMOTE
set -euo pipefail
cd ${REMOTE_DIR}

if [ -n "${FIXED_NODELIST}" ]; then
  NODELIST="${FIXED_NODELIST}"
else
  NODELIST=\$(bash cluster/pick_nodes.sh ${N}) || {
    echo "ERROR: no ${N} idle nodes found — aborting submission" >&2; exit 1
  }
fi
echo "Submitting experiment=${EXPERIMENT} N=${N} nodelist=\${NODELIST}"

# cu128 torch wheels are multi-GB; keep uv's cache off the tight home quota.
export UV_CACHE_DIR="\${UV_CACHE_DIR:-/tmp/uv_cache_\${USER}}"
uv sync --quiet

SBATCH_OUT=\$(sbatch \
  --nodes=${N} --ntasks=${N} \
  --nodelist="\${NODELIST}" \
  cluster/slurm/train.sbatch ${EXPERIMENT} ${OVERRIDES})
echo "\${SBATCH_OUT}"

# Ground truth for the training watcher's "dead" detector. Without this, a job
# that never appears in squeue (accepted by sbatch, then purged before the
# first poll) looks the same as a job that was never submitted.
JOB_ID=\$(echo "\${SBATCH_OUT}" | awk '/Submitted batch job/ {print \$NF}')
if [ -n "\${JOB_ID}" ]; then
  PYTHONPATH=. uv run python stern/watcher/append_job_registry.py \
    --job-id "\${JOB_ID}" --experiment "${EXPERIMENT}" \
    --run-tag "${RUN_TAG}" --nodelist "\${NODELIST}"
else
  echo "WARNING: could not parse job id from sbatch output; jobs_registry not updated" >&2
fi

# Converge this host onto the watcher designation in runs/state/watcher_host.
# Installing the timer unconditionally would make every submitting host a
# second watcher, and flock does not exclude hosts from each other on this NFS
# home. The installer enables the timer only on the designated host and stops
# it everywhere else. Idempotent either way.
#
# Cron is no longer used for the watcher. The legacy crontab entry is removed
# here for hosts that still have one.
if crontab -l 2>/dev/null | grep -q "watch_training"; then
  (crontab -l 2>/dev/null | grep -v "watch_training") | crontab - || crontab -r
  echo "Removed legacy training watcher crontab entry."
fi
\$HOME/stern/cluster/install_watcher.sh
REMOTE
