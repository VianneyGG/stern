#!/usr/bin/env bash
# video_encoding/scripts/submit_login-node.sh
# Full round trip: pull results, push code, pick idle nodes, submit sbatch.
#
# Usage: scripts/submit_login-node.sh <experiment-name> <N> [extra hydra overrides...]
#   <experiment-name>  name of a file in conf/experiment/ (without .yaml),
#                      e.g. videomae_ssl_pretrain_voxmini
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
#    FIXED_NODELIST (env) pins to specific nodes and skips pick_nodes — required
#    when data is staged node-locally (e.g. VoxCeleb2 on node001,node002,...).
# Extract run_tag from Hydra overrides (e.g. "+run_tag=p2-eb256"), if present,
# so the jobs registry entry can be joined back to sweep_ssl.sh's per-cell
# tagging by video_encoding/watcher/results.py.
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
  NODELIST=\$(bash video_encoding/scripts/pick_nodes.sh ${N}) || {
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
  video_encoding/slurm/train.sbatch ${EXPERIMENT} ${OVERRIDES})
echo "\${SBATCH_OUT}"

# Ground truth for the training watcher's "dead" detector (spec §3): without
# this, a job that never appears in squeue at all (e.g. sbatch accepted it
# but it was purged before the first poll) is indistinguishable from "no job
# was ever submitted."
JOB_ID=\$(echo "\${SBATCH_OUT}" | awk '/Submitted batch job/ {print \$NF}')
if [ -n "\${JOB_ID}" ]; then
  uv run python video_encoding/scripts/append_job_registry.py \
    --job-id "\${JOB_ID}" --experiment "${EXPERIMENT}" \
    --run-tag "${RUN_TAG}" --nodelist "\${NODELIST}"
else
  echo "WARNING: could not parse job id from sbatch output; jobs_registry not updated" >&2
fi

# Ensure the training watcher's cron entry is installed (video_encoding/watcher/*,
# commit ec1a3be). It's cron-driven and stateless between ticks by design
# (watcher/state.py: "no long-lived process -- it's cron-driven"), so
# installing/verifying the crontab line is the correct launch mechanism --
# not a PID file or screen/tmux session. grep-guarded so re-running
# submit_login-node.sh never installs a duplicate line.
if ! crontab -l 2>/dev/null | grep -q "watch_training"; then
  # cron's own PATH (/usr/bin:/bin on this cluster) doesn't include
  # ~/.local/bin, where uv lives -- an unqualified "uv" here fails silently
  # every tick with no watcher.log entry at all. \$HOME/.local/bin/uv is
  # robust since cron always sets HOME for the crontab owner. cron_stderr.log
  # captures anything that fails before our own logging can run.
  (crontab -l 2>/dev/null; echo "* * * * * cd ${REMOTE_DIR} && \$HOME/.local/bin/uv run python -m video_encoding.scripts.watch_training >> ${REMOTE_DIR}/.claude/state/cron_stderr.log 2>&1") | crontab -
  echo "Installed training watcher crontab entry."
fi
REMOTE
