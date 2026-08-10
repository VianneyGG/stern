#!/usr/bin/env bash
# video_encoding/scripts/view_mesh_tensorboard.sh
# On-demand TensorBoard viewer for a decentralized mesh (DiLoCo) training run.
#
# TensorBoard scalars are written once per round by the merge step
# (video_encoding/scripts/run_outer_step.py), which always runs on the
# login-node login node (video_encoding/watcher/mesh_tick.py::_do_merging),
# into /Data/$USER/stern_mesh_local/<run_id>/tb -- watcher-local NVMe, never
# an ephemeral compute node. Viewing is therefore always a single SSH hop to
# login-node: TensorBoard keeps running there independent of this laptop's
# connection, so recovering after an SSH drop is just rerunning this script.
#
# The remote command below is base64-encoded before transmission: piping a
# literal multi-line/quoted command through SSH re-parses it in the remote
# shell and silently mangles quoting (the same hazard flagged in
# seed_mesh.py's nested-ssh comments) -- base64 sidesteps that entirely.
#
# Usage: video_encoding/scripts/view_mesh_tensorboard.sh <run_id>
set -euo pipefail

RUN_ID="${1:?usage: view_mesh_tensorboard.sh <run_id>}"
REMOTE_HOST="login-node"
LOCAL_PORT=3003

RUN_ID_Q="$(printf '%q' "${RUN_ID}")"

# --- Step 1: idempotently ensure TensorBoard is running on login-node ---------
INNER_SCRIPT=$(cat <<'EOF'
set -euo pipefail
RUN_ID="$1"
TB_DIR="/Data/${USER}/stern_mesh_local/${RUN_ID}/tb"
if [[ ! -d "${TB_DIR}" ]]; then
  echo "view_mesh_tensorboard: ERROR -- no TB dir at ${TB_DIR} (unknown run_id, or its first merge hasn't landed yet)" >&2
  exit 1
fi
if ss -H -tln 'sport = :3003' 2>/dev/null | grep -q .; then
  echo "view_mesh_tensorboard: tensorboard already listening on port 3003 on $(hostname)"
else
  cd ~/stern
  nohup setsid uv run tensorboard \
    --logdir "${TB_DIR}" \
    --port 3003 --host 127.0.0.1 \
    > /tmp/mesh_tensorboard.log 2>&1 < /dev/null &
  disown
  echo "view_mesh_tensorboard: launched tensorboard on $(hostname)"
fi
EOF
)
ENCODED_INNER="$(printf '%s' "${INNER_SCRIPT}" | base64 | tr -d '\n')"

ssh -o ServerAliveInterval=30 "${REMOTE_HOST}" \
    "echo ${ENCODED_INNER} | base64 -d | bash -s -- ${RUN_ID_Q}"

# --- Step 2: local tunnel (laptop side), kill-and-replace -------------------
# Port 3003 is global across run_ids -- silently reusing a stale tunnel from a
# *different* run_id would show the wrong run's data, so always tear down any
# existing tunnel before opening this one.
pkill -f "L ${LOCAL_PORT}:localhost:${LOCAL_PORT}" 2>/dev/null || true

ssh "${REMOTE_HOST}" -N -f -L "${LOCAL_PORT}:localhost:${LOCAL_PORT}"

# --- Step 3: final message --------------------------------------------------
echo "view_mesh_tensorboard: http://localhost:${LOCAL_PORT}"
echo "view_mesh_tensorboard: tear down with: pkill -f \"L ${LOCAL_PORT}:localhost:${LOCAL_PORT}\""
