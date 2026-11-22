#!/usr/bin/env bash
# cluster/view_mesh_tensorboard.sh
# On-demand TensorBoard viewer for decentralized mesh (DiLoCo) training runs.
#
# TensorBoard scalars are written once per round by the merge step
# (stern/mesh/run_outer_step.py) into
# /Data/$USER/stern_mesh_local/<run_id>/tb on whichever host runs the merge.
# --logdir points at the *parent* of that (/Data/$USER/stern_mesh_local), so
# every run_id's tb/ subdir shows up as its own run in the TensorBoard run
# selector -- runs can be compared side by side, and no run_id/exp name needs
# to be passed to this script.
#
# Host: the results host is preferred and the login node is the fallback. Exactly
# one of them serves TensorBoard at a time, picked by SSH reachability, since the
# results host may be offline. On failover, any stray instance left running on
# the other candidate host is killed so viewers are never split across two
# hosts or ports.
#
# The remote command below is base64-encoded before transmission. Piping a
# literal multi-line, quoted command through SSH re-parses it in the remote
# shell and silently mangles quoting; base64 avoids that.
#
# Usage: cluster/view_mesh_tensorboard.sh
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/cluster_env.sh"

LOCAL_PORT=3003
CANDIDATE_HOSTS=("${STERN_RESULTS_HOST}" "${STERN_LOGIN_HOST}")

# --- Step 0: pick the first reachable host, preferring the results host -------------
REMOTE_HOST=""
for host in "${CANDIDATE_HOSTS[@]}"; do
  if ssh -o ConnectTimeout=5 -o BatchMode=yes "${host}" true 2>/dev/null; then
    REMOTE_HOST="${host}"
    break
  fi
done
if [[ -z "${REMOTE_HOST}" ]]; then
  echo "view_mesh_tensorboard: ERROR -- neither ${STERN_RESULTS_HOST} nor ${STERN_LOGIN_HOST} is reachable" >&2
  exit 1
fi
echo "view_mesh_tensorboard: using ${REMOTE_HOST}"

# Best-effort: stop any stray instance on the other candidate host so only
# one node ever serves TensorBoard at a time.
for host in "${CANDIDATE_HOSTS[@]}"; do
  [[ "${host}" == "${REMOTE_HOST}" ]] && continue
  ssh -o ConnectTimeout=5 -o BatchMode=yes "${host}" \
    "pkill -f 'tensorboard.*--port ${LOCAL_PORT}' 2>/dev/null || true" \
    >/dev/null 2>&1 || true
done

# --- Step 1: idempotently ensure TensorBoard is running on ${REMOTE_HOST} --
INNER_SCRIPT=$(cat <<'EOF'
set -euo pipefail
TB_DIR="/Data/${USER}/stern_mesh_local"
if [[ ! -d "${TB_DIR}" ]]; then
  echo "view_mesh_tensorboard: ERROR -- no TB root at ${TB_DIR} on $(hostname)" >&2
  exit 1
fi
if ss -H -tln 'sport = :__LOCAL_PORT__' 2>/dev/null | grep -q .; then
  echo "view_mesh_tensorboard: tensorboard already listening on port __LOCAL_PORT__ on $(hostname)"
else
  cd ~/stern
  nohup setsid uv run tensorboard \
    --logdir "${TB_DIR}" \
    --port __LOCAL_PORT__ --host 127.0.0.1 \
    > /tmp/mesh_tensorboard.log 2>&1 < /dev/null &
  disown
  echo "view_mesh_tensorboard: launched tensorboard on $(hostname), showing all runs under ${TB_DIR}"
fi
EOF
)
# __LOCAL_PORT__ substituted here (not interpolated in the heredoc above,
# which stays single-quoted so ${TB_DIR}/${USER}/$(hostname) expand
# remotely, not on this laptop) -- one LOCAL_PORT value drives every use of
# the port in this script, so a future port change can't drift between the
# tensorboard --port flag, the ss listen check, and the failover pkill above.
INNER_SCRIPT="${INNER_SCRIPT//__LOCAL_PORT__/${LOCAL_PORT}}"
ENCODED_INNER="$(printf '%s' "${INNER_SCRIPT}" | base64 | tr -d '\n')"

ssh -o ServerAliveInterval=30 "${REMOTE_HOST}" \
    "echo ${ENCODED_INNER} | base64 -d | bash -s --"

# --- Step 2: local tunnel (laptop side), kill-and-replace -------------------
# The tunnel target may have changed hosts since the last run (results host <->
# login node failover) -- always tear down any existing tunnel before opening
# this one so it can't silently point at a dead host.
pkill -f "L ${LOCAL_PORT}:localhost:${LOCAL_PORT}" 2>/dev/null || true

ssh "${REMOTE_HOST}" -N -f -L "${LOCAL_PORT}:localhost:${LOCAL_PORT}"

# --- Step 3: final message --------------------------------------------------
echo "view_mesh_tensorboard: http://localhost:${LOCAL_PORT}  (host: ${REMOTE_HOST})"
echo "view_mesh_tensorboard: tear down with: pkill -f \"L ${LOCAL_PORT}:localhost:${LOCAL_PORT}\""
