#!/usr/bin/env bash
# video_encoding/scripts/view_mesh_tensorboard.sh
# On-demand TensorBoard viewer for a decentralized mesh (DiLoCo) training run.
#
# Mesh workers write TB event files to node-local /Data on whichever SLURM
# compute node hosts them (video_encoding/scripts/run_mesh_worker.sh), reachable
# only via SSH from the login-node login node -- laptop -> login-node -> compute node.
# This script looks up the run's fixed node roster (MeshRunConfig.nodes[0],
# video_encoding/watcher/mesh_tick.py -- stable for the run's whole lifetime,
# not re-picked per round), starts TensorBoard there if not already running,
# and opens a local SSH tunnel so the dashboard is reachable at
# http://localhost:3003.
#
# Both remote command hops below are base64-encoded before transmission: piping
# a literal multi-line/quoted command through two nested SSH hops re-parses it
# in the remote shell and silently mangles quoting (the same hazard flagged in
# seed_mesh.py's nested-ssh comments) -- base64 sidesteps that entirely.
#
# Usage: video_encoding/scripts/view_mesh_tensorboard.sh <run_id>
set -euo pipefail

RUN_ID="${1:?usage: view_mesh_tensorboard.sh <run_id>}"
REMOTE_HOST="login-node"
LOCAL_PORT=3003

RUN_ID_Q="$(printf '%q' "${RUN_ID}")"

# --- Step 1: resolve node + status in one SSH round trip to login-node --------
echo "view_mesh_tensorboard: looking up node for run_id ${RUN_ID_Q} on ${REMOTE_HOST}..." >&2

RESOLVE_PY=$(cat <<'PYEOF'
import json, sys
from pathlib import Path

run_id = sys.argv[1]
state_dir = Path.home() / "stern" / ".claude" / "state"

configs = json.loads((state_dir / "mesh_run_configs.json").read_text())
match = next((c for c in configs if c["run_id"] == run_id), None)
if match is None:
    print(f"ERROR: run_id {run_id!r} not found in mesh_run_configs.json", file=sys.stderr)
    sys.exit(1)
node = match["nodes"][0]

status = "UNKNOWN"
excluded_reason = ""
watcher_state_path = state_dir / "watcher_state.json"
if watcher_state_path.exists():
    watcher_state = json.loads(watcher_state_path.read_text())
    run_state = watcher_state.get("runs", {}).get(run_id)
    if run_state:
        status = run_state.get("status", status)
    excluded_reason = watcher_state.get("excluded_nodes", {}).get(node, "")

print(f"NODE={node}")
print(f"STATUS={status}")
print(f"EXCLUDED={excluded_reason}")
PYEOF
)
ENCODED_RESOLVE="$(printf '%s' "${RESOLVE_PY}" | base64 | tr -d '\n')"

RESOLVED="$(ssh "${REMOTE_HOST}" "echo ${ENCODED_RESOLVE} | base64 -d | python3 - ${RUN_ID_Q}")"
NODE="$(echo "${RESOLVED}" | sed -n 's/^NODE=//p')"
STATUS="$(echo "${RESOLVED}" | sed -n 's/^STATUS=//p')"
EXCLUDED="$(echo "${RESOLVED}" | sed -n 's/^EXCLUDED=//p')"

if [[ -z "${NODE}" ]]; then
    echo "view_mesh_tensorboard: failed to resolve a node for run_id ${RUN_ID_Q}" >&2
    exit 1
fi

echo "view_mesh_tensorboard: run ${RUN_ID_Q} -> node '${NODE}' (status: ${STATUS})" >&2
if [[ "${STATUS}" == "DONE" ]]; then
    echo "view_mesh_tensorboard: note -- run is DONE, showing final results" >&2
fi
if [[ -n "${EXCLUDED}" ]]; then
    echo "view_mesh_tensorboard: WARNING -- node '${NODE}' is in the watcher's excluded_nodes list (${EXCLUDED}); it may not be running a worker right now" >&2
fi

# --- Step 2: idempotently ensure TensorBoard is running on that node -------
INNER_SCRIPT=$(cat <<'EOF'
set -euo pipefail
if ss -H -tln 'sport = :3003' 2>/dev/null | grep -q .; then
  echo "view_mesh_tensorboard: tensorboard already listening on port 3003 on $(hostname)"
else
  cd ~/stern
  nohup setsid uv run tensorboard \
    --logdir "/Data/${USER}/stern_checkpoints/tb" \
    --port 3003 --host 127.0.0.1 \
    > /tmp/mesh_tensorboard.log 2>&1 < /dev/null &
  disown
  echo "view_mesh_tensorboard: launched tensorboard on $(hostname)"
fi
EOF
)
ENCODED_INNER="$(printf '%s' "${INNER_SCRIPT}" | base64 | tr -d '\n')"

ssh -o ServerAliveInterval=30 "${REMOTE_HOST}" \
    "ssh ${NODE} 'echo ${ENCODED_INNER} | base64 -d | bash'"

# --- Step 3: local tunnel (laptop side), kill-and-replace ------------------
# Port 3003 is global across run_ids -- silently reusing a stale tunnel from a
# *different* run_id would show the wrong run's data, so always tear down any
# existing tunnel before opening this one.
pkill -f "L ${LOCAL_PORT}:localhost:${LOCAL_PORT}" 2>/dev/null || true

if ! ssh -J "${REMOTE_HOST}" "${NODE}" -N -f -L "${LOCAL_PORT}:localhost:${LOCAL_PORT}"; then
    echo "view_mesh_tensorboard: ssh -J (ProxyJump) failed -- requires OpenSSH >= 7.3" >&2
    echo "view_mesh_tensorboard: fallback: ssh -o ProxyCommand=\"ssh ${REMOTE_HOST} -W %h:%p\" ${NODE} -N -f -L ${LOCAL_PORT}:localhost:${LOCAL_PORT}" >&2
    exit 1
fi

# --- Step 4: final message -------------------------------------------------
echo "view_mesh_tensorboard: http://localhost:${LOCAL_PORT}"
echo "view_mesh_tensorboard: tear down with: pkill -f \"L ${LOCAL_PORT}:localhost:${LOCAL_PORT}\""
