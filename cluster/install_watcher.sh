#!/usr/bin/env bash
# Converge THIS host onto the mesh watcher designation.
#
#   install_watcher.sh <host>   designate <host>, then converge this machine
#   install_watcher.sh          converge this machine against the existing
#                               designation (used by submit_to_cluster.sh)
#
# Converge means: install and enable the systemd --user timer if this machine is
# the designated host, disable it otherwise. Designation itself lives in
# runs/state/watcher_host on the shared NFS home and is read by
# watch_training.py on every tick, so a host whose timer survives a
# decommission still refuses to touch shared state -- this script only spares
# it from waking up once a minute to do so.
#
# Why a file and not flock: fcntl.flock provides no cross-host mutual exclusion
# on this NFSv3 export, so nothing stops two enabled timers from ticking at once.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
STATE_DIR="${REPO_ROOT}/runs/state"
HOST_FILE="${STATE_DIR}/watcher_host"
UNIT_DIR="${HOME}/.config/systemd/user"

# Short form on both sides: `hostname` may return the FQDN.
THIS_HOST="$(hostname)"; THIS_HOST="${THIS_HOST%%.*}"

if [[ $# -ge 1 ]]; then
  DESIGNATED="${1%%.*}"
  mkdir -p "${STATE_DIR}"
  printf '%s\n' "${DESIGNATED}" > "${HOST_FILE}"
  echo "Designated watcher host: ${DESIGNATED}"
else
  [[ -f "${HOST_FILE}" ]] || { echo "ERROR: no ${HOST_FILE}; run with a host argument to designate one" >&2; exit 1; }
  DESIGNATED="$(tr -d '[:space:]' < "${HOST_FILE}")"
  DESIGNATED="${DESIGNATED%%.*}"
  [[ -n "${DESIGNATED}" ]] || { echo "ERROR: ${HOST_FILE} is empty" >&2; exit 1; }
fi

if [[ "${THIS_HOST}" != "${DESIGNATED}" ]]; then
  # `stop`, never `disable`: ~/.config/systemd/user is on the shared NFS home,
  # so the timers.target.wants symlink that `enable` creates is visible to every
  # host. `disable` would remove the designated host's enablement too, and
  # whichever host converged last would win. `stop` is per-host runtime state;
  # it does not survive a reboot, which is fine because watch_training.py checks
  # the designation itself and exits without touching shared state.
  systemctl --user stop stern-watcher.timer 2>/dev/null || true
  echo "${THIS_HOST} is not the designated watcher (${DESIGNATED}); local timer stopped."
  exit 0
fi

mkdir -p "${UNIT_DIR}"
cat > "${UNIT_DIR}/stern-watcher.service" <<'UNIT'
[Unit]
Description=STERN mesh/watcher tick (one-shot, timer-driven)

[Service]
Type=oneshot
WorkingDirectory=%h/stern
# --no-sync is load-bearing: without it every tick re-syncs the 9.4 GB NFS venv,
# and overlapping syncs raced each other into ModuleNotFoundError crash loops.
ExecStart=%h/.local/bin/uv run --no-sync python -m stern.watcher.watch_training
StandardOutput=append:%h/stern/runs/state/watcher_systemd.log
StandardError=append:%h/stern/runs/state/watcher_systemd.log
UNIT
cat > "${UNIT_DIR}/stern-watcher.timer" <<'UNIT'
[Unit]
Description=Run stern-watcher.service every minute

[Timer]
# Wall-clock anchored, NOT OnUnitActiveSec: a monotonic-only timer
# (OnUnitActiveSec + OnBootSec) has no anchor to re-arm from once an
# activation is missed and the boot is long past, and then it stops silently.
# OnCalendar always has a next elapse.
OnCalendar=*:*:00
Persistent=true
AccuracySec=5s

[Install]
WantedBy=timers.target
UNIT

systemctl --user daemon-reload
systemctl --user enable --now stern-watcher.timer
# daemon-reload alone does not re-read an already-running timer's schedule.
systemctl --user restart stern-watcher.timer
echo "${THIS_HOST} is the designated watcher; timer installed and enabled."
