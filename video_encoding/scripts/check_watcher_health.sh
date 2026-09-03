#!/usr/bin/env bash
# video_encoding/scripts/check_watcher_health.sh
#
# Self-check for the designated watcher host: re-converges stern-watcher.timer
# if it isn't active. Runs from system cron, deliberately NOT systemd --user --
# a dead/restarted user manager (the failure mode that caused the 2026-09-03
# outage) is invisible to anything that itself depends on that same manager,
# so this needs an independent trigger. Install with `crontab -e` on the
# designated host only (currently node089):
#
#   */15 * * * * XDG_RUNTIME_DIR=/run/user/$(id -u) $HOME/stern/video_encoding/scripts/check_watcher_health.sh >> $HOME/stern/runs/state/watcher_health_cron.log 2>&1
#
# Requires loginctl linger enabled for XDG_RUNTIME_DIR to exist outside a login
# session (`loginctl enable-linger $USER`; already on as of 2026-09-03).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HOST_FILE="${REPO_ROOT}/runs/state/watcher_host"

THIS_HOST="$(hostname)"; THIS_HOST="${THIS_HOST%%.*}"

[[ -f "${HOST_FILE}" ]] || exit 0
DESIGNATED="$(tr -d '[:space:]' < "${HOST_FILE}")"
[[ "${THIS_HOST}" == "${DESIGNATED}" ]] || exit 0

STATE="$(systemctl --user is-active stern-watcher.timer 2>/dev/null || true)"
if [[ "${STATE}" != "active" ]]; then
  echo "$(date -Is) [${THIS_HOST}] stern-watcher.timer state='${STATE}', re-converging"
  "${SCRIPT_DIR}/install_watcher.sh"
fi
