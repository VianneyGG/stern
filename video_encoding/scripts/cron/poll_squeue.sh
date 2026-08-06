#!/usr/bin/env bash
# video_encoding/scripts/cron/poll_squeue.sh
# Snapshot `squeue` state to runs/state/squeue.json. No model call --
# this is pure data capture, meant to run every 60s on login-node once jobs
# are being submitted. Not installed to crontab yet; see conf/CONFIGS.md.
#
# Usage: scripts/cron/poll_squeue.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
STATE_DIR="${REPO_ROOT}/runs/state"
LOG="${STATE_DIR}/cron.log"
LOCK="${STATE_DIR}/poll_squeue.lock"
mkdir -p "${STATE_DIR}"

exec 200>"${LOCK}"
if ! flock -n 200; then
  echo "$(date -Iseconds) poll_squeue skip: previous run still holding lock" >> "${LOG}"
  exit 0
fi

TMP="${STATE_DIR}/squeue.json.tmp"
OUT="${STATE_DIR}/squeue.json"

if squeue -u "${USER}" --json > "${TMP}" 2>>"${LOG}"; then
  mv "${TMP}" "${OUT}"
  echo "$(date -Iseconds) poll_squeue ok" >> "${LOG}"
else
  rc=$?
  echo "$(date -Iseconds) poll_squeue FAILED (squeue exit ${rc})" >> "${LOG}"
  rm -f "${TMP}"
fi
