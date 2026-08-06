#!/usr/bin/env bash
# video_encoding/scripts/sync_login-node.sh
# Bidirectional sync with login-node: pull cluster results first, then push
# source code. The pull-before-push order prevents --delete from wiping
# result files written by SLURM jobs between syncs.
#
# Usage: scripts/sync_login-node.sh [extra rsync flags forwarded to the push step]
set -euo pipefail

REMOTE_HOST="login-node"
REMOTE_DIR="~/stern"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# 1. Pull results from cluster → local (non-fatal: dir may not exist yet)
mkdir -p "${REPO_ROOT}/runs/results"
rsync -avz \
  "${REMOTE_HOST}:${REMOTE_DIR}/runs/results/" \
  "${REPO_ROOT}/runs/results/" || true

# 2. Push source → login-node (--delete is safe: results now exist locally)
rsync -avz --delete \
  --exclude-from="${REPO_ROOT}/.rsync-exclude" \
  "${REPO_ROOT}/" "${REMOTE_HOST}:${REMOTE_DIR}/" \
  "$@"
