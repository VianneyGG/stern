#!/usr/bin/env bash
# Per-node HTTP server for the P2P mesh, scoped to the mesh dir only.
#
# Serves ONLY /Data/$USER/mesh, never /Data itself: /Data is
# shared sticky world-writable scratch holding other students' data, and
# http.server serves everything under its target directory to any peer
# that connects. chmod 700 restricts local filesystem access but does NOT
# restrict this server -- pointing it at /Data would expose all of it to
# the subnet. No authentication (declined during design; the cluster
# subnet is treated as trusted).
set -euo pipefail

MESH_DIR="/Data/$USER/mesh"
PORT="${MESH_PORT:-8000}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

usage() {
  echo "usage: $(basename "$0") [start|install-cron]" >&2
  exit 2
}

cmd_start() {
  mkdir -p "${MESH_DIR}"
  chmod 700 "${MESH_DIR}"

  if curl -fs --max-time 2 "http://127.0.0.1:${PORT}/" >/dev/null 2>&1; then
    echo "serve_shards: already serving on port ${PORT}"
    return 0
  fi

  (cd "${MESH_DIR}" && nohup python3 -m http.server "${PORT}" >/dev/null 2>&1 &)
  echo "serve_shards: started http.server on port ${PORT}, serving ${MESH_DIR}"
}

cmd_install_cron() {
  local start_cmd="cd ${REPO_DIR} && MESH_PORT=${PORT} video_encoding/scripts/serve_shards.sh start"

  if ! crontab -l 2>/dev/null | grep -q "serve_shards.sh start"; then
    (
      crontab -l 2>/dev/null
      echo "@reboot ${start_cmd}"
      echo "*/5 * * * * ${start_cmd}"
    ) | crontab -
    echo "serve_shards: installed @reboot and */5 liveness crontab entries"
  else
    echo "serve_shards: crontab entries already installed"
  fi
}

case "${1:-}" in
  start) cmd_start ;;
  install-cron) cmd_install_cron ;;
  *) usage ;;
esac
