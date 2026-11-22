#!/usr/bin/env bash
# cluster/serve_shards_public.sh
#
# NOT the mesh's serve_shards.sh (that one serves /Data/<user>/mesh to the
# trusted cluster subnet, unauthenticated, via crontab). This script serves
# a VoxCeleb2-derived shard directory to the public internet via
# cloudflared, so it puts a random token in the URL path -- an
# unauthenticated bare tunnel would publish licensed corpus data at a
# guessable/loggable URL.
#
# Usage: scripts/serve_shards_public.sh <shards-dir> <manifest-csv-path> [port] [--local-only]
#   <shards-dir>        directory of shard-*.tar + manifest.json (build_wds_shards.py output)
#   <manifest-csv-path> path to the dataset's manifest.csv (ManifestRecord CSV)
#   [port]              default 8080
#   --local-only         skip the cloudflared tunnel; serve on localhost only (used by tests)
set -euo pipefail

usage() {
  echo "usage: $(basename "$0") <shards-dir> <manifest-csv-path> [port] [--local-only]" >&2
  exit 2
}

LOCAL_ONLY=0
POSITIONAL=()
for arg in "$@"; do
  if [ "$arg" = "--local-only" ]; then
    LOCAL_ONLY=1
  else
    POSITIONAL+=("$arg")
  fi
done
set -- "${POSITIONAL[@]}"

[ $# -ge 2 ] || usage
SHARDS_DIR="$1"
MANIFEST_CSV="$2"
PORT="${3:-8080}"

SHARDS_DIR="$(cd "$SHARDS_DIR" && pwd)"
MANIFEST_CSV="$(cd "$(dirname "$MANIFEST_CSV")" && pwd)/$(basename "$MANIFEST_CSV")"

TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')"
SERVE_ROOT="$(mktemp -d)"
SERVE_DIR="$SERVE_ROOT/$TOKEN"
mkdir -p "$SERVE_DIR"

shopt -s nullglob
tar_files=("$SHARDS_DIR"/*.tar)
shopt -u nullglob
if [ "${#tar_files[@]}" -eq 0 ]; then
  echo "serve_shards_public: no *.tar files found under $SHARDS_DIR" >&2
  exit 1
fi
for f in "${tar_files[@]}"; do
  ln -s "$f" "$SERVE_DIR/"
done
ln -s "$SHARDS_DIR/manifest.json" "$SERVE_DIR/manifest.json"
ln -s "$MANIFEST_CSV" "$SERVE_DIR/manifest.csv"

HTTP_PID=""
cleanup() {
  [ -n "$HTTP_PID" ] && kill "$HTTP_PID" 2>/dev/null || true
  rm -rf "$SERVE_ROOT"
}
trap cleanup EXIT

python3 - "$PORT" "$SERVE_ROOT" <<'PYEOF' &
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

port, root = int(sys.argv[1]), sys.argv[2]


class NoListingHandler(SimpleHTTPRequestHandler):
    # Directory listing would hand the token to anyone who requests
    # the bare tunnel root; every real request in this feature names
    # a file, so listings are refused outright.
    def list_directory(self, path):
        self.send_error(404, "File not found")
        return None


ThreadingHTTPServer(("0.0.0.0", port), partial(NoListingHandler, directory=root)).serve_forever()
PYEOF
HTTP_PID=$!

echo "serve_shards_public: serving $SHARDS_DIR + $MANIFEST_CSV"
echo "serve_shards_public: local base URL: http://localhost:$PORT/$TOKEN"

if [ "$LOCAL_ONLY" -eq 1 ]; then
  wait "$HTTP_PID"
else
  echo "serve_shards_public: starting cloudflared tunnel..."
  echo "serve_shards_public: once it prints a trycloudflare.com URL, the notebook's base URL is <that-url>/$TOKEN"
  cloudflared tunnel --url "http://localhost:$PORT"
fi
