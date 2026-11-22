#!/usr/bin/env bash
# cluster/clean_tier_data.sh
# Empty the data root on every node in one tier and lock it to chmod 700,
# keeping the directory itself. Run before reseeding a tier's mesh pool from
# scratch: stale mesh, checkpoint or shard files must not leak into a fresh
# run's shard placement or quota accounting.
#
# Usage: clean_tier_data.sh <fish|birds|cars>
#
# Node roster comes from the tiers file via tiers_lib.sh.
set -euo pipefail

TIER="${1:?usage: clean_tier_data.sh <fish|birds|cars>}"

source "$(dirname "${BASH_SOURCE[0]}")/tiers_lib.sh"
NODES=$(tier_nodes "${TIER}")

# mkdir -p first makes this idempotent on nodes where the data root doesn't
# exist yet; find -mindepth 1 -delete empties it (files, dotfiles and subdirs)
# without removing the directory itself.
echo "${NODES}" | tr ',' '\n' | xargs -P 10 -I{} bash -c '
  node="{}"
  if ssh -o BatchMode=yes -o ConnectTimeout=10 "${node}" \
      "mkdir -p /Data/\${USER} && find /Data/\${USER} -mindepth 1 -delete && chmod 700 /Data/\${USER}" \
      >/dev/null 2>&1; then
    echo "PASS ${node}"
  else
    echo "FAIL ${node}" >&2
  fi
'
