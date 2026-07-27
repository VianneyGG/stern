#!/usr/bin/env bash
# video_encoding/scripts/clean_tier_data.sh
# Empty /Data/$USER on every node in one tier and lock it to chmod 700,
# without deleting the directory itself. Run before reseeding a tier's
# mesh pool from scratch (stale mesh/checkpoints/shards must not leak into
# a fresh run's shard placement or quota accounting).
#
# Usage: clean_tier_data.sh <fish|birds|cars>
#
# Tier node lists copied verbatim from pick_nodes.sh / seed_mesh.py --
# same convention, do not diverge.
set -euo pipefail

TIER="${1:?usage: clean_tier_data.sh <fish|birds|cars>}"

TIER_FISH="node001,node002,node003,node004,node005,node006,node007,node008,node009,node010,node011,node012,node013,node014,node015,node016,node017,node018,node019,node020,node021,node022,node023"
TIER_BIRDS="node024,node025,node026,node027,node028,node029,node030,node031,node032,node033,node034,node035,node036,node037,node038,node039,node040,node041,node042,node043,node044,node045,node046"
TIER_CARS="node047,node048,node049,node050,node051,node052,node053,node054,node055,node056,node057,node058,node059,node060,node061,node062,node063,node064,node065,node066,node067,node068,node069"

case "${TIER}" in
  fish) NODES="${TIER_FISH}" ;;
  birds) NODES="${TIER_BIRDS}" ;;
  cars) NODES="${TIER_CARS}" ;;
  *) echo "ERROR: unknown tier '${TIER}' -- expected fish|birds|cars" >&2; exit 1 ;;
esac

# mkdir -p first makes this idempotent on nodes where /Data/$USER doesn't
# exist yet; find -mindepth 1 -delete empties it (files + dotfiles +
# subdirs) without removing the directory itself.
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
