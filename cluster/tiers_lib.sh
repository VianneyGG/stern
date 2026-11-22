#!/usr/bin/env bash
# cluster/tiers_lib.sh
# Shell-side reader for cluster/tiers.tsv -- the single source of
# truth for cluster node topology, shared with stern/mesh/topology.py.
#
# Source it, don't execute it:
#   source "$(dirname "${BASH_SOURCE[0]}")/tiers_lib.sh"
#
# The roster used to be pasted verbatim into pick_nodes.sh, seed_mesh.py,
# clean_tier_data.sh and launch_mesh_pool.sh. Adding a node to one and not the
# others desynced SLURM node picking from mesh seeding, so a job could be placed
# on a node that had no seeded shard payload.
#
# Deliberately parses the TSV with awk rather than calling Python: pick_nodes.sh
# runs on the login node where no venv is guaranteed active.

_TIERS_TSV="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/tiers.tsv"

if [[ ! -f "${_TIERS_TSV}" ]]; then
  echo "ERROR: tiers_lib.sh: roster not found at ${_TIERS_TSV}" >&2
  exit 1
fi

# tier_nodes <tier> -> comma-separated nodelist on stdout; rc=1 if unknown.
tier_nodes() {
  local tier="$1" nodes
  nodes=$(awk -F'\t' -v t="${tier}" '!/^#/ && NF==3 && $1==t {print $3}' "${_TIERS_TSV}")
  if [[ -z "${nodes}" ]]; then
    echo "ERROR: unknown tier '${tier}' -- known: $(all_tiers | tr '\n' ' ')" >&2
    return 1
  fi
  echo "${nodes}"
}

# all_tiers -> tier names, one per line, in preference order (best GPU first).
all_tiers() {
  awk -F'\t' '!/^#/ && NF==3 {print $1}' "${_TIERS_TSV}"
}

# tier_subnet <tier> -> subnet (e.g. "/254") on stdout.
tier_subnet() {
  awk -F'\t' -v t="$1" '!/^#/ && NF==3 && $1==t {print $2}' "${_TIERS_TSV}"
}
