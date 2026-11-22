#!/usr/bin/env bash
# cluster/tiers_lib.sh
# Shell-side reader for the tiers file, the single source of truth for cluster
# node topology. stern/mesh/topology.py reads the same file.
#
# Source it, don't execute it:
#   source "$(dirname "${BASH_SOURCE[0]}")/tiers_lib.sh"
#
# Keep the roster in one place. A copy pasted into several scripts drifts: a
# node added to one and not the others could get a SLURM job with no seeded
# shard payload.
#
# Deliberately parses the TSV with awk rather than calling Python: pick_nodes.sh
# runs on the login node where no venv is guaranteed active.

source "$(dirname "${BASH_SOURCE[0]}")/cluster_env.sh"
_TIERS_TSV="${STERN_TIERS_FILE}"

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
