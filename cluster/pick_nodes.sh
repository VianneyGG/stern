#!/usr/bin/env bash
# cluster/pick_nodes.sh
# Select N idle nodes from the best available tier. Run ON login-node (requires sinfo).
#
# Enforces three rules from the deep network protocol
# (docs/superpowers/specs/2026-06-30-cluster-tier1-findings.md):
#   1. Intra-subnet only — never mix nodes from different subnets (T2.3: a single
#      cross-switch uplink caps at 0.94 Gbps total; intra-switch backplane
#      is non-blocking at N×0.94 Gbps).
#   2. Homogeneous GPU tier — mixing tiers inflates step time to the slowest
#      rank (T5.3 straggler risk).
#   3. Always exclude node114 (sits on /252 but with high-latency cross-subnet path).
#      Also exclude login-node (login node).
#
# Tier ordering (best → worst), confirmed by T1.7 VRAM ramp + T2 subnet probes:
#   Tier 1  fish       /254  RTX 3090   25.3 GB  fwd_bwd batch=4   (primary)
#   Tier 2  birds      /253  A5000      25.3 GB  fwd_bwd batch=4   (same VRAM, intra-/253)
#   Tier 3  cars       /252  RTX4000Ada 21.0 GB  fwd_bwd batch=2
#   Tier 4  countries  /254  A4000      16.7 GB  fwd_bwd batch=2   (grad_accum required)
#   Tier 5  orchids    /253  RTX2000Ada 16.7 GB  fwd_bwd batch=2
#
# Note: countries share /254 with fish and orchids share /253 with birds. Never
# mix tiers even if they share a subnet — GPU heterogeneity causes stragglers.
#
# Usage:  pick_nodes.sh <N>
# Output: comma-separated nodelist for sbatch --nodelist, e.g. "node001,node002,..."
# Exit 1 if no single tier can supply N idle nodes.

set -euo pipefail

N="${1:?usage: pick_nodes.sh <N>}"

# Tier node lists come from cluster/tiers.tsv via tiers_lib.sh, which
# also fixes the preference order below. That file records why the river/department
# nodes (ain, allier, ardennes…) are absent: GPU type unconfirmed.
source "$(dirname "${BASH_SOURCE[0]}")/tiers_lib.sh"

# Snapshot idle nodes once (avoids repeated sinfo calls per tier).
IDLE=$(sinfo -N -o "%N %T" --state=idle 2>/dev/null | awk '$2=="idle"{print $1}' | sort)

# pick_from_tier <comma-node-list> <N> → prints comma-list of N idle nodes, rc=0;
# rc=1 if fewer than N idle nodes are in this tier.
pick_from_tier() {
    local candidates="$1"
    local need="$2"
    local found=()

    IFS=',' read -ra nodes <<< "$candidates"
    for node in "${nodes[@]}"; do
        if echo "$IDLE" | grep -qx "$node"; then
            found+=("$node")
            [[ "${#found[@]}" -ge "$need" ]] && break
        fi
    done

    if [[ "${#found[@]}" -ge "$need" ]]; then
        local IFS=','
        echo "${found[*]}"
        return 0
    fi
    return 1
}

while read -r tier_name; do
    tier_hosts="$(tier_nodes "$tier_name")"

    if nodelist=$(pick_from_tier "$tier_hosts" "$N"); then
        echo "# pick_nodes: selected tier=${tier_name} N=${N}" >&2
        echo "$nodelist"
        exit 0
    else
        # Count available in this tier for the diagnostic message.
        available=0
        IFS=',' read -ra nodes <<< "$tier_hosts"
        for node in "${nodes[@]}"; do
            echo "$IDLE" | grep -qx "$node" && ((available++)) || true
        done
        echo "# pick_nodes: tier=${tier_name} has ${available}/${N} idle — skipping" >&2
    fi
done < <(all_tiers)

echo "ERROR: no tier has ${N} idle nodes available. Run 'sinfo -N -o \"%N %T\" --state=idle' to inspect." >&2
exit 1
