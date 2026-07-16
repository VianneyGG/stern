#!/usr/bin/env bash
# video_encoding/scripts/pick_nodes.sh
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

# Tier node lists — derived from T1.7 GPU ramp (T2 confirmed /254 and /253 subnets).
# River/department nodes (ain, allier, ardennes…) are omitted: GPU type unconfirmed.
TIER_FISH="node001,node002,node003,node004,node005,node006,node007,node008,node009,node010,node011,node012,node013,node014,node015,node016,node017,node018,node019,node020,node021,node022,node023"
TIER_BIRDS="node024,node025,node026,node027,node028,node029,node030,node031,node032,node033,node034,node035,node036,node037,node038,node039,node040,node041,node042,node043,node044,node045,node046"
TIER_CARS="node047,node048,node049,node050,node051,node052,node053,node054,node055,node056,node057,node058,node059,node060,node061,node062,node063,node064,node065,node066,node067,node068,node069"
TIER_COUNTRIES="node070,node071,node072,node073,node074,node075,node076,node077,node078,node079,node080,node081,node082,node083,node084,node085,node086,node087,node088"
TIER_ORCHIDS="node089,node090,node091,node092,node093,node094,node095,node096,node097,node098,node099,node100,node101,node102,node103,node104,node105,node106,node107,node108,node109,node110,node111,node112,node113"

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

for tier_name_nodes in \
    "fish:${TIER_FISH}" \
    "birds:${TIER_BIRDS}" \
    "cars:${TIER_CARS}" \
    "countries:${TIER_COUNTRIES}" \
    "orchids:${TIER_ORCHIDS}"
do
    tier_name="${tier_name_nodes%%:*}"
    tier_nodes="${tier_name_nodes#*:}"

    if nodelist=$(pick_from_tier "$tier_nodes" "$N"); then
        echo "# pick_nodes: selected tier=${tier_name} N=${N}" >&2
        echo "$nodelist"
        exit 0
    else
        # Count available in this tier for the diagnostic message.
        available=0
        IFS=',' read -ra nodes <<< "$tier_nodes"
        for node in "${nodes[@]}"; do
            echo "$IDLE" | grep -qx "$node" && ((available++)) || true
        done
        echo "# pick_nodes: tier=${tier_name} has ${available}/${N} idle — skipping" >&2
    fi
done

echo "ERROR: no tier has ${N} idle nodes available. Run 'sinfo -N -o \"%N %T\" --state=idle' to inspect." >&2
exit 1
