#!/usr/bin/env bash
# video_encoding/scripts/register_mesh_run.sh
# Add (or replace) one run's entry in runs/state/mesh_run_configs.json --
# the file the watcher's load_run_configs() reads. This is the ONLY writer of
# that file; launch_mesh_pool.sh and sweep_queue.sh both call it.
#
# Usage: register_mesh_run.sh <run_id> <active_csv> <rounds> <outer_lr> <outer_momentum> [quorum_k] [weighting]
#
# [quorum_k] (optional): mesh_tick.MeshRunConfig.quorum_k -- merge once this
#   many nodes check in instead of waiting for all-or-ROUND_TIMEOUT_S (see
#   mesh/quorum.py). Omitted/empty keeps the old all-or-timeout behavior
#   (quorum_k absent from the JSON entry -> dataclass default None).
# [weighting] (optional, default "data"): one of run_outer_step.py's
#   WEIGHTING_CHOICES ("data", "inverse_loss", "token"). Pair "token" with a
#   quorum_k setting -- token weighting is what corrects a partial merge's
#   bias instead of the plain renormalization "data" falls back to.
#
# Registration alone is enough to start a run on an ALREADY-SEEDED pool: the
# watcher's _do_assigning writes shard_assignments_<run_id>.json itself from
# this entry's node list plus the on-disk placement in shard_map.json, then
# submits the workers. Re-seeding is only needed when the pool's physical
# shard placement changed.
set -euo pipefail

USAGE="usage: register_mesh_run.sh <run_id> <active_csv> <rounds> <outer_lr> <outer_momentum> [quorum_k] [weighting]"
RUN_ID="${1:?${USAGE}}"
ACTIVE_CSV="${2:?${USAGE}}"
ROUNDS="${3:?${USAGE}}"
OUTER_LR="${4:?${USAGE}}"
OUTER_MOMENTUM="${5:?${USAGE}}"
QUORUM_K="${6:-}"
WEIGHTING="${7:-data}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

python3 - "${RUN_ID}" "${ACTIVE_CSV}" "${ROUNDS}" "${OUTER_LR}" "${OUTER_MOMENTUM}" "${REPO_ROOT}" "${QUORUM_K}" "${WEIGHTING}" <<'PYEOF'
import json
import sys
from pathlib import Path

run_id, active_csv, rounds, outer_lr, outer_momentum, repo_root, quorum_k, weighting = (
    sys.argv[1],
    sys.argv[2],
    int(sys.argv[3]),
    float(sys.argv[4]),
    float(sys.argv[5]),
    Path(sys.argv[6]),
    sys.argv[7],
    sys.argv[8],
)
WEIGHTING_CHOICES = ("data", "inverse_loss", "token")
if weighting not in WEIGHTING_CHOICES:
    sys.exit(
        f"register_mesh_run: unknown weighting {weighting!r}, expected one of "
        f"{WEIGHTING_CHOICES}"
    )
nodes = active_csv.split(",")

path = repo_root / "runs" / "state" / "mesh_run_configs.json"
cfgs = json.loads(path.read_text()) if path.exists() else []
cfgs = [c for c in cfgs if c["run_id"] != run_id]
entry = {
    "run_id": run_id,
    "experiment": "videomae_mesh_ssl",
    "nodes": nodes,
    "n_shards": 260,
    "outer_lr": outer_lr,
    "outer_momentum": outer_momentum,
    "weighting": weighting,
    "rounds": rounds,
    "merge": True,
}
if quorum_k:
    entry["quorum_k"] = int(quorum_k)
cfgs.append(entry)
path.parent.mkdir(parents=True, exist_ok=True)
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(cfgs, indent=2))
tmp.replace(path)
print(f"register_mesh_run: {run_id} registered with {len(nodes)} nodes")
PYEOF
