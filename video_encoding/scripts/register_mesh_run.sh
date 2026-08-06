#!/usr/bin/env bash
# video_encoding/scripts/register_mesh_run.sh
# Add (or replace) one run's entry in runs/state/mesh_run_configs.json --
# the file the watcher's load_run_configs() reads. This is the ONLY writer of
# that file; launch_mesh_pool.sh and sweep_queue.sh both call it.
#
# Usage: register_mesh_run.sh <run_id> <active_csv> <rounds> <outer_lr> <outer_momentum>
#
# Registration alone is enough to start a run on an ALREADY-SEEDED pool: the
# watcher's _do_assigning writes shard_assignments_<run_id>.json itself from
# this entry's node list plus the on-disk placement in shard_map.json, then
# submits the workers. Re-seeding is only needed when the pool's physical
# shard placement changed.
set -euo pipefail

RUN_ID="${1:?usage: register_mesh_run.sh <run_id> <active_csv> <rounds> <outer_lr> <outer_momentum>}"
ACTIVE_CSV="${2:?usage: register_mesh_run.sh <run_id> <active_csv> <rounds> <outer_lr> <outer_momentum>}"
ROUNDS="${3:?usage: register_mesh_run.sh <run_id> <active_csv> <rounds> <outer_lr> <outer_momentum>}"
OUTER_LR="${4:?usage: register_mesh_run.sh <run_id> <active_csv> <rounds> <outer_lr> <outer_momentum>}"
OUTER_MOMENTUM="${5:?usage: register_mesh_run.sh <run_id> <active_csv> <rounds> <outer_lr> <outer_momentum>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

python3 - "${RUN_ID}" "${ACTIVE_CSV}" "${ROUNDS}" "${OUTER_LR}" "${OUTER_MOMENTUM}" "${REPO_ROOT}" <<'PYEOF'
import json
import sys
from pathlib import Path

run_id, active_csv, rounds, outer_lr, outer_momentum, repo_root = (
    sys.argv[1],
    sys.argv[2],
    int(sys.argv[3]),
    float(sys.argv[4]),
    float(sys.argv[5]),
    Path(sys.argv[6]),
)
nodes = active_csv.split(",")

path = repo_root / "runs" / "state" / "mesh_run_configs.json"
cfgs = json.loads(path.read_text()) if path.exists() else []
cfgs = [c for c in cfgs if c["run_id"] != run_id]
cfgs.append(
    {
        "run_id": run_id,
        "experiment": "videomae_mesh_ssl",
        "nodes": nodes,
        "n_shards": 260,
        "outer_lr": outer_lr,
        "outer_momentum": outer_momentum,
        "weighting": "data",
        "rounds": rounds,
        "merge": True,
    }
)
path.parent.mkdir(parents=True, exist_ok=True)
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(cfgs, indent=2))
tmp.replace(path)
print(f"register_mesh_run: {run_id} registered with {len(nodes)} nodes")
PYEOF
