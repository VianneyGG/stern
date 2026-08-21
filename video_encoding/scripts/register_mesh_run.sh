#!/usr/bin/env bash
# video_encoding/scripts/register_mesh_run.sh
# Add (or replace) one run's entry in runs/state/mesh_run_configs.json --
# the file the watcher's load_run_configs() reads. This is the ONLY writer of
# that file; launch_mesh_pool.sh and sweep_queue.sh both call it.
#
# Usage: register_mesh_run.sh <run_id> <active_csv> <rounds> <outer_lr> <outer_momentum> [quorum_k] [weighting] [extra_overrides]
#        register_mesh_run.sh --remove <run_id>
#
# --remove <run_id>: delete that run's entry, leaving every other entry
#   untouched and in order. Removal lives here because this script is the
#   only writer of mesh_run_configs.json -- tearing a sweep wave down by
#   hand-editing live coordinator state has caused incidents before. An
#   absent run_id exits 0 so a multi-cell teardown can be re-run; a missing
#   configs file exits 1, because that is a broken deployment, not a no-op.
#
# [quorum_k] (REQUIRED unless ALLOW_NO_QUORUM=1): mesh_tick.MeshRunConfig.quorum_k
#   -- merge once this many nodes check in instead of waiting for
#   all-or-ROUND_TIMEOUT_S (see mesh/quorum.py). Must not exceed the node
#   roster, or the quorum can never be reached. ALLOW_NO_QUORUM=1 restores
#   the old all-or-timeout behavior (quorum_k absent from the JSON entry ->
#   dataclass default None).
# [weighting] (optional, default "data"): one of run_outer_step.py's
#   WEIGHTING_CHOICES ("data", "inverse_loss", "token"). Pair "token" with a
#   quorum_k setting -- token weighting is what corrects a partial merge's
#   bias instead of the plain renormalization "data" falls back to.
# [extra_overrides] (optional): a single space-separated string of raw Hydra
#   override tokens (e.g. "train.accum_steps=128 train.lr_mult=1.5"),
#   written verbatim into MeshRunConfig.extra_overrides and appended to
#   every _submit_worker_job sbatch call for this run -- on initial
#   assignment AND every later node replenishment, since it's stored in
#   the run's own persisted JSON config rather than passed per-launch.
#   Omitted/empty -> no "extra_overrides" key in the JSON entry.
#
# Registration alone is enough to start a run on an ALREADY-SEEDED pool: the
# watcher's _do_assigning writes shard_assignments_<run_id>.json itself from
# this entry's node list plus each node's verified holdings in
# verified_holdings.json, then submits the workers. That file comes from
# validate_seeding.py, and _do_assigning refuses to launch without it.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ "${1:-}" == "--remove" ]]; then
  if [[ $# -ne 2 || -z "${2:-}" ]]; then
    echo "usage: register_mesh_run.sh --remove <run_id> (--remove takes exactly one argument and no registration arguments)" >&2
    exit 1
  fi
  python3 - "${2}" "${REPO_ROOT}" <<'REMOVEPY'
import json
import sys
from pathlib import Path

run_id, repo_root = sys.argv[1], Path(sys.argv[2])
path = repo_root / "runs" / "state" / "mesh_run_configs.json"
if not path.exists():
    sys.exit(
        f"register_mesh_run: {path} does not exist -- refusing to treat a "
        "missing mesh_run_configs.json as an empty one"
    )
cfgs = json.loads(path.read_text())
kept = [c for c in cfgs if c["run_id"] != run_id]
if len(kept) == len(cfgs):
    # Idempotent on purpose: a teardown covering several cells must not
    # abort midway because one cell was already removed.
    print(f"register_mesh_run: {run_id} not registered -- nothing to remove")
else:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(kept, indent=2))
    tmp.replace(path)
    print(f"register_mesh_run: {run_id} removed, {len(kept)} run(s) left")
REMOVEPY
  exit 0
fi

USAGE="usage: register_mesh_run.sh <run_id> <active_csv> <rounds> <outer_lr> <outer_momentum> [quorum_k] [weighting] [extra_overrides]"
RUN_ID="${1:?${USAGE}}"
ACTIVE_CSV="${2:?${USAGE}}"
ROUNDS="${3:?${USAGE}}"
OUTER_LR="${4:?${USAGE}}"
OUTER_MOMENTUM="${5:?${USAGE}}"
QUORUM_K="${6:-}"
WEIGHTING="${7:-data}"
EXTRA_OVERRIDES="${8:-}"

python3 - "${RUN_ID}" "${ACTIVE_CSV}" "${ROUNDS}" "${OUTER_LR}" "${OUTER_MOMENTUM}" "${REPO_ROOT}" "${QUORUM_K}" "${WEIGHTING}" "${EXTRA_OVERRIDES}" <<'PYEOF'
import json
import os
import re
import sys
from pathlib import Path

run_id, active_csv, rounds, outer_lr, outer_momentum, repo_root, quorum_k, weighting, extra_overrides = (
    sys.argv[1],
    sys.argv[2],
    int(sys.argv[3]),
    float(sys.argv[4]),
    float(sys.argv[5]),
    Path(sys.argv[6]),
    sys.argv[7],
    sys.argv[8],
    sys.argv[9],
)
WEIGHTING_CHOICES = ("data", "inverse_loss", "token")
if weighting not in WEIGHTING_CHOICES:
    sys.exit(
        f"register_mesh_run: unknown weighting {weighting!r}, expected one of "
        f"{WEIGHTING_CHOICES}"
    )
overrides = extra_overrides.split()
override_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*=.+$")
for token in overrides:
    if not override_re.match(token):
        sys.exit(f"register_mesh_run: malformed extra override {token!r}, expected key=value")
nodes = active_csv.split(",")

# Every entry this script writes has merge=True, so a run with no quorum_k
# waits for all-or-ROUND_TIMEOUT_S at each merge: one dead node costs the
# whole pool a 30-minute stall per round, repeatedly. Refuse to register that
# silently. ALLOW_NO_QUORUM=1 is the deliberate opt-out.
if not quorum_k and not os.environ.get("ALLOW_NO_QUORUM"):
    sys.exit(
        "register_mesh_run: quorum_k is required on a merge-enabled run "
        "(every run registered here has merge=True). Pass it as argument 6, "
        "or set ALLOW_NO_QUORUM=1 to accept all-or-timeout merges."
    )
if quorum_k and int(quorum_k) > len(nodes):
    sys.exit(
        f"register_mesh_run: quorum_k={int(quorum_k)} exceeds the {len(nodes)}-node "
        "roster -- the quorum could never be reached and every round would "
        "run to ROUND_TIMEOUT_S"
    )

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
if overrides:
    entry["extra_overrides"] = overrides
cfgs.append(entry)
path.parent.mkdir(parents=True, exist_ok=True)
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(cfgs, indent=2))
tmp.replace(path)
print(f"register_mesh_run: {run_id} registered with {len(nodes)} nodes")
PYEOF
