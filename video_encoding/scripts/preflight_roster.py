#!/usr/bin/env python
"""Refuse a pool launch whose roster will not train in full.

Run before seeding. `eligible_nodes` drops any node in the watcher's global
`excluded_nodes` at ASSIGNING time -- long after a 223 GB seed has run and
validate_seeding has reported full coverage over the *seeded* roster. That
gap is how cars-real10r trained at 246/260 and fish-real10r at 204/260.

Exits non-zero when any requested node would be dropped, so `set -e` in
launch_mesh_pool.sh aborts before the seed rather than after it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from video_encoding.mesh.preflight import (  # noqa: E402
    screen_roster,
    stale_exclusions,
)
from video_encoding.watcher.cluster import query_idle_nodes  # noqa: E402
from video_encoding.watcher.node_health import (  # noqa: E402
    DEFAULT_MIN_GPU_FREE_MIB,
    query_node_health,
)
from video_encoding.watcher.state import load_state, save_state  # noqa: E402

STATE_DIR = REPO_ROOT / "runs" / "state"


def _live_run_ids(configs_path: Path) -> set[str]:
    """Every run_id still registered. A missing file means nothing is live,
    which makes every run-scoped exclusion reclaimable -- correct, and the
    caller still has to ask for that with --clear-stale."""
    if not configs_path.exists():
        return set()
    configs = json.loads(configs_path.read_text())
    return {c["run_id"] for c in configs if c.get("run_id")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", required=True, help="comma-separated roster")
    parser.add_argument("--state", type=Path, default=STATE_DIR / "watcher_state.json")
    parser.add_argument(
        "--configs", type=Path, default=STATE_DIR / "mesh_run_configs.json"
    )
    parser.add_argument(
        "--min-gpu-free-mib", type=int, default=DEFAULT_MIN_GPU_FREE_MIB
    )
    parser.add_argument(
        "--no-gpu-probe",
        action="store_true",
        help="screen exclusions only, without one ssh per node",
    )
    parser.add_argument(
        "--clear-stale",
        action="store_true",
        help=(
            "reclaim exclusions owned by a run that no longer exists. "
            "clear_run_state only fires at DONE, so an abandoned run bans "
            "its nodes forever. Never touches a 'manual:' flag."
        ),
    )
    parser.add_argument(
        "--warn-only", action="store_true", help="report but exit 0 anyway"
    )
    args = parser.parse_args(argv)

    requested = [h.strip() for h in args.nodes.split(",") if h.strip()]
    if not requested:
        print("preflight_roster: --nodes was empty", file=sys.stderr)
        return 1

    state = load_state(args.state)

    if args.clear_stale:
        live = _live_run_ids(args.configs)
        stale = stale_exclusions(state.excluded_nodes, live_run_ids=live)
        for node, reason in sorted(stale.items()):
            print(f"preflight_roster: reclaiming {node} (was: {reason})")
            state.excluded_nodes.pop(node, None)
        if stale:
            save_state(args.state, state)

    gpu_free_mib = None
    if not args.no_gpu_probe:
        healths = query_node_health(requested, idle=query_idle_nodes())
        gpu_free_mib = {h.node: h.gpu_free_mib for h in healths}

    verdict = screen_roster(
        requested,
        excluded_nodes=state.excluded_nodes,
        gpu_free_mib=gpu_free_mib,
        min_gpu_free_mib=args.min_gpu_free_mib,
    )

    print(
        f"preflight_roster: {len(verdict.usable)}/{len(requested)} node(s) will train"
    )
    for node, reason in verdict.rejected:
        print(f"  REJECT {node}: {reason}", file=sys.stderr)

    if verdict.ok:
        return 0
    print(
        "preflight_roster: refusing to seed a roster that will not train in "
        "full -- clear the exclusion (or rerun with --clear-stale if its run "
        "is gone), free the GPU, or drop the node from POOL_NODES",
        file=sys.stderr,
    )
    return 0 if args.warn_only else 1


if __name__ == "__main__":
    raise SystemExit(main())
