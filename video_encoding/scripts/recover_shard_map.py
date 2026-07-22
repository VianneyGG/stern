#!/usr/bin/env python
"""Rebuild shard_map.json from each already-seeded mesh node's live holdings.

Use when the shard_map metadata file was lost (or never written) even
though the underlying shard placement across nodes is still correct --
this SSHes into each node, lists what it actually has on disk, and inverts
that into the shard -> holders map `mesh/fetch.py` needs at training time.
Do NOT re-run `seed_mesh.py` for this: free space drifts over time, and a
fresh placement computation would not match what is physically on disk.

Reuses `seed_mesh.py`'s existing `_list_remote_shards`/`_run_ssh` SSH glue
rather than a second SSH wrapper.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from video_encoding.mesh.shard_map import save_shard_map
from video_encoding.mesh.shard_map_recovery import build_shard_map_from_holdings
from video_encoding.scripts.seed_mesh import _list_remote_shards


def _parse_subnet_args(pairs: list[str]) -> dict[str, str]:
    node_subnets: dict[str, str] = {}
    for pair in pairs:
        host, sep, subnet = pair.partition("=")
        if not sep:
            raise ValueError(f"--subnet must be HOST=SUBNET, got {pair!r}")
        node_subnets[host] = subnet
    return node_subnets


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild shard_map.json from live node holdings (no re-seeding, "
            "no placement recomputation)."
        )
    )
    parser.add_argument(
        "--node",
        action="append",
        required=True,
        dest="nodes",
        help="hostname to list shards on (repeatable)",
    )
    parser.add_argument(
        "--mesh-dir",
        default="/Data/$USER/mesh",
        help="remote shard directory on each node",
    )
    parser.add_argument(
        "--subnet",
        action="append",
        default=[],
        dest="subnets",
        help="HOST=SUBNET (repeatable, required for every --node)",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="local path to write the recovered shard_map JSON",
    )
    args = parser.parse_args()

    node_subnets = _parse_subnet_args(args.subnets)

    holdings: dict[str, list[str]] = {}
    val_shard_sets: dict[str, frozenset[str]] = {}
    for host in args.nodes:
        try:
            holdings[host] = _list_remote_shards(host, args.mesh_dir)
            val_shard_sets[host] = frozenset(
                _list_remote_shards(host, f"{args.mesh_dir.rstrip('/')}/val")
            )
        except (OSError, subprocess.SubprocessError) as exc:
            print(
                f"recover_shard_map: failed to list shards on {host!r}: {exc}",
                file=sys.stderr,
            )
            return 1

    # design spec S5 requires byte-identical val shard sets across nodes --
    # a naming mismatch here is the cheap, catchable proxy for that.
    reference_host, reference_val = next(iter(val_shard_sets.items()))
    for host, val_set in val_shard_sets.items():
        if val_set != reference_val:
            print(
                f"recover_shard_map: val shard set on {host!r} does not match "
                f"{reference_host!r} (design spec S5 requires byte-identical "
                f"val sets) -- diverging entries: {sorted(val_set ^ reference_val)}",
                file=sys.stderr,
            )
            return 1

    shard_map = build_shard_map_from_holdings(holdings, node_subnets)
    save_shard_map(args.out, shard_map)
    print(
        f"recover_shard_map: wrote {len(shard_map)} shards from "
        f"{len(holdings)} node(s) to {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
