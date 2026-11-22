#!/usr/bin/env python
"""Verify a mesh pool's on-disk shard payload against the golden reference.

Exit 0 means every node's seeded payload matches the source node's reference set by
name and byte size, val/ is complete, and both manifest.json files are
present. Exit 1 means something is wrong and no run should be launched.

Two artifacts are written to --metadata-dir on every run, pass or fail:

  shard_manifest.json     {"train": {name: size}, "val": {name: size}}
  verified_holdings.json  {pool: {node: [shard, ...]}}

`verified_holdings.json` is written ONLY from a successful scan. It is never
hand-edited and never inferred from a placement plan -- that inference is
what produced the stale shard map behind the 2026-08-20 incident.

Cluster/cron usage:
    ~/.local/bin/uv run --no-sync python -m stern.mesh.validate_seeding \
        --pool fish
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from stern.cluster_env import setting
from stern.mesh.seeding import holdings_from_scan, validate
from stern.mesh.topology import tier_nodes
from stern.mesh.seed_nodes import MESH_DIR, _VAL_SUBDIR

GOLDEN_HOST = setting("STERN_SOURCE_NODE")
GOLDEN_DIR = "/Data/voxceleb2_shards_8fps"  # 8fps corpus, default since 2026-09-19
DEFAULT_METADATA_DIR = Path.home() / "stern_mesh" / "metadata"

# `-L` follows symlinks: seed_mesh._symlink_source_node links the source
# node's own shards rather than copying them, so without -L every one of
# them would report as a wrong-size (symlink-sized) shard.
_LIST_CMD = "ls -lL --time-style=+ {d} 2>/dev/null | awk '{{print $5, $NF}}'"


def _run_ssh(host: str, command: str, *, timeout_s: float = 60.0) -> str:
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, command],
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if result.returncode != 0:
        raise OSError(
            f"ssh {host} exited {result.returncode}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout


def parse_listing(output: str) -> dict[str, int]:
    """Parse "<size> <name>" lines into {shard name: byte size}.

    Only `shard-*.tar` entries are kept -- `manifest.json` and the `val`
    directory itself also appear in the listing.
    """
    sizes: dict[str, int] = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        size, name = parts
        if not (name.startswith("shard-") and name.endswith(".tar")):
            continue
        try:
            sizes[name] = int(size)
        except ValueError:
            continue
    return sizes


def _listing(host: str, directory: str, run_ssh) -> dict[str, int]:
    return parse_listing(run_ssh(host, _LIST_CMD.format(d=shlex.quote(directory))))


def build_manifest(
    golden_host: str, golden_dir: str, *, run_ssh=_run_ssh
) -> tuple[dict[str, int], dict[str, int]]:
    """Read the golden reference's train and val shard sizes."""
    train = _listing(golden_host, golden_dir, run_ssh)
    val = _listing(golden_host, f"{golden_dir.rstrip('/')}/{_VAL_SUBDIR}", run_ssh)
    return train, val


def _exists(host: str, path: str, run_ssh) -> bool:
    try:
        run_ssh(host, f"test -f {shlex.quote(path)} && echo ok")
        return True
    except OSError:
        return False


def scan_pool(
    nodes: list[str], *, mesh_dir: str = MESH_DIR, run_ssh=_run_ssh
) -> tuple[
    dict[str, dict[str, int]], dict[str, dict[str, int]], dict[str, tuple[bool, bool]]
]:
    """Scan every node's seeded payload.

    An unreachable node is recorded as holding nothing rather than being
    dropped: a node missing from the report is a node nobody checks.
    """
    val_dir = f"{mesh_dir.rstrip('/')}/{_VAL_SUBDIR}"
    scan: dict[str, dict[str, int]] = {}
    val_scan: dict[str, dict[str, int]] = {}
    has_manifest: dict[str, tuple[bool, bool]] = {}

    for node in nodes:
        try:
            scan[node] = _listing(node, mesh_dir, run_ssh)
            val_scan[node] = _listing(node, val_dir, run_ssh)
            has_manifest[node] = (
                _exists(node, f"{mesh_dir.rstrip('/')}/manifest.json", run_ssh),
                _exists(node, f"{val_dir}/manifest.json", run_ssh),
            )
        except OSError as exc:
            print(
                f"WARNING: validate_seeding: {node!r} unreachable ({exc})",
                file=sys.stderr,
            )
            scan[node] = {}
            val_scan[node] = {}
            has_manifest[node] = (False, False)

    return scan, val_scan, has_manifest


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify a mesh pool's seeded shards against the golden reference."
    )
    parser.add_argument(
        "--pool",
        required=True,
        help=(
            "pool label -- a tier name (fish) or, for a cross-tier roster, "
            "the arm name. Used as the key under which verified holdings are "
            "recorded, and as the tier to scan when --nodes is omitted."
        ),
    )
    parser.add_argument(
        "--nodes",
        default=None,
        help=(
            "comma-separated roster to scan. Defaults to every node in the "
            "tier named by --pool. Pass this for a roster that is not a whole "
            "tier -- a cross-tier pool has no tier to look up, and scanning a "
            "whole tier would gate the launch on nodes the run never uses."
        ),
    )
    parser.add_argument("--golden-host", default=GOLDEN_HOST)
    parser.add_argument("--golden-dir", default=GOLDEN_DIR)
    parser.add_argument("--mesh-dir", default=MESH_DIR)
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    args = parser.parse_args(argv)

    train_manifest, val_manifest = build_manifest(
        args.golden_host, args.golden_dir, run_ssh=_run_ssh
    )
    if not train_manifest:
        print(
            f"validate_seeding: golden reference {args.golden_host}:{args.golden_dir} "
            "listed no shards -- refusing to validate against an empty manifest",
            file=sys.stderr,
        )
        return 1

    if args.nodes is not None:
        nodes = [h.strip() for h in args.nodes.split(",") if h.strip()]
        if not nodes:
            print(
                "validate_seeding: --nodes was empty -- refusing to report a "
                "pool with no nodes as valid",
                file=sys.stderr,
            )
            return 1
    else:
        nodes = tier_nodes(args.pool)
    scan, val_scan, has_manifest = scan_pool(
        nodes, mesh_dir=args.mesh_dir, run_ssh=_run_ssh
    )
    report = validate(args.pool, train_manifest, val_manifest, scan, val_scan, has_manifest)

    _write_json(
        args.metadata_dir / "shard_manifest.json",
        {"train": train_manifest, "val": val_manifest},
    )
    holdings_path = args.metadata_dir / "verified_holdings.json"
    all_holdings = json.loads(holdings_path.read_text()) if holdings_path.exists() else {}
    fresh = holdings_from_scan(scan)
    # A node holds one payload, so it belongs to one pool. Drop it from every
    # other pool's entry: mesh_tick._verified_holdings flattens all pools into
    # {node: shards} and the last key wins, and keys are written sorted -- so a
    # node moved from "cars" to an arm-named pool would silently resolve to its
    # stale "cars" holdings purely because "cars" sorts later. That is the
    # 2026-08-20 failure mode (attribution disagreeing with reality) with a
    # different cause.
    for pool, held in all_holdings.items():
        if pool == args.pool:
            continue
        for node in fresh:
            held.pop(node, None)
    all_holdings[args.pool] = fresh
    _write_json(holdings_path, all_holdings)

    print(
        f"validate_seeding: pool={report.pool} nodes={len(report.nodes)} "
        f"coverage={report.coverage:.1%} "
        f"({len(train_manifest) - len(report.missing_pool_wide)}/{len(train_manifest)})"
    )
    for node in report.nodes:
        if node.ok:
            continue
        print(
            f"  {node.node}: held={len(scan[node.node])} "
            f"wrong_size={len(node.wrong_size)} val_missing={len(node.val_missing)} "
            f"manifest_missing={list(node.manifest_missing)}"
        )
    if report.missing_pool_wide:
        print(
            f"  MISSING POOL-WIDE ({len(report.missing_pool_wide)}): "
            f"{', '.join(report.missing_pool_wide[:10])}"
            + (" ..." if len(report.missing_pool_wide) > 10 else ""),
            file=sys.stderr,
        )
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
