#!/usr/bin/env python
"""Seed the decentralized P2P training mesh: probe live capacity, plan
placement, distribute shards, and write the shard map.

Ties together the pure logic modules built in Tasks 1-3
(`stern.mesh.shard_map`, `.scoring`, `.placement`) to the real
fish cluster:

1. `probe_nodes` SSHes into each candidate host and reads live free space
   on `/Data` (a shared 1 TB scratch mount that drifts as other students'
   jobs use it -- 90 GB on node024 to 905 GB on node089 was one measured
   spread) rather than trusting a static assumption.
2. `plan_placement` (Task 3) turns that live capacity plus the compute
   weights (Task 2) into a `hostname -> [shard, ...]` layout.
3. `build_shard_map` inverts that into the `shard -> [Holder, ...]`
   lookup `mesh/fetch.py` (Task 7) needs at training time.
4. `rsync_plan` / `main`'s transfer loop physically move the shards --
   pulled *on the target* over the cluster's internal network so nothing
   routes through the submitting laptop, matching the existing
   `distribute_shards.sh` precedent.

`NODE_GROUPS` / `NODE_SUBNETS` come from `stern/mesh/topology.py`,
which reads `cluster/tiers.tsv` -- the single source of truth,
also read directly by `pick_nodes.sh`, `clean_tier_data.sh` and
`launch_mesh_pool.sh`. Add a node there and every consumer sees it; there is
no longer a second roster to keep in sync.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from stern.mesh.placement import InsufficientCapacityError, plan_placement
from stern.mesh.scoring import NodeInfo
from stern.mesh.shard_map import Holder, save_shard_map
from stern.mesh.topology import NODE_GROUPS, NODE_SUBNETS

MESH_DIR = "/Data/$USER/mesh"

# Shared-home quota constraint: N x 172 MB (bf16 ViT-B checkpoint) + master +
# momentum buffer must fit. This caps how many nodes may PUSH checkpoints to
# shared NFS concurrently (--active-nodes below) -- it does not cap how many
# may hold a seeded shard payload, since seeded shards live on node-local
# /Data, which is not quota'd. That's MAX_SEEDED_NODES.
#
# Raised 16 -> 20 on 2026-08-26 so a pool can take the largest divisor of the
# 800-shard corpus that fits a tier (20 -> 40 shards/node exactly; 16 would
# force 50 and waste three of fish's 23 nodes).
#
# CAUTION: this is a PER-INVOCATION guard and the quota is GLOBAL. Four
# concurrent pools at 20 active nodes each would be 80 x 172 MB = 13.8 GB and
# blow a 12-13 GB budget while every individual run passes this check. Size
# concurrent runs against `quota -s`, not against this constant.
MAX_MESH_NODES = 20

# Nodes that may be seeded with a local shard payload: the 16 active nodes
# plus 6 backup nodes the watcher promotes into --active-nodes on
# replenishment. Without pre-seeding, a promoted backup has no
# /Data/$USER/mesh payload and dies on read_my_shards' KeyError.
MAX_SEEDED_NODES = 22

# Re-exported for the callers that historically imported the roster from this
# script. The roster itself now lives in cluster/tiers.tsv.
__all__ = ["NODE_GROUPS", "NODE_SUBNETS", "MAX_MESH_NODES", "MAX_SEEDED_NODES"]

_DF_COMMAND = "df -BG /Data | awk 'NR==2{print $2, $4}'"
_VAL_SUBDIR = "val"


def _run_ssh(host: str, command: str, *, timeout_s: float = 20.0) -> str:
    """Default `run_ssh`: real `ssh <host> <command>`, returning stdout.

    Raises `OSError` on any non-zero exit or connection failure so callers
    (`probe_nodes`) can treat every failure mode uniformly.
    """
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


def probe_nodes(hostnames: list[str], *, run_ssh=_run_ssh) -> list[NodeInfo]:
    """SSH-probe each host's live free space on `/Data`.

    Unreachable hosts (a dead node, a transient SSH failure -- orchids has
    a known-unreachable node) are skipped with a warning rather than
    raising: one bad host must not abort an entire seeding run.
    """
    nodes: list[NodeInfo] = []
    for host in hostnames:
        group = NODE_GROUPS.get(host)
        subnet = NODE_SUBNETS.get(host)
        if group is None or subnet is None:
            print(
                f"WARNING: probe_nodes: {host!r} is not in the known roster "
                "(NODE_GROUPS/NODE_SUBNETS) -- skipping",
                file=sys.stderr,
            )
            continue

        try:
            output = run_ssh(host, _DF_COMMAND)
        except (OSError, subprocess.SubprocessError) as exc:
            print(
                f"WARNING: probe_nodes: {host!r} unreachable ({exc}) -- skipping",
                file=sys.stderr,
            )
            continue

        parts = output.split()
        if len(parts) < 2:
            print(
                f"WARNING: probe_nodes: unexpected df output from {host!r}: "
                f"{output!r} -- skipping",
                file=sys.stderr,
            )
            continue

        try:
            free_gb = float(parts[1].rstrip("Gg"))
        except ValueError:
            print(
                f"WARNING: probe_nodes: could not parse free space from "
                f"{host!r} df output {output!r} -- skipping",
                file=sys.stderr,
            )
            continue

        nodes.append(
            NodeInfo(hostname=host, group=group, subnet=subnet, free_gb=free_gb)
        )

    return nodes


def build_shard_map(
    placement: dict[str, list[str]], nodes: list[NodeInfo], *, port: int = 8000
) -> dict[str, list[Holder]]:
    """Invert a `hostname -> [shard, ...]` placement into `shard -> [Holder, ...]`.

    Holder order for a given shard follows `placement`'s host iteration
    order, so it is deterministic for a given placement dict.
    """
    by_host = {node.hostname: node for node in nodes}
    shard_map: dict[str, list[Holder]] = {}
    for host, shards in placement.items():
        node = by_host.get(host)
        if node is None:
            raise ValueError(
                f"build_shard_map: placement references host {host!r} with no "
                "matching NodeInfo in `nodes`"
            )
        holder = Holder(host=host, subnet=node.subnet, port=port)
        for shard in shards:
            shard_map.setdefault(shard, []).append(holder)
    return shard_map


def rsync_plan(
    placement: dict[str, list[str]], source_node: str, *, mesh_dir: str = MESH_DIR
) -> list[tuple[str, str, list[str]]]:
    """Plan target-pulls (never a laptop push) for every non-source holder.

    Returns `(target, source, shards)` triples, one per host in `placement`
    other than `source_node` that holds at least one shard. `main` turns
    each triple into a `ssh target "rsync --files-from=- source:... ..."`
    pull so the transfer runs node-to-node over the cluster's internal
    network.
    """
    plan: list[tuple[str, str, list[str]]] = []
    for host in sorted(placement):
        if host == source_node:
            continue
        shards = placement[host]
        if not shards:
            continue
        plan.append((host, source_node, list(shards)))
    return plan


def _execute_rsync_pull(
    target: str,
    source_node: str,
    shards: list[str],
    *,
    mesh_dir: str,
    remote_source_dir: str,
    subdir: str | None = None,
    timeout_s: float = 1800.0,
) -> None:
    """Run one target-side pull: `ssh target "mkdir/chmod && rsync --files-from=- ..."`.

    `subdir=None` (train shards) lands files flat under `mesh_dir`, matching
    `resolve_shard`'s `mesh_root / shard` lookup (Task 8) and `Holder.url()`'s
    flat `http://host:port/{shard}` scheme (Task 1) -- a `shards/` subdirectory
    here would make every seeded train shard invisible to both. `subdir="val"`
    keeps replicated val shards isolated from train shards, which share the
    same `shard-{i:06d}.tar` naming from `build_wds_shards.py` and would
    otherwise collide if both were flattened into `mesh_dir`.

    The shard filenames are piped over stdin to the remote rsync's
    `--files-from=-` rather than interpolated into the command string --
    the nested `ssh target "rsync source:..."` quoting is a real hazard
    with many filenames, so only the handful of path-level arguments
    (mesh_dir, source_node, remote_source_dir) are `shlex.quote`d into the
    command text at all.
    """
    dest_dir = f"{mesh_dir}/{subdir}" if subdir else mesh_dir
    if source_node == target:
        source_spec = f"{shlex.quote(remote_source_dir)}/"
    else:
        source_spec = f"{shlex.quote(source_node)}:{shlex.quote(remote_source_dir)}/"

    mkdir_targets = " ".join(
        shlex.quote(d) for d in dict.fromkeys([mesh_dir, dest_dir])
    )
    chmod_targets = mkdir_targets
    remote_cmd = (
        f"mkdir -p {mkdir_targets} && "
        f"chmod 700 {chmod_targets} && "
        f"rsync -az --files-from=- {source_spec} {shlex.quote(dest_dir)}/"
    )

    # ServerAlive* forces a dead-but-not-closed session (observed after a
    # completed 65-shard/54GB transfer to node090: files landed correctly but
    # the ssh process itself hung past the 1800s timeout with nothing left
    # to send) to be detected and killed in seconds rather than silently
    # blocking until timeout_s.
    result = subprocess.run(
        [
            "ssh",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=3",
            target,
            remote_cmd,
        ],
        input="\n".join(shards) + "\n",
        text=True,
        capture_output=True,
        timeout=timeout_s,
    )
    if result.returncode != 0:
        raise OSError(
            f"rsync pull {source_node} -> {target} ({subdir}) failed: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )


def _symlink_source_node(
    source_node: str,
    own_train_shards: list[str],
    val_shards: list[str],
    *,
    mesh_dir: str,
    remote_source_dir: str,
) -> None:
    """Make `source_node`'s own placement holdings resolvable under `mesh_dir`.

    `rsync_plan` never targets `source_node` (it already holds every shard
    under `remote_source_dir`, not `mesh_dir`), but `assign_locality_first`
    still treats `placement[source_node]` as this node's local holdings, and
    `resolve_shard`'s `mesh_root / shard` lookup only checks `mesh_dir` --
    without this, the source node would report its own assigned shards as
    missing and fetch them over HTTP from a replica for zero reason. Symlinks
    (not a second rsync copy) since the bytes already exist locally under
    `remote_source_dir`; `Path.exists()` (what `resolve_shard` calls) follows
    them transparently.
    """
    val_dir = f"{mesh_dir}/{_VAL_SUBDIR}"
    remote_val_dir = f"{remote_source_dir.rstrip('/')}/{_VAL_SUBDIR}"
    remote_cmd = (
        f"mkdir -p {shlex.quote(mesh_dir)} {shlex.quote(val_dir)} && "
        f"chmod 700 {shlex.quote(mesh_dir)} {shlex.quote(val_dir)} && "
        "while IFS= read -r f; do "
        f'ln -sf {shlex.quote(remote_source_dir)}/"$f" {shlex.quote(mesh_dir)}/"$f"; '
        "done"
    )
    stdin_lines = "\n".join(own_train_shards + ["manifest.json"]) + "\n"
    result = subprocess.run(
        [
            "ssh",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=3",
            source_node,
            remote_cmd,
        ],
        input=stdin_lines,
        text=True,
        capture_output=True,
        timeout=120.0,
    )
    if result.returncode != 0:
        raise OSError(
            f"symlink source-node shards on {source_node!r} failed: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )

    remote_cmd_val = (
        f"mkdir -p {shlex.quote(val_dir)} && chmod 700 {shlex.quote(val_dir)} && "
        "while IFS= read -r f; do "
        f'ln -sf {shlex.quote(remote_val_dir)}/"$f" {shlex.quote(val_dir)}/"$f"; '
        "done"
    )
    # val/ has its own manifest.json (a distinct clip_key -> shard index built
    # by build_wds_shards.py for the val split, not interchangeable with the
    # train manifest symlinked above) -- MeshShardIterableDataset(mesh_root=
    # val_root) loads it unconditionally, so it must be linked here too.
    result = subprocess.run(
        [
            "ssh",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=3",
            source_node,
            remote_cmd_val,
        ],
        input="\n".join(val_shards + ["manifest.json"]) + "\n",
        text=True,
        capture_output=True,
        timeout=120.0,
    )
    if result.returncode != 0:
        raise OSError(
            f"symlink source-node val shards on {source_node!r} failed: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )


def _list_remote_shards(host: str, remote_dir: str) -> list[str]:
    """`ls` a remote directory on `host` and return `shard-*.tar` entries, sorted."""
    output = _run_ssh(host, f"ls -1 {shlex.quote(remote_dir)}")
    return sorted(
        line.strip()
        for line in output.splitlines()
        if line.strip().startswith("shard-") and line.strip().endswith(".tar")
    )


def _remote_file_exists(host: str, remote_path: str) -> bool:
    try:
        _run_ssh(host, f"test -f {shlex.quote(remote_path)}")
        return True
    except OSError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Probe live free space on candidate mesh nodes, plan capacity-aware "
            "shard placement, distribute shards via target-pull rsync, and "
            "write the resulting shard map."
        )
    )
    parser.add_argument(
        "--nodes",
        required=True,
        help="comma-separated candidate hostnames to probe (e.g. pick_nodes.sh output)",
    )
    parser.add_argument(
        "--source-node",
        required=True,
        help=(
            "hostname that already holds the full built shard set (train + "
            "val/). May be outside --nodes, in which case it serves shards "
            "without being seeded or counted in the placement split."
        ),
    )
    parser.add_argument(
        "--active-nodes",
        default=None,
        help=(
            "comma-separated subset of --nodes that will push checkpoints to "
            "shared NFS this run (<= MAX_MESH_NODES). Defaults to all of "
            "--nodes, matching prior behaviour for callers that never seed "
            "backup nodes. Pass this when --nodes includes backup nodes that "
            "should only hold a seeded local payload."
        ),
    )
    parser.add_argument(
        "--shards-dir",
        required=True,
        help="remote directory on --source-node with shard-*.tar and a val/ subdir",
    )
    parser.add_argument(
        "--shard-map",
        required=True,
        type=Path,
        help="local path to write shard_map JSON",
    )
    parser.add_argument("--dataset-gb", required=True, type=float)
    parser.add_argument("--replication", type=int, default=2)
    parser.add_argument("--min-free-gb", type=float, default=50.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print per-node shard counts and exit without transferring",
    )
    args = parser.parse_args()

    hostnames = [h.strip() for h in args.nodes.split(",") if h.strip()]
    active_nodes = (
        [h.strip() for h in args.active_nodes.split(",") if h.strip()]
        if args.active_nodes is not None
        else hostnames
    )
    if not set(active_nodes) <= set(hostnames):
        print(
            "seed_mesh: --active-nodes must be a subset of --nodes -- "
            f"extra: {sorted(set(active_nodes) - set(hostnames))}",
            file=sys.stderr,
        )
        return 1
    if len(active_nodes) > MAX_MESH_NODES:
        print(
            f"seed_mesh: {len(active_nodes)} --active-nodes exceeds "
            f"MAX_MESH_NODES={MAX_MESH_NODES} (shared home quota: N x 172 MB "
            "checkpoint + master + momentum buffer) -- reduce --active-nodes",
            file=sys.stderr,
        )
        return 1

    nodes = probe_nodes(hostnames)
    if not nodes:
        print("seed_mesh: no reachable nodes among --nodes; aborting", file=sys.stderr)
        return 1
    if len(nodes) > MAX_SEEDED_NODES:
        print(
            f"seed_mesh: {len(nodes)} reachable node(s) exceeds MAX_SEEDED_NODES="
            f"{MAX_SEEDED_NODES} (nodes that may hold a seeded local shard "
            "payload) -- reduce --nodes",
            file=sys.stderr,
        )
        return 1
    # The source node only has to *serve* shards, so it may sit outside
    # --nodes entirely (the corpus is built on a file server like node114, which
    # is deliberately absent from tiers.tsv and would otherwise be skipped by
    # probe_nodes and abort the run). Keeping it out of --nodes also keeps
    # n_shards divisible by the pool size. Reachability is still enforced --
    # _list_remote_shards below fails loudly on an unreachable source.
    if args.source_node in set(hostnames) and args.source_node not in {
        n.hostname for n in nodes
    }:
        print(
            f"seed_mesh: --source-node {args.source_node!r} was listed in "
            "--nodes but did not probe successfully",
            file=sys.stderr,
        )
        return 1

    try:
        train_shards = _list_remote_shards(args.source_node, args.shards_dir)
        val_shards = _list_remote_shards(
            args.source_node, f"{args.shards_dir.rstrip('/')}/val"
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(
            f"seed_mesh: failed to list shards on {args.source_node!r}: {exc}",
            file=sys.stderr,
        )
        return 1

    # manifest.json is the single whole-set index MeshShardIterableDataset.__iter__
    # loads unconditionally on every node before reading a single shard (Task 8's
    # load_shard_index(mesh_root)) -- without it every worker fails immediately.
    manifest_path = f"{args.shards_dir.rstrip('/')}/manifest.json"
    if not _remote_file_exists(args.source_node, manifest_path):
        print(
            f"seed_mesh: manifest.json not found at {manifest_path!r} on "
            f"{args.source_node!r} -- run build_wds_shards.py first",
            file=sys.stderr,
        )
        return 1

    # val/ has its own manifest.json (built separately for the val shard set,
    # not interchangeable with the train one above) -- evaluate_val_loss's
    # MeshShardIterableDataset(mesh_root=mesh_root/"val") loads it
    # unconditionally on every node, so it must exist on source before we
    # plan any replication.
    val_manifest_path = f"{args.shards_dir.rstrip('/')}/val/manifest.json"
    if not _remote_file_exists(args.source_node, val_manifest_path):
        print(
            f"seed_mesh: val manifest.json not found at {val_manifest_path!r} "
            f"on {args.source_node!r} -- run build_wds_shards.py for the val "
            "split first",
            file=sys.stderr,
        )
        return 1

    try:
        placement = plan_placement(
            nodes,
            train_shards,
            dataset_gb=args.dataset_gb,
            replication=args.replication,
            min_free_gb=args.min_free_gb,
        )
    except (InsufficientCapacityError, ValueError) as exc:
        print(f"seed_mesh: placement failed: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(
            f"seed_mesh: dry-run -- {len(train_shards)} train shard(s), "
            f"{len(val_shards)} val shard(s) across {len(nodes)} probed node(s)"
        )
        for host in sorted(placement):
            print(f"  {host}: {len(placement[host])} shard(s)")
        return 0

    shard_map = build_shard_map(placement, nodes)
    plan = rsync_plan(placement, args.source_node)

    if args.source_node in placement:
        print(
            f"seed_mesh: symlinking {args.source_node}'s own holdings into {MESH_DIR}"
        )
        _symlink_source_node(
            args.source_node,
            placement[args.source_node],
            val_shards,
            mesh_dir=MESH_DIR,
            remote_source_dir=args.shards_dir,
        )
    else:
        print(
            f"seed_mesh: {args.source_node} is a serve-only source (outside "
            "--nodes); not seeding a mesh payload onto it"
        )

    for target, source, shards in plan:
        print(f"seed_mesh: pulling {len(shards)} shard(s) {source} -> {target}")
        _execute_rsync_pull(
            target,
            source,
            shards,
            mesh_dir=MESH_DIR,
            remote_source_dir=args.shards_dir,
        )

    for node in nodes:
        if node.hostname == args.source_node:
            continue
        print(
            f"seed_mesh: replicating {len(val_shards)} val shard(s) + "
            f"val manifest.json {args.source_node} -> {node.hostname}"
        )
        _execute_rsync_pull(
            node.hostname,
            args.source_node,
            val_shards + ["manifest.json"],
            mesh_dir=MESH_DIR,
            remote_source_dir=f"{args.shards_dir.rstrip('/')}/val",
            subdir=_VAL_SUBDIR,
        )

    for node in nodes:
        if node.hostname == args.source_node:
            continue
        print(
            f"seed_mesh: replicating manifest.json {args.source_node} -> {node.hostname}"
        )
        _execute_rsync_pull(
            node.hostname,
            args.source_node,
            ["manifest.json"],
            mesh_dir=MESH_DIR,
            remote_source_dir=args.shards_dir,
        )

    save_shard_map(args.shard_map, shard_map)
    print(f"seed_mesh: wrote shard map ({len(shard_map)} shard(s)) to {args.shard_map}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
