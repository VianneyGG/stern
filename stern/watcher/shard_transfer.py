"""Copy a departed node's shards onto its replacement.

`mesh.assignment.reassign_failed` can only route an orphaned shard to a
survivor that already holds a replica, and drops it otherwise -- correct,
since a node cannot open a shard it does not have and the P2P fetch path is
gone. But replication is 1x for 40-68% of the corpus on every tier except
fish, so dropping is the common case. One real run lost 46 of 260 shards
(17.7%) that way.

The shards were never gone. The excluded nodes stayed up and reachable the
whole time -- SLURM still called them idle -- they simply could not run a
worker, because a foreign process held their GPU or they had never been
seeded. Copying the files to a node that *can* run is what turns a permanent
coverage loss back into a transfer.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from collections.abc import Sequence

from stern.cluster_env import data_root

MESH_DATA_DIR = str(data_root() / "mesh")

# Shards are ~350 MB each over 1GbE (every tier is 1GbE -- there is no
# higher-bandwidth tier), so a handful of them is minutes, not seconds.
DEFAULT_TIMEOUT_S = 900.0


def copy_shards(
    src_node: str,
    dst_node: str,
    shards: Sequence[str],
    *,
    mesh_data_dir: str = MESH_DATA_DIR,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> list[str]:
    """rsync `shards` from `src_node` to `dst_node`; return what landed.

    The transfer is a target-side PULL -- `ssh dst_node "rsync src_node:... ."`
    -- not `rsync src_node:... dst_node:...`. rsync refuses a command with two
    remote endpoints ("The source and destination cannot both be remote"), so
    the direct form fails on every call. This function never raises, so that
    failure was silent: every transfer returned [] with nothing logged. The
    pull also keeps the bytes node-to-node instead of routing them through the
    watcher's host.

    Shard names go to the remote rsync over stdin via `--files-from=-` rather
    than being interpolated into the command string: the nested
    `ssh dst "rsync src:..."` quoting is a real hazard once there are many
    filenames.

    All-or-nothing per call: rsync either reports success for the batch or
    this returns `[]`. A partial-success report would be worse than useless,
    because the caller assigns exactly what comes back. A worker must never be
    assigned a shard that did not arrive, and it raises rather than skipping.

    Never raises: a transfer failure must leave the pool under-filled, not
    crash the watcher tick. It does report the reason on stderr, which lands
    in the watcher log -- returning [] with no explanation is what let the
    remote-to-remote bug survive undetected.
    """
    if not shards:
        return []

    quoted_dir = shlex.quote(mesh_data_dir)
    remote_cmd = (
        f"mkdir -p {quoted_dir} && "
        # -L dereferences: a seeded shard is often a symlink into the source
        # node's own shard directory, and a reboot can wipe that target. Plain
        # -a would copy the dangling link, so the shard would list as present
        # and fail to open.
        "rsync -aL --partial --files-from=- "
        f"-e {shlex.quote('ssh -o StrictHostKeyChecking=no')} "
        f"{shlex.quote(src_node)}:{quoted_dir}/ {quoted_dir}/"
    )
    cmd = ["ssh", "-o", "StrictHostKeyChecking=no", dst_node, remote_cmd]
    try:
        result = subprocess.run(
            cmd,
            input="\n".join(shards) + "\n",
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(
            f"WARNING: copy_shards: {src_node} -> {dst_node} failed to run ({exc})",
            file=sys.stderr,
        )
        return []
    if result.returncode != 0:
        print(
            f"WARNING: copy_shards: {src_node} -> {dst_node} exited "
            f"{result.returncode}: {result.stderr.strip() or result.stdout.strip()}",
            file=sys.stderr,
        )
        return []
    return list(shards)
