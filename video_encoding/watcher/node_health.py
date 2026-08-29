"""Real node readiness, as opposed to what SLURM is willing to say.

`cluster.query_idle_nodes` asks sinfo and believes the answer. On 2026-08-22
that answer was wrong for 10 of 12 birds nodes: sinfo reported every excluded
node "idle" while 7 still had ~21.7 GiB of their 24.5 GiB GPU held by a
foreign process and 3 had never been seeded. `_replenish_pool` picked 6 of
them and all 6 crashed on submit -- 3 with `torch.OutOfMemoryError`, 3 with
`FileNotFoundError: .../mesh/manifest.json` -- each burning the two missed
rounds that permanently exclude a node.

Readiness is three independent facts from three sources: SLURM says nobody
else is scheduled here, nvidia-smi says the GPU is actually free, and the
filesystem says the shards are present. Checking one of the three is what
made a guaranteed-crash job look like a healthy pick, so `truly_idle` wants
all three and treats unknown as unhealthy -- an unreachable probe is exactly
when submitting a doomed job is most likely.
"""

from __future__ import annotations

import subprocess
from collections.abc import Collection
from dataclasses import dataclass

# A worker needs ~2 GiB for the model plus headroom for activations; a node
# holding a 21 GiB foreign process leaves ~2.9 GiB and OOMs during the first
# forward pass, not at allocation time.
#
# NOT "most of the card": cards differ by tier, so a fraction-of-total rule
# written against 24 GiB silently bans whole tiers. Free MiB on an IDLE node,
# measured across all four tiers on 2026-08-22: fish 23973-24080, birds 23954,
# cars 17560-19860, countries 13392-15817. An occupied node reads 1495-2453.
# The old 20_000 floor therefore rejected every cars and countries node as
# unhealthy -- on a cross-tier roster that leaves replenishment able to draw
# from only half the cluster, while the gap it actually has to detect (2453 vs
# 13392) is enormous. 10 GiB sits in that gap with room on both sides.
DEFAULT_MIN_GPU_FREE_MIB = 10_000

_MESH_DATA_DIR = "/Data/$USER/mesh"

# One ssh round-trip per node yielding exactly two lines: free GPU MiB, then
# the seeded shard count (0 when the manifest is missing, which is how an
# unseeded node reports itself rather than failing the whole probe).
_PROBE = (
    "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; "
    f"if [ -f {_MESH_DATA_DIR}/manifest.json ]; then ls {_MESH_DATA_DIR}/*.tar "
    "2>/dev/null | wc -l; else echo 0; fi"
)


@dataclass(frozen=True)
class NodeHealth:
    """`None` means the probe could not answer, never "fine"."""

    node: str
    slurm_idle: bool
    gpu_free_mib: int | None
    seeded_shards: int | None


def truly_idle(
    healths: Collection[NodeHealth],
    *,
    min_gpu_free_mib: int = DEFAULT_MIN_GPU_FREE_MIB,
    require_seeded: bool = True,
) -> set[str]:
    """Nodes that can actually run a worker right now.

    Fails closed: a node whose GPU or seeding is unknown is excluded, so a
    probe timeout under-fills the pool instead of submitting a crash. That
    mirrors `_replenish_pool`'s existing stance -- an under-filled pool is a
    safe failure mode, a guaranteed-crash job is not.
    """
    ready = set()
    for health in healths:
        if not health.slurm_idle:
            continue
        if health.gpu_free_mib is None or health.gpu_free_mib < min_gpu_free_mib:
            continue
        if require_seeded and not health.seeded_shards:
            continue
        ready.add(health.node)
    return ready


def _parse_probe(stdout: str) -> tuple[int | None, int | None]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    values: list[int | None] = [None, None]
    for i in range(min(2, len(lines))):
        try:
            values[i] = int(lines[i])
        except ValueError:
            values[i] = None
    return values[0], values[1]


def query_node_health(
    nodes: Collection[str],
    *,
    idle: Collection[str],
    timeout_s: float = 10.0,
) -> list[NodeHealth]:
    """Probe each node's GPU and seeding over ssh; never raises.

    `idle` is the SLURM view (`cluster.query_idle_nodes`), passed in rather
    than queried here so the caller keeps its single sinfo call. Any probe
    failure -- timeout, nonzero exit, unparseable output -- yields `None`
    fields, which `truly_idle` reads as unhealthy.
    """
    idle_set = set(idle)
    healths = []
    for node in nodes:
        gpu_free_mib: int | None = None
        seeded_shards: int | None = None
        try:
            result = subprocess.run(
                [
                    "ssh",
                    "-o",
                    "StrictHostKeyChecking=no",
                    "-o",
                    f"ConnectTimeout={int(timeout_s)}",
                    node,
                    _PROBE,
                ],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            if result.returncode == 0:
                gpu_free_mib, seeded_shards = _parse_probe(result.stdout)
        except (OSError, subprocess.SubprocessError):
            pass
        healths.append(
            NodeHealth(
                node=node,
                slurm_idle=node in idle_set,
                gpu_free_mib=gpu_free_mib,
                seeded_shards=seeded_shards,
            )
        )
    return healths


def foreign_gpu_pids(
    gpu_pids: Collection[int], job_pid_tree: Collection[int]
) -> set[int]:
    """GPU compute PIDs not in this SLURM job's own process tree."""
    return set(gpu_pids) - set(job_pid_tree)


_GPU_PROC_PROBE = (
    "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader"
)


def query_gpu_process_pids(node: str, *, timeout_s: float = 10.0) -> set[int] | None:
    """GPU compute PIDs currently running on `node`. None on any probe
    failure -- never raises, mirrors query_node_health."""
    try:
        result = subprocess.run(
            [
                "ssh",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                f"ConnectTimeout={int(timeout_s)}",
                node,
                _GPU_PROC_PROBE,
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if result.returncode != 0:
            return None
        pids = set()
        for line in result.stdout.splitlines():
            pid_str = line.split(",")[0].strip()
            if pid_str.isdigit():
                pids.add(int(pid_str))
        return pids
    except (OSError, subprocess.SubprocessError):
        return None


def query_job_pid_tree(
    node: str, slurm_job_id: str, *, timeout_s: float = 10.0
) -> set[int] | None:
    """PIDs SLURM attributes to `slurm_job_id` on `node` (`scontrol listpids`).
    None on any probe failure."""
    try:
        result = subprocess.run(
            [
                "ssh",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                f"ConnectTimeout={int(timeout_s)}",
                node,
                f"scontrol listpids {slurm_job_id}",
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if result.returncode != 0:
            return None
        pids = set()
        for line in result.stdout.splitlines()[1:]:  # header row
            parts = line.split()
            if parts and parts[0].isdigit():
                pids.add(int(parts[0]))
        return pids
    except (OSError, subprocess.SubprocessError):
        return None
