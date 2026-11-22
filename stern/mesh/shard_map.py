"""Shard holder registry for the decentralized P2P training mesh.

Maps a WebDataset shard filename to the list of nodes ("holders") that have
it on local disk, and picks which holder a fetcher should pull from. No I/O
beyond the explicit save/load functions; selection logic is pure so it can
be fixture-tested like `stern/watcher/detectors.py`.

Selection prefers a same-subnet holder: the cross-switch uplink caps at
~0.94 Gbps *in total* while the intra-switch backplane is non-blocking, so a
cross-subnet pull contends with every other job sharing that uplink. Among equally-preferred
holders, a hash of the shard name spreads fetchers across replicas instead
of funneling everyone onto `holders[0]`'s single 1GbE NIC.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Holder:
    host: str
    subnet: str  # "/254", "/253", "/252"
    port: int = 8000


def save_shard_map(path: Path, shard_map: dict[str, list[Holder]]) -> None:
    """Write `shard_map` atomically (tmp + rename), matching `watcher/registry.py`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        shard: [asdict(holder) for holder in holders]
        for shard, holders in shard_map.items()
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def load_shard_map(path: Path) -> dict[str, list[Holder]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return {
        shard: [Holder(**holder) for holder in holders]
        for shard, holders in payload.items()
    }


def find_holders(shard_map: dict[str, list[Holder]], shard: str) -> list[Holder]:
    return shard_map.get(shard, [])
