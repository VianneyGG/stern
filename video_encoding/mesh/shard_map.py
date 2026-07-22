"""Shard holder registry for the decentralized P2P training mesh.

Maps a WebDataset shard filename to the list of nodes ("holders") that have
it on local disk, and picks which holder a fetcher should pull from. No I/O
beyond the explicit save/load functions; selection logic is pure so it can
be fixture-tested like `video_encoding/watcher/detectors.py`.

Selection prefers a same-subnet holder: the cross-switch uplink caps at
~0.94 Gbps *in total* while the intra-switch backplane is non-blocking
(`2026-06-30-cluster-tier1-findings.md` T2.3), so a cross-subnet pull
contends with every other job sharing that uplink. Among equally-preferred
holders, a hash of the shard name spreads fetchers across replicas instead
of funneling everyone onto `holders[0]`'s single 1GbE NIC.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Holder:
    host: str
    subnet: str  # "/254", "/253", "/252"
    port: int = 8000

    def url(self, shard: str) -> str:
        return f"http://{self.host}:{self.port}/{shard}"


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


def _spread_index(shard: str, n: int) -> int:
    """Deterministic pseudo-random index in [0, n), coordinator-free.

    Uses a content hash of the shard name (not Python's `hash()`, which is
    salted per-process) so every node picks the same holder for a given
    shard without needing to talk to each other, while different shards
    land on different holders.
    """
    digest = hashlib.sha256(shard.encode("utf-8")).digest()
    return int.from_bytes(digest, "big") % n


def pick_holder(
    shard_map: dict[str, list[Holder]],
    shard: str,
    requester_subnet: str,
    *,
    exclude: frozenset[str] = frozenset(),
) -> Holder | None:
    candidates = [h for h in find_holders(shard_map, shard) if h.host not in exclude]
    if not candidates:
        return None

    same_subnet = [h for h in candidates if h.subnet == requester_subnet]
    pool = same_subnet if same_subnet else candidates

    # Sort so the spread index doesn't depend on the input list's order.
    pool = sorted(pool, key=lambda h: h.host)
    return pool[_spread_index(shard, len(pool))]
