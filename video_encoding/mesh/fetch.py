"""P2P shard fetch with retries and holder fallback.

This is the **exception path** for the decentralized mesh: locality-first
assignment (`mesh/assignment.py`) means most shards a node trains are
already local, so `fetch_shard` only runs for the delta produced by
throughput recalibration, a capacity-capped node shedding work, or a failed
node's orphans (`mesh/shard_map.py`).

A shard is ~1 GB, so the transfer is streamed in ~1 MiB chunks rather than
buffered in RAM, and written to a `.part` file that is only renamed to its
final name on success -- an interrupted fetch must never be mistakable for
a complete shard by whatever trains from it next.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import requests

from video_encoding.mesh.shard_map import Holder, pick_holder

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 1 * 1024 * 1024  # ~1 MiB


class ShardFetchError(RuntimeError):
    """Raised when a shard cannot be fetched from any known holder."""


def _download(url: str, part_path: Path, timeout_s: float) -> None:
    """Stream `url` to `part_path`, ~1 MiB at a time. Raises on any failure."""
    with requests.get(url, stream=True, timeout=timeout_s) as response:
        response.raise_for_status()
        with open(part_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
                if chunk:
                    f.write(chunk)


def fetch_shard(
    shard: str,
    shard_map: dict[str, list[Holder]],
    requester_subnet: str,
    dest_dir: Path,
    *,
    attempts: int = 3,  # per holder
    timeout_s: float = 60.0,
    sleep=time.sleep,  # injected so tests don't actually sleep
) -> Path:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_dir.chmod(0o700)

    final_path = dest_dir / shard
    if final_path.exists():
        return final_path

    part_path = dest_dir / f"{shard}.part"

    tried: set[str] = set()
    last_exc: Exception | None = None

    while True:
        holder = pick_holder(
            shard_map, shard, requester_subnet, exclude=frozenset(tried)
        )
        if holder is None:
            if not tried:
                raise ShardFetchError(f"no holder known for shard {shard!r}")
            raise ShardFetchError(
                f"all holders exhausted for shard {shard!r}: last error: {last_exc}"
            )

        tried.add(holder.host)

        for attempt in range(1, attempts + 1):
            try:
                _download(holder.url(shard), part_path, timeout_s)
            except (requests.RequestException, OSError) as exc:
                last_exc = exc
                part_path.unlink(missing_ok=True)
                logger.warning(
                    "fetch_shard: attempt %d/%d for %s from %s failed: %s",
                    attempt,
                    attempts,
                    shard,
                    holder.host,
                    exc,
                )
                sleep(2**attempt)
            else:
                part_path.rename(final_path)
                return final_path
        # attempts exhausted on this holder -- fall back to a different one
