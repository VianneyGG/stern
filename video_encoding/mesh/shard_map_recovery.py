"""Rebuild `shard_map.json` from each mesh node's actual on-disk holdings.

Used when the shard_map metadata file was lost (or never written) even
though the underlying shard placement across nodes is still correct --
recomputing from live holdings avoids a fresh `seed_mesh.py` placement run,
which would drift from what's physically on disk as free space changes.
"""

from __future__ import annotations

from video_encoding.mesh.shard_map import Holder


def build_shard_map_from_holdings(
    holdings: dict[str, list[str]],
    node_subnets: dict[str, str],
    *,
    port: int = 8000,
) -> dict[str, list[Holder]]:
    """Invert `{host: [shard, ...]}` into `{shard: [Holder, ...]}`.

    Every host in `holdings` must have an entry in `node_subnets` --
    `pick_holder`'s same-subnet preference silently degrades to always
    cross-subnet if a subnet is missing or defaulted, so a missing entry
    raises `KeyError` naming the host rather than guessing.

    Iterates hosts in sorted order so holder lists come out in the same
    order regardless of `holdings`' insertion order.
    """
    shard_map: dict[str, list[Holder]] = {}
    for host in sorted(holdings):
        if host not in node_subnets:
            raise KeyError(host)
        subnet = node_subnets[host]
        for shard in holdings[host]:
            shard_map.setdefault(shard, []).append(
                Holder(host=host, subnet=subnet, port=port)
            )
    return shard_map
