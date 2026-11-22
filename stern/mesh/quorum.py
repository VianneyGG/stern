"""Quorum merge: replaces the all-or-timeout wait in watcher/mesh_tick.py.

Waiting for every node in the round (or the full ROUND_TIMEOUT_S) before
merging is what costs 297s/round on average (spec: 2604.21428 Decoupled
DiLoCo, 2501.18512 Streaming DiLoCo). `ready_for_merge` lets a round merge
once `quorum_k` nodes have checked in, without waiting on stragglers.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass


@dataclass(frozen=True)
class NodeStats:
    c_tokens: int  # samples actually trained on this round (0 if no shards fetched)
    c_steps: int  # local optimizer steps actually completed this round


def token_weights(stats: dict[str, NodeStats]) -> dict[str, float]:
    """`w_i = c_tokens_i * (c_tokens_i / c_steps_i)`, normalized to sum to 1.

    "Quantity x quality" (spec 2.2): squares the token count and divides by
    steps taken to earn it, so a node that trained more data per step (a
    healthy, well-fed worker) is weighted up relative to one that needed
    more steps for the same tokens. Supersedes weighting="data"
    (mesh/outer_step.data_weights), which renormalizes only over whoever's
    checkpoint happened to arrive -- silently biased whenever the round
    didn't get a full check-in. A node with c_tokens=0 (fetched no shards)
    is weighted to exactly 0.0 without ever computing 0/c_steps, so a
    zero-token, zero-step node (the common case -- it never trained at all)
    can't produce a 0/0 NaN.
    """
    raw = {
        name: 0.0
        if s.c_tokens <= 0 or s.c_steps <= 0
        else s.c_tokens * (s.c_tokens / s.c_steps)
        for name, s in stats.items()
    }
    total = sum(raw.values())
    if total <= 0:
        raise ValueError(
            f"token_weights: total weight is zero (stats={stats}); "
            "cannot compute proportional weights"
        )
    return {name: w / total for name, w in raw.items()}


def ready_for_merge(
    checked_in: Collection[str], *, quorum_k: int, grace_s: float, now: float
) -> bool:
    """True once `quorum_k` nodes have checked in, or once `now` (elapsed
    seconds since the round started) reaches `grace_s`, whichever is first.

    Quorum short-circuits the wait: it returns True the instant quorum_k is
    met, even if grace_s hasn't elapsed -- grace_s only matters for the
    below-quorum fallback, where it plays the role ROUND_TIMEOUT_S played
    before quorum merge (merge with whatever arrived rather than wait
    indefinitely for a straggler that may never check in).
    """
    if len(checked_in) >= quorum_k:
        return True
    return now >= grace_s
