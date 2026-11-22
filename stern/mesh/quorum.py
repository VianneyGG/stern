"""Quorum merge: merge a round once `quorum_k` nodes have checked in.

Waiting for every node, or for ROUND_TIMEOUT_S, cost about 297 s per round.
The decoupled and streaming DiLoCo designs (arXiv 2604.21428, 2501.18512)
avoid that barrier. `ready_for_merge` lets stragglers miss the round.
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

    Quantity times quality: a node that trained more data per step is weighted
    up relative to one that needed more steps for the same tokens. Replaces
    `weighting="data"` (`outer_step.data_weights`), which renormalizes only over
    the checkpoints that arrived and so is biased when a round is incomplete.
    A node with no tokens or no steps gets weight exactly 0.0 without computing
    0/c_steps, so it cannot produce a NaN.
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
