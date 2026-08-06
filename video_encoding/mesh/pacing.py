"""Per-node step budgets: make every node's round take the same *time*.

Measured over 1455 production rounds, `flag_wait_s` averaged 296.9s against
4.8s of real NFS I/O -- the mesh spends 19.1% of its wall-clock with fast
nodes sitting idle at the barrier. The cause is structural: `steps_per_round`
is uniform for every node while measured throughput within a single node
group spans 4.5x (birds) to 9.8x (cars), so a round's duration is
`steps_per_round / rate_i` and the fast nodes wait out the difference.

Overlapping the wait with extra local training (the "train through the wait"
approach) does not fix this: filling the mean 297s barrier at the median
9.91 steps/s needs ~2900 steps, 30% of a 9600-step round, so each node would
carry a third of a round's worth of un-merged divergence into the next
round's pseudo-gradient -- breaking DiLoCo's assumption that every worker
starts an inner phase from the merged parameters.

Sizing the work to the node instead removes the idle rather than filling it.
Each node independently trains for `budget_s` seconds' worth of steps at its
own observed rate; the target duration is a constant from config, so no
coordination or protocol change is needed. Unequal step counts are already
handled correctly at merge time by `outer_step.data_weights`, which weights
each node's pseudo-gradient by the samples it actually trained on.
"""

from __future__ import annotations


def steps_for_budget(
    *,
    rate_steps_per_s: float,
    budget_s: float,
    floor_steps: int,
    ceil_steps: int,
) -> int:
    """How many steps this node can fit in `budget_s`, clamped to [floor, ceil].

    Both clamps are load-bearing. Without `floor_steps`, a badly degraded node
    contributes a handful of steps whose pseudo-gradient is mostly noise but
    still carries its `data_weights` share. Without `ceil_steps`, a single
    over-estimated rate lets one node train an unbounded round and stall the
    whole mesh at the barrier it was meant to remove.

    Raises `ValueError` on a non-positive rate: that means the throughput
    measurement failed, which is a different condition from "this node is
    slow" and must not be silently absorbed into the floor.
    """
    if rate_steps_per_s <= 0:
        raise ValueError(
            f"steps_for_budget: rate_steps_per_s must be positive, got "
            f"{rate_steps_per_s} -- a non-positive rate means the throughput "
            "measurement failed, not that the node is slow"
        )
    return max(floor_steps, min(ceil_steps, int(rate_steps_per_s * budget_s)))


def update_rate_ema(
    prev: float | None, observed: float, *, alpha: float = 0.3
) -> float:
    """`alpha * observed + (1 - alpha) * prev`, or `observed` on the first call.

    Throughput here is not stationary -- the fish nodes are shared with other
    students' jobs and NFS contention moves per-round rates around -- so the
    budget tracks a smoothed rate rather than the last round's, which would
    make step counts oscillate with transient contention.
    """
    if prev is None:
        return observed
    return alpha * observed + (1.0 - alpha) * prev
