"""DiLoCo outer optimizer core: pseudo-gradient merge of divergent worker weights.

After each node in the decentralized P2P mesh trains locally for H steps on
its assigned shards (Tasks 1-4 decide *which* shards; this module is
orthogonal to that), the resulting worker weights have diverged from the
shared master. DiLoCo (arXiv 2311.08105, OpenDiLoCo arXiv 2407.07852) treats
`master - worker` as a "pseudo-gradient" -- the direction the master would
have moved had it seen that worker's data -- averages those pseudo-gradients
across workers weighted by how much data each one trained on (FedAvg-style,
`n_i / sum(n_j)`), and then takes an SGD-with-Nesterov-momentum step on the
master using that averaged pseudo-gradient in place of a real gradient.

Momentum persists *across* rounds (the caller is expected to carry
`new_momentum_buf` into the next round's call): that's what distinguishes
DiLoCo from plain federated averaging, which is memoryless and discards
whatever the momentum buffer learned about the descent direction in prior
rounds. At `lr=1.0, momentum=0.0` the Nesterov step degenerates to
`master - delta`, which -- because `delta` is itself an alpha-weighted
average of `master - worker_i` -- is algebraically identical to plain
weighted averaging `sum(alpha_i * worker_i)`. That equivalence makes naive
averaging a free ablation baseline (Tasks 13/14) and is the trip-wire that
would catch a flipped pseudo-gradient sign: get the sign backwards and this
specific equivalence breaks while every other test here still passes.
"""

from __future__ import annotations

import torch

StateDict = dict[str, torch.Tensor]


def data_weights(n_samples: dict[str, int]) -> dict[str, float]:
    """FedAvg data-proportional weights: `alpha_i = n_i / sum(n_j)`.

    Deliberately *not* inverse-val-loss or any quality-based weighting: the
    averaged pseudo-gradient is meant to estimate the full-data gradient, and
    weighting by model quality would bias that estimator away from a true
    data-proportional average.
    """
    total = sum(n_samples.values())
    if total <= 0:
        raise ValueError(
            f"data_weights: total sample count is zero (n_samples={n_samples}); "
            "cannot compute proportional weights"
        )
    return {name: n / total for name, n in n_samples.items()}


def pseudo_gradient(master: StateDict, worker: StateDict) -> StateDict:
    """`theta_master - theta_worker`, in that order.

    Sign matters: the outer step later computes `master - lr * delta`, which
    must move the master *toward* the worker. Reversing this subtraction
    would train the whole mesh uphill while every downstream test still runs.
    """
    return {key: master[key] - worker[key] for key in master}


def average_deltas(deltas: dict[str, StateDict], alphas: dict[str, float]) -> StateDict:
    """`sum_i alpha_i * delta_i`, accumulated in fp32."""
    names = list(deltas)
    first = deltas[names[0]]
    acc = {
        key: torch.zeros_like(tensor, dtype=torch.float32)
        for key, tensor in first.items()
    }
    for name in names:
        alpha = alphas[name]
        for key, tensor in deltas[name].items():
            acc[key] += alpha * tensor.to(torch.float32)
    return acc


def nesterov_outer_step(
    master: StateDict,
    delta: StateDict,
    momentum_buf: StateDict | None,
    *,
    lr: float,
    momentum: float,
) -> tuple[StateDict, StateDict]:
    """SGD-with-Nesterov-momentum step on `master` using `delta` as the gradient.

    Follows `torch.optim.SGD`'s Nesterov convention:
        buf  = momentum * buf + grad      # buf = grad.clone() on the first step
        step = grad + momentum * buf
        p    = p - lr * step

    Accumulates in fp32 regardless of the master's on-disk dtype (checkpoints
    are bf16) and casts the result back to the master's original dtype on
    return. Raises `KeyError` if `delta` is missing a key present in `master`.
    """
    new_master: StateDict = {}
    new_buf: StateDict = {}

    for key, p in master.items():
        if key not in delta:
            raise KeyError(
                f"nesterov_outer_step: delta is missing key '{key}' present in master"
            )
        grad = delta[key].to(torch.float32)
        p32 = p.to(torch.float32)

        if momentum_buf is None or key not in momentum_buf:
            buf = grad.clone()
        else:
            buf = momentum * momentum_buf[key].to(torch.float32) + grad

        step = grad + momentum * buf
        new_p32 = p32 - lr * step

        new_master[key] = new_p32.to(p.dtype)
        new_buf[key] = buf

    return new_master, new_buf
