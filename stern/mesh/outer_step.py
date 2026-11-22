"""DiLoCo outer optimizer core: pseudo-gradient merge of divergent worker weights.

After each worker trains locally for H steps on its assigned shards, its
weights have diverged from the shared master. DiLoCo (arXiv 2311.08105,
OpenDiLoCo arXiv 2407.07852) treats `master - worker` as a pseudo-gradient,
averages those across workers weighted by data count (FedAvg-style,
`n_i / sum(n_j)`), and takes an SGD-with-Nesterov-momentum step on the master
using the average in place of a real gradient.

Momentum persists across rounds: the caller carries `new_momentum_buf` into
the next call. Plain federated averaging is memoryless and discards that. At
`lr=1.0, momentum=0.0` the Nesterov step reduces to `master - delta`, which
equals plain weighted averaging `sum(alpha_i * worker_i)`. That identity is a
useful test: a flipped pseudo-gradient sign breaks it while other tests still
pass.
"""

from __future__ import annotations

from typing import Any

import torch

StateDict = dict[str, torch.Tensor]

# Version that watcher/mesh_tick.py::_bootstrap_master stamps on the master
# before round 0 merges. Negative, so it cannot collide with a real round_num:
# `filter_stale_deltas` must keep a worker's first diff against it, and the
# already-merged check in run_outer_step must not mistake it for a finished merge.
BOOTSTRAP_MASTER_VERSION = -1


def to_fp32_state_dict(state_dict: StateDict) -> StateDict:
    """Upcast every floating-point tensor to fp32; non-float buffers are untouched.

    Needed before any delta or momentum arithmetic, on the merge side (a
    checkpoint or master.pt may be fp16) and on the worker side (the master
    snapshot a pseudo-gradient push diffs against). A no-op for fp32 input.
    """
    return {
        k: (v.float() if torch.is_floating_point(v) else v)
        for k, v in state_dict.items()
    }


def data_weights(n_samples: dict[str, int]) -> dict[str, float]:
    """FedAvg data-proportional weights: `alpha_i = n_i / sum(n_j)`.

    Not inverse-val-loss or any other quality weighting. The average should
    estimate the full-data gradient, and quality weighting would bias it.
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


def filter_stale_deltas(
    checkpoints: dict[str, dict[str, Any]], current_master_version: int | None
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Drop checkpoints whose `master_version` differs from `current_master_version`.

    A pseudo-gradient computed against an outdated master must not be applied
    to the current one. The check is skipped when `current_master_version` is
    None (an unversioned master). A checkpoint with no `master_version` pushed
    full weights rather than a delta, so it is not comparable and is always kept.

    Returns `(survivors, dropped_names)`. The caller logs the drops.
    """
    if current_master_version is None:
        return dict(checkpoints), []
    survivors: dict[str, dict[str, Any]] = {}
    dropped: list[str] = []
    for name, ckpt in checkpoints.items():
        version = ckpt.get("master_version")
        if version is None or version == current_master_version:
            survivors[name] = ckpt
        else:
            dropped.append(name)
    return survivors, dropped


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
