from __future__ import annotations

import pytest
import torch

from video_encoding.mesh.outer_step import (
    average_deltas,
    data_weights,
    nesterov_outer_step,
    pseudo_gradient,
)


def _sd(v: float) -> dict[str, torch.Tensor]:
    return {"w": torch.full((2, 2), v), "b": torch.full((2,), v)}


def test_data_weights_are_proportional_to_samples_and_sum_to_one():
    alphas = data_weights({"a": 300, "b": 100})

    assert alphas["a"] == pytest.approx(0.75)
    assert alphas["b"] == pytest.approx(0.25)
    assert sum(alphas.values()) == pytest.approx(1.0)


def test_data_weights_rejects_zero_total():
    with pytest.raises(ValueError, match="zero"):
        data_weights({"a": 0})


def test_pseudo_gradient_is_master_minus_worker():
    master = _sd(1.0)
    worker = _sd(0.25)

    delta = pseudo_gradient(master, worker)

    for key in master:
        assert torch.allclose(delta[key], torch.full_like(delta[key], 0.75))


def test_average_deltas_applies_alphas():
    deltas = {"a": _sd(1.0), "b": _sd(0.0)}
    alphas = {"a": 0.25, "b": 0.75}

    avg = average_deltas(deltas, alphas)

    for key in avg:
        assert torch.allclose(avg[key], torch.full_like(avg[key], 0.25))


def test_naive_averaging_is_the_lr1_momentum0_special_case():
    # master is deliberately far from Sum(alpha_i * worker_i) (== 1.0 here):
    # if master coincided with that weighted average, a standalone flipped
    # pseudo_gradient sign (worker - master instead of master - worker) would
    # produce new_master = 2*master - Sum(alpha_i*worker_i), which collapses
    # to the same (wrong-for-the-wrong-reason) answer only in that degenerate
    # case. master=5.0 breaks the coincidence so this test is a genuine
    # standalone trip-wire for the sign, independent of
    # test_pseudo_gradient_is_master_minus_worker.
    master = _sd(5.0)
    workers = {"a": _sd(0.0), "b": _sd(2.0)}
    alphas = {"a": 0.5, "b": 0.5}

    deltas = {name: pseudo_gradient(master, w) for name, w in workers.items()}
    delta = average_deltas(deltas, alphas)

    new_master, _ = nesterov_outer_step(master, delta, None, lr=1.0, momentum=0.0)

    expected = _sd(0.5 * 0.0 + 0.5 * 2.0)
    for key in new_master:
        assert torch.allclose(new_master[key], expected[key])


def test_momentum_accelerates_across_repeated_identical_rounds():
    master = _sd(1.0)
    delta = _sd(0.1)

    master1, buf1 = nesterov_outer_step(master, delta, None, lr=0.7, momentum=0.9)
    step1 = master["w"][0, 0] - master1["w"][0, 0]

    master2, _buf2 = nesterov_outer_step(master1, delta, buf1, lr=0.7, momentum=0.9)
    step2 = master1["w"][0, 0] - master2["w"][0, 0]

    assert step2 > step1


def test_zero_momentum_gives_constant_step_size():
    master = _sd(1.0)
    delta = _sd(0.1)

    master1, buf1 = nesterov_outer_step(master, delta, None, lr=1.0, momentum=0.0)
    step1 = master["w"][0, 0] - master1["w"][0, 0]

    master2, _buf2 = nesterov_outer_step(master1, delta, buf1, lr=1.0, momentum=0.0)
    step2 = master1["w"][0, 0] - master2["w"][0, 0]

    assert step1 == pytest.approx(step2.item())


def test_step_rejects_mismatched_keys():
    master = _sd(1.0)
    delta = _sd(0.1)
    del delta["b"]

    with pytest.raises(KeyError):
        nesterov_outer_step(master, delta, None, lr=1.0, momentum=0.0)
