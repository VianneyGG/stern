from __future__ import annotations

import pytest
import torch

from stern.mesh.outer_step import (
    average_deltas,
    data_weights,
    filter_stale_deltas,
    nesterov_outer_step,
    pseudo_gradient,
    to_fp32_state_dict,
)


def _sd(v: float) -> dict[str, torch.Tensor]:
    return {"w": torch.full((2, 2), v), "b": torch.full((2,), v)}


def test_filter_stale_deltas_keeps_everything_when_versions_match():
    checkpoints = {
        "a": {"master_version": 5, "n_samples": 1},
        "b": {"master_version": 5, "n_samples": 1},
    }
    survivors, dropped = filter_stale_deltas(checkpoints, current_master_version=5)
    assert survivors == checkpoints
    assert dropped == []


def test_filter_stale_deltas_drops_exactly_the_mismatched_checkpoint():
    checkpoints = {
        "a": {"master_version": 5, "n_samples": 1},
        "b": {"master_version": 4, "n_samples": 1},
    }
    survivors, dropped = filter_stale_deltas(checkpoints, current_master_version=5)
    assert set(survivors) == {"a"}
    assert dropped == ["b"]


def test_filter_stale_deltas_skips_the_check_when_current_version_is_none():
    """An unversioned master predates Phase 3.2 -- treat every checkpoint as
    compatible rather than rejecting a mesh mid-upgrade."""
    checkpoints = {
        "a": {"master_version": 5, "n_samples": 1},
        "b": {"n_samples": 1},  # full-weight push, no master_version key at all
    }
    survivors, dropped = filter_stale_deltas(checkpoints, current_master_version=None)
    assert survivors == checkpoints
    assert dropped == []


def test_filter_stale_deltas_keeps_a_checkpoint_with_no_master_version_key():
    """A full-weight (round-0) push never had a master to diff against, so it
    has no master_version at all -- that is not the same as "stale"."""
    checkpoints = {"a": {"n_samples": 1}}
    survivors, dropped = filter_stale_deltas(checkpoints, current_master_version=5)
    assert survivors == checkpoints
    assert dropped == []


def test_to_fp32_state_dict_upcasts_float_and_passes_non_float_through():
    sd = {
        "w": torch.full((2, 2), 0.5).half(),
        "step": torch.tensor([7], dtype=torch.int64),
    }
    out = to_fp32_state_dict(sd)
    assert out["w"].dtype == torch.float32
    assert out["step"].dtype == torch.int64
    assert out["step"].item() == 7


def test_data_weights_are_proportional_to_samples_and_sum_to_one():
    alphas = data_weights({"a": 300, "b": 100})

    assert alphas["a"] == pytest.approx(0.75)
    assert alphas["b"] == pytest.approx(0.25)
    assert sum(alphas.values()) == pytest.approx(1.0)


def test_data_weights_rejects_zero_total():
    with pytest.raises(ValueError, match="zero"):
        data_weights({"a": 0})


def test_data_weights_accepts_a_total_of_exactly_one_sample():
    alphas = data_weights({"a": 1})
    assert alphas["a"] == pytest.approx(1.0)


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


def test_average_deltas_accumulates_in_fp32_even_for_fp16_input():
    """The module contract is 'accumulated in fp32' -- fp16 deltas summed
    in fp16 lose precision that fp32 accumulation would not."""
    fp16_deltas = {
        name: {k: t.to(torch.float16) for k, t in sd.items()}
        for name, sd in {"a": _sd(1.0), "b": _sd(0.0)}.items()
    }
    alphas = {"a": 0.25, "b": 0.75}

    avg = average_deltas(fp16_deltas, alphas)

    for tensor in avg.values():
        assert tensor.dtype == torch.float32


def test_nesterov_outer_step_with_existing_momentum_buffer():
    """Exercises the momentum_buf-provided branch (buf = momentum*buf + grad),
    which every other test here skips by using momentum=0 or a fresh buffer.
    Hand-computed: buf = 0.9*0.5 + 1.0 = 1.45; step = 1.0 + 0.9*1.45 = 2.305;
    new_p = 2.0 - 0.1*2.305 = 1.7695."""
    master = {"w": torch.full((1,), 2.0)}
    delta = {"w": torch.full((1,), 1.0)}
    momentum_buf = {"w": torch.full((1,), 0.5)}

    new_master, new_buf = nesterov_outer_step(
        master, delta, momentum_buf, lr=0.1, momentum=0.9
    )

    assert torch.allclose(new_master["w"], torch.full((1,), 1.7695))
    assert torch.allclose(new_buf["w"], torch.full((1,), 1.45))


def test_nesterov_outer_step_casts_result_back_to_the_master_dtype():
    master = {"w": torch.full((1,), 2.0, dtype=torch.float16)}
    delta = {"w": torch.full((1,), 1.0)}

    new_master, _ = nesterov_outer_step(master, delta, None, lr=0.1, momentum=0.0)

    assert new_master["w"].dtype == torch.float16


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

    with pytest.raises(KeyError, match="b"):
        nesterov_outer_step(master, delta, None, lr=1.0, momentum=0.0)
