"""Unit tests for `_train_epoch`'s accumulation/clipping mechanics in isolation
from the full training stage (a tiny linear model + synthetic batches, no
VideoMAE backbone) -- these are the behaviors that were silently no-ops before
this session's fixes: `accum_steps` didn't accumulate, and `grad_clip` didn't
clip.
"""

from __future__ import annotations

from unittest import mock

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from stern.training.finetune import _train_epoch

pytestmark = pytest.mark.integration


def _loader(n_batches: int, batch_size: int = 4) -> DataLoader:
    x = torch.randn(n_batches * batch_size, 3, 2, 4, 4)
    y = torch.tanh(torch.randn(n_batches * batch_size, 2))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3, 2)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.lin(pixel_values.mean(dim=[2, 3, 4])))


@pytest.mark.parametrize(
    ("accum_steps", "expected_steps"),
    [
        # 4 batches: boundaries fall at batches 2 and 4.
        pytest.param(2, 2, id="accum_2_steps_on_boundaries"),
        # Every batch is a boundary.
        pytest.param(1, 4, id="accum_1_steps_every_batch"),
    ],
)
def test_accum_steps_sets_optimizer_step_count(accum_steps, expected_steps):
    model = _TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loader = _loader(n_batches=4)

    with mock.patch.object(optimizer, "step", wraps=optimizer.step) as step_spy:
        _train_epoch(
            model, loader, optimizer, torch.device("cpu"), accum_steps=accum_steps
        )

    assert step_spy.call_count == expected_steps


def _grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.norm(2).item() ** 2
    return total**0.5


def _pre_step_grad_norm(grad_clip: float) -> float:
    """Runs one real (unmocked) epoch, capturing the gradient norm right
    before optimizer.step() -- if grad_clip is applied, that's the point at
    which it has already shrunk the gradient in place."""
    torch.manual_seed(0)
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    loader = _loader(n_batches=1, batch_size=4)
    norms: list[float] = []
    real_step = optimizer.step

    def _capturing_step(*args, **kwargs):
        norms.append(_grad_norm(model))
        return real_step(*args, **kwargs)

    optimizer.step = _capturing_step
    _train_epoch(model, loader, optimizer, torch.device("cpu"), grad_clip=grad_clip)
    return norms[0]


def test_grad_clip_zero_disables_clipping():
    """grad_clip=0.0 must leave the gradient at its raw, unclipped magnitude
    -- verified by comparing against the same seeded run with a tiny
    grad_clip, which would visibly shrink the pre-step norm if clipping were
    still applied when disabled."""
    unclipped_norm = _pre_step_grad_norm(grad_clip=0.0)
    clipped_norm = _pre_step_grad_norm(grad_clip=1e-3)

    assert clipped_norm == pytest.approx(1e-3, rel=0.1)
    assert unclipped_norm > clipped_norm


def test_grad_clip_positive_clips_each_optimizer_step():
    """With grad_clip=1e-3 every step's gradient ends at norm exactly 1e-3.

    clip_grad_norm_ returns the pre-clip norm, so the test first checks that
    the raw norm exceeds 1e-3 (clipping really engages). It then reads the
    gradient norm at optimizer.step(), the gradient the optimizer consumes,
    and expects 1e-3. A clip that is a no-op, or any later scaling, fails it.
    """
    model = _TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loader = _loader(n_batches=3)
    real_clip = torch.nn.utils.clip_grad_norm_
    real_step = optimizer.step
    pre_norms: list[float] = []
    stepped_norms: list[float] = []

    def _recording_clip(parameters, max_norm, *args, **kwargs):
        pre_norms.append(float(real_clip(parameters, max_norm, *args, **kwargs)))
        return pre_norms[-1]

    def _recording_step(*args, **kwargs):
        stepped_norms.append(_grad_norm(model))
        return real_step(*args, **kwargs)

    optimizer.step = _recording_step
    with mock.patch(
        "torch.nn.utils.clip_grad_norm_", side_effect=_recording_clip
    ) as clip_spy:
        _train_epoch(model, loader, optimizer, torch.device("cpu"), grad_clip=1e-3)

    assert clip_spy.call_count == 3
    for call in clip_spy.call_args_list:
        assert call.args[1] == 1e-3 or call.kwargs.get("max_norm") == 1e-3
    assert all(n > 1e-3 for n in pre_norms)
    assert stepped_norms == pytest.approx([1e-3] * 3, rel=1e-4)
