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


def test_accum_steps_reduces_optimizer_step_count():
    model = _TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loader = _loader(n_batches=4)

    with mock.patch.object(optimizer, "step", wraps=optimizer.step) as step_spy:
        _train_epoch(model, loader, optimizer, torch.device("cpu"), accum_steps=2)

    # 4 batches at accum_steps=2 -> optimizer.step() fires on boundaries only (2, 4).
    assert step_spy.call_count == 2


def test_accum_steps_one_steps_every_batch():
    model = _TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loader = _loader(n_batches=4)

    with mock.patch.object(optimizer, "step", wraps=optimizer.step) as step_spy:
        _train_epoch(model, loader, optimizer, torch.device("cpu"), accum_steps=1)

    assert step_spy.call_count == 4


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
    model = _TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loader = _loader(n_batches=3)

    with mock.patch(
        "torch.nn.utils.clip_grad_norm_", wraps=torch.nn.utils.clip_grad_norm_
    ) as clip_spy:
        _train_epoch(model, loader, optimizer, torch.device("cpu"), grad_clip=0.5)

    assert clip_spy.call_count == 3
    for call in clip_spy.call_args_list:
        assert call.args[1] == 0.5 or call.kwargs.get("max_norm") == 0.5
