"""Unit tests for training/loop_utils.py's WeightEMA."""

import torch
from torch import nn

from stern.training.loop_utils import WeightEMA


def test_ema_initializes_to_the_model_weights():
    model = nn.Linear(4, 4, bias=False)
    ema = WeightEMA(model, decay=0.9)
    assert torch.equal(ema.state_dict()["weight"], model.weight.detach().float())


def test_ema_converges_toward_a_fixed_target_under_repeated_updates():
    torch.manual_seed(0)
    model = nn.Linear(4, 4, bias=False)
    ema = WeightEMA(model, decay=0.9)

    target = torch.randn_like(model.weight)
    with torch.no_grad():
        model.weight.copy_(target)

    start_gap = (ema.state_dict()["weight"] - target).abs().sum()
    for _ in range(200):
        ema.update(model)
    end_gap = (ema.state_dict()["weight"] - target).abs().sum()

    assert end_gap < start_gap
    assert torch.allclose(ema.state_dict()["weight"], target, atol=1e-4)


def test_ema_update_does_not_mutate_the_live_model():
    model = nn.Linear(4, 4, bias=False)
    live_before = model.weight.detach().clone()
    ema = WeightEMA(model, decay=0.5)

    with torch.no_grad():
        model.weight.add_(1.0)
    ema.update(model)

    assert not torch.equal(model.weight.detach(), live_before)
    # ema shadow moved only halfway (decay=0.5) toward the new live weights.
    assert not torch.allclose(ema.state_dict()["weight"], model.weight.detach())
