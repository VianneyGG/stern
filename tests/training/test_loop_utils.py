"""Unit tests for training/loop_utils.py's WeightEMA."""

import pytest
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


def test_ema_update_is_one_step_formula_and_leaves_live_weights_alone():
    # Hand-picked: shadow starts at 2.0, live weight is set to 6.0, decay 0.9.
    # ema = 0.9 * 2.0 + 0.1 * 6.0 = 1.8 + 0.6 = 2.4 (swapping d and 1-d gives 5.6).
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.0)
    ema = WeightEMA(model, decay=0.9)
    with torch.no_grad():
        model.weight.fill_(6.0)

    ema.update(model)

    assert ema.state_dict()["weight"].item() == pytest.approx(2.4)
    assert model.weight.item() == 6.0
