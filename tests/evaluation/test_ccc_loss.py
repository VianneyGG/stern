import numpy as np
import pytest
import torch

from stern.evaluation.ccc import ccc_loss, concordance_correlation_coefficient

# x = [1, 2, 3, 4]: mean 2.5, population var 1.25 (deviations +-1.5, +-0.5).
X = [1.0, 2.0, 3.0, 4.0]


def test_numpy_ccc_matches_hand_value():
    def ccc(y):
        return concordance_correlation_coefficient(np.array(X), np.array(y))

    assert ccc([1.0, 2.0, 3.0, 4.0]) == pytest.approx(1.0, abs=1e-12)
    # y = x + 1: mean diff 1, covar 1.25, denom 1.25 + 1.25 + 1 = 3.5 -> 2.5 / 3.5
    assert ccc([2.0, 3.0, 4.0, 5.0]) == pytest.approx(5 / 7, abs=1e-12)
    # y = 2x: var_y 5, covar 2.5, mean diff 2.5, denom 1.25 + 5 + 6.25 = 12.5 -> 5 / 12.5
    assert ccc([2.0, 4.0, 6.0, 8.0]) == pytest.approx(0.4, abs=1e-12)


def test_ccc_loss_equals_negative_ccc_on_hand_cases():
    # Same input pair, both label columns identical: loss = -CCC (hand values above).
    def loss(y):
        pred = torch.tensor([X, X], dtype=torch.float64).T
        target = torch.tensor([y, y], dtype=torch.float64).T
        return ccc_loss(pred, target).item()

    assert loss([1.0, 2.0, 3.0, 4.0]) == pytest.approx(-1.0, abs=1e-12)
    assert loss([2.0, 3.0, 4.0, 5.0]) == pytest.approx(-5 / 7, abs=1e-12)
    assert loss([2.0, 4.0, 6.0, 8.0]) == pytest.approx(-0.4, abs=1e-12)


def test_ccc_loss_averages_valence_and_arousal_independently():
    # Valence: target x+1 -> CCC 5/7. Arousal: target 2x -> CCC 0.4.
    # loss = -(5/7 + 2/5) / 2 = -39/70.
    pred = torch.tensor([X, X], dtype=torch.float64).T
    target = torch.tensor([[2.0, 3.0, 4.0, 5.0], [2.0, 4.0, 6.0, 8.0]]).T
    assert ccc_loss(pred, target).item() == pytest.approx(-39 / 70, abs=1e-12)


def test_constant_prediction_gets_zero_ccc_not_full_credit():
    # Valence prediction constant at 3.0 vs target x: covar 0, denom
    # 0 + 1.25 + (3 - 2.5)^2 = 1.5, so CCC_v = 0 and contributes nothing.
    # Arousal matches exactly (CCC 1). loss = -(0 + 1) / 2 = -0.5, not -1.
    pred = torch.tensor([[3.0, 1.0], [3.0, 2.0], [3.0, 3.0], [3.0, 4.0]], dtype=torch.float64)
    target = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]], dtype=torch.float64)
    assert ccc_loss(pred, target).item() == pytest.approx(-0.5, abs=1e-12)
    assert concordance_correlation_coefficient(np.full(4, 3.0), np.array(X)) == 0.0


def test_ccc_loss_pairs_each_prediction_column_with_its_own_target():
    # Valence: pred x vs target x -> CCC 1. Arousal: pred constant 3 vs target x+1 -> CCC 0.
    # loss = -(1 + 0) / 2 = -0.5. Swapping the target columns gives
    # CCC(x, x+1) = 5/7 and CCC(3, x) = 0 -> -5/14, so a swap fails here.
    pred = torch.tensor([[1.0, 3.0], [2.0, 3.0], [3.0, 3.0], [4.0, 3.0]], dtype=torch.float64)
    target = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]], dtype=torch.float64)
    assert ccc_loss(pred, target).item() == pytest.approx(-0.5, abs=1e-12)


def test_ccc_loss_is_differentiable_in_prediction():
    pred = torch.tensor([X, X], dtype=torch.float64).T.clone().requires_grad_(True)
    target = torch.tensor([[2.0, 3.0, 4.0, 5.0], [2.0, 4.0, 6.0, 8.0]]).T
    ccc_loss(pred, target).backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum().item() > 0.0
