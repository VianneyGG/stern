import numpy as np
import pytest

from eval.ccc import concordance_correlation_coefficient


def test_ccc_is_one_for_identical_signals():
    x = np.array([0.1, 0.5, -0.3, 0.8, -0.9])
    ccc = concordance_correlation_coefficient(x, x)
    np.testing.assert_allclose(ccc, 1.0, atol=1e-8)


def test_ccc_is_negative_one_for_perfectly_inverted_signal():
    # Mean-zero by construction (1 + 2 - 1 - 2 == 0): CCC(x, -x) == -1 exactly
    # only when mean(x) == mean(-x) == 0, since the formula's bias term
    # (mean_x - mean_y)^2 must vanish for the bound to be tight.
    x = np.array([1.0, 2.0, -1.0, -2.0])
    assert x.mean() == 0.0
    ccc = concordance_correlation_coefficient(x, -x)
    np.testing.assert_allclose(ccc, -1.0, atol=1e-8)


def test_ccc_matches_hand_computed_value():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([2.0, 3.0, 4.0, 5.0])
    mean_x, mean_y = x.mean(), y.mean()
    var_x, var_y = x.var(), y.var()
    covar = np.mean((x - mean_x) * (y - mean_y))
    expected = (2 * covar) / (var_x + var_y + (mean_x - mean_y) ** 2)
    ccc = concordance_correlation_coefficient(x, y)
    np.testing.assert_allclose(ccc, expected, atol=1e-8)


def test_ccc_penalizes_uncorrelated_signal():
    rng = np.random.default_rng(0)
    x = rng.normal(size=1000)
    y = rng.normal(size=1000)
    ccc = concordance_correlation_coefficient(x, y)
    assert abs(ccc) < 0.15  # near zero for independent signals


def test_ccc_rejects_non_1d_input():
    x = np.array([[0.1, 0.5], [-0.3, 0.8]])
    y = np.array([[0.1, 0.5], [-0.3, 0.8]])
    with pytest.raises(ValueError):
        concordance_correlation_coefficient(x, y)


def test_ccc_rejects_empty_input():
    x = np.array([])
    y = np.array([])
    with pytest.raises(ValueError):
        concordance_correlation_coefficient(x, y)


def test_ccc_rejects_nan():
    x = np.array([0.1, 0.5, np.nan])
    y = np.array([0.1, 0.5, -0.3])
    with pytest.raises(ValueError):
        concordance_correlation_coefficient(x, y)


def test_ccc_rejects_inf():
    x = np.array([0.1, 0.5, np.inf])
    y = np.array([0.1, 0.5, -0.3])
    with pytest.raises(ValueError):
        concordance_correlation_coefficient(x, y)


def test_ccc_handles_identical_constant_signals():
    x = np.full(5, 3.0)
    y = np.full(5, 3.0)
    ccc = concordance_correlation_coefficient(x, y)
    assert ccc == 1.0


def test_ccc_handles_different_constant_signals():
    x = np.full(5, 3.0)
    y = np.full(5, 7.0)
    ccc = concordance_correlation_coefficient(x, y)
    assert ccc == 0.0
