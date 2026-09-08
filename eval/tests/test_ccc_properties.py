"""Property tests for eval/ccc.concordance_correlation_coefficient."""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from eval.ccc import concordance_correlation_coefficient as ccc

_finite_float = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
)


def _signal(n):
    return arrays(dtype=np.float64, shape=n, elements=_finite_float)


@st.composite
def _matched_signal_pair(draw):
    n = draw(st.integers(min_value=2, max_value=50))
    return draw(_signal(n)), draw(_signal(n))


@given(st.integers(min_value=2, max_value=50).flatmap(_signal))
def test_ccc_of_a_signal_with_itself_is_one(x):
    assert abs(ccc(x, x) - 1.0) < 1e-6


@given(_matched_signal_pair())
def test_ccc_is_bounded_between_minus_one_and_one(pair):
    x, y = pair
    value = ccc(x, y)
    assert -1.0 - 1e-9 <= value <= 1.0 + 1e-9


@given(_matched_signal_pair())
def test_ccc_is_symmetric(pair):
    x, y = pair
    assert abs(ccc(x, y) - ccc(y, x)) < 1e-9
