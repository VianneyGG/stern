"""Property tests for data/harmonise.py."""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from data.harmonise import TARGET_VA_RANGE, harmonise_va

_finite_float = st.floats(
    min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False
)


@st.composite
def _values_and_source_range(draw):
    lo = draw(_finite_float)
    hi = draw(_finite_float.filter(lambda h: h > lo))
    n = draw(st.integers(min_value=1, max_value=30))
    values = draw(
        arrays(
            dtype=np.float64,
            shape=n,
            elements=st.floats(
                min_value=lo, max_value=hi, allow_nan=False, allow_infinity=False
            ),
        )
    )
    return values, (lo, hi)


@given(_values_and_source_range())
def test_output_stays_within_the_target_range(values_and_range):
    values, source_range = values_and_range
    out = harmonise_va(values, source_range)
    lo, hi = TARGET_VA_RANGE
    assert np.all(out >= lo - 1e-9)
    assert np.all(out <= hi + 1e-9)


@given(_values_and_source_range())
def test_reharmonising_an_already_target_range_signal_is_a_no_op(values_and_range):
    values, source_range = values_and_range
    once = harmonise_va(values, source_range)
    twice = harmonise_va(once, TARGET_VA_RANGE)
    np.testing.assert_allclose(once, twice, atol=1e-9)
