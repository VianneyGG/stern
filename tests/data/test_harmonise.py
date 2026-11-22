import numpy as np
import pytest

from stern.data.harmonise import harmonise_va, harmonise_rate


def test_harmonise_va_passthrough_for_pm1_range():
    va = np.array([-1.0, 0.0, 0.5, 1.0])
    out = harmonise_va(va, source_range=(-1.0, 1.0))
    np.testing.assert_allclose(out, va)


def test_harmonise_va_rescales_zero_one_to_pm1():
    va = np.array([0.0, 0.25, 0.5, 1.0])
    out = harmonise_va(va, source_range=(0.0, 1.0))
    np.testing.assert_allclose(out, np.array([-1.0, -0.5, 0.0, 1.0]))


def test_harmonise_va_rejects_out_of_range_input():
    va = np.array([0.0, 1.5])
    with pytest.raises(ValueError, match="fall outside"):
        harmonise_va(va, source_range=(0.0, 1.0))


def test_harmonise_va_rejects_input_below_the_source_range():
    va = np.array([-0.5, 0.5])
    with pytest.raises(ValueError, match="fall outside"):
        harmonise_va(va, source_range=(0.0, 1.0))


def test_harmonise_rate_downsamples_to_target_hz():
    # 100 samples at 50Hz -> resample to 25Hz -> 50 samples
    t = np.linspace(0, 1.98, 100)  # 50Hz over ~2s
    signal = np.sin(2 * np.pi * 1.0 * t)
    out = harmonise_rate(signal, source_hz=50.0, target_hz=25.0)
    assert out.shape[0] == 50


def test_harmonise_rate_noop_when_rates_match():
    signal = np.arange(10, dtype=float)
    out = harmonise_rate(signal, source_hz=25.0, target_hz=25.0)
    np.testing.assert_allclose(out, signal)


def test_harmonise_rate_upsamples_to_target_hz():
    # 10 samples at 10Hz -> resample to 20Hz -> 20 samples
    t = np.arange(10) / 10.0
    signal = np.sin(2 * np.pi * 1.0 * t)
    out = harmonise_rate(signal, source_hz=10.0, target_hz=20.0)
    assert out.shape[0] == 20
    # even-ratio upsampling: target sample 2*k lands exactly on source sample k
    np.testing.assert_allclose(out[0::2], signal)


def test_harmonise_rate_handles_non_integer_ratio():
    # 30 samples at 30Hz -> resample to 25Hz -> 25 samples
    n_source = 30
    source_t = np.arange(n_source) / 30.0
    signal = source_t.copy()  # identity ramp: value == time
    out = harmonise_rate(signal, source_hz=30.0, target_hz=25.0)
    assert out.shape[0] == 25
    # signal == source_t, so linear interpolation at target_t reproduces
    # target_t exactly (all target_t fall within source_t's range here)
    target_t = np.arange(out.shape[0]) / 25.0
    np.testing.assert_allclose(out, target_t)


def test_harmonise_rate_rejects_empty_signal():
    signal = np.array([])
    with pytest.raises(ValueError, match="non-empty"):
        harmonise_rate(signal, source_hz=25.0, target_hz=10.0)


def test_harmonise_va_rejects_invalid_source_range():
    va = np.array([0.0, 0.5])
    with pytest.raises(ValueError, match="hi > lo"):
        harmonise_va(va, source_range=(1.0, 0.0))


def test_harmonise_va_rejects_equal_hi_and_lo():
    va = np.array([0.5])
    with pytest.raises(ValueError, match="hi > lo"):
        harmonise_va(va, source_range=(0.5, 0.5))


def test_harmonise_va_rejects_nan():
    va = np.array([0.0, np.nan])
    with pytest.raises(ValueError, match="finite"):
        harmonise_va(va, source_range=(0.0, 1.0))


def test_harmonise_rate_rejects_zero_target_samples():
    signal = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="zero target samples"):
        harmonise_rate(signal, source_hz=100.0, target_hz=1.0)


def test_harmonise_rate_rejects_nonpositive_source_hz():
    signal = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="rates must be positive"):
        harmonise_rate(signal, source_hz=0.0, target_hz=10.0)


def test_harmonise_rate_accepts_a_sub_one_hz_rate():
    # source_hz=0.5 is positive but < 1 -- must not be rejected as if it
    # were <= 0.
    signal = np.array([1.0, 2.0, 3.0, 4.0])
    out = harmonise_rate(signal, source_hz=0.5, target_hz=0.5)
    np.testing.assert_allclose(out, signal)


def test_harmonise_rate_rejects_nonpositive_target_hz():
    signal = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="rates must be positive"):
        harmonise_rate(signal, source_hz=10.0, target_hz=0.0)
