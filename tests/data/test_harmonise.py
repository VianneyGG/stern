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


@pytest.mark.parametrize(
    "values",
    [
        pytest.param([0.0, 1.5], id="above_range"),
        pytest.param([-0.5, 0.5], id="below_range"),
    ],
)
def test_harmonise_va_rejects_values_outside_source_range(values):
    with pytest.raises(ValueError, match="fall outside"):
        harmonise_va(np.array(values), source_range=(0.0, 1.0))


@pytest.mark.parametrize(
    "source_range",
    [
        pytest.param((1.0, 0.0), id="reversed"),
        pytest.param((0.5, 0.5), id="equal_hi_lo"),
    ],
)
def test_harmonise_va_rejects_source_range_without_hi_above_lo(source_range):
    va = np.array([0.0, 0.5])
    with pytest.raises(ValueError, match="hi > lo"):
        harmonise_va(va, source_range=source_range)


def test_harmonise_va_rejects_nan():
    va = np.array([0.0, np.nan])
    with pytest.raises(ValueError, match="finite"):
        harmonise_va(va, source_range=(0.0, 1.0))


@pytest.mark.parametrize(
    "n_source, source_hz, target_hz, expected_len",
    [
        # 100 samples at 50Hz -> 25Hz: 2s of signal -> 50 samples
        pytest.param(100, 50.0, 25.0, 50, id="downsample_2_to_1"),
        # 10 samples at 10Hz -> 20Hz: even upsampling doubles the length
        pytest.param(10, 10.0, 20.0, 20, id="upsample_1_to_2"),
        # 30 samples at 30Hz -> 25Hz: non-integer ratio, 1.2s -> 25 samples
        pytest.param(30, 30.0, 25.0, 25, id="non_integer_ratio"),
    ],
)
def test_harmonise_rate_output_length_matches_target_hz(
    n_source, source_hz, target_hz, expected_len
):
    signal = np.sin(2 * np.pi * np.arange(n_source) / source_hz)
    out = harmonise_rate(signal, source_hz=source_hz, target_hz=target_hz)
    assert out.shape[0] == expected_len


def test_harmonise_rate_upsample_puts_source_samples_on_even_targets():
    # 10 samples at 10Hz -> 20Hz: target sample 2*k lands exactly on source sample k
    t = np.arange(10) / 10.0
    signal = np.sin(2 * np.pi * 1.0 * t)
    out = harmonise_rate(signal, source_hz=10.0, target_hz=20.0)
    np.testing.assert_allclose(out[0::2], signal)


def test_harmonise_rate_non_integer_ratio_reproduces_identity_ramp():
    # signal == source_t, so linear interpolation at target_t reproduces
    # target_t exactly (all target_t fall within source_t's range here)
    source_t = np.arange(30) / 30.0
    out = harmonise_rate(source_t.copy(), source_hz=30.0, target_hz=25.0)
    target_t = np.arange(out.shape[0]) / 25.0
    np.testing.assert_allclose(out, target_t)


@pytest.mark.parametrize(
    "signal, rate",
    [
        pytest.param(np.arange(10, dtype=float), 25.0, id="same_rate_25hz"),
        # source_hz=0.5 is positive but < 1 -- must not be rejected as if it were <= 0
        pytest.param(np.arange(1.0, 5.0), 0.5, id="same_rate_sub_one_hz"),
    ],
)
def test_harmonise_rate_is_identity_when_rates_match(signal, rate):
    out = harmonise_rate(signal, source_hz=rate, target_hz=rate)
    np.testing.assert_allclose(out, signal)


def test_harmonise_rate_rejects_empty_signal():
    signal = np.array([])
    with pytest.raises(ValueError, match="non-empty"):
        harmonise_rate(signal, source_hz=25.0, target_hz=10.0)


def test_harmonise_rate_rejects_zero_target_samples():
    signal = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="zero target samples"):
        harmonise_rate(signal, source_hz=100.0, target_hz=1.0)


@pytest.mark.parametrize(
    "source_hz, target_hz",
    [
        pytest.param(0.0, 10.0, id="source_zero"),
        pytest.param(10.0, 0.0, id="target_zero"),
    ],
)
def test_harmonise_rate_rejects_nonpositive_rate(source_hz, target_hz):
    signal = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="rates must be positive"):
        harmonise_rate(signal, source_hz=source_hz, target_hz=target_hz)
