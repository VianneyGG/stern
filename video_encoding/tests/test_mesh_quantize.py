from __future__ import annotations

import torch

from video_encoding.mesh.quantize import (
    dequantize_e3m0_state_dict,
    quantize_e3m0_with_error_feedback,
    quantize_with_error_feedback,
)


def test_quantize_with_error_feedback_halves_precision_and_passes_non_float_through():
    state_dict = {
        "w": torch.full((2, 2), 0.5),
        "step": torch.tensor([7], dtype=torch.int64),
    }
    quantized = quantize_with_error_feedback(state_dict, residual_path=None)
    assert quantized["w"].dtype == torch.float16
    assert quantized["step"].dtype == torch.int64
    assert quantized["step"].item() == 7


def test_quantize_with_error_feedback_compensates_across_calls(tmp_path):
    """Repeated calls on the same state dict should average out the fp16
    rounding bias via the persisted residual, same guarantee the master
    broadcast direction already relies on -- this is the new (Phase 3.2)
    caller, the worker's pseudo-gradient push, exercising the same module."""
    residual_path = tmp_path / "residual.pt"
    true_value = 1.0 + 0.3 * (2**-10)
    state_dict = {"w": torch.full((4, 4), true_value, dtype=torch.float32)}

    plain_err = (state_dict["w"].half().float() - state_dict["w"]).abs().mean().item()

    total = torch.zeros_like(state_dict["w"])
    n_calls = 50
    for _ in range(n_calls):
        q = quantize_with_error_feedback(state_dict, residual_path)
        total += q["w"].float()
    feedback_err = (total / n_calls - state_dict["w"]).abs().mean().item()

    assert feedback_err < plain_err / 5


def test_e3m0_round_trips_exact_powers_of_two():
    """E3M0 (1 sign, 3 exponent, 0 mantissa) is exact only at its own grid
    points -- powers of two -- so those must survive quantize+dequantize
    without any rounding error, unlike arbitrary values."""
    state_dict = {"w": torch.tensor([0.125, 0.5, 1.0, 2.0, 8.0, -0.25, -4.0])}
    codes = quantize_e3m0_with_error_feedback(state_dict, residual_path=None)
    dequantized = dequantize_e3m0_state_dict(codes)
    assert torch.allclose(dequantized["w"], state_dict["w"])


def test_e3m0_codes_are_packed_one_per_uint8_byte():
    """Half of fp16's 2 bytes/element -- real, if not the full 4x a 2-per-byte
    packed layout would give (see quantize.py's module docstring)."""
    state_dict = {"w": torch.full((4, 4), 0.5)}
    codes = quantize_e3m0_with_error_feedback(state_dict, residual_path=None)
    assert codes["w"].dtype == torch.uint8
    assert codes["w"].element_size() == 1


def test_e3m0_passes_non_float_tensors_through_unchanged():
    state_dict = {
        "w": torch.full((2, 2), 0.5),
        "step": torch.tensor([7], dtype=torch.int64),
    }
    codes = quantize_e3m0_with_error_feedback(state_dict, residual_path=None)
    assert codes["step"].dtype == torch.int64
    assert codes["step"].item() == 7
    dequantized = dequantize_e3m0_state_dict(codes)
    assert dequantized["step"].dtype == torch.int64
    assert dequantized["step"].item() == 7


def test_e3m0_preserves_sign():
    state_dict = {"w": torch.tensor([1.0, -1.0])}
    codes = quantize_e3m0_with_error_feedback(state_dict, residual_path=None)
    dequantized = dequantize_e3m0_state_dict(codes)
    assert dequantized["w"][0].item() > 0
    assert dequantized["w"][1].item() < 0


def test_e3m0_exact_zero_round_trips_to_zero():
    state_dict = {"w": torch.tensor([0.0])}
    codes = quantize_e3m0_with_error_feedback(state_dict, residual_path=None)
    dequantized = dequantize_e3m0_state_dict(codes)
    assert dequantized["w"].item() == 0.0


def test_e3m0_clamps_rather_than_overflows_outside_representable_range():
    """A value far outside E3M0's representable magnitudes (its dynamic
    range is a handful of powers of two) must clamp to the nearest
    representable level, not wrap or produce a nonsense code."""
    state_dict = {"w": torch.tensor([1e6, -1e6])}
    codes = quantize_e3m0_with_error_feedback(state_dict, residual_path=None)
    dequantized = dequantize_e3m0_state_dict(codes)
    assert torch.isfinite(dequantized["w"]).all()
    assert dequantized["w"][0].item() > 0
    assert dequantized["w"][1].item() < 0


def test_e3m0_error_feedback_compensates_across_calls(tmp_path):
    """Same bias-correction guarantee as the fp16 error-feedback path,
    exercised against E3M0's much coarser (~4-bit) quantization grid."""
    residual_path = tmp_path / "residual.pt"
    true_value = 0.3
    state_dict = {"w": torch.full((8, 8), true_value, dtype=torch.float32)}

    single = dequantize_e3m0_state_dict(
        quantize_e3m0_with_error_feedback(state_dict, residual_path=None)
    )
    plain_err = (single["w"] - true_value).abs().mean().item()

    total = torch.zeros_like(state_dict["w"])
    n_calls = 50
    for _ in range(n_calls):
        codes = quantize_e3m0_with_error_feedback(state_dict, residual_path)
        total += dequantize_e3m0_state_dict(codes)["w"]
    feedback_err = (total / n_calls - true_value).abs().mean().item()

    assert feedback_err < plain_err / 5
