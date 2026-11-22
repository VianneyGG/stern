from __future__ import annotations

import torch

from video_encoding.mesh.quantize import (
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
