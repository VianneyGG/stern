"""Error-feedback fp16 quantization, shared by both DiLoCo NFS directions:
the master broadcast (`run_outer_step.py`) and, from Phase 3.2, the
worker's pseudo-gradient push (`training/diloco/mesh_round.py`) -- one
primitive instead of two copies that would drift apart.

Plain fp16 casting biases every round in the same direction, and both the
master weights and the pseudo-gradient DiLoCo computes from them are small
differences of large numbers -- repeated correlated rounding accumulates
over a multi-day run. Persisting the residual `(x - fp16(x))` and adding it
back on the next call makes the compression unbiased over time (standard
error feedback from the gradient-compression literature).
"""

from __future__ import annotations

from pathlib import Path

import torch

from video_encoding.mesh.outer_step import StateDict
from video_encoding.training.checkpoint import save as _atomic_save


def quantize_with_error_feedback(
    state_dict: StateDict, residual_path: Path | None
) -> StateDict:
    """Cast every floating-point tensor in `state_dict` to fp16, carrying
    the rounding residual across calls via `residual_path`.

    `residual_path` lives on local disk, never NFS: it is a pure
    optimization. If missing (first call, or a wiped local dir) this
    degrades to a plain cast rather than failing -- unlike the outer
    momentum buffer, losing it costs precision, not correctness.
    """
    residual: StateDict = {}
    if residual_path is not None and residual_path.exists():
        residual = torch.load(residual_path, map_location="cpu", weights_only=False)[
            "residual"
        ]

    quantized_sd: StateDict = {}
    new_residual: StateDict = {}
    for key, value in state_dict.items():
        if not torch.is_floating_point(value):
            quantized_sd[key] = value
            continue
        compensated = value.float()
        prior = residual.get(key)
        # A shape change (model surgery mid-run) invalidates the residual
        # for that tensor -- drop it rather than crashing the caller.
        if prior is not None and prior.shape == compensated.shape:
            compensated = compensated + prior.float()
        quantized = compensated.half()
        quantized_sd[key] = quantized
        new_residual[key] = compensated - quantized.float()

    if residual_path is not None:
        _atomic_save(residual_path, {"residual": new_residual})

    return quantized_sd
