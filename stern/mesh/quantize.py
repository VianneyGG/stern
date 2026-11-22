"""Error-feedback fp16 quantization for both DiLoCo NFS directions.

Used for the master broadcast (`run_outer_step.py`) and the worker's
pseudo-gradient push (`training/mesh_round.py`), so the two cannot drift apart.

Plain fp16 casting biases every round in the same direction. Master weights
and pseudo-gradients are small differences of large numbers, so the bias
accumulates over a multi-day run. Persisting the residual `(x - fp16(x))` and
adding it back on the next call makes the compression unbiased over time
(standard error feedback from the gradient-compression literature).
"""

from __future__ import annotations

from pathlib import Path

import torch

from stern.mesh.outer_step import StateDict
from stern.training.checkpoint import save as _atomic_save


def quantize_with_error_feedback(
    state_dict: StateDict, residual_path: Path | None
) -> StateDict:
    """Cast every floating-point tensor to fp16, carrying the rounding residual.

    `residual_path` must be on local disk, never NFS. It is only an
    optimization: if it is missing, this falls back to a plain cast. Losing it
    costs precision, not correctness (unlike the outer momentum buffer).
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
            compensated = compensated + prior.float().to(compensated.device)
        quantized = compensated.half()
        quantized_sd[key] = quantized
        new_residual[key] = compensated - quantized.float()

    if residual_path is not None:
        _atomic_save(residual_path, {"residual": new_residual})

    return quantized_sd
