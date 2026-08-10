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
            compensated = compensated + prior.float().to(compensated.device)
        quantized = compensated.half()
        quantized_sd[key] = quantized
        new_residual[key] = compensated - quantized.float()

    if residual_path is not None:
        _atomic_save(residual_path, {"residual": new_residual})

    return quantized_sd


# E3M0 (1 sign bit, 3 exponent bits, 0 mantissa bits, arXiv 2501.18512
# section 3): representable nonzero magnitudes are exactly 2**(code - BIAS)
# for code in 1..7 (code 0 reserved for exact zero), i.e. BIAS=4 gives a
# representable range of 0.125 .. 8.0 -- NOT 0.0625, which would be code 0
# and collides with the reserved zero code. Log-scale (not fixed-point) is
# what makes 4 bits usable at all here -- weight/gradient tensors are
# heavy-tailed around zero with a long tail of larger-magnitude outliers,
# and a floating-point grid keeps *relative* error bounded across that whole
# dynamic range, where a linear 4-bit grid would blow up absolute error for
# the (common) small values or clip the (rare) large ones. Worst-case
# relative error is ~41% (rounding UP from the geometric midpoint between
# adjacent powers of two, e.g. true value sqrt(2)*2**k quantized to 2**(k+1):
# (2 - sqrt(2))/sqrt(2) ~= 0.414 -- the down-rounding case is only ~29%, so
# 41% is the correct bound, not the smaller of the two). BIAS=4 centers the
# representable range on weight scale, matching the existing fp16 broadcast
# path's "[-1, 1]" framing.
#
# NOT WIRED into run_outer_step.py's BROADCAST_DTYPE_CHOICES or the worker
# loader (training/diloco/mesh_round.py) yet -- both would need to branch on
# an explicit dtype tag to call `dequantize_e3m0_state_dict` instead of
# `outer_step.to_fp32_state_dict`'s blind upcast (a uint8 code tensor is not
# a real float dtype, unlike fp16, so it can't just be `.float()`-cast).
# fp16 broadcast itself still carries an unresolved "measure ||delta||/||w||
# per tensor on a real round before flipping it" bar (run_outer_step.py);
# E3M0 is far coarser than fp16 and STERN's own throughput audit found I/O
# is 4.8s of a ~297s average round barrier (mesh/pacing.py) -- not the
# bottleneck. This primitive exists tested and ready for that measurement
# pass, not for a default-path flip ahead of it.
E3M0_BIAS = 4


def _quantize_e3m0(x: torch.Tensor) -> torch.Tensor:
    """Round `x` to the nearest E3M0 level, packed one code per uint8 byte
    (half fp16's footprint; a 2-per-byte bit-packed layout would halve it
    again but is a pure bit-shifting follow-up, not a numeric one -- left
    for once this quantization is validated on a real round).

    Code layout: bit 3 = sign, bits 2-0 = exponent code (0 = zero, 1-7 =
    magnitude 2**(code - E3M0_BIAS)).
    """
    sign = (x < 0).to(torch.uint8)
    mag = x.abs()
    # log2(0) = -inf; goes through torch.round unchanged (-inf) and clamps
    # to 0 below, landing on the reserved zero code -- no separate branch
    # needed for exact zeros.
    log2_mag = torch.where(
        mag > 0, torch.log2(mag), torch.full_like(mag, float("-inf"))
    )
    code = torch.round(log2_mag + E3M0_BIAS).clamp(min=0, max=7).to(torch.uint8)
    return (sign << 3) | code


def _dequantize_e3m0(codes: torch.Tensor) -> torch.Tensor:
    """Inverse of `_quantize_e3m0`. Exact only at E3M0's own grid points
    (powers of two); returns fp32."""
    sign = ((codes >> 3) & 1).to(torch.float32)
    exp_code = (codes & 0x7).to(torch.float32)
    mag = torch.where(
        exp_code == 0, torch.zeros_like(exp_code), torch.exp2(exp_code - E3M0_BIAS)
    )
    return torch.where(sign > 0, -mag, mag)


def quantize_e3m0_with_error_feedback(
    state_dict: StateDict, residual_path: Path | None
) -> StateDict:
    """E3M0 counterpart to `quantize_with_error_feedback`: same residual
    carry-across-calls mechanics (see that function's docstring), but each
    floating-point tensor becomes a `torch.uint8` code tensor instead of
    fp16 -- the caller must `dequantize_e3m0_state_dict` before any
    delta/momentum arithmetic, unlike fp16 tensors which upcast for free.
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
        if prior is not None and prior.shape == compensated.shape:
            compensated = compensated + prior.float()
        codes = _quantize_e3m0(compensated)
        quantized_sd[key] = codes
        new_residual[key] = compensated - _dequantize_e3m0(codes)

    if residual_path is not None:
        _atomic_save(residual_path, {"residual": new_residual})

    return quantized_sd


def dequantize_e3m0_state_dict(state_dict: StateDict) -> StateDict:
    """Inverse of `quantize_e3m0_with_error_feedback`'s per-tensor coding.
    Non-`uint8` tensors (the non-float buffers `quantize_e3m0_with_error_
    feedback` passed through unchanged) pass through unchanged here too.

    KNOWN LIMITATION (part of why this is unwired -- see the module-level
    comment above): dtype is the only signal this function has for "was
    this key coded." A state dict containing a genuine, always-uint8 buffer
    (not one `quantize_e3m0_with_error_feedback` produced) would be silently
    misdecoded as E3M0 codes. Wiring this into a live checkpoint format
    needs an explicit coded-keys marker traveling with the payload, not
    dtype-sniffing -- deferred to whoever does that wiring, since the right
    marker shape depends on the dtype-tag scheme that wiring picks.
    """
    return {
        k: (_dequantize_e3m0(v) if v.dtype == torch.uint8 else v)
        for k, v in state_dict.items()
    }
