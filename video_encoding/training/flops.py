"""FLOPs / parameter counting for isoFLOP result JSONs.

Ports the interface of What_Happens_Next's `src/flops.py` so the two
projects' result JSONs slot into the same isoFLOP comparison tables. Only
the fields consumed by the result JSON are exposed here; more elaborate
per-layer breakdowns can be added later if needed.

Fallback policy: if `torch.utils.flop_counter.FlopCounterMode` fails
(model runs a control-flow branch it can't trace, custom kernel without a
FLOP formula, etc.), we return `flops_fwd_per_clip=None` and let the
caller record just the parameter counts — a partial result JSON beats a
crashed run.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from video_encoding.training.checkpoint import unwrap as _unwrap


def measure_cost(
    model: nn.Module,
    *,
    sample_shape: tuple[int, ...],
    device: torch.device | str,
) -> dict[str, Any]:
    """Return `{params_total, params_encoder, flops_fwd_per_clip}`.

    `params_encoder` is best-effort: we look for a common `encoder`
    attribute (VideoMAE, ViT), else a `videomae` attribute (the
    HF `VideoMAEForPreTraining` layout used by SSL), else fall back to
    the total. This matters for isoFLOP tables that price backbone
    weights separately from heads.
    """
    raw = _unwrap(model)
    params_total = _count_params(raw)
    params_encoder = _count_encoder_params(raw, fallback=params_total)
    flops_fwd = _flops_forward(raw, sample_shape=sample_shape, device=device)
    return {
        "params_total": params_total,
        "params_encoder": params_encoder,
        "flops_fwd_per_clip": flops_fwd,
    }


def compute_flops(flops_fwd_per_clip: int | None, samples_seen: int) -> int | None:
    """Total compute FLOPs = 3 × fwd × samples (fwd + 2× for backward).

    The 3× factor is the standard Kaplan/Chinchilla convention. Returns
    `None` if `flops_fwd_per_clip` was unavailable so downstream JSONs can
    still record `samples_seen` without a fabricated FLOPs number.
    """
    if flops_fwd_per_clip is None:
        return None
    return 3 * int(flops_fwd_per_clip) * int(samples_seen)


def _flops_forward(
    model: nn.Module,
    *,
    sample_shape: tuple[int, ...],
    device: torch.device | str,
) -> int | None:
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return None

    model = model.to(device)
    was_training = model.training
    model.eval()
    x = torch.randn(*sample_shape, device=device)
    try:
        with FlopCounterMode(display=False) as counter, torch.no_grad():
            model(x)
        # `.get_total_flops()` returns the total across all recorded ops.
        return int(counter.get_total_flops())
    except Exception:
        return None
    finally:
        if was_training:
            model.train()


def _count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _count_encoder_params(module: nn.Module, *, fallback: int) -> int:
    for attr in ("encoder", "videomae", "backbone"):
        sub = getattr(module, attr, None)
        if isinstance(sub, nn.Module):
            return _count_params(sub)
    return fallback
