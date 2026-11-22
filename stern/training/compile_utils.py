"""torch.compile wiring for the training loops.

Applied BEFORE wrap_ddp, never after: checkpoint.py's unwrap() checks `.module`
before `_orig_mod`, which only unwraps correctly if compile wrapped the raw
model first (DDP(torch.compile(model)), not torch.compile(DDP(model))).
"""

from __future__ import annotations

import torch


def maybe_compile(model: torch.nn.Module, cfg) -> torch.nn.Module:
    """Wrap `model` in torch.compile per cfg.train.compile/compile_mode.

    No-op (returns `model` unchanged, same object) when cfg.train.compile is
    falsy -- the default, until compile is benchmarked on-cluster.
    """
    if not bool(_get(cfg, "train.compile", default=False)):
        return model
    mode = str(_get(cfg, "train.compile_mode", default="default"))
    return torch.compile(model, mode=mode)


def _get(cfg, dotted: str, *, default):
    node = cfg
    for part in dotted.split("."):
        try:
            node = node[part] if isinstance(node, dict) else getattr(node, part)
        except (KeyError, AttributeError):
            return default
    return node
