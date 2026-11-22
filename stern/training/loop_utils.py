"""Small helpers shared by the training loops (finetune.py and pretrain.py)."""

from __future__ import annotations

import torch
from torch import nn


def set_sampler_epoch(loader, epoch: int) -> None:
    """Reshuffle a DistributedSampler-backed loader for a new epoch (no-op otherwise)."""
    sampler = getattr(loader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


class WeightEMA:
    """Polyak-averaged shadow of a model's float parameters/buffers.

    Eval/checkpoint-only smoothing -- never fed back into the optimized
    weights or, in the mesh path, into the pushed checkpoint that DiLoCo's
    outer step merges.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone().float()
            for k, v in model.state_dict().items()
            if torch.is_floating_point(v)
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(
                    v.detach().float(), alpha=1 - self.decay
                )

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}
