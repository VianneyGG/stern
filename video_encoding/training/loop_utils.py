"""Small helpers shared by training/train.py and training/pretrain.py."""

from __future__ import annotations


def set_sampler_epoch(loader, epoch: int) -> None:
    """Reshuffle a DistributedSampler-backed loader for a new epoch (no-op otherwise)."""
    sampler = getattr(loader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)
