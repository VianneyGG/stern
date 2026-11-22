"""Uniform weight average of several mesh master snapshots.

The mesh keeps no Polyak/EMA average of the merged master, while the LeVJEPA
paper evaluates one. Averaging saved round snapshots is the closest
post-hoc substitute.

Float tensors are averaged in fp32 and cast back to their dtype. Non-float
tensors (e.g. BatchNorm's num_batches_tracked) are taken from the last
snapshot. The output keeps the `{"model_state_dict": ...}` layout the gate and
Stage 2 load.

Usage:
    uv run --no-sync python -m video_encoding.scripts.average_checkpoints \
        --out avg.pt run/master_round30.pt run/master_round36.pt ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def average_state_dicts(state_dicts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not state_dicts:
        raise ValueError("need at least one state dict")
    keys = state_dicts[0].keys()
    for i, sd in enumerate(state_dicts[1:], start=1):
        if sd.keys() != keys:
            diff = sorted(set(sd) ^ set(keys))[:5]
            raise ValueError(f"state dict {i} has different keys, e.g. {diff}")
    out = {}
    for k in keys:
        last = state_dicts[-1][k]
        if last.is_floating_point():
            out[k] = (sum(sd[k].float() for sd in state_dicts) / len(state_dicts)).to(last.dtype)
        else:
            out[k] = last.clone()
    return out


def _load(path: Path) -> dict[str, torch.Tensor]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    return raw.get("model_state_dict", raw)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("snapshots", nargs="+", type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    avg = average_state_dicts([_load(s) for s in a.snapshots])
    torch.save(
        {"model_state_dict": avg, "averaged_from": [str(s) for s in a.snapshots]}, a.out
    )
    print(f"averaged {len(a.snapshots)} snapshots -> {a.out}")


if __name__ == "__main__":
    main()
