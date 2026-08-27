"""Group-consistent spatial augmentation for MAE pretraining windows.

VideoMAE's GroupMultiScaleCrop samples one crop scale + position and one flip
decision per clip, then applies it identically to every frame. Independent
per-frame augmentation would inject motion the model didn't actually see,
corrupting the exact signal masked-reconstruction is meant to learn.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from PIL import Image

_SCALES = (1.0, 0.875, 0.75, 0.66)


@dataclass(frozen=True)
class GroupAugmentParams:
    crop_box: tuple[int, int, int, int]  # (left, top, right, bottom)
    flip: bool


def sample_group_augment(
    image_size: int, rng: random.Random, scales: tuple[float, ...] = _SCALES
) -> GroupAugmentParams:
    scale = rng.choice(scales)
    crop_size = max(1, round(image_size * scale))
    max_offset = image_size - crop_size
    left = rng.randint(0, max_offset)
    top = rng.randint(0, max_offset)
    return GroupAugmentParams(
        crop_box=(left, top, left + crop_size, top + crop_size),
        flip=rng.random() < 0.5,
    )


def apply_group_augment(
    frames: list[np.ndarray], params: GroupAugmentParams, image_size: int
) -> list[np.ndarray]:
    out = []
    for frame in frames:
        img = (
            Image.fromarray(frame)
            .crop(params.crop_box)
            .resize((image_size, image_size), Image.BILINEAR)
        )
        if params.flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        out.append(np.array(img))
    return out
