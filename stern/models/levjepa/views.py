"""Multi-view augmentation for LeVJEPA: one global view and several local crops per clip.

Each local crop's box is returned too, so the encoder can place its RoPE
coordinates in the global view's frame.
"""

from __future__ import annotations

import hashlib

import torch
import torchvision.transforms.v2.functional as TF

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _normalize(frames: torch.Tensor) -> torch.Tensor:
    # frames: (T, 3, H, W) float in [0, 1]
    mean = torch.tensor(_IMAGENET_MEAN, dtype=frames.dtype, device=frames.device).view(
        1, 3, 1, 1
    )
    std = torch.tensor(_IMAGENET_STD, dtype=frames.dtype, device=frames.device).view(
        1, 3, 1, 1
    )
    return (frames - mean) / std


def _to_tchw_float(clip: torch.Tensor) -> torch.Tensor:
    # clip: (T, H, W, 3) uint8 -> (T, 3, H, W) float in [0, 1]
    return clip.permute(0, 3, 1, 2).float() / 255.0


def derive_seed(*parts) -> int:
    """Stable 63-bit seed from arbitrary parts, e.g. `(seed, "views", round)`.
    Python's `hash()` is salted per process, so it cannot key a generator
    that must agree across arms and relaunches."""
    digest = hashlib.sha256(":".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little") >> 1


def build_views(
    clip: torch.Tensor,
    num_local: int = 4,
    global_size: int = 224,
    local_size: int = 96,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns `(global_view, local_views, crop_boxes)`.

    `generator`: draw crops and jitter from it instead of the global RNG, so
    views do not shift when something else (model init, token drop) changes
    how much global randomness is consumed. None = global RNG, as before.

    `crop_boxes`: `(num_local, 4)`, fractional `(top, left, height, width)`
    of the original frame each local view was cropped from -- consumed by
    `LeVJEPAEncoder.forward(..., crop_box=...)` to place a local view's RoPE
    coordinates correctly within the global view's grid frame.
    """
    frames = _to_tchw_float(clip)  # (T, 3, H, W)
    orig_h, orig_w = frames.shape[-2], frames.shape[-1]

    global_resized = TF.resize(frames, [global_size, global_size], antialias=True)
    global_view = _normalize(global_resized).permute(1, 0, 2, 3)  # (3, T, H, W)

    local_views = []
    crop_boxes = []
    for _ in range(num_local):
        top, left, height, width = _random_crop_box(orig_h, orig_w, generator)
        cropped = TF.resized_crop(
            frames, top, left, height, width, [local_size, local_size], antialias=True
        )
        brightness = 0.6 + torch.rand(1, generator=generator).item() * 0.8
        contrast = 0.6 + torch.rand(1, generator=generator).item() * 0.8
        saturation = 0.6 + torch.rand(1, generator=generator).item() * 0.8
        cropped = TF.adjust_brightness(cropped, brightness)
        cropped = TF.adjust_contrast(cropped, contrast)
        cropped = TF.adjust_saturation(cropped, saturation)
        local_views.append(_normalize(cropped).permute(1, 0, 2, 3))  # (3, T, H, W)
        crop_boxes.append(
            [top / orig_h, left / orig_w, height / orig_h, width / orig_w]
        )

    return (
        global_view,
        torch.stack(local_views, dim=0),
        torch.tensor(crop_boxes, dtype=torch.float32),
    )


def _random_crop_box(
    height: int, width: int, generator: torch.Generator | None = None
) -> tuple[int, int, int, int]:
    scale = 0.2 + torch.rand(1, generator=generator).item() * 0.5  # area fraction in [0.2, 0.7]
    target_area = scale * height * width
    aspect = 0.75 + torch.rand(1, generator=generator).item() * 0.5  # in [0.75, 1.25]
    crop_h = min(height, max(1, int(round((target_area / aspect) ** 0.5))))
    crop_w = min(width, max(1, int(round((target_area * aspect) ** 0.5))))
    top = torch.randint(0, height - crop_h + 1, (1,), generator=generator).item()
    left = torch.randint(0, width - crop_w + 1, (1,), generator=generator).item()
    return top, left, crop_h, crop_w
