"""Labelled dataset for VA-probe training.

Mirrors SSLWindowDataset (stern/data/ssl_windows.py) but:
- Labelled (VA harmonised to [-1, 1]), not unlabelled video
- Reads from AFEW-VA-style frame directories only, no live-decode fallback
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset

from stern.data.labels import load_per_frame_va
from stern.data.frame_decoder import list_frame_files, read_frames_at
from stern.data.windows import build_frame_windows
from stern.models.videomae.encoder import preprocess_window


def shuffle_frames(window: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Randomly permute a window's frames along T.

    Destroys temporal order but keeps every frame. A CCC drop under this
    control therefore reflects the order, not the content or domain shift.
    """
    perm = torch.randperm(window.shape[1], generator=generator)
    return window[:, perm]


def freeze_frames(window: torch.Tensor, index: int | None = None) -> torch.Tensor:
    """Replace every frame with one source frame (default: the middle one).

    A secondary reference, not the gate. Unlike shuffling, it also changes the
    content distribution.
    """
    t = window.shape[1]
    i = t // 2 if index is None else index
    return window[:, i : i + 1].expand_as(window).clone()


def make_temporal_transform(
    condition: str, seed: int
) -> Callable[[torch.Tensor], torch.Tensor] | None:
    """Build the per-window transform for one probe condition."""
    if condition == "normal":
        return None
    if condition == "shuffled":
        # One generator per DataLoader worker, created lazily inside that worker.
        # A generator closed over before fork would be copied into every worker,
        # and they would replay the same permutations. The shuffled condition is
        # the control for the temporal gate, so its randomness must not depend
        # on how the loader assigns workers.
        generators: dict[int, torch.Generator] = {}

        def _shuffle(window: torch.Tensor) -> torch.Tensor:
            info = torch.utils.data.get_worker_info()
            worker_id = 0 if info is None else int(info.id)
            generator = generators.get(worker_id)
            if generator is None:
                generator = torch.Generator().manual_seed(seed + 1_000_003 * worker_id)
                generators[worker_id] = generator
            return shuffle_frames(window, generator)

        return _shuffle
    if condition == "frozen":
        return freeze_frames
    raise ValueError(
        f"unknown temporal condition {condition!r}: expected normal|shuffled|frozen"
    )


class VAWindowDataset(Dataset):
    """Flattens (clip -> fixed 16-frame windows) into individual
    `(pixel_values, label)` training examples.

    Each window's label is the mean VA over its 16 frames (harmonised to
    `[-1, 1]`). Frame and label windows share the same start index, so pixels
    and targets stay aligned. Only frame *directories* (AFEW-VA's PNG layout)
    are supported here; the model always consumes exactly `num_frames` frames.
    """

    def __init__(
        self,
        labelled_dataset,
        *,
        num_frames: int,
        image_size: int,
        overlap: float,
        va_range: tuple[float, float],
        frame_stride: int = 1,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        self._num_frames = num_frames
        self._image_size = image_size
        self._transform = transform
        self._items: list[tuple[list[Path], list[int], np.ndarray]] = []
        skipped: list[str] = []
        for record in labelled_dataset.records:
            clip_dir = labelled_dataset.root / record.clip_path
            try:
                frame_files = list_frame_files(clip_dir)
                va = load_per_frame_va(
                    labelled_dataset.root / record.label_path, va_range
                )
            except (ValueError, OSError) as exc:
                # A manifest can list a clip whose frames failed to extract or
                # whose label file is empty/malformed -- skip that one clip
                # rather than aborting construction for every other good clip.
                skipped.append(f"{record.clip_path} ({exc})")
                continue
            n = min(len(frame_files), va.shape[0])
            for _, idx in build_frame_windows(
                clip_dir,
                num_frames=num_frames,
                overlap=overlap,
                n_frames=n,
                frame_stride=frame_stride,
            ):
                # Averaged over the sampled rows, not over the span: under a
                # stride the span's other rows describe frames the encoder was
                # never shown, so including them would label the window with
                # motion it cannot see. Identical to the contiguous slice at
                # stride 1.
                label = va[idx].mean(axis=0)
                self._items.append((frame_files, idx, label.astype(np.float32)))
        if skipped:
            print(
                f"WARNING: skipped {len(skipped)} malformed clip(s): {skipped}",
                flush=True,
            )
        if not self._items:
            raise ValueError(
                "no usable clips: every manifest record failed to load "
                f"(skipped: {skipped})"
                if skipped
                else "no usable clips: manifest is empty"
            )

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        frame_files, idx, label = self._items[i]
        frames = read_frames_at(frame_files, idx, size=self._image_size)
        pixel_values = preprocess_window(frames, self._num_frames, self._image_size)
        window = pixel_values.squeeze(0)
        if self._transform is not None:
            window = self._transform(window)
        return window, torch.from_numpy(label)
