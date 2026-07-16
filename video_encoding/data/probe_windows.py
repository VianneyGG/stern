"""Labelled dataset for VA-probe training.

Mirrors SSLWindowDataset (video_encoding/data/ssl_windows.py) but:
- Labelled (VA harmonised to [-1, 1]), not unlabelled video
- Reads from AFEW-VA-style frame directories only, no live-decode fallback
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data.labels import load_per_frame_va
from video_encoding.data.frame_decoder import list_frame_files, read_frames_at
from video_encoding.data.windows import build_frame_windows
from video_encoding.models.videomaev2.encoder import preprocess_window


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
    ):
        self._num_frames = num_frames
        self._image_size = image_size
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
                clip_dir, num_frames=num_frames, overlap=overlap, n_frames=n
            ):
                label = va[idx[0] : idx[0] + num_frames].mean(axis=0)
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
        return pixel_values.squeeze(0), torch.from_numpy(label)
