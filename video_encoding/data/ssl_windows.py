"""Label-free dataset for SSL pretraining on unlabelled video.

Mirrors _VAWindowDataset from probe/train.py but:
- No labels (unlabelled video like VoxCeleb2)
- Reads from cached JPEG frames (Workstream A4b) when available
- Falls back to live mp4 decode if cache miss
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

from video_encoding.data.frame_decoder import decode_frames, read_frames_at
from video_encoding.data.loaders import SSLDataset
from video_encoding.models.videomaev2.encoder import preprocess_window
from video_encoding.data.windows import build_frame_windows


class SSLWindowDataset(Dataset):
    """Unlabelled, fixed-window dataset from SSLDataset.

    Iterates identity→clip, reads fixed `num_frames`-long windows, emits only
    pixel_values (no labels). Reads from the pre-decoded JPEG cache
    (Workstream A4b, `VOXMINI_FRAMES_ROOT`) when set, using
    `build_frame_windows` (shared with `VAWindowDataset`) -- multiple
    overlapping windows per clip, each reading only its own slice of frame
    files.

    Falls back to one live-decoded window per clip (via `decode_frames`'s
    whole-clip uniform sampling) when `VOXMINI_FRAMES_ROOT` is unset -- this
    path is for the small local smoke test only, where paying mp4 decode
    once per clip is cheap and multi-window overlap isn't worth the
    complexity of decoding+caching a whole clip's frames in memory just to
    slice windows out of it.
    """

    def __init__(
        self,
        ssl_dataset: SSLDataset,
        *,
        num_frames: int,
        image_size: int,
        overlap: float,
    ):
        self._num_frames = num_frames
        self._image_size = image_size
        self._ssl_dataset = ssl_dataset
        # Cache-backed window: (frame_files, idx). Live-decode window: (None, clip_path).
        self._items: list[tuple[list[Path] | None, list[int] | str]] = []

        cache_root = os.environ.get("VOXMINI_FRAMES_ROOT")

        skipped: list[str] = []
        for records in ssl_dataset.records_by_identity.values():
            for record in records:
                clip_path = record.clip_path
                try:
                    if cache_root:
                        cache_dir = Path(cache_root) / Path(clip_path).with_suffix("")
                        self._items.extend(
                            build_frame_windows(
                                cache_dir, num_frames=num_frames, overlap=overlap
                            )
                        )
                    else:
                        # decode_frames already uniformly samples num_frames
                        # from the whole clip -- one window, no start offset.
                        self._items.append((None, clip_path))
                except (ValueError, OSError) as exc:
                    skipped.append(f"{clip_path} ({exc})")
                    continue

        if skipped:
            print(
                f"WARNING: skipped {len(skipped)} clips during SSL dataset load: "
                f"{skipped[:5]}{'...' if len(skipped) > 5 else ''}",
                flush=True,
            )
        if not self._items:
            raise ValueError(
                f"no usable SSL clips (skipped: {len(skipped)}). "
                "Check dataset presence and cache path (VOXMINI_FRAMES_ROOT)."
            )

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, i: int) -> torch.Tensor:
        frame_files, idx = self._items[i]
        try:
            if frame_files is not None:
                frames = read_frames_at(frame_files, idx, size=self._image_size)
            else:
                clip_file = self._ssl_dataset.root / idx
                frames = decode_frames(clip_file, self._num_frames)
        except (ValueError, OSError) as exc:
            raise RuntimeError(f"Failed to load frames for item {i}") from exc

        pixel_values = preprocess_window(frames, self._num_frames, self._image_size)
        return pixel_values.squeeze(0)
