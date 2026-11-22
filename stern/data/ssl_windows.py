"""Label-free dataset for SSL pretraining on unlabelled video.

Mirrors _VAWindowDataset from probe/train.py but:
- No labels (unlabelled video like VoxCeleb2)
- Reads from WDS shards (VOXMINI_SHARDS_ROOT, preferred) or the legacy
  cached-JPEG-directory layout (VOXMINI_FRAMES_ROOT) when available
- Falls back to live mp4 decode if neither is set
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from stern.data.frame_decoder import decode_frames, read_frames_at
from stern.data.loaders import SSLDataset
from stern.data.wds_frames import load_shard_index, read_shard_frames_indexed
from stern.data.windows import window_starts
from stern.models.videomae.encoder import preprocess_window
from stern.data.windows import build_frame_windows


class SSLWindowDataset(Dataset):
    """Unlabelled, fixed-window dataset from SSLDataset.

    Iterates identity->clip, reads fixed `num_frames`-long windows, emits only
    pixel_values (no labels). Three source priorities, checked in order:

    1. VOXMINI_SHARDS_ROOT set: read from WDS shards (build_wds_shards.py).
       `_items` entries are `((shard_path, clip_key), idx)` -- a tuple in
       position 0 distinguishes this from the directory-cache case below.
    2. VOXMINI_FRAMES_ROOT set: read from the legacy loose-JPEG-directory
       cache via `build_frame_windows` -- `_items` entries are
       `(frame_files, idx)`, unchanged from before.
    3. Neither set: one live-decoded window per clip via `decode_frames`,
       `_items` entries are `(None, clip_path)` -- for the small local smoke
       test only, unchanged from before.
    """

    def __init__(
        self,
        ssl_dataset: SSLDataset,
        *,
        num_frames: int,
        image_size: int,
        overlap: float,
        frame_stride: int = 1,
        raw_frames: bool = False,
    ):
        self._num_frames = num_frames
        self._image_size = image_size
        self._raw_frames = raw_frames
        self._ssl_dataset = ssl_dataset
        self._items: list[tuple[object, list[int] | str]] = []

        shards_root = os.environ.get("VOXMINI_SHARDS_ROOT")
        cache_root = os.environ.get("VOXMINI_FRAMES_ROOT")
        shard_index = load_shard_index(Path(shards_root)) if shards_root else None

        skipped: list[str] = []
        for records in ssl_dataset.records_by_identity.values():
            for record in records:
                clip_path = record.clip_path
                try:
                    if shard_index is not None:
                        clip_key = str(Path(clip_path).with_suffix(""))
                        entry = shard_index[clip_key]
                        shard_path = Path(shards_root) / entry["shard"]
                        for start in window_starts(
                            entry["num_frames"], num_frames, overlap, frame_stride
                        ):
                            idx = list(
                                range(
                                    start,
                                    start + num_frames * frame_stride,
                                    frame_stride,
                                )
                            )
                            self._items.append(((shard_path, clip_key), idx))
                    elif cache_root:
                        cache_dir = Path(cache_root) / Path(clip_path).with_suffix("")
                        self._items.extend(
                            build_frame_windows(
                                cache_dir,
                                num_frames=num_frames,
                                overlap=overlap,
                                frame_stride=frame_stride,
                            )
                        )
                    else:
                        self._items.append((None, clip_path))
                except (ValueError, OSError, KeyError) as exc:
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
                "Check dataset presence and cache path (VOXMINI_SHARDS_ROOT/VOXMINI_FRAMES_ROOT)."
            )

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, i: int) -> torch.Tensor:
        source, idx = self._items[i]
        try:
            if isinstance(source, tuple):
                shard_path, clip_key = source
                frames = read_shard_frames_indexed(
                    shard_path, clip_key, idx, size=self._image_size
                )
            elif source is not None:
                frames = read_frames_at(source, idx, size=self._image_size)
            else:
                clip_file = self._ssl_dataset.root / idx
                frames = decode_frames(clip_file, self._num_frames)
        except (ValueError, OSError) as exc:
            raise RuntimeError(f"Failed to load frames for item {i}") from exc

        if self._raw_frames:
            # LeVJEPA's build_views does its own uint8 normalisation and
            # photometric augmentation -- handing it preprocess_window's
            # already-ImageNet-normalised float tensor would double up
            # (and its (C, T, H, W) layout doesn't match build_views's
            # expected (T, H, W, 3) input either).
            return torch.from_numpy(np.stack(frames))

        pixel_values = preprocess_window(frames, self._num_frames, self._image_size)
        return pixel_values.squeeze(0)
