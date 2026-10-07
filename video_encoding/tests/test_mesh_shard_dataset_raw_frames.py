"""Tests for MeshShardIterableDataset's raw_frames mode (LeVJEPA mesh path).

LeVJEPAViewDataset/build_views need un-normalized (T, H, W, 3) uint8 clips,
matching SSLWindowDataset(raw_frames=True)'s contract -- not the normalized
(C, T, H, W) float pixel_values preprocess_window produces for VideoMAE.
"""

from __future__ import annotations

import numpy as np
import torch

from video_encoding.data.mesh_shard_dataset import MeshShardIterableDataset
from video_encoding.scripts.build_wds_shards import (
    BalancedShardWriter,
    encode_frames_jpeg,
)


def _write_shard(root, clip_key: str, n_frames: int, image_size: int = 16) -> str:
    writer = BalancedShardWriter(root, n_shards=1)
    rows = np.arange(image_size, dtype=np.int32)[:, None]
    cols = np.arange(image_size, dtype=np.int32)[None, :]
    frames = [
        np.repeat(
            (((rows * 7 + cols * 11 + i * 3) % 256).astype(np.uint8))[:, :, None],
            3,
            axis=2,
        )
        for i in range(n_frames)
    ]
    writer.write(clip_key, encode_frames_jpeg(frames))
    writer.close()
    return "shard-000000.tar"


def test_raw_frames_yields_unnormalized_uint8_clip(tmp_path):
    root = tmp_path / "mesh"
    _write_shard(root, "clip0", n_frames=20, image_size=16)
    dataset = MeshShardIterableDataset(
        ["shard-000000.tar"],
        mesh_root=root,
        shard_map={},
        subnet="/254",
        num_frames=4,
        image_size=16,
        overlap=0.0,
        seed=0,
        raw_frames=True,
    )

    clip = next(iter(dataset))

    assert clip.dtype == torch.uint8
    assert clip.shape == (4, 16, 16, 3)
    # A normalized float tensor would never hit this range; a raw decoded
    # JPEG frame (built from a non-constant rows/cols/i pattern above) will.
    assert int(clip.max()) > 1


def test_raw_frames_and_normalized_modes_agree_on_source_pixels(tmp_path):
    """raw_frames=True must decode the same underlying frames as the default
    path -- only skip the preprocess_window normalization step, not change
    which frames/windows are read."""
    root = tmp_path / "mesh"
    _write_shard(root, "clip0", n_frames=20, image_size=16)

    def _kwargs(**overrides):
        base = dict(
            shard_names=["shard-000000.tar"],
            mesh_root=root,
            shard_map={},
            subnet="/254",
            num_frames=4,
            image_size=16,
            overlap=0.0,
            seed=0,
        )
        base.update(overrides)
        return base

    normalized = next(
        iter(MeshShardIterableDataset(**_kwargs()))
    )  # (1,C,T,H,W)... squeezed to (C,T,H,W)
    raw = next(
        iter(MeshShardIterableDataset(**_kwargs(raw_frames=True)))
    )  # (T,H,W,3) uint8

    # Undo preprocess_window's ImageNet normalize + (C,T,H,W) permute to
    # recover approximate raw pixel values, and compare against the raw path.
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
    denorm = ((normalized * std + mean) * 255.0).round().clamp(0, 255).byte()
    denorm = denorm.permute(1, 2, 3, 0)  # (T, H, W, 3)

    assert denorm.shape == raw.shape
    assert torch.allclose(denorm.float(), raw.float(), atol=2.0)
