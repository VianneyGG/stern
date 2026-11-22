"""Tests for MeshShardIterableDataset's frame_stride parameter."""

from unittest.mock import patch

import numpy as np
import torch

from stern.data.mesh_shard_dataset import MeshShardIterableDataset
from stern.prep.build_wds_shards import (
    BalancedShardWriter,
    encode_frames_jpeg,
)


def _write_shard(root, clip_key: str, n_frames: int, image_size: int = 16) -> str:
    writer = BalancedShardWriter(root, n_shards=1)
    frames = [
        np.full((image_size, image_size, 3), i % 256, dtype=np.uint8)
        for i in range(n_frames)
    ]
    writer.write(clip_key, encode_frames_jpeg(frames))
    writer.close()
    return "shard-000000.tar"


def test_frame_stride_spaces_out_read_indices(tmp_path):
    root = tmp_path / "mesh"
    _write_shard(root, "clip0", n_frames=53, image_size=16)
    dataset = MeshShardIterableDataset(
        ["shard-000000.tar"],
        mesh_root=root,
        shard_map={},
        subnet="/254",
        num_frames=4,
        image_size=16,
        overlap=0.0,
        frame_stride=4,
        seed=0,
    )

    from stern.data import mesh_shard_dataset as mod

    seen_idx = []
    real_read = mod.read_shard_frames_indexed

    def _spy(path, clip_key, idx, size):
        seen_idx.append(list(idx))
        return real_read(path, clip_key, idx, size=size)

    with patch.object(mod, "read_shard_frames_indexed", side_effect=_spy):
        windows = list(dataset)

    assert len(windows) > 0
    assert len(seen_idx) > 0
    for idx in seen_idx:
        assert idx == list(range(idx[0], idx[0] + 4 * 4, 4))


def _gradient_shard(root, clip_key: str, n_frames: int, image_size: int = 16) -> str:
    """Like _write_shard but with position-varying content, so a crop or flip
    is actually visible in the decoded pixels (a solid frame is invariant to
    both, and would hide a no-op augmentation path)."""
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


def _kwargs(root):
    return dict(
        shard_names=["shard-000000.tar"],
        mesh_root=root,
        shard_map={},
        subnet="/254",
        num_frames=4,
        image_size=16,
        overlap=0.0,
        seed=0,
    )


def test_augment_flag_changes_pixels_but_not_shape(tmp_path):
    """augment=True must reach the pixels, not just survive the call.

    The shape check alone would pass with the augmentation path dead.
    """
    root = tmp_path / "mesh"
    _gradient_shard(root, "clip0", n_frames=53)
    plain = next(iter(MeshShardIterableDataset(**_kwargs(root), augment=False)))
    augmented = next(iter(MeshShardIterableDataset(**_kwargs(root), augment=True)))

    assert plain.shape == augmented.shape
    assert not torch.equal(plain, augmented), (
        "augment=True produced byte-identical output -- the augmentation path "
        "is a no-op"
    )


def _write_multi_shard(root, n_shards: int, n_clips: int, n_frames: int = 53):
    """Several clips spread over several shards.

    The order test below needs >1 shard to have any power: within a single
    shard the window shuffle happens before that shard's augment draws, so a
    shared RNG only shows up as divergence in the SECOND and later shards.
    """
    writer = BalancedShardWriter(root, n_shards=n_shards)
    frames = [
        np.full((16, 16, 3), i % 256, dtype=np.uint8) for i in range(n_frames)
    ]
    payload = encode_frames_jpeg(frames)
    for c in range(n_clips):
        writer.write(f"clip{c}", payload)
    writer.close()
    return sorted(p.name for p in root.glob("shard-*.tar"))


def test_augment_does_not_change_window_order(tmp_path):
    """Toggling augment must not reorder the training data.

    Augmentation draws share the worker's RNG lineage; if they consumed the
    same stream as the per-shard window shuffle, flipping this flag would
    silently change *which* windows are seen in *which* order, confounding
    any augment-on/augment-off comparison.
    """
    root = tmp_path / "mesh"
    shard_names = _write_multi_shard(root, n_shards=3, n_clips=6)
    assert len(shard_names) > 1, "fixture must span multiple shards"

    from stern.data import mesh_shard_dataset as mod

    def _order(augment: bool) -> list[tuple[str, list[int]]]:
        seen: list[tuple[str, list[int]]] = []
        real_read = mod.read_shard_frames_indexed

        def _spy(path, clip_key, idx, size):
            seen.append((clip_key, list(idx)))
            return real_read(path, clip_key, idx, size=size)

        kwargs = _kwargs(root) | {"shard_names": shard_names}
        with patch.object(mod, "read_shard_frames_indexed", side_effect=_spy):
            list(MeshShardIterableDataset(**kwargs, augment=augment))
        return seen

    assert _order(augment=False) == _order(augment=True)
