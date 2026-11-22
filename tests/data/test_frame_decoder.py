from pathlib import Path

import av
import numpy as np
from PIL import Image

from stern.data.frame_decoder import decode_frames, load_clip_frames


def _write_synthetic_mp4(
    path: Path, num_frames: int, width: int = 64, height: int = 64
) -> None:
    """Encode a tiny real mp4 whose frame N is solid color value N (mod 256)."""
    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=25)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    for i in range(num_frames):
        arr = np.full((height, width, 3), fill_value=i % 256, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def test_decode_frames_returns_requested_count_and_shape(tmp_path: Path):
    video_path = tmp_path / "clip.mp4"
    _write_synthetic_mp4(video_path, num_frames=40)

    frames = decode_frames(video_path, num_frames=16)

    assert len(frames) == 16
    for frame in frames:
        assert frame.shape == (224, 224, 3)
        assert frame.dtype == np.uint8


def test_decode_frames_samples_uniformly_across_full_duration(tmp_path: Path):
    video_path = tmp_path / "clip.mp4"
    _write_synthetic_mp4(video_path, num_frames=40)

    frames = decode_frames(video_path, num_frames=4)

    # mpeg4 lossy encoding perturbs exact pixel values, so compare with
    # tolerance instead of exact equality; the key behavior under test is
    # that samples span near-0 to near-39, not clustered at the start.
    sampled_values = [int(round(f[0, 0, 0])) for f in frames]
    assert sampled_values[0] <= 5
    assert sampled_values[-1] >= 34
    assert sampled_values == sorted(sampled_values)


def test_decode_frames_handles_fewer_source_frames_than_requested(tmp_path: Path):
    video_path = tmp_path / "clip.mp4"
    _write_synthetic_mp4(video_path, num_frames=3)

    frames = decode_frames(video_path, num_frames=16)

    assert len(frames) == 16


def test_load_clip_frames_dispatches_to_frame_directory(tmp_path: Path):
    # AFEW-VA/VEATIC layout: clip_path is a directory of frame images, not a
    # video file -- LabelledDataset.load_frames() used to call decode_frames()
    # unconditionally here and crash (av.open() can't open a directory).
    clip_dir = tmp_path / "001"
    clip_dir.mkdir()
    for i in range(10):
        Image.fromarray(np.full((8, 8, 3), i, dtype=np.uint8)).save(
            clip_dir / f"{i}.png"
        )

    frames = load_clip_frames(clip_dir, num_frames=4, size=8)

    assert len(frames) == 4
    for frame in frames:
        assert frame.shape == (8, 8, 3)


def test_load_clip_frames_dispatches_to_video_file(tmp_path: Path):
    video_path = tmp_path / "clip.mp4"
    _write_synthetic_mp4(video_path, num_frames=20)

    frames = load_clip_frames(video_path, num_frames=4)

    assert len(frames) == 4
    assert frames[0].shape == (224, 224, 3)
