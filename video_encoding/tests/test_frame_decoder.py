from pathlib import Path

import av
import numpy as np
import pytest

from video_encoding.data.frame_decoder import decode_frames


def _write_synthetic_mp4(path: Path, num_frames: int, width: int = 64, height: int = 64) -> None:
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
