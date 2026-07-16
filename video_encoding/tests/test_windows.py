from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from video_encoding.data.windows import build_frame_windows, make_windows


def _write_frames(clip_dir: Path, n: int) -> None:
    clip_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(
            clip_dir / f"{i:04d}.jpg"
        )


def test_build_frame_windows_matches_window_starts(tmp_path):
    clip_dir = tmp_path / "clip"
    _write_frames(clip_dir, n=24)

    windows = build_frame_windows(clip_dir, num_frames=16, overlap=0.5)

    # window_starts(24, 16, 0.5) -> step=8 -> starts at [0, 8]
    assert len(windows) == 2
    assert windows[0][1] == list(range(0, 16))
    assert windows[1][1] == list(range(8, 24))
    assert len(windows[0][0]) == 24  # frame_files listing, shared across windows


def test_build_frame_windows_n_frames_override_clamps_below_disk_count(tmp_path):
    clip_dir = tmp_path / "clip"
    _write_frames(clip_dir, n=24)

    # Labels only cover 20 frames -- windows must not run past that.
    windows = build_frame_windows(clip_dir, num_frames=16, overlap=0.0, n_frames=20)

    assert len(windows) == 1
    assert windows[0][1] == list(range(0, 16))


def test_make_windows_no_overlap_exact_division():
    frames = np.arange(100)  # 100 frames at fps=50 -> 2.0s total
    windows = make_windows(frames, fps=50.0, len_s=1.0, overlap=0.0)
    assert len(windows) == 2
    np.testing.assert_array_equal(windows[0], frames[0:50])
    np.testing.assert_array_equal(windows[1], frames[50:100])


def test_make_windows_with_50pct_overlap():
    frames = np.arange(150)  # 3.0s at fps=50
    windows = make_windows(frames, fps=50.0, len_s=1.0, overlap=0.5)
    # step = 50 * (1 - 0.5) = 25 frames; windows start at 0, 25, 50, 75, 100 (last full window ends at 150)
    assert len(windows) == 5
    np.testing.assert_array_equal(windows[1], frames[25:75])


def test_make_windows_drops_trailing_partial_window():
    frames = np.arange(
        120
    )  # 2.4s at fps=50, window=1s -> 2 full windows, partial dropped
    windows = make_windows(frames, fps=50.0, len_s=1.0, overlap=0.0)
    assert len(windows) == 2


def test_make_windows_rejects_overlap_out_of_bounds():
    frames = np.arange(50)
    with pytest.raises(ValueError):
        make_windows(frames, fps=50.0, len_s=1.0, overlap=1.0)


def test_make_windows_rejects_negative_overlap():
    frames = np.arange(50)
    with pytest.raises(ValueError):
        make_windows(frames, fps=50.0, len_s=1.0, overlap=-0.1)


def test_make_windows_empty_frames_returns_empty_list():
    frames = np.arange(0)
    windows = make_windows(frames, fps=50.0, len_s=1.0, overlap=0.0)
    assert windows == []


def test_make_windows_rejects_nonpositive_fps():
    frames = np.arange(50)
    with pytest.raises(ValueError):
        make_windows(frames, fps=0.0, len_s=1.0, overlap=0.0)
    with pytest.raises(ValueError):
        make_windows(frames, fps=-10.0, len_s=1.0, overlap=0.0)


def test_make_windows_rejects_nonpositive_len_s():
    frames = np.arange(50)
    with pytest.raises(ValueError):
        make_windows(frames, fps=50.0, len_s=0.0, overlap=0.0)
    with pytest.raises(ValueError):
        make_windows(frames, fps=50.0, len_s=-1.0, overlap=0.0)


def test_make_windows_fewer_frames_than_one_window():
    frames = np.arange(10)  # 0.2s at fps=50, window=1s -> no full window
    windows = make_windows(frames, fps=50.0, len_s=1.0, overlap=0.0)
    assert windows == []
