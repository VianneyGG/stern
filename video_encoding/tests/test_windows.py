import numpy as np
import pytest

from video_encoding.data.windows import make_windows


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
