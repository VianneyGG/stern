"""Unit tests for VAWindowDataset (video_encoding/data/probe_windows.py).

This is the Stage-2 dataset: every labelled training example the VA probe ever
sees comes through here, so its per-window (pixels, label) alignment is what the
headline CCC number ultimately rests on. It had no test coverage at all until
now, while the tube-masking sampler next door had exhaustive coverage -- the
inverse of where the scientific risk actually is.

Frames are written as solid images whose pixel value equals the frame index, and
labels as `valence = arousal = frame index`. That makes "which frames did this
window return" and "which rows did its label average" both directly observable,
which is the only way to catch an off-by-one between the two.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from data.labels import write_label_csv
from data.manifest import ManifestRecord
from video_encoding.data.loaders import LabelledDataset
from video_encoding.data.probe_windows import VAWindowDataset
from video_encoding.models.videomaev2.encoder import preprocess_window

NUM_FRAMES = 4
IMAGE_SIZE = 8


def _solid(value: int, size: int = IMAGE_SIZE) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def _write_indexed_frames(clip_dir: Path, n: int) -> None:
    """Frame i is a solid image of pixel value i, so a returned window's
    identity is readable straight off its pixels."""
    clip_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.fromarray(_solid(i)).save(clip_dir / f"{i:04d}.png")


def _labelled(
    root: Path, clips: list[str], *, n_frames: int, n_labels: int | None = None
) -> LabelledDataset:
    """One record per clip; frame i and label row i both carry the value i."""
    n_labels = n_frames if n_labels is None else n_labels
    records = []
    for clip in clips:
        _write_indexed_frames(root / clip, n_frames)
        label_path = f"{clip}.csv"
        write_label_csv(
            root / label_path, [(float(i), float(i)) for i in range(n_labels)]
        )
        records.append(
            ManifestRecord(
                identity_id=clip.split("/")[0],
                clip_path=clip,
                duration_s=float(n_frames),
                role="probe",
                label_path=label_path,
            )
        )
    return LabelledDataset(records=records, window_cfg=OmegaConf.create({}), root=root)


def _dataset(
    labelled: LabelledDataset, *, overlap: float = 0.0, hi: float
) -> VAWindowDataset:
    return VAWindowDataset(
        labelled,
        num_frames=NUM_FRAMES,
        image_size=IMAGE_SIZE,
        overlap=overlap,
        va_range=(0.0, hi),
    )


def _expected_label(start: int, hi: float) -> float:
    """harmonise_va maps [0, hi] -> [-1, 1], so frame value v -> 2v/hi - 1.
    The window label is the mean of that over frames [start, start+NUM_FRAMES)."""
    values = np.arange(start, start + NUM_FRAMES, dtype=np.float64)
    return float((2.0 * values / hi - 1.0).mean())


def test_pixels_and_label_come_from_the_same_frame_range(tmp_path):
    """The alignment contract: window k must return the pixels of frames
    [start, start+num_frames) AND a label averaged over those same rows.

    A shared start index is the whole point of the class. If pixels and labels
    ever drift apart the model trains on mislabelled windows and nothing
    downstream -- shapes, dtypes, loss magnitude -- looks wrong.
    """
    labelled = _labelled(tmp_path, ["id0/clip0"], n_frames=12)
    ds = _dataset(labelled, hi=11.0)

    assert len(ds) == 3  # starts 0, 4, 8

    for k, start in enumerate([0, 4, 8]):
        pixel_values, label = ds[k]
        expected_pixels = preprocess_window(
            [_solid(start + t) for t in range(NUM_FRAMES)], NUM_FRAMES, IMAGE_SIZE
        ).squeeze(0)
        assert torch.allclose(pixel_values, expected_pixels), (
            f"window {k} returned the wrong frames"
        )
        assert label[0].item() == pytest.approx(_expected_label(start, 11.0))
        assert label[1].item() == pytest.approx(_expected_label(start, 11.0))


def test_label_is_the_mean_not_the_first_or_last_frame(tmp_path):
    """Guards the specific degenerate implementations that still produce a
    plausible-looking (2,) label in [-1, 1]: taking va[idx[0]] alone, or the
    window's final frame, instead of the mean across the window."""
    labelled = _labelled(tmp_path, ["id0/clip0"], n_frames=8)
    ds = _dataset(labelled, hi=7.0)

    _, label = ds[0]  # frames 0..3, raw values 0,1,2,3 -> mean 1.5
    mean_of_window = 2.0 * 1.5 / 7.0 - 1.0
    first_frame_only = 2.0 * 0.0 / 7.0 - 1.0
    last_frame_only = 2.0 * 3.0 / 7.0 - 1.0

    assert label[0].item() == pytest.approx(mean_of_window)
    assert label[0].item() != pytest.approx(first_frame_only)
    assert label[0].item() != pytest.approx(last_frame_only)


def test_windows_never_run_past_a_shorter_label_array(tmp_path):
    """`n = min(len(frame_files), va.shape[0])` is load-bearing: with more
    frames than label rows, an unclamped window count would emit a window whose
    label slice is short, so `.mean(axis=0)` would silently average fewer rows
    than the window actually covers rather than failing."""
    labelled = _labelled(tmp_path, ["id0/clip0"], n_frames=10, n_labels=6)
    ds = _dataset(labelled, hi=9.0)

    # n = min(10, 6) = 6 -> starts = range(0, 6-4+1, 4) = [0]. Unclamped it
    # would be range(0, 10-4+1, 4) = [0, 4], and window 1 would average
    # va[4:8] -- only 2 of the 4 rows it claims to cover.
    assert len(ds) == 1
    _, label = ds[0]
    assert label[0].item() == pytest.approx(_expected_label(0, 9.0))


def test_va_is_harmonised_from_the_declared_source_range(tmp_path):
    """The dataset must apply the manifest's va_range, not assume [-1, 1] input.
    Same raw frame values under a wider declared range must yield smaller
    magnitudes -- an ignored va_range would return identical labels."""
    labelled = _labelled(tmp_path, ["id0/clip0"], n_frames=8)

    narrow = _dataset(labelled, hi=7.0)[0][1]
    wide = _dataset(labelled, hi=70.0)[0][1]

    assert narrow[0].item() == pytest.approx(2.0 * 1.5 / 7.0 - 1.0)
    assert wide[0].item() == pytest.approx(2.0 * 1.5 / 70.0 - 1.0)
    assert abs(wide[0].item()) > abs(narrow[0].item())
    assert -1.0 <= narrow[0].item() <= 1.0
    assert -1.0 <= wide[0].item() <= 1.0


def test_overlap_shifts_window_starts_by_the_expected_step(tmp_path):
    labelled = _labelled(tmp_path, ["id0/clip0"], n_frames=12)

    ds = _dataset(labelled, overlap=0.5, hi=11.0)

    # step = round(4 * (1 - 0.5)) = 2 -> starts = range(0, 9, 2) = 0,2,4,6,8
    assert len(ds) == 5
    for k, start in enumerate([0, 2, 4, 6, 8]):
        assert ds[k][1][0].item() == pytest.approx(_expected_label(start, 11.0))


def test_windows_from_every_clip_are_flattened_into_one_index_space(tmp_path):
    """Two clips of 8 frames give 2 windows each; the dataset must expose all
    four as a flat 0..3 index, not nest per clip."""
    labelled = _labelled(tmp_path, ["id0/clip0", "id1/clip1"], n_frames=8)

    ds = _dataset(labelled, hi=7.0)

    assert len(ds) == 4
    starts = [0, 4, 0, 4]
    for k, start in enumerate(starts):
        assert ds[k][1][0].item() == pytest.approx(_expected_label(start, 7.0))


def test_a_malformed_clip_is_skipped_and_the_good_clips_survive(tmp_path):
    """A manifest can name a clip whose frames failed to extract. That must cost
    one clip, not the whole training set."""
    labelled = _labelled(tmp_path, ["id0/good"], n_frames=8)
    broken_dir = tmp_path / "id1/broken"
    broken_dir.mkdir(parents=True)  # exists but holds no image files
    write_label_csv(tmp_path / "id1/broken.csv", [(0.0, 0.0)] * 8)
    labelled.records.append(
        ManifestRecord(
            identity_id="id1",
            clip_path="id1/broken",
            duration_s=8.0,
            role="probe",
            label_path="id1/broken.csv",
        )
    )

    ds = _dataset(labelled, hi=7.0)

    assert len(ds) == 2  # only the good clip's two windows
    assert ds[0][1][0].item() == pytest.approx(_expected_label(0, 7.0))


def test_every_clip_malformed_raises_rather_than_yielding_an_empty_dataset(tmp_path):
    """An empty VAWindowDataset would train for zero steps and report a
    finished run, so construction must fail loudly instead."""
    (tmp_path / "id0/broken").mkdir(parents=True)
    write_label_csv(tmp_path / "id0/broken.csv", [(0.0, 0.0)] * 8)
    labelled = LabelledDataset(
        records=[
            ManifestRecord(
                identity_id="id0",
                clip_path="id0/broken",
                duration_s=8.0,
                role="probe",
                label_path="id0/broken.csv",
            )
        ],
        window_cfg=OmegaConf.create({}),
        root=tmp_path,
    )

    with pytest.raises(ValueError, match="every manifest record failed to load"):
        _dataset(labelled, hi=7.0)


def test_empty_manifest_raises_a_distinct_message(tmp_path):
    """'manifest is empty' and 'every record failed' need different messages --
    one means the acquisition script never ran, the other means the data on
    disk is broken, and they call for different fixes."""
    labelled = LabelledDataset(
        records=[], window_cfg=OmegaConf.create({}), root=tmp_path
    )

    with pytest.raises(ValueError, match="manifest is empty"):
        _dataset(labelled, hi=7.0)


def test_clip_shorter_than_one_window_contributes_nothing(tmp_path):
    """3 frames with num_frames=4 yields no window; the clip is dropped without
    an error, but a dataset of only such clips must still raise."""
    labelled = _labelled(tmp_path, ["id0/short"], n_frames=3)

    with pytest.raises(ValueError, match="manifest is empty|failed to load"):
        _dataset(labelled, hi=2.0)


def test_getitem_returns_model_ready_shapes_and_dtypes(tmp_path):
    """`train.py` feeds these straight to the encoder and to ccc_loss: pixels
    must be (C, T, H, W) with the batch dim squeezed off, and the label a
    float32 (2,) -- not (1, 2), which would broadcast wrongly in the loss."""
    labelled = _labelled(tmp_path, ["id0/clip0"], n_frames=8)

    pixel_values, label = _dataset(labelled, hi=7.0)[0]

    assert pixel_values.shape == (3, NUM_FRAMES, IMAGE_SIZE, IMAGE_SIZE)
    assert pixel_values.dtype == torch.float32
    assert label.shape == (2,)
    assert label.dtype == torch.float32
