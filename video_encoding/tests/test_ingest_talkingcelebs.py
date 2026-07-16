from pathlib import Path

import pytest

from video_encoding.scripts.ingest_talkingcelebs import list_clips


def test_list_clips_walks_split_identity_video_hierarchy(tmp_path: Path):
    (tmp_path / "train" / "id00000" / "vid001").mkdir(parents=True)
    (tmp_path / "train" / "id00000" / "vid001" / "001.mp4").touch()
    (tmp_path / "train" / "id00001" / "vid002").mkdir(parents=True)
    (tmp_path / "train" / "id00001" / "vid002" / "001.mp4").touch()

    result = list_clips(tmp_path, probe_duration_s=lambda _: 30.0)

    assert set(result.keys()) == {"id00000", "id00001"}
    assert len(result["id00000"]) == 1
    assert result["id00000"][0][1] == 30.0


def test_list_clips_aggregates_clips_across_splits(tmp_path: Path):
    for split in ["train", "test"]:
        (tmp_path / split / "id00000" / "vid001").mkdir(parents=True)
        (tmp_path / split / "id00000" / "vid001" / "001.mp4").touch()

    result = list_clips(tmp_path, probe_duration_s=lambda _: 30.0)

    assert len(result["id00000"]) == 2


def test_list_clips_path_is_relative_to_root(tmp_path: Path):
    (tmp_path / "train" / "id00000" / "vid001").mkdir(parents=True)
    (tmp_path / "train" / "id00000" / "vid001" / "001.mp4").touch()

    result = list_clips(tmp_path, probe_duration_s=lambda _: 30.0)

    expected = str(Path("train") / "id00000" / "vid001" / "001.mp4")
    assert result["id00000"][0][0] == expected


def test_list_clips_ignores_non_mp4_files(tmp_path: Path):
    (tmp_path / "train" / "id00000" / "vid001").mkdir(parents=True)
    (tmp_path / "train" / "id00000" / "vid001" / "001.mp4").touch()
    (tmp_path / "train" / "id00000" / "vid001" / "README.txt").touch()

    result = list_clips(tmp_path, probe_duration_s=lambda _: 30.0)

    assert len(result["id00000"]) == 1


def test_list_clips_uses_injected_duration_prober(tmp_path: Path):
    (tmp_path / "train" / "id00000" / "vid001").mkdir(parents=True)
    clip = tmp_path / "train" / "id00000" / "vid001" / "001.mp4"
    clip.touch()
    probed = []

    def prober(p: Path) -> float:
        probed.append(p)
        return 29.97

    result = list_clips(tmp_path, probe_duration_s=prober)

    assert probed == [clip]
    assert result["id00000"][0][1] == 29.97


def test_list_clips_returns_empty_for_empty_root(tmp_path: Path):
    result = list_clips(tmp_path, probe_duration_s=lambda _: 30.0)
    assert result == {}
