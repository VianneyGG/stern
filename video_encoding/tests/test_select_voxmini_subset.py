# video_encoding/tests/test_select_voxmini_subset.py
from pathlib import Path

import pytest

from video_encoding.scripts.select_voxmini_subset import (
    Identity,
    list_clips,
    load_identities,
    select_stratified_subset,
)


def test_list_clips_walks_identity_directories_and_uses_injected_prober(tmp_path: Path):
    (tmp_path / "id00012" / "vid001").mkdir(parents=True)
    (tmp_path / "id00012" / "vid001" / "00001.mp4").touch()
    (tmp_path / "id00012" / "vid001" / "00002.mp4").touch()
    (tmp_path / "id00013" / "vid002").mkdir(parents=True)
    (tmp_path / "id00013" / "vid002" / "00001.mp4").touch()

    result = list_clips(tmp_path, probe_duration_s=lambda path: 3.5)

    assert len(result["id00012"]) == 2
    assert len(result["id00013"]) == 1
    assert all(duration == 3.5 for clips in result.values() for _, duration in clips)


def test_load_identities_joins_meta_and_clips(tmp_path: Path):
    meta_csv = tmp_path / "vox2_meta.csv"
    meta_csv.write_text(
        "VoxCeleb2 ID,VGGFace2 ID,Gender,Set\n"
        "id00012,n000001,m,dev\n"
        "id00013,n000002,f,dev\n"
    )
    raw_root = tmp_path / "raw"
    (raw_root / "id00012" / "vid001").mkdir(parents=True)
    (raw_root / "id00012" / "vid001" / "00001.mp4").touch()
    (raw_root / "id00013" / "vid002").mkdir(parents=True)
    (raw_root / "id00013" / "vid002" / "00001.mp4").touch()

    identities = load_identities(meta_csv, raw_root, probe_duration_s=lambda path: 5.0)

    by_id = {identity.identity_id: identity for identity in identities}
    assert by_id["id00012"].gender == "m"
    assert by_id["id00013"].gender == "f"
    assert by_id["id00012"].clips[0][1] == 5.0


def test_load_identities_skips_clips_with_no_metadata(tmp_path: Path):
    meta_csv = tmp_path / "vox2_meta.csv"
    meta_csv.write_text("VoxCeleb2 ID,VGGFace2 ID,Gender,Set\nid00012,n000001,m,dev\n")
    raw_root = tmp_path / "raw"
    (raw_root / "id00012" / "vid001").mkdir(parents=True)
    (raw_root / "id00012" / "vid001" / "00001.mp4").touch()
    (raw_root / "id99999" / "vid999").mkdir(parents=True)  # not in meta_csv
    (raw_root / "id99999" / "vid999" / "00001.mp4").touch()

    identities = load_identities(meta_csv, raw_root, probe_duration_s=lambda path: 5.0)

    assert [i.identity_id for i in identities] == ["id00012"]


def _make_identity(
    identity_id: str, gender: str, n_clips: int, clip_duration_s: float
) -> Identity:
    clips = [
        (f"{identity_id}/clip{i:03d}.mp4", clip_duration_s) for i in range(n_clips)
    ]
    return Identity(identity_id=identity_id, gender=gender, clips=clips)


def test_select_stratified_subset_hits_target_hours_within_tolerance():
    identities = [
        _make_identity(f"m{i}", "m", n_clips=10, clip_duration_s=600.0)
        for i in range(50)
    ] + [
        _make_identity(f"f{i}", "f", n_clips=10, clip_duration_s=600.0)
        for i in range(50)
    ]
    # corpus: 100 identities * 10 clips * 600s = 166.67h total

    selected = select_stratified_subset(identities, target_hours=100.0, seed=42)

    total_s = sum(i.total_duration_s for i in selected)
    assert (
        100.0 * 3600.0 <= total_s < 100.0 * 3600.0 + 6000.0
    )  # within one identity's worth (10 clips * 600s)


def test_select_stratified_subset_preserves_gender_ratio():
    identities = [
        _make_identity(f"m{i}", "m", n_clips=10, clip_duration_s=600.0)
        for i in range(80)
    ] + [
        _make_identity(f"f{i}", "f", n_clips=10, clip_duration_s=600.0)
        for i in range(20)
    ]
    # corpus is 80% male / 20% female by duration

    selected = select_stratified_subset(identities, target_hours=50.0, seed=42)

    male_s = sum(i.total_duration_s for i in selected if i.gender == "m")
    female_s = sum(i.total_duration_s for i in selected if i.gender == "f")
    male_fraction = male_s / (male_s + female_s)
    assert 0.75 <= male_fraction <= 0.85


def test_select_stratified_subset_is_deterministic_given_seed():
    identities = [
        _make_identity(
            f"id{i}", "m" if i % 2 == 0 else "f", n_clips=5, clip_duration_s=30.0
        )
        for i in range(40)
    ]

    first = select_stratified_subset(identities, target_hours=1.0, seed=7)
    second = select_stratified_subset(identities, target_hours=1.0, seed=7)

    assert [i.identity_id for i in first] == [i.identity_id for i in second]


def test_select_stratified_subset_keeps_each_identitys_clips_together():
    identities = [
        _make_identity(f"id{i}", "m", n_clips=5, clip_duration_s=30.0)
        for i in range(20)
    ]

    selected = select_stratified_subset(identities, target_hours=0.5, seed=1)

    for identity in selected:
        assert len(identity.clips) == 5  # never a partial identity


def test_select_stratified_subset_raises_when_corpus_too_small():
    identities = [_make_identity("id0", "m", n_clips=1, clip_duration_s=60.0)]

    with pytest.raises(ValueError, match="cannot reach target_hours"):
        select_stratified_subset(identities, target_hours=218.0, seed=42)
