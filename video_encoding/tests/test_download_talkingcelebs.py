from pathlib import Path

import pytest
import yaml

from video_encoding.scripts.download_talkingcelebs import ClipSpec, limit_specs, parse_sources


def _spec(identity_id: str, clip_num: int) -> ClipSpec:
    return ClipSpec(
        identity_id=identity_id,
        video_id=f"vid_{identity_id}",
        clip_num=clip_num,
        start="00:00:00",
        end="00:00:30",
        split="train",
    )


def _write_sources(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "sources.yaml"
    p.write_text(yaml.dump(data))
    return p


def _minimal_sources(*, splits: list[str] | None = None) -> dict:
    splits = splits or ["train"]
    return {
        "subdatasets": [
            {
                "name": split,
                "identities": [
                    {
                        "id": 0,
                        "name": "Obama",
                        "videos": [
                            {
                                "id": "ABC123DEFGH",
                                "clips": [
                                    {"id": 1, "start": "00:00:30", "end": "00:01:00"},
                                    {"id": 2, "start": "00:01:00", "end": "00:01:30"},
                                ],
                            }
                        ],
                    }
                ],
            }
            for split in splits
        ]
    }


def test_parse_sources_returns_one_spec_per_clip(tmp_path: Path):
    sources_path = _write_sources(tmp_path, _minimal_sources())

    specs = parse_sources(sources_path)

    assert len(specs) == 2


def test_parse_sources_formats_identity_id_with_zero_padding(tmp_path: Path):
    data = {
        "subdatasets": [
            {
                "name": "train",
                "identities": [
                    {
                        "id": 42,
                        "name": "Test",
                        "videos": [
                            {"id": "XYZ", "clips": [{"id": 1, "start": "00:00:00", "end": "00:00:30"}]},
                        ],
                    }
                ],
            }
        ]
    }
    sources_path = _write_sources(tmp_path, data)

    specs = parse_sources(sources_path)

    assert specs[0].identity_id == "id00042"


def test_parse_sources_sets_correct_fields(tmp_path: Path):
    sources_path = _write_sources(tmp_path, _minimal_sources())

    specs = parse_sources(sources_path)

    assert specs[0] == ClipSpec(
        identity_id="id00000",
        video_id="ABC123DEFGH",
        clip_num=1,
        start="00:00:30",
        end="00:01:00",
        split="train",
    )


def test_parse_sources_handles_multiple_splits(tmp_path: Path):
    sources_path = _write_sources(tmp_path, _minimal_sources(splits=["train", "test"]))

    specs = parse_sources(sources_path)

    assert {s.split for s in specs} == {"train", "test"}
    assert len(specs) == 4  # 2 clips × 2 splits


def test_parse_sources_handles_multiple_identities(tmp_path: Path):
    data = {
        "subdatasets": [
            {
                "name": "train",
                "identities": [
                    {
                        "id": i,
                        "name": f"Person{i}",
                        "videos": [
                            {"id": f"VID{i}", "clips": [{"id": 1, "start": "00:00:00", "end": "00:00:30"}]},
                        ],
                    }
                    for i in range(5)
                ],
            }
        ]
    }
    sources_path = _write_sources(tmp_path, data)

    specs = parse_sources(sources_path)

    identity_ids = [s.identity_id for s in specs]
    assert identity_ids == [f"id{i:05d}" for i in range(5)]


def test_limit_specs_returns_all_when_no_caps_given():
    specs = [_spec("id0", 1), _spec("id1", 1)]

    result = limit_specs(specs)

    assert result == specs


def test_limit_specs_caps_number_of_identities():
    specs = [_spec("id0", 1), _spec("id1", 1), _spec("id2", 1)]

    result = limit_specs(specs, max_identities=2)

    assert {s.identity_id for s in result} == {"id0", "id1"}


def test_limit_specs_caps_clips_per_identity():
    specs = [_spec("id0", 1), _spec("id0", 2), _spec("id0", 3)]

    result = limit_specs(specs, max_clips_per_identity=2)

    assert len(result) == 2
    assert [s.clip_num for s in result] == [1, 2]


def test_limit_specs_applies_both_caps_together():
    specs = [
        _spec("id0", 1), _spec("id0", 2), _spec("id0", 3),
        _spec("id1", 1), _spec("id1", 2),
        _spec("id2", 1),
    ]

    result = limit_specs(specs, max_identities=2, max_clips_per_identity=2)

    assert {s.identity_id for s in result} == {"id0", "id1"}
    assert len(result) == 4


def test_limit_specs_preserves_first_seen_order():
    specs = [_spec("id0", 1), _spec("id1", 1), _spec("id0", 2)]

    result = limit_specs(specs, max_identities=5, max_clips_per_identity=5)

    assert result == specs
