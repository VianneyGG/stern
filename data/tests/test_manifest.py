from pathlib import Path

import pytest

from data.manifest import (
    ManifestRecord,
    ManifestRoleMismatchError,
    ManifestSchemaError,
    read_manifest,
    write_manifest,
)


def test_write_then_read_manifest_round_trips(tmp_path: Path):
    records = [
        ManifestRecord(
            identity_id="id00012",
            clip_path="id00012/clip001.mp4",
            duration_s=4.2,
            role="ssl",
        ),
        ManifestRecord(
            identity_id="id00012",
            clip_path="id00012/clip002.mp4",
            duration_s=3.8,
            role="ssl",
        ),
    ]
    manifest_path = tmp_path / "manifest.csv"

    write_manifest(manifest_path, records)
    result = read_manifest(manifest_path, expected_role="ssl")

    assert result == records


def test_read_manifest_preserves_label_path_for_labelled_roles(tmp_path: Path):
    records = [
        ManifestRecord(
            identity_id="vid001",
            clip_path="vid001.mp4",
            duration_s=120.0,
            role="probe",
            label_path="vid001_va.csv",
        ),
    ]
    manifest_path = tmp_path / "manifest.csv"

    write_manifest(manifest_path, records)
    result = read_manifest(manifest_path, expected_role="probe")

    assert result[0].label_path == "vid001_va.csv"


def test_read_manifest_rejects_role_mismatch(tmp_path: Path):
    records = [
        ManifestRecord(identity_id="id1", clip_path="a.mp4", duration_s=1.0, role="ssl")
    ]
    manifest_path = tmp_path / "manifest.csv"
    write_manifest(manifest_path, records)

    with pytest.raises(ManifestRoleMismatchError, match="role="):
        read_manifest(manifest_path, expected_role="probe")


def test_read_manifest_rejects_missing_required_columns(tmp_path: Path):
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("identity_id,clip_path\nid1,a.mp4\n")

    with pytest.raises(ManifestSchemaError, match="missing required"):
        read_manifest(manifest_path, expected_role="ssl")


def test_read_manifest_silently_accepts_duplicate_clip_path_rows(tmp_path: Path):
    """Pins current behavior: `clip_path` has no uniqueness check, so a
    manifest built by concatenating two acquisition runs (or a copy-paste
    error) can silently double-count a clip downstream. Not asserted as
    correct -- if `read_manifest` grows a duplicate-rejection check, update
    this test deliberately rather than treating a failure here as a
    regression."""
    records = [
        ManifestRecord(
            identity_id="id00012",
            clip_path="id00012/clip001.mp4",
            duration_s=4.2,
            role="ssl",
        ),
        ManifestRecord(
            identity_id="id00012",
            clip_path="id00012/clip001.mp4",
            duration_s=4.2,
            role="ssl",
        ),
    ]
    manifest_path = tmp_path / "manifest.csv"
    write_manifest(manifest_path, records)

    result = read_manifest(manifest_path, expected_role="ssl")

    assert len(result) == 2
    assert result[0].clip_path == result[1].clip_path == "id00012/clip001.mp4"
