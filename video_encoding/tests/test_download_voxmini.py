# video_encoding/tests/test_download_voxmini.py
from pathlib import Path

import pytest

from video_encoding.scripts.download_voxmini import (
    MissingCredentialsError,
    load_parts,
    md5sum,
    require_credentials,
)

pytestmark = pytest.mark.integration


def test_require_credentials_raises_when_unset(monkeypatch):
    monkeypatch.delenv("VOXCELEB_USER", raising=False)
    monkeypatch.delenv("VOXCELEB_PASS", raising=False)

    with pytest.raises(MissingCredentialsError, match="must both be set"):
        require_credentials()


def test_require_credentials_raises_when_only_one_set(monkeypatch):
    monkeypatch.setenv("VOXCELEB_USER", "alice")
    monkeypatch.delenv("VOXCELEB_PASS", raising=False)

    with pytest.raises(MissingCredentialsError, match="must both be set"):
        require_credentials()


def test_require_credentials_returns_values_when_set(monkeypatch):
    monkeypatch.setenv("VOXCELEB_USER", "alice")
    monkeypatch.setenv("VOXCELEB_PASS", "secret")

    user, password = require_credentials()

    assert (user, password) == ("alice", "secret")


def test_md5sum_matches_known_value(tmp_path: Path):
    file_path = tmp_path / "sample.txt"
    file_path.write_bytes(b"hello world")

    assert md5sum(file_path) == "5eb63bbbe01eeed093cb22bb8f5acdc3"


def test_load_parts_rejects_empty_parts_csv(tmp_path: Path):
    parts_csv = tmp_path / "parts.csv"
    parts_csv.write_text("filename,url,md5\n")

    with pytest.raises(ValueError, match="no part rows"):
        load_parts(parts_csv)


def test_load_parts_reads_rows(tmp_path: Path):
    parts_csv = tmp_path / "parts.csv"
    parts_csv.write_text(
        "filename,url,md5\n"
        "vox2_dev_mp4_partaa,https://example.invalid/vox2_dev_mp4_partaa,abc123\n"
    )

    parts = load_parts(parts_csv)

    assert parts == [
        {
            "filename": "vox2_dev_mp4_partaa",
            "url": "https://example.invalid/vox2_dev_mp4_partaa",
            "md5": "abc123",
        }
    ]
