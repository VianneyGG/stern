"""Download VoxCeleb2 dev-set zip parts from the official VGG server.

Modeled on clovaai/voxceleb_trainer's dataprep.py download approach: read a
declarative list of (filename, url, md5) parts, fetch each with HTTP Basic
Auth, verify MD5, concatenate, unzip. The actual part URLs/checksums are NOT
hardcoded here -- they live in conf/voxceleb2_parts.csv, which ships
header-only. TODO(human): after the VGG license-agreement form is approved,
fill in one row per dev-set zip part (filename, download URL, MD5) from the
official VoxCeleb download page before running this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import zipfile
from pathlib import Path

import requests


class MissingCredentialsError(RuntimeError):
    pass


def require_credentials() -> tuple[str, str]:
    """Read VOXCELEB_USER/VOXCELEB_PASS from env. Raises before any network call
    if either is unset, so a misconfigured run fails immediately, not after
    partially downloading hundreds of GB."""
    user = os.environ.get("VOXCELEB_USER")
    password = os.environ.get("VOXCELEB_PASS")
    if not user or not password:
        raise MissingCredentialsError(
            "VOXCELEB_USER and VOXCELEB_PASS must both be set (from the VGG "
            "license-agreement form approval email) before downloading."
        )
    return user, password


def md5sum(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_parts(parts_csv: Path) -> list[dict[str, str]]:
    with parts_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        parts = list(reader)
    if not parts:
        raise ValueError(
            f"{parts_csv} has no part rows -- fill in filename/url/md5 from the "
            "official VGG VoxCeleb download page before running the download."
        )
    return parts


def download_part(url: str, dest: Path, user: str, password: str) -> None:
    with requests.get(url, auth=(user, password), stream=True, timeout=60) as response:
        response.raise_for_status()
        with dest.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)


def download_parts(
    parts: list[dict[str, str]], dest_dir: Path, user: str, password: str
) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for part in parts:
        dest = dest_dir / part["filename"]
        download_part(part["url"], dest, user, password)
        actual_md5 = md5sum(dest)
        if actual_md5 != part["md5"]:
            raise ValueError(
                f"MD5 mismatch for {part['filename']}: expected {part['md5']}, got {actual_md5}"
            )


def concatenate_and_unzip(
    dest_dir: Path, parts: list[dict[str, str]], combined_name: str, extract_to: Path
) -> None:
    combined_path = dest_dir / combined_name
    with combined_path.open("wb") as out:
        for part in parts:
            with (dest_dir / part["filename"]).open("rb") as f:
                out.write(f.read())
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(combined_path) as zf:
        zf.extractall(extract_to)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parts-csv",
        type=Path,
        default=Path(__file__).parent.parent / "conf" / "voxceleb2_parts.csv",
    )
    parser.add_argument("--dest-dir", type=Path, required=True, help="VOXMINI_ROOT/raw")
    parser.add_argument("--combined-name", default="vox2_dev_mp4.zip")
    args = parser.parse_args()

    user, password = require_credentials()
    parts = load_parts(args.parts_csv)
    download_parts(parts, args.dest_dir, user, password)
    concatenate_and_unzip(args.dest_dir, parts, args.combined_name, args.dest_dir)
    print(f"downloaded + extracted {len(parts)} parts to {args.dest_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
