#!/usr/bin/env python
"""Unzip a downloaded VEATIC.zip and extract frames into the STERN data root.

VEATIC (WACV 2024, github.com/AlbusPeter/VEATIC) ships as a single Google Drive
file, `VEATIC.zip` -- no signed EULA, but no scriptable/anonymous download URL
either:
    https://drive.google.com/file/d/1HZIw8RGsRwwENhJlhNJRL88YyfiE442N/view
Download it by hand, then pass the .zip path to this script.

This reimplements the repo's own `video_frame.py` (which uses cv2) with `av`
(already a STERN dependency) so we don't add opencv just for frame extraction.
Frame files are written unpadded (`0.jpg, 1.jpg, ..., 10.jpg`, matching the
original tool) -- `frame_decoder.list_frame_files` sorts these numerically.

Defaults to `.data/veatic` (see `.gitignore`'s "local bulk dataset downloads"
entry) for local work. On the login-node/SallesInfo cluster, override with
`VEATIC_ROOT=/Data/$USER/veatic` -- bulk data belongs on per-node `/Data`
there, never the shared NFS mount.

Usage (local):
    uv run python stern/prep/fetch_veatic.py --zip ~/Downloads/VEATIC.zip

Usage (cluster):
    VEATIC_ROOT=/Data/$USER/veatic \\
      uv run python stern/prep/fetch_veatic.py \\
        --zip ~/Downloads/VEATIC.zip --dest $VEATIC_ROOT

Then build the manifest (PYTHONPATH=. is needed for build_veatic_manifest.py's
`stern.data.manifest` import; this fetch script doesn't need it):
    PYTHONPATH=. uv run python stern/prep/build_veatic_manifest.py --veatic-root $VEATIC_ROOT
"""

from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from pathlib import Path

import av


def extract_frames(video_path: Path, dest_dir: Path) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frame.to_image().save(dest_dir / f"{n}.jpg")
            n += 1
    return n


def _find_dir(root: Path, *names: str) -> Path:
    """Locate a directory matching any of `names` under `root`, at any depth.

    The zip's actual top-level layout is `videos/` (plural) + `rating_averaged/`
    -- the README's tree names it `video/` (singular), confirmed wrong against
    the real downloaded VEATIC.zip this session. Accept both spellings and
    search subdirectories too, rather than assume either is exact.
    """
    for name in names:
        if (root / name).is_dir():
            return root / name
    matches = [p for name in names for p in root.rglob(name) if p.is_dir()]
    if not matches:
        raise ValueError(f"none of {names} found as a directory under {root}")
    return matches[0]


def build(zip_path: Path, dest: Path) -> None:
    unzip_dir = dest / "_unzipped"
    unzip_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(unzip_dir)

    video_dir = _find_dir(unzip_dir, "videos", "video")
    ratings_dir = _find_dir(unzip_dir, "rating_averaged")

    frame_root = dest / "frame"
    for video_path in sorted(video_dir.glob("*.mp4")):
        video_id = video_path.stem
        n = extract_frames(video_path, frame_root / video_id)
        print(f"{video_id}: extracted {n} frames")

    dest_ratings = dest / "rating_averaged"
    dest_ratings.mkdir(parents=True, exist_ok=True)
    for csv_path in ratings_dir.glob("*.csv"):
        shutil.copy2(csv_path, dest_ratings / csv_path.name)

    shutil.rmtree(unzip_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True, help="path to VEATIC.zip")
    parser.add_argument(
        "--dest", type=Path, default=Path(os.environ.get("VEATIC_ROOT", ".data/veatic"))
    )
    args = parser.parse_args()

    build(args.zip, args.dest)
    print(
        "next: PYTHONPATH=. uv run python stern/prep/build_veatic_manifest.py "
        f"--veatic-root {args.dest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
