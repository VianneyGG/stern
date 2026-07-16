"""Download TalkingCelebs video clips from YouTube via the samarrik/talkingcelebs
sources manifest on HuggingFace.

Three phases:
  1. Fetch sources.yaml from HF (public repo, no auth required).
  2. Download each unique source video once via yt-dlp's native downloader
     (cached under dest_dir/.video_cache/<video_id>.mp4, reused across all of
     that video's clips).
  3. Locally trim each clip's time range + transcode to 720p / 25 fps H.264
     via a separate ffmpeg pass over the already-downloaded file.

Phases 2/3 are split (rather than yt-dlp's built-in --download-sections)
because --download-sections hands the network fetch itself to ffmpeg for
keyframe-accurate cutting, which fails on hosts where ffmpeg was built
without TLS support ("https or dtls protocol not found") -- e.g. the
login-node cluster's system ffmpeg. Trimming a local file needs no network
protocol support, so it works regardless.

After this script completes, run ingest_talkingcelebs.py to build the manifest.

Usage:
    TALKINGCELEBS_ROOT=/data/talkingcelebs
    uv run python video_encoding/scripts/download_talkingcelebs.py \\
        --dest-dir $TALKINGCELEBS_ROOT/raw
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import NamedTuple

import yaml

HF_REPO_ID = "samarrik/talkingcelebs"
SOURCES_FILENAME = "sources.yaml"


class ClipSpec(NamedTuple):
    identity_id: str  # "id00000", "id00001", …
    video_id: str     # YouTube video ID
    clip_num: int     # 1-based clip number within the video
    start: str        # "HH:MM:SS"
    end: str          # "HH:MM:SS"
    split: str        # "train" or "test"


def fetch_sources_yaml(dest_dir: Path) -> Path:
    """Download sources.yaml from the HF repo to dest_dir. Returns local path."""
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=SOURCES_FILENAME,
        repo_type="dataset",
        local_dir=dest_dir,
    )
    return Path(local)


def parse_sources(sources_path: Path) -> list[ClipSpec]:
    """Parse sources.yaml into a flat list of ClipSpec."""
    with sources_path.open() as f:
        data = yaml.safe_load(f)

    specs: list[ClipSpec] = []
    for subdataset in data["subdatasets"]:
        split = subdataset["name"]
        for identity in subdataset["identities"]:
            identity_id = f"id{identity['id']:05d}"
            for video in identity["videos"]:
                video_id = video["id"]
                for clip in video["clips"]:
                    specs.append(
                        ClipSpec(
                            identity_id=identity_id,
                            video_id=video_id,
                            clip_num=int(clip["id"]),
                            start=clip["start"],
                            end=clip["end"],
                            split=split,
                        )
                    )
    return specs


def limit_specs(
    specs: list[ClipSpec],
    max_identities: int | None = None,
    max_clips_per_identity: int | None = None,
) -> list[ClipSpec]:
    """Cap to at most `max_identities` distinct identities and at most
    `max_clips_per_identity` clips per identity (both in first-seen order),
    for a small, fast-to-iterate mini subset."""
    if max_identities is None and max_clips_per_identity is None:
        return specs

    seen_identities: set[str] = set()
    clip_count: dict[str, int] = {}
    result: list[ClipSpec] = []
    for spec in specs:
        if spec.identity_id not in seen_identities:
            if max_identities is not None and len(seen_identities) >= max_identities:
                continue
            seen_identities.add(spec.identity_id)
            clip_count[spec.identity_id] = 0
        if max_clips_per_identity is not None and clip_count[spec.identity_id] >= max_clips_per_identity:
            continue
        clip_count[spec.identity_id] += 1
        result.append(spec)
    return result


def download_full_video(video_id: str, cache_dir: Path) -> Path:
    """Download the full source video once via yt-dlp's native downloader
    (no --download-sections -- see module docstring for why). Cached at
    cache_dir/<video_id>.mp4; reused across all of that video's clips."""
    cache_path = cache_dir / f"{video_id}.mp4"
    if cache_path.exists():
        return cache_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-playlist",
        # current YouTube extraction requires a JS runtime for signature
        # decryption; deno isn't installed on the cluster, but node is.
        "--js-runtimes", "node",
        "-f", "best[ext=mp4]/best",
        "-o", str(cache_path),
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed for {url}:\n{result.stderr}")
    return cache_path


def download_clip(spec: ClipSpec, dest_dir: Path, cache_dir: Path) -> Path:
    """Trim `spec`'s time range out of its (cached) full source video and
    transcode to 720p / 25 fps H.264 via a local ffmpeg pass.

    Output: dest_dir/<split>/<identity_id>/<video_id>/<NNN>.mp4
    Skips if the file already exists (safe to re-run after partial failure).
    """
    out_path = (
        dest_dir
        / spec.split
        / spec.identity_id
        / spec.video_id
        / f"{spec.clip_num:03d}.mp4"
    )
    if out_path.exists():
        return out_path

    full_video_path = download_full_video(spec.video_id, cache_dir)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-ss", spec.start,
        "-to", spec.end,
        "-i", str(full_video_path),
        "-vf", "scale=-2:720",
        "-r", "25",
        "-c:v", "libx264",
        "-c:a", "aac",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg trim failed for {spec.video_id} [{spec.start}-{spec.end}]:\n{result.stderr}"
        )
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest-dir", type=Path, required=True,
                        help="root output directory (TALKINGCELEBS_ROOT/raw)")
    parser.add_argument(
        "--splits", nargs="+", default=["train", "test"],
        help="subdatasets to download (default: train test)",
    )
    parser.add_argument(
        "--max-identities", type=int, default=None,
        help="cap on distinct identities, for a small mini subset (default: no cap)",
    )
    parser.add_argument(
        "--max-clips-per-identity", type=int, default=None,
        help="cap on clips per identity, for a small mini subset (default: no cap)",
    )
    args = parser.parse_args()

    args.dest_dir.mkdir(parents=True, exist_ok=True)
    sources_path = fetch_sources_yaml(args.dest_dir)
    specs = parse_sources(sources_path)
    specs = [s for s in specs if s.split in args.splits]
    specs = limit_specs(
        specs,
        max_identities=args.max_identities,
        max_clips_per_identity=args.max_clips_per_identity,
    )

    cache_dir = args.dest_dir / ".video_cache"

    print(f"downloading {len(specs)} clips to {args.dest_dir}...")
    failed: list[ClipSpec] = []
    for i, spec in enumerate(specs, 1):
        try:
            out = download_clip(spec, args.dest_dir, cache_dir)
            print(f"[{i}/{len(specs)}] {out.relative_to(args.dest_dir)}")
        except RuntimeError as exc:
            print(f"[{i}/{len(specs)}] FAILED: {exc}")
            failed.append(spec)

    shutil.rmtree(cache_dir, ignore_errors=True)

    if failed:
        print(f"\n{len(failed)} clips failed.")
        return 1
    print(f"\nall {len(specs)} clips downloaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
