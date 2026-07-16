"""Decode-once JPEG frame cache for VoxCeleb2 SSL pretraining.

Per the transfer plan (Workstream A4b): mp4 decode is CPU-bound and would
otherwise repeat on every SSL epoch, on every node. Pay the decode cost once
here, writing subsampled (default 8fps) 224px JPEG frames in the same
`list_frame_files`/`read_frames_at` layout `frame_decoder.py` already reads
for VEATIC -- so the SSL window dataset can read VoxCeleb2 the same way,
mp4-free.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable

from data.manifest import read_manifest

_TranscodeJob = tuple[Path, Path]
_TranscodeResult = str | None
_RunTranscode = Callable[[list[_TranscodeJob]], list[_TranscodeResult]]


def compute_frame_step(native_fps: float, target_fps: float) -> int:
    """How many native frames to advance between kept frames.

    Never 0 (a clip with `native_fps < target_fps` still keeps every frame,
    it just can't be *upsampled* to `target_fps`).
    """
    if native_fps <= 0:
        return 1
    return max(1, round(native_fps / target_fps))


def transcode_clip(
    clip_path: Path,
    out_dir: Path,
    target_fps: float = 8.0,
    size: int = 224,
    quality: int = 90,
) -> int:
    """Decode `clip_path`, keep every `compute_frame_step`-th frame resized to
    `size`x`size`, write as `out_dir/0000.jpg, 0001.jpg, ...`. Returns the
    number of frames written."""
    import av
    from PIL import Image

    with av.open(str(clip_path)) as container:
        stream = container.streams.video[0]
        native_fps = float(stream.average_rate) if stream.average_rate else target_fps
        step = compute_frame_step(native_fps, target_fps)

        out_dir.mkdir(parents=True, exist_ok=True)
        n_written = 0
        for i, frame in enumerate(container.decode(stream)):
            if i % step != 0:
                continue
            img = Image.fromarray(frame.to_ndarray(format="rgb24")).resize(
                (size, size), Image.BILINEAR
            )
            img.save(out_dir / f"{n_written:04d}.jpg", quality=quality)
            n_written += 1
    return n_written


def is_already_cached(out_dir: Path) -> bool:
    """Resumability check: a clip is done if its frame dir exists and has
    at least one frame (an interrupted run leaves an empty/partial dir, which
    does not count as done)."""
    return out_dir.is_dir() and any(out_dir.iterdir())


def _safe_transcode(
    clip_path: Path,
    out_dir: Path,
    target_fps: float,
    size: int,
    quality: int,
) -> str | None:
    """Wrap `transcode_clip` so one corrupt/unreadable clip (a real occurrence
    in the VoxCeleb2 corpus) doesn't crash the whole caching run. Returns an
    error string on failure, `None` on success -- mirrors the
    `select_voxmini_subset._safe_probe` skip-don't-crash pattern."""
    try:
        transcode_clip(
            clip_path, out_dir, target_fps=target_fps, size=size, quality=quality
        )
        return None
    except Exception as exc:  # noqa: BLE001 -- best-effort over an untrusted external corpus
        return f"{clip_path}: {exc}"


def _default_run_transcode(
    jobs: list[_TranscodeJob],
    target_fps: float,
    size: int,
    quality: int,
    max_workers: int,
) -> list[_TranscodeResult]:
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(_safe_transcode, clip_path, out_dir, target_fps, size, quality)
            for clip_path, out_dir in jobs
        ]
        return [f.result() for f in futures]


def cache_frames(
    manifest_path: Path,
    raw_root: Path,
    frames_root: Path,
    target_fps: float = 8.0,
    size: int = 224,
    quality: int = 90,
    max_workers: int = 8,
    run_transcode: _RunTranscode | None = None,
) -> tuple[int, int]:
    """Transcode every clip in `manifest_path` from `raw_root` (mp4s) to
    `frames_root` (JPEG frame dirs), skipping clips already cached. Returns
    `(n_transcoded, n_skipped_as_unreadable)`.

    `run_transcode` is injectable (defaults to a real `ProcessPoolExecutor`
    run) so tests can exercise the resumability/failure-counting logic with a
    synchronous stand-in instead of real multiprocessing + video files --
    mirrors `select_voxmini_subset`'s injectable-prober pattern.
    """
    records = read_manifest(manifest_path, expected_role="ssl")

    todo: list[_TranscodeJob] = []
    for record in records:
        clip_path = raw_root / record.clip_path
        out_dir = frames_root / Path(record.clip_path).with_suffix("")
        if not is_already_cached(out_dir):
            todo.append((clip_path, out_dir))

    if run_transcode is None:
        errors = _default_run_transcode(todo, target_fps, size, quality, max_workers)
    else:
        errors = run_transcode(todo)

    n_failed = 0
    for error in errors:
        if error is not None:
            n_failed += 1
            print(f"skipping unreadable clip {error}", file=sys.stderr)

    return len(todo) - n_failed, n_failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help="mp4 root the manifest's clip_path is relative to",
    )
    parser.add_argument(
        "--frames-root", type=Path, required=True, help="output JPEG-cache root"
    )
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    n_ok, n_failed = cache_frames(
        args.manifest,
        args.raw_root,
        args.frames_root,
        target_fps=args.fps,
        size=args.size,
        quality=args.quality,
        max_workers=args.workers,
    )
    print(
        f"cached {n_ok} clips ({n_failed} skipped as unreadable) to {args.frames_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
