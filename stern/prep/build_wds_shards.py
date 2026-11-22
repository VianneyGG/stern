"""Decode-once WDS shard cache for VoxCeleb2 SSL pretraining.

Writes 100 size-balanced WebDataset .tar shards. Each member is one clip's
frames at 2 fps, stored as a pickled list of JPEG byte strings (quality 95).
Frames stay compressed on disk, and ssl_windows.py decodes them at read time.

decord seeks straight to the target frame indices rather than decoding every
native frame. It runs CPU-only, so there is no CUDA wheel coupling.
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from stern.data.manifest import read_manifest

_EncodeJob = tuple[Path, str]
_EncodeResult = tuple[list[bytes], str | None]
_RunEncode = Callable[[list[_EncodeJob], float, int], list[_EncodeResult]]


def compute_frame_step(native_fps: float, target_fps: float) -> int:
    """How many native frames to advance between kept frames. Never 0."""
    if native_fps <= 0:
        return 1
    return max(1, round(native_fps / target_fps))


def encode_frames_jpeg(frames: list[np.ndarray], quality: int = 95) -> list[bytes]:
    """JPEG-encode each RGB uint8 frame array. Pillow's stock wheels are
    libjpeg-turbo-backed already; nothing extra to install for this."""
    out = []
    for frame in frames:
        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, format="JPEG", quality=quality, optimize=True)
        out.append(buf.getvalue())
    return out


def decode_clip_frames(
    clip_path: Path, target_fps: float, size: int
) -> list[np.ndarray]:
    """Decode with decord, seeking only to the kept frames, resized to size x size RGB uint8."""
    import decord

    vr = decord.VideoReader(str(clip_path), num_threads=1)
    native_fps = vr.get_avg_fps() or target_fps
    step = compute_frame_step(native_fps, target_fps)
    indices = list(range(0, len(vr), step))
    batch = vr.get_batch(indices).asnumpy()  # (N, H, W, 3) uint8
    return [
        np.array(Image.fromarray(f).resize((size, size), Image.BILINEAR)) for f in batch
    ]


def encode_clip(clip_path: Path, target_fps: float, size: int) -> list[bytes]:
    """Full per-clip pipeline: decode+resize (decord) -> JPEG-encode (Pillow)."""
    frames = decode_clip_frames(clip_path, target_fps, size)
    return encode_frames_jpeg(frames, quality=95)


class BalancedShardWriter:
    """Greedy min-size writer across a fixed number of .tar shards.

    A clip's frames never split across shards. `{out_dir}/manifest.json` records
    what is written, so a restarted run neither re-encodes nor duplicates clips.
    """

    def __init__(self, out_dir: Path, n_shards: int = 100):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.n_shards = n_shards
        self._manifest_path = self.out_dir / "manifest.json"
        self._manifest: dict[str, dict] = {}
        if self._manifest_path.exists():
            self._manifest = json.loads(self._manifest_path.read_text())

        self.sizes = [0] * n_shards
        for entry in self._manifest.values():
            shard_idx = int(entry["shard"].removeprefix("shard-").removesuffix(".tar"))
            self.sizes[shard_idx] += entry["size"]

        self._tars = [
            tarfile.open(
                self._shard_path(i), "a" if self._shard_path(i).exists() else "w"
            )
            for i in range(n_shards)
        ]

    def _shard_path(self, i: int) -> Path:
        return self.out_dir / f"shard-{i:06d}.tar"

    @property
    def already_written(self) -> set[str]:
        return set(self._manifest)

    def write(self, clip_key: str, frame_jpegs: list[bytes]) -> None:
        if clip_key in self._manifest:
            return
        payload = pickle.dumps(frame_jpegs)
        shard_idx = min(range(self.n_shards), key=lambda i: self.sizes[i])

        info = tarfile.TarInfo(name=f"{clip_key}.pickle")
        info.size = len(payload)
        self._tars[shard_idx].addfile(info, io.BytesIO(payload))

        self.sizes[shard_idx] += len(payload)
        self._manifest[clip_key] = {
            "shard": self._shard_path(shard_idx).name,
            "num_frames": len(frame_jpegs),
            "size": len(payload),
        }
        # Persist the manifest in flush(), not here. Re-serialising the whole dict
        # on every write costs more as the run grows. close() flushes too, so a
        # writer used without flush() still ends with a correct manifest.json.

    def flush(self) -> None:
        """Persist the in-memory manifest. Call once per batch of writes, not per write."""
        # Atomic write: a kill mid-write must never leave manifest.json
        # truncated/invalid, since __init__ crashes the whole restart if
        # json.loads() fails on it (losing resumability entirely).
        tmp_path = self._manifest_path.parent / (self._manifest_path.name + ".tmp")
        tmp_path.write_text(json.dumps(self._manifest))
        tmp_path.replace(self._manifest_path)

    def close(self) -> None:
        self.flush()
        for tar in self._tars:
            tar.close()


def _safe_encode(clip_path: Path, target_fps: float, size: int) -> _EncodeResult:
    """Encode one clip; return (frames, None) or ([], error message)."""
    try:
        return encode_clip(clip_path, target_fps, size), None
    except Exception as exc:  # noqa: BLE001 -- best-effort over an untrusted external corpus
        return [], f"{clip_path}: {exc}"


def _default_run_encode(
    jobs: list[_EncodeJob], target_fps: float, size: int, max_workers: int = 8
) -> list[_EncodeResult]:
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(_safe_encode, clip_path, target_fps, size)
            for clip_path, _clip_key in jobs
        ]
        return [f.result() for f in futures]


def build_shards(
    manifest_path: Path,
    raw_root: Path,
    out_dir: Path,
    target_fps: float = 2.0,
    size: int = 224,
    n_shards: int = 100,
    max_workers: int = 8,
    run_encode: _RunEncode | None = None,
    chunk_size: int = 64,
) -> tuple[int, int]:
    """Encode every clip in `manifest_path` from `raw_root` into balanced shards.

    Clips already in the output manifest are skipped, so a killed run resumes.
    Returns (n_encoded, n_skipped_as_unreadable).

    `run_encode` takes a batch of jobs and returns their results. Tests inject a
    synchronous version so they need neither multiprocessing nor video files.

    Work runs in `chunk_size` chunks, and the manifest is flushed after each one.
    That bounds the work lost to a kill and the memory held for encoded frames.
    """
    records = read_manifest(manifest_path, expected_role="ssl")
    writer = BalancedShardWriter(out_dir, n_shards=n_shards)

    todo: list[_EncodeJob] = []
    for record in records:
        clip_key = str(Path(record.clip_path).with_suffix(""))
        if clip_key not in writer.already_written:
            todo.append((raw_root / record.clip_path, clip_key))

    n_failed = 0
    n_ok = 0
    for chunk_start in range(0, len(todo), chunk_size):
        chunk = todo[chunk_start : chunk_start + chunk_size]

        if run_encode is None:
            results = _default_run_encode(chunk, target_fps, size, max_workers)
        else:
            results = run_encode(chunk, target_fps, size)

        for (_clip_path, clip_key), (frame_jpegs, error) in zip(chunk, results):
            if error is not None:
                n_failed += 1
                print(f"skipping unreadable clip {error}", file=sys.stderr)
                continue
            writer.write(clip_key, frame_jpegs)
            n_ok += 1

        writer.flush()

    writer.close()
    return n_ok, n_failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--n-shards", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=64)
    args = parser.parse_args()

    n_ok, n_failed = build_shards(
        args.manifest,
        args.raw_root,
        args.out_dir,
        target_fps=args.fps,
        size=args.size,
        n_shards=args.n_shards,
        max_workers=args.workers,
        chunk_size=args.chunk_size,
    )
    print(
        f"encoded {n_ok} clips ({n_failed} skipped as unreadable) into {args.out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
