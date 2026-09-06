"""Tests for video_encoding.data.shard_stream: shard-major HTTP streaming
for molab notebooks. Uses a real http.server.ThreadingHTTPServer fixture
serving hand-built shard tars (same pickle-per-clip format as
test_wds_frames.py's _make_shard helper)."""

from __future__ import annotations

import io
import json
import pickle
import tarfile
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from video_encoding.data.shard_stream import StreamingShardDataset, fetch_shard


def _make_shard(shard_path: Path, clips: dict[str, list[np.ndarray]]) -> None:
    with tarfile.open(shard_path, "w") as tar:
        for clip_key, frames in clips.items():
            jpegs = []
            for frame in frames:
                buf = io.BytesIO()
                Image.fromarray(frame).save(buf, format="JPEG")
                jpegs.append(buf.getvalue())
            payload = pickle.dumps(jpegs)
            info = tarfile.TarInfo(name=f"{clip_key}.pickle")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))


def _serve(root: Path) -> ThreadingHTTPServer:
    handler = partial(SimpleHTTPRequestHandler, directory=str(root))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _base_url(server: ThreadingHTTPServer) -> str:
    return f"http://127.0.0.1:{server.server_port}"


def _build_corpus(root: Path, num_shards: int, clips_per_shard: int = 1) -> dict:
    manifest: dict[str, dict] = {}
    for s in range(num_shards):
        shard_name = f"shard-{s:06d}.tar"
        clips = {}
        for c in range(clips_per_shard):
            clip_key = f"id{s}/vid{c}/clip{c}"
            frames = [
                np.full((8, 8, 3), (s * 10 + c * 3 + i) % 255, dtype=np.uint8)
                for i in range(4)
            ]
            clips[clip_key] = frames
            manifest[clip_key] = {"shard": shard_name, "num_frames": 4, "size": 4}
        _make_shard(root / shard_name, clips)
    (root / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_fetch_shard_downloads_then_reuses_cache(tmp_path):
    origin = tmp_path / "origin"
    origin.mkdir()
    _build_corpus(origin, num_shards=1)
    server = _serve(origin)
    try:
        cache_dir = tmp_path / "cache"
        first = fetch_shard(
            "shard-000000.tar", base_url=_base_url(server), cache_dir=cache_dir
        )
        assert first.exists()
        expected_bytes = (origin / "shard-000000.tar").read_bytes()
        assert first.read_bytes() == expected_bytes

        second = fetch_shard(
            "shard-000000.tar", base_url=_base_url(server), cache_dir=cache_dir
        )
        assert second == first
    finally:
        server.shutdown()


def test_fetch_shard_raises_on_truncated_transfer(tmp_path, monkeypatch):
    origin = tmp_path / "origin"
    origin.mkdir()
    _build_corpus(origin, num_shards=1)
    server = _serve(origin)
    try:
        cache_dir = tmp_path / "cache"

        class _TruncatedResponse:
            headers = {"Content-Length": "999999999"}

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def raise_for_status(self):
                pass

            def iter_content(self, chunk_size):
                yield b"only a few bytes"

        monkeypatch.setattr(
            "video_encoding.data.shard_stream.requests.get",
            lambda *a, **k: _TruncatedResponse(),
        )

        with pytest.raises(IOError):
            fetch_shard(
                "shard-000000.tar", base_url=_base_url(server), cache_dir=cache_dir
            )

        assert not (cache_dir / "shard-000000.tar").exists()
        assert list(cache_dir.glob(".*")) == []
    finally:
        server.shutdown()


def test_streaming_dataset_evicts_shard_after_consuming_it(tmp_path):
    origin = tmp_path / "origin"
    origin.mkdir()
    manifest = _build_corpus(origin, num_shards=2, clips_per_shard=1)
    server = _serve(origin)
    try:
        cache_dir = tmp_path / "cache"
        dataset = StreamingShardDataset(
            base_url=_base_url(server),
            cache_dir=cache_dir,
            num_frames=4,
            image_size=8,
            overlap=0.0,
        )
        windows = list(dataset)
        assert len(windows) == len(manifest)
        assert list(cache_dir.glob("*.tar")) == []
    finally:
        server.shutdown()


def test_streaming_dataset_never_holds_more_than_two_shards(tmp_path):
    origin = tmp_path / "origin"
    origin.mkdir()
    _build_corpus(origin, num_shards=4, clips_per_shard=1)
    server = _serve(origin)
    try:
        cache_dir = tmp_path / "cache"
        dataset = StreamingShardDataset(
            base_url=_base_url(server),
            cache_dir=cache_dir,
            num_frames=4,
            image_size=8,
            overlap=0.0,
        )
        max_resident = 0
        for _ in dataset:
            max_resident = max(max_resident, len(list(cache_dir.glob("*.tar"))))
        assert max_resident <= 2
    finally:
        server.shutdown()
