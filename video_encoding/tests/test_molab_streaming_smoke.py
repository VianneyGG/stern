"""End-to-end local proof that a molab-style session can stream SSL
training windows entirely from an HTTP shard source (StreamingShardDataset,
not a local path). Real cloudflared tunnel + molab GPU + SSH push-back are
manual verification -- see the design doc's Open questions #1-3 -- and are
not exercised here."""

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
from PIL import Image

from video_encoding.data.shard_stream import StreamingShardDataset


def _make_shard(shard_path: Path, clip_key: str, frames: list[np.ndarray]) -> None:
    with tarfile.open(shard_path, "w") as tar:
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


def test_streaming_dataset_produces_expected_pixel_shape(tmp_path):
    origin = tmp_path / "origin"
    origin.mkdir()
    clip_key = "id0/vid0/clip0"
    frames = [np.full((8, 8, 3), i, dtype=np.uint8) for i in range(4)]
    _make_shard(origin / "shard-000000.tar", clip_key, frames)
    manifest = {clip_key: {"shard": "shard-000000.tar", "num_frames": 4, "size": 4}}
    (origin / "manifest.json").write_text(json.dumps(manifest))

    server = _serve(origin)
    try:
        cache_dir = tmp_path / "cache"
        dataset = StreamingShardDataset(
            base_url=f"http://127.0.0.1:{server.server_port}",
            cache_dir=cache_dir,
            num_frames=4,
            image_size=8,
            overlap=0.0,
        )
        windows = list(dataset)

        assert len(windows) == 1
        assert windows[0].shape == (3, 4, 8, 8)
        assert list(cache_dir.glob("*.tar")) == []
    finally:
        server.shutdown()
