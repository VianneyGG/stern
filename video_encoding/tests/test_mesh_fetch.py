"""Tests for `video_encoding.mesh.fetch`.

Uses a real `http.server.ThreadingHTTPServer` (not mocks) so these tests
exercise the same `requests`+HTTP mechanism the cluster uses for P2P shard
transfer.
"""

from __future__ import annotations

import functools
import http.server
import os
import socket
import threading

import pytest

from video_encoding.mesh.fetch import ShardFetchError, fetch_shard
from video_encoding.mesh.shard_map import Holder

SHARD_NAME = "shard-000042.tar"
SUBNET = "/254"


def _dead_port() -> int:
    """Return a port with nothing listening on it (bind then immediately close)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def live_server(tmp_path):
    """Real ThreadingHTTPServer on a background thread, serving SHARD_NAME."""
    serve_dir = tmp_path / "served"
    serve_dir.mkdir()
    content = os.urandom(256 * 1024)
    (serve_dir / SHARD_NAME).write_bytes(content)

    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(serve_dir)
    )
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    port = httpd.server_address[1]
    holder = Holder(host="127.0.0.1", subnet=SUBNET, port=port)
    try:
        yield holder, content
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


@pytest.fixture
def fake_sleep():
    """Injected `sleep` that records calls instead of actually sleeping."""
    calls: list[float] = []

    def _sleep(seconds: float) -> None:
        calls.append(seconds)

    _sleep.calls = calls
    return _sleep


def test_fetches_shard_from_a_live_holder(tmp_path, live_server, fake_sleep):
    holder, content = live_server
    dest_dir = tmp_path / "dest"
    shard_map = {SHARD_NAME: [holder]}

    result = fetch_shard(SHARD_NAME, shard_map, SUBNET, dest_dir, sleep=fake_sleep)

    assert result == dest_dir / SHARD_NAME
    assert result.read_bytes() == content


def test_no_partial_file_is_left_visible_on_success(tmp_path, live_server, fake_sleep):
    holder, _content = live_server
    dest_dir = tmp_path / "dest"
    shard_map = {SHARD_NAME: [holder]}

    fetch_shard(SHARD_NAME, shard_map, SUBNET, dest_dir, sleep=fake_sleep)

    assert list(dest_dir.glob("*.part")) == []


def test_falls_back_to_a_second_holder_when_the_first_is_dead(
    tmp_path, live_server, fake_sleep
):
    holder, content = live_server
    # Distinct host from the live holder: pick_holder's `exclude` filters by
    # `.host`, so a same-host dead/live pair would wrongly exclude both once
    # either is tried.
    dead = Holder(host="127.0.0.2", subnet=SUBNET, port=_dead_port())
    dest_dir = tmp_path / "dest"
    shard_map = {SHARD_NAME: [dead, holder]}

    result = fetch_shard(
        SHARD_NAME, shard_map, SUBNET, dest_dir, attempts=2, sleep=fake_sleep
    )

    assert result.read_bytes() == content
    assert fake_sleep.calls  # backoff was invoked while the dead holder was tried
    assert list(dest_dir.glob("*.part")) == []


def test_raises_when_every_holder_is_exhausted(tmp_path, fake_sleep):
    dead = Holder(host="127.0.0.1", subnet=SUBNET, port=_dead_port())
    dest_dir = tmp_path / "dest"
    shard_map = {SHARD_NAME: [dead]}

    with pytest.raises(ShardFetchError, match=SHARD_NAME):
        fetch_shard(
            SHARD_NAME, shard_map, SUBNET, dest_dir, attempts=2, sleep=fake_sleep
        )

    assert list(dest_dir.glob("*.part")) == []


def test_raises_when_no_holder_is_known(tmp_path, fake_sleep):
    dest_dir = tmp_path / "dest"

    with pytest.raises(ShardFetchError, match="no holder"):
        fetch_shard(SHARD_NAME, {}, SUBNET, dest_dir, sleep=fake_sleep)


def test_existing_local_file_is_returned_without_refetching(tmp_path, fake_sleep):
    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()
    content = os.urandom(1024)
    (dest_dir / SHARD_NAME).write_bytes(content)

    # empty shard_map: fetch_shard would raise if it tried to actually fetch
    result = fetch_shard(SHARD_NAME, {}, SUBNET, dest_dir, sleep=fake_sleep)

    assert result == dest_dir / SHARD_NAME
    assert result.read_bytes() == content
    assert not fake_sleep.calls
