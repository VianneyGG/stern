from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from video_encoding.watcher.cluster import query_idle_nodes

pytestmark = pytest.mark.integration


def test_query_idle_nodes_parses_idle_hosts():
    fake = MagicMock(
        returncode=0, stdout="node001 idle\nanchois mixed\nanguille idle\n"
    )
    with patch("video_encoding.watcher.cluster.subprocess.run", return_value=fake):
        assert query_idle_nodes() == {"node001", "node003"}


def test_query_idle_nodes_returns_empty_set_on_nonzero_exit():
    fake = MagicMock(returncode=1, stdout="")
    with patch("video_encoding.watcher.cluster.subprocess.run", return_value=fake):
        assert query_idle_nodes() == set()


def test_query_idle_nodes_returns_empty_set_on_subprocess_error():
    with patch(
        "video_encoding.watcher.cluster.subprocess.run",
        side_effect=OSError("sinfo missing"),
    ):
        assert query_idle_nodes() == set()
