from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from video_encoding.watcher.cluster import (
    query_idle_nodes,
    query_job_states,
    query_node_states,
)


@pytest.mark.integration
def test_query_idle_nodes_parses_idle_hosts():
    fake = MagicMock(
        returncode=0, stdout="node001 idle\nanchois mixed\nanguille idle\n"
    )
    with patch("video_encoding.watcher.cluster.subprocess.run", return_value=fake):
        assert query_idle_nodes() == {"node001", "node003"}


@pytest.mark.integration
def test_query_idle_nodes_returns_empty_set_on_nonzero_exit():
    fake = MagicMock(returncode=1, stdout="")
    with patch("video_encoding.watcher.cluster.subprocess.run", return_value=fake):
        assert query_idle_nodes() == set()


@pytest.mark.integration
def test_query_idle_nodes_returns_empty_set_on_subprocess_error():
    with patch(
        "video_encoding.watcher.cluster.subprocess.run",
        side_effect=OSError("sinfo missing"),
    ):
        assert query_idle_nodes() == set()


def test_query_job_states_parses_job_id_and_state():
    fake = MagicMock(returncode=0, stdout="101|RUNNING\n102|NODE_FAIL\n")
    with patch("video_encoding.watcher.cluster.subprocess.run", return_value=fake):
        assert query_job_states(["101", "102"]) == {
            "101": "RUNNING",
            "102": "NODE_FAIL",
        }


def test_query_job_states_returns_empty_dict_on_nonzero_exit():
    fake = MagicMock(returncode=1, stdout="")
    with patch("video_encoding.watcher.cluster.subprocess.run", return_value=fake):
        assert query_job_states(["101"]) == {}


def test_query_job_states_returns_empty_dict_on_subprocess_error():
    with patch(
        "video_encoding.watcher.cluster.subprocess.run",
        side_effect=OSError("squeue missing"),
    ):
        assert query_job_states(["101"]) == {}


def test_query_job_states_returns_empty_dict_for_empty_job_id_list():
    assert query_job_states([]) == {}


def test_query_node_states_parses_all_node_states():
    fake = MagicMock(
        returncode=0, stdout="node001 idle\nanchois drained\nanguille down\n"
    )
    with patch("video_encoding.watcher.cluster.subprocess.run", return_value=fake):
        assert query_node_states() == {
            "node001": "idle",
            "node002": "drained",
            "node003": "down",
        }


def test_query_node_states_returns_empty_dict_on_nonzero_exit():
    fake = MagicMock(returncode=1, stdout="")
    with patch("video_encoding.watcher.cluster.subprocess.run", return_value=fake):
        assert query_node_states() == {}


def test_query_node_states_returns_empty_dict_on_subprocess_error():
    with patch(
        "video_encoding.watcher.cluster.subprocess.run",
        side_effect=OSError("sinfo missing"),
    ):
        assert query_node_states() == {}
