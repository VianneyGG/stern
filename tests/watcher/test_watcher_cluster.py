from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from stern.watcher.cluster import (
    query_idle_nodes,
    query_job_states,
    query_node_states,
)


@pytest.mark.integration
def test_query_idle_nodes_parses_idle_hosts():
    fake = MagicMock(
        returncode=0, stdout="node001 idle\nnode002 mixed\nnode003 idle\n"
    )
    with patch("stern.watcher.cluster.subprocess.run", return_value=fake):
        assert query_idle_nodes() == {"node001", "node003"}


def test_query_job_states_parses_job_id_and_state():
    fake = MagicMock(returncode=0, stdout="101|RUNNING\n102|NODE_FAIL\n")
    with patch("stern.watcher.cluster.subprocess.run", return_value=fake):
        assert query_job_states(["101", "102"]) == {
            "101": "RUNNING",
            "102": "NODE_FAIL",
        }


def test_query_job_states_returns_empty_dict_for_empty_job_id_list():
    assert query_job_states([]) == {}


def test_query_node_states_parses_all_node_states():
    fake = MagicMock(
        returncode=0, stdout="node001 idle\nnode002 drained\nnode003 down\n"
    )
    with patch("stern.watcher.cluster.subprocess.run", return_value=fake):
        assert query_node_states() == {
            "node001": "idle",
            "node002": "drained",
            "node003": "down",
        }


@pytest.mark.parametrize(
    "query, args, empty, failure",
    [
        pytest.param(
            query_idle_nodes,
            (),
            set(),
            "nonzero_exit",
            id="idle-nodes-nonzero-exit",
            marks=pytest.mark.integration,
        ),
        pytest.param(
            query_idle_nodes,
            (),
            set(),
            "os_error",
            id="idle-nodes-os-error",
            marks=pytest.mark.integration,
        ),
        pytest.param(
            query_job_states,
            (["101"],),
            {},
            "nonzero_exit",
            id="job-states-nonzero-exit",
        ),
        pytest.param(
            query_job_states,
            (["101"],),
            {},
            "os_error",
            id="job-states-os-error",
        ),
        pytest.param(
            query_node_states,
            (),
            {},
            "nonzero_exit",
            id="node-states-nonzero-exit",
        ),
        pytest.param(
            query_node_states,
            (),
            {},
            "os_error",
            id="node-states-os-error",
        ),
    ],
)
def test_query_returns_empty_on_failure(query, args, empty, failure):
    if failure == "nonzero_exit":
        fake = MagicMock(returncode=1, stdout="")
        patcher = patch("stern.watcher.cluster.subprocess.run", return_value=fake)
    else:
        patcher = patch(
            "stern.watcher.cluster.subprocess.run",
            side_effect=OSError("command missing"),
        )
    with patcher:
        assert query(*args) == empty
