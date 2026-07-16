from __future__ import annotations

from unittest.mock import MagicMock, patch

from video_encoding.watcher.quota import (
    collect_disk_usage_pct,
    home_quota_pct,
    node_data_usage_pct,
)


def test_home_quota_pct_parses_df_output():
    fake = MagicMock(stdout="Use%\n 87%\n", returncode=0)
    with patch("subprocess.run", return_value=fake) as mock_run:
        pct = home_quota_pct()
    assert pct == 87.0
    assert mock_run.call_args[0][0][:2] == ["df", "--output=pcent"]


def test_home_quota_pct_returns_none_on_failure():
    with patch("subprocess.run", side_effect=OSError("df missing")):
        assert home_quota_pct() is None


def test_node_data_usage_pct_ssh_per_node():
    fake = MagicMock(stdout="Use%\n 42%\n", returncode=0)
    with patch("subprocess.run", return_value=fake) as mock_run:
        result = node_data_usage_pct(["node001", "node047"])
    assert result == {"node001": 42.0, "node047": 42.0}
    assert mock_run.call_count == 2
    first_cmd = mock_run.call_args_list[0][0][0]
    assert first_cmd[:2] == ["ssh", "node001"]


def test_node_data_usage_pct_skips_unreachable_node():
    with patch("subprocess.run", side_effect=OSError("ssh timeout")):
        result = node_data_usage_pct(["deadnode"])
    assert result == {}


def test_collect_disk_usage_pct_merges_home_and_nodes():
    with (
        patch("video_encoding.watcher.quota.home_quota_pct", return_value=91.0),
        patch(
            "video_encoding.watcher.quota.node_data_usage_pct",
            return_value={"node001": 30.0},
        ),
    ):
        merged = collect_disk_usage_pct(["node001"])
    assert merged == {"home": 91.0, "node:node001": 30.0}


def test_collect_disk_usage_pct_omits_home_when_unavailable():
    with (
        patch("video_encoding.watcher.quota.home_quota_pct", return_value=None),
        patch("video_encoding.watcher.quota.node_data_usage_pct", return_value={}),
    ):
        assert collect_disk_usage_pct([]) == {}
