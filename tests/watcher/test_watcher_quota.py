from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from stern.watcher.quota import (
    collect_disk_usage_pct,
    home_quota_pct,
    node_data_usage_pct,
    parse_quota_output,
)

# Captured from `quota -p` on login-node during the 2026-08-14 incident
# investigation: 19955708/31457280 blocks (~63.4%), 87843/300000 files (~29.3%).
_REAL_QUOTA_OUTPUT = (
    "Disk quotas for user stern-user (uid 13828): \n"
    "     Filesystem  blocks   quota   limit   grace   files   quota   limit   grace\n"
    "storage-node.example.edu:/students\n"
    "                19955708  31457280 31457280       0   87843  300000  300000       0\n"
    "storage-node.example.edu:/students/.snapshot/daily.2026-08-09_0010\n"
    "                19955708  31457280 31457280       0   87843  300000  300000       0\n"
)

pytestmark = pytest.mark.integration


def test_parse_quota_output_returns_block_pct_when_it_dominates():
    assert parse_quota_output(_REAL_QUOTA_OUTPUT) == 100.0 * 19955708 / 31457280


def test_parse_quota_output_returns_inode_pct_when_it_dominates():
    stdout = (
        "Disk quotas for user x (uid 1): \n"
        "     Filesystem  blocks   quota   limit   grace   files   quota   limit   grace\n"
        "fs:/mount\n"
        "                1000  1000000 1000000       0   290000  300000  300000       0\n"
    )
    assert parse_quota_output(stdout) == 100.0 * 290000 / 300000


def test_parse_quota_output_ignores_snapshot_duplicate_rows():
    # _REAL_QUOTA_OUTPUT already has two identical data rows (real + one
    # .snapshot repeat) -- only the first is used, per parse_quota_output's
    # docstring; asserting equality to the single-row value proves it.
    single_row = (
        "storage-node.example.edu:/students\n"
        "                19955708  31457280 31457280       0   87843  300000  300000       0\n"
    )
    expected = 100.0 * 19955708 / 31457280
    assert parse_quota_output(single_row) == expected
    assert parse_quota_output(_REAL_QUOTA_OUTPUT) == expected


def test_parse_quota_output_skips_unlimited_hard_limit():
    stdout = (
        "fs:/mount\n"
        "                1000  0 0       0   290000  300000  300000       0\n"
    )
    assert parse_quota_output(stdout) == 100.0 * 290000 / 300000


def test_parse_quota_output_returns_none_for_header_only():
    stdout = (
        "Disk quotas for user x (uid 1): \n"
        "     Filesystem  blocks   quota   limit   grace   files   quota   limit   grace\n"
    )
    assert parse_quota_output(stdout) is None


def test_parse_quota_output_returns_none_for_empty_or_garbage():
    assert parse_quota_output("") is None
    assert parse_quota_output("quota: command not found\n") is None


def test_parse_quota_output_handles_no_wrap_single_line_row():
    # `quota -w -p` keeps a long filesystem name on the SAME line as the 8
    # numeric fields instead of wrapping it onto its own line first --
    # regression for the 2026-08-17 bug where field-count-== 8 matching
    # depended on whether the name happened to wrap, and silently returned
    # None (falling back to df, the exact wrong number the -p switch to
    # `quota` was meant to stop using) whenever it didn't.
    stdout = (
        "Disk quotas for user x (uid 1): \n"
        "     Filesystem  blocks   quota   limit   grace   files   quota   limit   grace\n"
        "storage-node.example.edu:/students 24148844  31457280 31457280       0   88326  300000  300000       0\n"
    )
    assert parse_quota_output(stdout) == 100.0 * 24148844 / 31457280


def test_home_quota_pct_reads_real_quota_not_df():
    fake = MagicMock(stdout=_REAL_QUOTA_OUTPUT, returncode=0)
    with patch("subprocess.run", return_value=fake) as mock_run:
        pct = home_quota_pct()
    assert pct == 100.0 * 19955708 / 31457280
    mock_run.assert_called_once_with(
        ["quota", "-w", "-p"], capture_output=True, text=True, timeout=10
    )


def test_home_quota_pct_falls_back_to_df_when_quota_output_unparseable():
    quota_fake = MagicMock(stdout="quota: command not found\n", returncode=1)
    df_fake = MagicMock(stdout="Use%\n 87%\n", returncode=0)
    with patch("subprocess.run", side_effect=[quota_fake, df_fake]) as mock_run:
        pct = home_quota_pct()
    assert pct == 87.0
    assert mock_run.call_count == 2
    assert mock_run.call_args_list[1][0][0][:2] == ["df", "--output=pcent"]


def test_home_quota_pct_falls_back_to_df_when_quota_binary_missing():
    df_fake = MagicMock(stdout="Use%\n 87%\n", returncode=0)
    with patch("subprocess.run", side_effect=[OSError("quota missing"), df_fake]):
        assert home_quota_pct() == 87.0


def test_home_quota_pct_returns_none_when_quota_and_df_both_fail():
    with patch("subprocess.run", side_effect=OSError("nothing available")):
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
        patch("stern.watcher.quota.home_quota_pct", return_value=91.0),
        patch(
            "stern.watcher.quota.node_data_usage_pct",
            return_value={"node001": 30.0},
        ),
    ):
        merged = collect_disk_usage_pct(["node001"])
    assert merged == {"home": 91.0, "node:node001": 30.0}


def test_collect_disk_usage_pct_omits_home_when_unavailable():
    with (
        patch("stern.watcher.quota.home_quota_pct", return_value=None),
        patch("stern.watcher.quota.node_data_usage_pct", return_value={}),
    ):
        assert collect_disk_usage_pct([]) == {}
