from __future__ import annotations

import os
import time

from stern.watcher.registry import JobRecord
from stern.watcher.results import result_json_mtimes


def test_result_json_mtimes_finds_existing_files(tmp_path):
    (tmp_path / "exp_a-tag1.json").write_text("{}")
    registry = [
        JobRecord("1", "exp_a", "tag1", "2026-07-15T09:00:00Z", "node001"),
        JobRecord("2", "exp_b", None, "2026-07-15T09:00:00Z", "node047"),
    ]

    mtimes = result_json_mtimes(registry, tmp_path)

    assert set(mtimes) == {"1"}
    assert mtimes["1"] == os.path.getmtime(tmp_path / "exp_a-tag1.json")


def test_result_json_mtimes_reflects_latest_write(tmp_path):
    path = tmp_path / "exp_a.json"
    path.write_text("{}")
    first = result_json_mtimes(
        [JobRecord("1", "exp_a", None, "2026-07-15T09:00:00Z", "")], tmp_path
    )["1"]

    time.sleep(0.01)
    path.write_text('{"updated": true}')
    second = result_json_mtimes(
        [JobRecord("1", "exp_a", None, "2026-07-15T09:00:00Z", "")], tmp_path
    )["1"]

    assert second >= first
