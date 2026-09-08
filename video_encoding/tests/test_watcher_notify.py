from unittest.mock import patch

import pytest

from video_encoding.watcher.notify import notify

pytestmark = pytest.mark.integration


def test_notify_posts_to_webhook(monkeypatch):
    monkeypatch.setenv("MESH_NOTIFY_WEBHOOK_URL", "http://example.invalid/hook")
    with patch("video_encoding.watcher.notify.urllib.request.urlopen") as mock_open:
        notify("test_subject", "test detail")
        mock_open.assert_called_once()


def test_notify_falls_back_to_log_file_on_no_url(tmp_path, monkeypatch):
    monkeypatch.delenv("MESH_NOTIFY_WEBHOOK_URL", raising=False)
    monkeypatch.setattr("video_encoding.watcher.notify._ESCALATIONS_DIR", tmp_path)
    notify("test_subject", "test detail")
    log = tmp_path / "NEEDS_HUMAN.log"
    assert log.exists()
    assert "test_subject" in log.read_text()


def test_notify_falls_back_to_log_file_on_network_error(tmp_path, monkeypatch):
    monkeypatch.setenv("MESH_NOTIFY_WEBHOOK_URL", "http://example.invalid/hook")
    monkeypatch.setattr("video_encoding.watcher.notify._ESCALATIONS_DIR", tmp_path)
    with patch(
        "video_encoding.watcher.notify.urllib.request.urlopen",
        side_effect=OSError("boom"),
    ):
        notify("test_subject", "test detail")
    log = tmp_path / "NEEDS_HUMAN.log"
    assert "test_subject" in log.read_text()
