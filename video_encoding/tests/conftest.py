"""Tests run at any hour, so the class-hours gate (scripts/class_hours.py) is
pinned open for every test except test_class_hours.py, which exercises it.
Test-only: production code has no way to switch the gate off."""

import pytest

from video_encoding.scripts import class_hours


@pytest.fixture(autouse=True)
def _class_hours_open(request, monkeypatch):
    if request.module.__name__.endswith("test_class_hours"):
        return
    monkeypatch.setattr(class_hours, "blocked", lambda *a, **k: None)
