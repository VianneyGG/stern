"""Shared test setup.

Cluster settings are pinned to the committed placeholders (cluster/*.example) before
any stern module is imported, so tests never depend on a local site.conf or roster.

Tests run at any hour, so the class-hours gate (stern/watcher/class_hours.py) is
pinned open for every test except test_class_hours.py, which exercises it.
Test-only: production code has no way to switch the gate off.
"""

import os
from pathlib import Path

_CLUSTER_DIR = Path(__file__).resolve().parents[1] / "cluster"
os.environ["STERN_TIERS_FILE"] = str(_CLUSTER_DIR / "tiers.example.tsv")
os.environ["STERN_DATA_ROOT"] = "/Data/stern-user"
for _line in (_CLUSTER_DIR / "site.conf.example").read_text().splitlines():
    if _line.startswith("STERN_") and "=" in _line:
        _key, _value = _line.split("=", 1)
        os.environ[_key] = _value.split("#", 1)[0].strip().strip('"')

import pytest  # noqa: E402

from stern.watcher import class_hours  # noqa: E402


@pytest.fixture(autouse=True)
def _class_hours_open(request, monkeypatch):
    if request.module.__name__.endswith("test_class_hours"):
        return
    monkeypatch.setattr(class_hours, "blocked", lambda *a, **k: None)
