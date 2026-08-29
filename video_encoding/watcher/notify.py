"""video_encoding/watcher/notify.py
Dependency-free-of-claude-p notification (design §3). Every automatic
escalation call site posts here instead of spawning `claude -p`; a broken
LLM invocation must never silently disable notification too.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path

_ESCALATIONS_DIR = (
    Path(__file__).resolve().parents[2] / "runs" / "state" / "escalations"
)
_TIMEOUT_S = 5


def notify(subject: str, detail: str, *, webhook_url: str | None = None) -> None:
    """POST {subject, detail, timestamp} to MESH_NOTIFY_WEBHOOK_URL, 5s
    timeout, no retry -- this must never hang the watcher's per-tick flock.
    On any failure (no URL, network error, timeout) appends to
    runs/state/escalations/NEEDS_HUMAN.log."""
    url = (
        webhook_url
        if webhook_url is not None
        else os.environ.get("MESH_NOTIFY_WEBHOOK_URL")
    )
    payload = {"subject": subject, "detail": detail, "timestamp": time.time()}
    if url:
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=_TIMEOUT_S)
            return
        except Exception:
            pass  # fall through to file log
    _log_to_file(subject, detail)


def _log_to_file(subject: str, detail: str) -> None:
    _ESCALATIONS_DIR.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {subject}: {detail}\n"
    with (_ESCALATIONS_DIR / "NEEDS_HUMAN.log").open("a") as fh:
        fh.write(line)
