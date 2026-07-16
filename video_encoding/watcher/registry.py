"""Jobs registry: append-only record of every job submit_login-node.sh submits.

Squeue-diffing alone can't detect "a job that never started" -- there's
nothing to diff against without a record of what *should* exist. This file
is that ground truth; the `dead` detector (video_encoding/watcher/detectors.py)
diffs squeue against it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class JobRecord:
    slurm_job_id: str
    experiment: str
    run_tag: str | None
    submitted_at: str  # ISO 8601 UTC
    nodelist: str


def load_registry(path: Path) -> list[JobRecord]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    return [JobRecord(**r) for r in payload]


def append_job_record(path: Path, record: JobRecord) -> None:
    """Append `record`, writing the whole list back atomically.

    The registry stays small (one line per submitted job, not per tick), so
    rewriting it wholesale on every append is simpler than a true append-only
    log and still atomic via tmp+rename.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    records = load_registry(path)
    records.append(record)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps([asdict(r) for r in records], indent=2))
    tmp.replace(path)
