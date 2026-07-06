from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Role = Literal["ssl", "probe", "eval"]

_REQUIRED_COLUMNS = ("identity_id", "clip_path", "duration_s", "role", "label_path")


@dataclass
class ManifestRecord:
    identity_id: str
    clip_path: str
    duration_s: float
    role: Role
    label_path: str | None = None


class ManifestRoleMismatchError(ValueError):
    """A manifest row's role doesn't match the role the caller expected."""


class ManifestSchemaError(ValueError):
    """A manifest file is missing required columns."""


def write_manifest(path: Path, records: list[ManifestRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_REQUIRED_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "identity_id": record.identity_id,
                    "clip_path": record.clip_path,
                    "duration_s": record.duration_s,
                    "role": record.role,
                    "label_path": record.label_path or "",
                }
            )


def read_manifest(path: Path, expected_role: Role) -> list[ManifestRecord]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        if not set(_REQUIRED_COLUMNS).issubset(fieldnames):
            missing = set(_REQUIRED_COLUMNS) - fieldnames
            raise ManifestSchemaError(
                f"manifest {path} missing required columns: {missing}"
            )
        records = []
        for row in reader:
            if row["role"] != expected_role:
                raise ManifestRoleMismatchError(
                    f"manifest {path} row for identity={row['identity_id']!r} has "
                    f"role={row['role']!r}, expected role={expected_role!r}"
                )
            records.append(
                ManifestRecord(
                    identity_id=row["identity_id"],
                    clip_path=row["clip_path"],
                    duration_s=float(row["duration_s"]),
                    role=row["role"],
                    label_path=row["label_path"] or None,
                )
            )
    return records
