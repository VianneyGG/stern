"""Per-role dataset loaders. ssl / probe / eval NEVER share a loader by
default. Burning Aff-Wild2's labels on label-free SSL would waste its main
value -- keeping loaders separate makes that mistake structurally hard to
make by accident.

Per docs/superpowers/specs/2026-07-01-training-pipeline-design.md, this
pipeline has no face-cropping stage (a shared crop/face_pipeline stage was
archived in SCAFFOLD_ARCHIVE.md, out of scope) -- .load_frames() decodes raw
whole-frame samples via video_encoding/data/frame_decoder.py instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from data.manifest import ManifestRecord, read_manifest
from video_encoding.data.frame_decoder import load_clip_frames


class DatasetNotAcquiredError(RuntimeError):
    """Raised when dataset_cfg.present is False -- data hasn't been acquired yet.

    Distinct from NotImplementedError on .load_frames(): this means "run the
    acquisition script first," that means "crop pipeline doesn't exist yet."
    """


@dataclass
class SSLDataset:
    records_by_identity: dict[str, list[ManifestRecord]]
    window_cfg: DictConfig
    root: Path

    def load_frames(
        self, identity_id: str, clip_path: str, num_frames: int = 16
    ) -> list[np.ndarray]:
        return load_clip_frames(self.root / clip_path, num_frames=num_frames)


@dataclass
class LabelledDataset:
    records: list[ManifestRecord]
    window_cfg: DictConfig
    root: Path

    def load_frames(self, clip_path: str, num_frames: int = 16) -> list[np.ndarray]:
        return load_clip_frames(self.root / clip_path, num_frames=num_frames)


def _require_acquired(dataset_cfg: DictConfig) -> None:
    if not dataset_cfg.present:
        raise DatasetNotAcquiredError(
            f"dataset '{dataset_cfg.name}' has present=false -- run its "
            "acquisition script and flip present=true before loading."
        )


def build_ssl_loader(dataset_cfg: DictConfig, window_cfg: DictConfig) -> SSLDataset:
    """Unlabelled loader for Recipe B SSL adaptation (role: ssl, e.g. VoxMini)."""
    if dataset_cfg.role != "ssl":
        raise ValueError(
            f"build_ssl_loader requires role='ssl', got '{dataset_cfg.role}'"
        )
    _require_acquired(dataset_cfg)
    records = read_manifest(Path(dataset_cfg.manifest_path), expected_role="ssl")
    records_by_identity: dict[str, list[ManifestRecord]] = {}
    for record in records:
        records_by_identity.setdefault(record.identity_id, []).append(record)
    return SSLDataset(
        records_by_identity=records_by_identity,
        window_cfg=window_cfg,
        root=Path(dataset_cfg.path),
    )


def build_probe_loader(
    dataset_cfg: DictConfig, window_cfg: DictConfig
) -> LabelledDataset:
    """Labelled VA loader for probe training (role: probe, e.g. Aff-Wild2)."""
    if dataset_cfg.role != "probe":
        raise ValueError(
            f"build_probe_loader requires role='probe', got '{dataset_cfg.role}'"
        )
    _require_acquired(dataset_cfg)
    records = read_manifest(Path(dataset_cfg.manifest_path), expected_role="probe")
    return LabelledDataset(
        records=records, window_cfg=window_cfg, root=Path(dataset_cfg.path)
    )


def build_eval_loader(
    dataset_cfg: DictConfig, window_cfg: DictConfig
) -> LabelledDataset:
    """Labelled domain-shift test loader (role: eval, e.g. SEWA, AFEW-VA)."""
    if dataset_cfg.role != "eval":
        raise ValueError(
            f"build_eval_loader requires role='eval', got '{dataset_cfg.role}'"
        )
    _require_acquired(dataset_cfg)
    records = read_manifest(Path(dataset_cfg.manifest_path), expected_role="eval")
    return LabelledDataset(
        records=records, window_cfg=window_cfg, root=Path(dataset_cfg.path)
    )
