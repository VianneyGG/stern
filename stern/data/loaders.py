"""Per-role dataset loaders. SSL, probe and eval data never share a loader.

Keeping them apart stops labelled data from leaking into label-free SSL by
accident. There is no face-cropping stage: `load_frames` decodes whole frames
through stern/data/frame_decoder.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from stern.data.manifest import ManifestRecord, read_manifest
from stern.data.frame_decoder import load_clip_frames


class DatasetNotAcquiredError(RuntimeError):
    """Raised when dataset_cfg.present is False: the acquisition script has not run."""


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


def identity_disjoint_split(
    records: list[ManifestRecord], split_fraction: float = 0.2
) -> tuple[set[str], set[str]]:
    """Partition `records` into (fit_ids, scored_ids) by `identity_id`.

    The split uses only the sorted identity ids, never a seed. Every seed and
    both arms therefore score the same identities, which keeps a
    checkpoint-minus-control difference paired. A seeded window-level split
    does not have this property, on purpose.
    """
    identity_ids = sorted({record.identity_id for record in records})
    n_scored = max(1, round(len(identity_ids) * split_fraction))
    scored_ids = set(identity_ids[-n_scored:])
    fit_ids = set(identity_ids) - scored_ids
    return fit_ids, scored_ids
