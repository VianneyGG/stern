"""Map jobs-registry entries to their result JSON's mtime, if it exists.

Both training/pretrain.py and training/train.py write via
tb_logger.write_result_json(overwrite=True), landing at the stable filename
tb_logger.result_json_filename(experiment, run_tag) computes -- this module
is the read-side counterpart, used by the `dead` and `stalled` detectors.
"""

from __future__ import annotations

import os
from pathlib import Path

from video_encoding.training.tb_logger import result_json_filename
from video_encoding.watcher.registry import JobRecord


def result_json_mtimes(
    registry: list[JobRecord], results_root: Path
) -> dict[str, float]:
    mtimes: dict[str, float] = {}
    for record in registry:
        path = results_root / result_json_filename(record.experiment, record.run_tag)
        if path.exists():
            mtimes[record.slurm_job_id] = os.path.getmtime(path)
    return mtimes
