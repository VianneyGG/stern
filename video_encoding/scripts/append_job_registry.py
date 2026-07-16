"""CLI wrapper around watcher.registry.append_job_record, called from
submit_login-node.sh's remote ssh block right after `sbatch` returns a job id.

Usage:
  uv run python video_encoding/scripts/append_job_registry.py \
      --registry-path .claude/state/jobs_registry.json \
      --job-id 36901 --experiment videomae_ssl_pretrain_voxmini \
      --run-tag p2-eb256 --nodelist node001
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from video_encoding.watcher.registry import JobRecord, append_job_record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry-path", type=Path, default=Path(".claude/state/jobs_registry.json")
    )
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--nodelist", default="")
    args = parser.parse_args(argv)

    record = JobRecord(
        slurm_job_id=args.job_id,
        experiment=args.experiment,
        run_tag=args.run_tag or None,
        submitted_at=datetime.now(timezone.utc).isoformat(),
        nodelist=args.nodelist,
    )
    append_job_record(args.registry_path, record)
    print(f"jobs_registry: appended {record.slurm_job_id} ({record.experiment})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
