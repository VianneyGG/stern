"""Static checks needing no cluster access: is per-node pacing configured,
is torch.compile on, and is fp16 error-feedback quantization wired on both
NFS directions (master broadcast, worker pseudo-gradient push)?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

_QUANTIZE_CALL = "quantize_with_error_feedback("


def read_pacing_config(experiment_cfg_text: str) -> dict:
    cfg = yaml.safe_load(experiment_cfg_text)
    round_budget_s = cfg.get("mesh", {}).get("round_budget_s")
    return {
        "round_budget_s": round_budget_s,
        "pacing_enabled": round_budget_s is not None,
    }


def read_compile_config(train_cfg_text: str) -> dict:
    cfg = yaml.safe_load(train_cfg_text)
    return {"compile": bool(cfg.get("train", {}).get("compile", False))}


def check_quantize_call_sites(repo_root: Path) -> dict:
    master_path = repo_root / "video_encoding" / "scripts" / "run_outer_step.py"
    worker_path = repo_root / "video_encoding" / "training" / "diloco" / "mesh_round.py"
    return {
        "master_broadcast": _file_contains(master_path, _QUANTIZE_CALL),
        "worker_push": _file_contains(worker_path, _QUANTIZE_CALL),
    }


def _file_contains(path: Path, needle: str) -> bool:
    if not path.exists():
        return False
    return needle in path.read_text()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_root", type=Path)
    parser.add_argument(
        "--experiment-cfg",
        type=Path,
        default=Path("video_encoding/conf/experiment/videomae_mesh_ssl.yaml"),
    )
    parser.add_argument(
        "--train-cfg", type=Path, default=Path("video_encoding/conf/train/ssl.yaml")
    )
    args = parser.parse_args()

    result = {
        "pacing": read_pacing_config(
            (args.repo_root / args.experiment_cfg).read_text()
        ),
        "compile": read_compile_config((args.repo_root / args.train_cfg).read_text()),
        "quantize": check_quantize_call_sites(args.repo_root),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
