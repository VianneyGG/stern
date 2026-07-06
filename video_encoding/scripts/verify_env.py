# video_encoding/scripts/verify_env.py
"""Smoke-test that torch and tensorflow can both load in the single root env
(REPO_SETUP.md design decision: single uv env, see design doc risk note).
Run after any `uv sync` that touches torch/tensorflow/cuda pins.
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        import tensorflow as tf
    except Exception as exc:  # noqa: BLE001 - surface any import-time failure
        print(f"tensorflow import FAILED: {exc}", file=sys.stderr)
        return 1

    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        print(f"torch import FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"tensorflow {tf.__version__} loaded OK")
    print(
        f"torch {torch.__version__} loaded OK, cuda available: {torch.cuda.is_available()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
