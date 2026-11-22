"""Static check: is quorum merge enabled for each registered run?

Reads runs/state/mesh_run_configs.json, or a local copy of it. In
MeshRunConfig (stern/watcher/mesh_tick.py), `quorum_k` defaults to None. That
default falls back to the old all-or-ROUND_TIMEOUT_S wait, which quorum merge
replaces.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def check_quorum_config(run_configs: list[dict]) -> list[dict]:
    return [
        {
            "run_id": rc["run_id"],
            "quorum_k": rc.get("quorum_k"),
            "grace_s": rc.get("grace_s"),
            "quorum_enabled": rc.get("quorum_k") is not None,
        }
        for rc in run_configs
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh_run_configs_path", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    run_configs = json.loads(args.mesh_run_configs_path.read_text())
    result = check_quorum_config(run_configs)
    output = json.dumps(result, indent=2)

    if args.out is not None:
        args.out.write_text(output)
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
