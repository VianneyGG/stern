"""Assemble the Markdown throughput audit report from the checkers' JSON evidence.

Pure rendering with no cluster access. The evidence must already be pulled
(see the ssh commands in main()'s help) and passed in as parsed JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def render_report(
    *,
    quorum_config: list[dict],
    merge_summary: dict,
    watcher_gaps: list[dict],
    pacing: dict,
    compile_status: dict,
    quantize: dict,
    unreachable_layers: list[str] = (),
) -> str:
    lines: list[str] = ["# DiLoCo Throughput Audit", ""]

    lines += ["## Watcher/scheduling", ""]
    if "watcher/scheduling" in unreachable_layers:
        lines += [
            "**unverified** -- `watcher/scheduling` was unreachable for this layer's evidence pull.",
            "",
        ]
    else:
        lines += [
            "Source: `runs/state/mesh_run_configs.json` (quorum_k/grace_s per run).",
            "",
        ]
        for entry in quorum_config:
            lines.append(
                f"- `{entry['run_id']}`: quorum_enabled: {entry['quorum_enabled']} "
                f"(quorum_k={entry['quorum_k']}, grace_s={entry['grace_s']})"
            )
        lines += [
            "",
            "Source: `runs/state/watcher.log` (MERGING event counts per run).",
            "",
        ]
        for run_id, counts in merge_summary.items():
            lines.append(
                f"- `{run_id}`: full={counts['full']} quorum={counts['quorum']} "
                f"timeout={counts['timeout']} wait_s_between_merges={counts['wait_s_between_merges']}"
            )
        lines += [
            "",
            "Cron-outage gap candidates (>120s between consecutive log lines). "
            "Caveat: watcher.log only gets a line on ticks that do something -- "
            "most ticks are silent fast-path returns -- so most gaps here reflect "
            "normal quiet periods, not real outages. Only exceptionally large "
            "gaps are meaningful without independently cross-referencing "
            "crontab/squeue liveness for that window.",
            "",
        ]
        if watcher_gaps:
            largest = sorted(watcher_gaps, key=lambda gap: -gap["gap_s"])[:5]
            lines.append(f"- {len(watcher_gaps)} gaps detected; 5 largest:")
            for gap in largest:
                lines.append(
                    f"  - {gap['gap_start']} -> {gap['gap_end']} ({gap['gap_s']}s)"
                )
        else:
            lines.append("- no gaps detected")
        lines.append("")

    lines += ["## Mesh comm/sync", ""]
    lines += [
        "Source: `configs/experiment/videomae_mesh_ssl.yaml` (`mesh.round_budget_s`).",
        f"- pacing_enabled: {pacing['pacing_enabled']} (round_budget_s={pacing['round_budget_s']})",
        "",
        "Source: static grep of `stern/mesh/run_outer_step.py` / `stern/training/mesh_round.py`.",
        f"- quantize on master broadcast: {quantize['master_broadcast']}",
        f"- quantize on worker push: {quantize['worker_push']}",
        "",
    ]

    lines += ["## Model compute", ""]
    lines += [
        "Source: `configs/train/ssl.yaml` (`train.compile`).",
        f"- compile enabled: {compile_status['compile']}",
        "- **Recommendation (lower-confidence than the above)**: benchmark "
        "`torch.compile` on-cluster -- never measured in production, so this "
        "is a recommendation to measure, not a measured finding.",
        "",
    ]

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            (__doc__ or "") + "\n\nPull evidence first:\n"
            "  ssh <login-node> 'cat ~/stern/runs/state/mesh_run_configs.json' > /tmp/mesh_run_configs.json\n"
            "  ssh <login-node> 'cat ~/stern/runs/state/watcher.log' > /tmp/watcher.log\n"
        )
    )
    parser.add_argument("quorum_config_json", type=Path)
    parser.add_argument(
        "watcher_log_summary_json",
        type=Path,
        help="Combined JSON from audit_watcher_log.py's stdout ({merge_summary, watcher_gaps} keys)",
    )
    parser.add_argument("compute_config_json", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--unreachable-layer",
        action="append",
        default=[],
        dest="unreachable_layers",
        help="Mark a layer unreachable instead of reading its evidence file (repeatable)",
    )
    args = parser.parse_args()

    compute = json.loads(args.compute_config_json.read_text())
    watcher_log_summary = json.loads(args.watcher_log_summary_json.read_text())
    report = render_report(
        quorum_config=json.loads(args.quorum_config_json.read_text()),
        merge_summary=watcher_log_summary["merge_summary"],
        watcher_gaps=watcher_log_summary["watcher_gaps"],
        pacing=compute["pacing"],
        compile_status=compute["compile"],
        quantize=compute["quantize"],
        unreachable_layers=args.unreachable_layers,
    )
    args.out.write_text(report)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
