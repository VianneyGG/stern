"""Pure rendering: diagnostics.json + plots/ -> one self-contained report.html.

No torch, no `_diag_common`, no metric computation: every verdict, bracket and
tier is precomputed by run_diagnostics.py, so the layout can be iterated on
against a fixture diagnostics.json (the same split plot_vit_internals.py keeps).
This module is also the single source of the constants both sides agree on.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
from html import escape
from pathlib import Path

# Emitted check names with a configured pass/fail control -- the ladder bracket.
# An envelope whose check is in neither this set nor NO_VERDICT renders
# struck-through: a metric without a control is not a diagnostic.
CONTROLLED: frozenset[str] = frozenset(
    {
        "residual_branch_norm",
        "layerscale_gamma",
        "branch_norm_ratio",
        "cka_matrix",
        "decode_from_depth",
        "delta_from_init",
        "attention_entropy_by_block",
        "labelfree",
        "probe_by_depth",
        # DiLoCo: synthetic split/noise brackets, block 1, 1.0, 0.0, three
        # baselines at 0.0, and the configured sweep cell, respectively.
        "agreement_matrix",
        "effective_replicas",
        "per_block_divergence",
        "merge_magnitude",
        "momentum_alignment",
        "merge_regression",
        "counterfactual_sweep",
        # The AFEW-VA gate: random_init under the identical fine-tune recipe.
        "finetune_afewva",
    }
)

# Qualitative-only checks: they never render a verdict, by design.
NO_VERDICT: frozenset[str] = frozenset({"diag_qualitative"})

# Emitted check name -> the report panel its plots go under. Unlisted checks
# default to "qualitative".
PANEL_OF: dict[str, str] = {
    "residual_branch_norm": "depth",
    "layerscale_gamma": "depth",
    "branch_norm_ratio": "depth",
    "delta_from_init": "depth",
    "decode_from_depth": "depth",
    "cka_matrix": "cka",
    "diag_qualitative": "qualitative",
    "agreement_matrix": "diloco",
    "effective_replicas": "diloco",
    "per_block_divergence": "diloco",
    "merge_magnitude": "diloco",
    "momentum_alignment": "diloco",
    "merge_regression": "diloco",
    "counterfactual_sweep": "diloco",
}


# (check, metric-name regex, direction): how a metric moves as the rung
# improves. Per metric, not per check -- one check emits metrics that disagree
# (labelfree's rankme rises, alpha_req is best near 1). First match
# wins. A metric no rule matches has no declared direction: tau is read
# rising, and it is never flagged inverted nor painted red.
# attention_entropy_by_block is deliberately undeclared: on the 2026-10-03
# ladder it fell from random init (uniform) to the ceiling, but entropy is not
# monotone in quality -- full collapse to one token is the low extreme too --
# so declaring "falling" from that ladder would be circular.
# branch_norm_ratio's ratios are undeclared for the same reason: declared
# "falling" (a branch overwriting the stream is the collapse signature), they
# rose from random init to the ceiling at nearly every block on that ladder --
# branches start small at init and grow in healthy training too.
DIRECTION_RULES: tuple[tuple[str, str, str], ...] = (
    ("residual_branch_norm", r"", "rising"),
    ("delta_from_init", r"", "rising"),
    ("layerscale_gamma", r"", "rising"),
    ("decode_from_depth", r"", "rising"),
    ("labelfree", r"^rankme$", "rising"),
    ("probe_by_depth", r"^layer_\d+_(checkpoint|diff)$", "rising"),
)


def good_direction(check: str, metric: str) -> str | None:
    """`"rising"`/`"falling"` for a metric with a declared direction, else None."""
    for rule_check, pattern, direction in DIRECTION_RULES:
        if rule_check == check and re.search(pattern, metric):
            return direction
    return None


def bad_direction(check: str, metric: str) -> str | None:
    """The side on which a value reads as a worse checkpoint, derived from
    `good_direction` so the red-cell rule and the tiering cannot disagree."""
    good = good_direction(check, metric)
    return None if good is None else ("low" if good == "rising" else "high")


ROLLOUT_CAVEAT = (
    "Attention rollout is a heuristic approximation of information flow, "
    "not a verified attribution method -- treat rollout maps as suggestive, "
    "never as proof of what the model actually uses."
)

# Check -> (red cell, what it implicates, how to confirm). Authored, not
# computed; a row renders only when its check has at least one red cell.
WHAT_TO_DO_NEXT: dict[str, tuple[str, str, str]] = {
    "residual_branch_norm": (
        "residual_branch_norm flat at init for deep blocks",
        "depth is not being trained -- inner LR, warmup, or depth-wise LR decay",
        "delta_from_init on the same blocks; causal ablation if they disagree",
    ),
    "branch_norm_ratio": (
        "branch_norm_ratio spiking in blocks 1-2",
        "the shallow attention branch is overwriting the residual stream -- "
        "init scale or missing LayerScale",
        "cka_matrix row 1; enabling layer_scale_init is the candidate change",
    ),
    "cka_matrix": (
        "cka_matrix saturated across deep blocks",
        "dead depth -- model is deeper than the data supports",
        "shorten depth, or decode_from_depth to find where the curve flattens",
    ),
    "labelfree": (
        "label-free metrics at the ladder floor with the probe above it",
        "the metric, not the checkpoint -- a tiering failure",
        "the metric's own ladder-tau; demote it",
    ),
    "probe_by_depth": (
        "probe CCC ~= its random_init probe control",
        "the encoder contributes nothing recoverable -- pretraining recipe, not depth",
        "the AFEW-VA gate, if available",
    ),
    "merge_regression": (
        "merge_regression persistently positive",
        "DiLoCo is hurting on this backbone",
        "counterfactual_sweep, starting below the configured gamma",
    ),
    "m_eff": (
        "M_eff ~= 1 on a large pool",
        "pool size is buying nothing -- shard homogeneity or H too small",
        "agreement_matrix against the synthetic ceiling control",
    ),
    "momentum_alignment": (
        "momentum_alignment oscillating about 0",
        "outer_momentum=0.9 is an LLM constant misapplied",
        "the (gamma, 0.0) column of the counterfactual sweep",
    ),
}


def metric_key(check: str, input_kind: str, metric: str) -> str:
    """The name a metric is tiered under in `ladder_tiers` -- input kind is part
    of it, since a metric on real and on random input are different metrics."""
    return f"{check}.{input_kind}.{metric}"


PANELS: tuple[tuple[str, str], ...] = (
    ("depth", "Depth panel"),
    ("cka", "CKA heatmaps"),
    ("qualitative", "Qualitative panels"),
    ("round_trajectory", "Round trajectories"),
    ("diloco", "DiLoCo diagnostics"),
)

_VERDICT_TEXT = {
    "SEPARATED": "SEPARATED",
    "INCONCLUSIVE": "INCONCLUSIVE",
    "OUT_OF_RANGE": "OUT OF LADDER RANGE",
}

_STYLE = """
body { font-family: system-ui, sans-serif; margin: 1.5rem; color: #1d1d1f; background: #fff; }
table { border-collapse: collapse; margin-bottom: 1rem; }
td, th { border: 1px solid #ccc; padding: 3px 8px; font-size: 0.9em; vertical-align: top; }
.banner h1 { padding: 0.5rem 0.8rem; background: #eee; }
td.red { background: #f8d7da; }
td.pass { background: #d4edda; }
td.inconclusive { background: #fff3cd; }
tr.tier-B { opacity: 0.75; font-style: italic; }
.bracket, .caveat, .plot-missing { color: #555; font-size: 0.85em; }
.caveat { font-style: italic; }
figure { display: inline-block; margin: 0.4rem; }
img { max-width: 420px; }
"""


def embed_png(path: Path) -> str | None:
    """`path` as a `data:` URI, or None when the file is absent."""
    path = Path(path)
    if not path.is_file():
        return None
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode(
        "ascii"
    )


def verdict(reports: list[dict]) -> str:
    """The banner verdict: the AFEW-VA gate's status, or GATE_NOT_MET."""
    for e in reports:
        if e["check"] == "finetune_afewva":
            return e["meta"]["status"]
    return "GATE_NOT_MET"


def format_value(metric: dict) -> str:
    """A value with its replication form -- never a bare point estimate."""
    if not metric["replicated"]:
        return f"{metric['mean']:.3g} (unreplicated)"
    if metric["deterministic"]:
        return f"{metric['mean']:.3g} (deterministic)"
    return f"{metric['mean']:.3g} ± {metric['sd']:.3g} (n={metric['n']})"


def _where(e: dict) -> str:
    label = e["rung"] or e["run_id"] or ""
    if e["round"] is not None:
        label += f" @ round {e['round']}"
    return label


def _tier_of(e: dict, name: str, metric: dict, ladder_tiers: dict) -> str:
    if not metric["replicated"]:
        return "C"
    entry = ladder_tiers.get(metric_key(e["check"], e["input_kind"], name))
    return entry["tier"] if entry else "C"


def _metric_cell(e: dict, metric: dict, tier: str) -> str:
    body = escape(format_value(metric))
    if metric.get("bracket"):
        body += f'<br><span class="bracket">{escape(metric["bracket"])}</span>'
    if metric.get("verdict"):
        body += f"<br>{_VERDICT_TEXT[metric['verdict']]}"
    if e["check"] not in CONTROLLED and e["check"] not in NO_VERDICT:
        return f"<td><s>{body}</s></td>"
    cls = None
    if tier in ("A", "B"):
        if metric.get("red"):
            cls = "red"
        elif metric.get("verdict") == "INCONCLUSIVE":
            cls = "inconclusive"
        elif metric.get("verdict") == "SEPARATED":
            cls = "pass"
    return f'<td class="{cls}">{body}</td>' if cls else f"<td>{body}</td>"


def _ladder_panel(ladder_tiers: dict) -> str:
    rows = []
    for key, t in ladder_tiers.items():
        rungs = "<br>".join(
            f"{escape(rung)}={escape(format_value({**v, 'replicated': True}))}"
            for rung, v in t["rungs"].items()
        )
        # .get: a ladder.json written before direction-aware tiering has
        # neither key, and reads as it always did.
        tier = t["tier"] + (" (inverted)" if t.get("inverted") else "")
        rows.append(
            f'<tr class="tier-{escape(t["tier"])}"><td>{escape(key)}</td>'
            f"<td>{t['tau']:.3g}</td><td>{escape(tier)}</td>"
            f"<td>{t['present_rungs']}</td><td>{rungs}</td></tr>"
        )
    return (
        "<h2>Ladder panel</h2><table><tr><th>metric</th><th>tau</th><th>tier</th>"
        f"<th>rungs present</th><th>per-rung values</th></tr>{''.join(rows)}</table>"
    )


def _metric_table(reports: list[dict], ladder_tiers: dict) -> str:
    rows = []
    for e in reports:
        for name, metric in e["metrics"].items():
            tier = _tier_of(e, name, metric, ladder_tiers)
            rows.append(
                f'<tr class="tier-{tier}"><td>'
                f"{escape(metric_key(e['check'], e['input_kind'], name))}</td>"
                f"<td>{escape(_where(e))}</td>{_metric_cell(e, metric, tier)}</tr>"
            )
    return (
        "<h2>Metric table</h2><table><tr><th>metric</th><th>run</th><th>value</th></tr>"
        f"{''.join(rows)}</table>"
    )


def _real_vs_random(reports: list[dict]) -> str:
    """Does the metric respond to data at all -- a separate question from the
    ladder verdict, so a separate table, never folded into it."""
    randoms = {
        (e["check"], e["run_id"], e["round"], e["rung"]): e
        for e in reports
        if e["input_kind"] == "random"
    }
    rows = []
    for e in reports:
        other = randoms.get((e["check"], e["run_id"], e["round"], e["rung"]))
        if e["input_kind"] != "real" or other is None:
            continue
        for name, metric in e["metrics"].items():
            if name in other["metrics"]:
                rows.append(
                    f"<tr><td>{escape(e['check'])}.{escape(name)}</td>"
                    f"<td>{escape(_where(e))}</td><td>{escape(format_value(metric))}</td>"
                    f"<td>{escape(format_value(other['metrics'][name]))}</td></tr>"
                )
    return (
        "<h2>Real vs random</h2><table><tr><th>metric</th><th>run</th><th>real</th>"
        f"<th>random</th></tr>{''.join(rows)}</table>"
    )


def _panel(key: str, title: str, reports: list[dict], plots_dir: Path) -> str:
    figures: list[str] = []
    caveats: list[str] = []
    for e in reports:
        meta = e["meta"]
        if meta.get("panel", "qualitative") != key:
            continue
        plots = meta.get("plot", [])
        for name in [plots] if isinstance(plots, str) else plots:
            uri = embed_png(Path(plots_dir) / name)
            figures.append(
                f'<figure><img src="{uri}" alt="{escape(name)}"><figcaption>'
                f"{escape(name)}</figcaption></figure>"
                if uri
                else f'<p class="plot-missing">plot missing: {escape(name)}</p>'
            )
        if meta.get("caveat") and meta["caveat"] not in caveats:
            caveats.append(meta["caveat"])
    if key == "qualitative" and any(e["check"] == "diag_qualitative" for e in reports):
        caveats.append(ROLLOUT_CAVEAT)
    body = "".join(f'<p class="caveat">{escape(c)}</p>' for c in caveats)
    body += "".join(figures) or "<p>no plots for this panel</p>"
    return f"<h2>{escape(title)}</h2>{body}"


def _what_to_do_next(reports: list[dict]) -> str:
    red_checks = sorted(
        {
            e["check"]
            for e in reports
            if any(m.get("red") for m in e["metrics"].values())
        }
    )
    if not red_checks:
        return ""
    # A red check with no authored row still gets one: a silent omission would
    # read as "nothing to do" for a cell the table just marked red.
    unauthored = ("no authored guidance", "see the design doc's diagnostic table")
    rows = "".join(
        "<tr>"
        + "".join(
            f"<td>{escape(c)}</td>"
            for c in WHAT_TO_DO_NEXT.get(check, (f"{check} red", *unauthored))
        )
        + "</tr>"
        for check in red_checks
    )
    return (
        "<h2>What to do next</h2><table><tr><th>red cell</th><th>implicates</th>"
        f"<th>confirm with</th></tr>{rows}</table>"
    )


def _footer(diagnostics: dict) -> str:
    def table(entries: list[dict]) -> str:
        rows = "".join(
            f"<tr><td>{escape(str(x.get('check') or ''))}</td>"
            f"<td>{escape(_where(x))}</td><td>{escape(x['reason'])}</td></tr>"
            for x in entries
        )
        return (
            f"<table><tr><th>check</th><th>run</th><th>reason</th></tr>{rows}</table>"
        )

    hyperparams = json.dumps(
        diagnostics.get("hyperparams", {}), indent=1, sort_keys=True
    )
    return (
        "<footer><h2>Provenance</h2>"
        f"<p>schema {escape(str(diagnostics['schema']))} &middot; ladder revision "
        f"{escape(diagnostics['ladder_revision'])}</p><pre>{escape(hyperparams)}</pre>"
        f"<h3>Errors</h3>{table(diagnostics['errors'])}"
        f"<h3>Skipped</h3>{table(diagnostics['skipped'])}</footer>"
    )


def render(diagnostics: dict, plots_dir: Path) -> str:
    """One self-contained HTML page from a schema-1 diagnostics.json and the
    PNGs it names. Deterministic: identical inputs give an identical string."""
    reports = diagnostics["reports"]
    ladder_tiers = diagnostics.get("ladder_tiers", {})
    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        "<title>SSL diagnostics report</title>"
        f"<style>{_STYLE}</style></head><body>"
        f'<header class="banner"><h1>Verdict: {escape(verdict(reports))}</h1></header>'
        + _ladder_panel(ladder_tiers)
        + _metric_table(reports, ladder_tiers)
        + _real_vs_random(reports)
        + "".join(_panel(key, title, reports, plots_dir) for key, title in PANELS)
        + _what_to_do_next(reports)
        + _footer(diagnostics)
        + "</body></html>\n"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument(
        "--plots-dir", type=Path, default=None, help="default: <diagnostics dir>/plots"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="default: <diagnostics dir>/report.html"
    )
    args = parser.parse_args(argv)
    plots_dir = args.plots_dir or args.diagnostics.parent / "plots"
    out = args.out or args.diagnostics.parent / "report.html"
    out.write_text(render(json.loads(args.diagnostics.read_text()), plots_dir))


if __name__ == "__main__":
    main()
