"""Behavioural tests for render_report.py -- pure rendering of a
diagnostics.json (schema 1) into one self-contained report.html.

The fixture is hand-built in the orchestrator's exact output shape, with the
verdict fields already stamped: the renderer's contract is that it computes
none of them, so every colouring asserted here must follow from those fields.
"""

from __future__ import annotations

import copy
import re
import subprocess
import sys

import pytest

from video_encoding.scripts.render_report import (
    ROLLOUT_CAVEAT,
    WHAT_TO_DO_NEXT,
    format_value,
    render,
    verdict,
)

_FAKE_PNG = b"\x89PNG\r\n\x1a\nnot-really-an-image"


def _metric(mean, sd=0.1, *, deterministic=False, replicated=True, **stamp):
    base = {
        "mean": mean,
        "sd": None if deterministic else sd,
        "n": None if deterministic else 3,
        "seeds": [] if deterministic else [1, 2, 3],
        "deterministic": deterministic,
        "replicated": replicated,
        "bracket": None,
        "verdict": None,
        "red": False,
    }
    base.update(stamp)
    return base


def _envelope(check, metrics, *, input_kind="real", run_id="r1", **meta):
    return {
        "check": check,
        "run_id": run_id,
        "round": None,
        "rung": None,
        "input_kind": input_kind,
        "metrics": metrics,
        "meta": {"LADDER_REVISION": "v1", "input_kind": input_kind, **meta},
    }


def _tier(tier):
    return {
        "tau": 1.0,
        "tier": tier,
        "present_rungs": 4,
        "rungs": {
            "random_init": {"mean": 0.5, "sd": 0.1, "n": 3, "deterministic": False},
            "overfit": {"mean": 2.0, "sd": 0.1, "n": 3, "deterministic": False},
        },
    }


def _diagnostics() -> dict:
    return {
        "schema": 1,
        "ladder_revision": "v1",
        "hyperparams": {},
        "ladder_tiers": {
            "residual_branch_norm.real.block_0_attn": _tier("A"),
            "branch_norm_ratio.real.block_0_attn_ratio": _tier("B"),
            "attention_entropy_by_block.real.block0": _tier("C"),
        },
        "reports": [
            _envelope(
                "residual_branch_norm",
                {
                    "block_0_attn": _metric(
                        1.0,
                        bracket="1 ± 0.1 · between random_init=0.5 and overfit=2",
                        verdict="SEPARATED",
                    )
                },
            ),
            _envelope(
                "branch_norm_ratio",
                {
                    "block_0_attn_ratio": _metric(
                        0.6,
                        bracket="0.6 ± 0.1 · between random_init=0.5 and overfit=2",
                        verdict="INCONCLUSIVE",
                    )
                },
            ),
            _envelope(
                "attention_entropy_by_block",
                {
                    "block0": _metric(
                        3.0,
                        deterministic=True,
                        replicated=False,
                        verdict="INCONCLUSIVE",
                    )
                },
            ),
            _envelope(
                "attention_entropy_by_block",
                {"block0": _metric(3.1, deterministic=True, replicated=False)},
                input_kind="random",
            ),
            _envelope(
                "delta_from_init",
                {"block_0": _metric(0.2, deterministic=True)},
            ),
            _envelope("cka_matrix", {}, plot=["r1__cka_full.png"], panel="cka"),
        ],
        "errors": [
            {
                "check": "probe_by_depth",
                "run_id": "r1",
                "round": None,
                "rung": None,
                "reason": "VEATIC_ROOT unset",
            }
        ],
        "skipped": [
            {
                "check": "residual_branch_norm",
                "run_id": "r2",
                "round": None,
                "rung": None,
                "reason": "no ref_bundle supplied",
            }
        ],
    }


@pytest.fixture
def plots_dir(tmp_path):
    (tmp_path / "r1__cka_full.png").write_bytes(_FAKE_PNG)
    return tmp_path


def _metric_table(html: str) -> str:
    start = html.index("<h2>Metric table</h2>")
    return html[start : html.index("</table>", start)]


def _row(html: str, anchor: str) -> str:
    """The whole <tr> containing `anchor`."""
    at = html.index(anchor)
    return html[html.rindex("<tr", 0, at) : html.index("</tr>", at)]


def test_output_is_self_contained(plots_dir):
    html = render(_diagnostics(), plots_dir)
    for scheme in ("http://", "https://", "file://"):
        assert scheme not in html
    srcs = re.findall(r'<img[^>]*\ssrc="([^"]*)"', html)
    assert len(srcs) == 1
    assert srcs[0].startswith("data:image/png;base64,")


def test_banner_is_gate_not_met_without_a_gate_report(plots_dir):
    d = _diagnostics()
    assert verdict(d["reports"]) == "GATE_NOT_MET"
    html = render(d, plots_dir)
    banner = html[html.index("<header") : html.index("</header>")]
    assert "Verdict: GATE_NOT_MET" in banner
    assert "PASS" not in banner


def test_banner_reports_the_gate_status():
    gate = _envelope("finetune_afewva", {}, status="PASS")
    assert verdict([gate]) == "PASS"


def test_format_value_has_three_forms():
    assert format_value(_metric(1.0, 0.1)) == "1 ± 0.1 (n=3)"
    assert format_value(_metric(0.2, deterministic=True)) == "0.2 (deterministic)"
    assert (
        format_value(_metric(3.0, deterministic=True, replicated=False))
        == "3 (unreplicated)"
    )


def test_no_metric_row_carries_a_bare_estimate(plots_dir):
    table = _metric_table(render(_diagnostics(), plots_dir))
    rows = re.findall(r"<tr class=\"tier-.\">.*?</tr>", table, flags=re.S)
    assert len(rows) == 5
    for row in rows:
        assert re.search(r"\(n=\d+\)|\(deterministic\)|\(unreplicated\)", row), row


def test_an_uncontrolled_check_renders_struck_through(plots_dir):
    d = _diagnostics()
    d["reports"].append(_envelope("no_control_metric", {"x": _metric(1.0)}))
    row = _row(_metric_table(render(d, plots_dir)), "no_control_metric.real.x")
    assert "<s>" in row
    assert "1 ± 0.1 (n=3)" in row


def test_inconclusive_colour_excludes_red_and_pass(plots_dir):
    row = _row(
        _metric_table(render(_diagnostics(), plots_dir)),
        "branch_norm_ratio.real.block_0_attn_ratio",
    )
    assert 'class="inconclusive"' in row
    assert 'class="red"' not in row and 'class="pass"' not in row


def test_a_separated_tier_a_cell_is_pass_and_a_red_one_is_red(plots_dir):
    d = _diagnostics()
    html = render(d, plots_dir)
    assert 'class="pass"' in _row(
        _metric_table(html), "residual_branch_norm.real.block_0_attn"
    )
    d["reports"][0]["metrics"]["block_0_attn"]["red"] = True
    row = _row(
        _metric_table(render(d, plots_dir)), "residual_branch_norm.real.block_0_attn"
    )
    assert 'class="red"' in row and 'class="pass"' not in row


def test_tier_c_has_no_decision_colouring_and_tier_b_is_marked(plots_dir):
    html = render(_diagnostics(), plots_dir)
    table = _metric_table(html)
    row = _row(table, "attention_entropy_by_block.real.block0")
    assert 'class="tier-C"' in row
    for cls in ("inconclusive", "red", "pass", "tier-A", "tier-B"):
        assert cls not in row
    assert 'class="tier-B"' in _row(table, "branch_norm_ratio.real.block_0_attn_ratio")


def test_ladder_panel_precedes_the_metric_table(plots_dir):
    html = render(_diagnostics(), plots_dir)
    assert html.index("Ladder panel") < html.index("Metric table")
    ladder_row = _row(html, "residual_branch_norm.real.block_0_attn")
    assert 'class="tier-A"' in ladder_row
    assert "overfit=2 ± 0.1 (n=3)" in ladder_row


def test_out_of_range_is_spelled_out(plots_dir):
    d = _diagnostics()
    d["reports"][0]["metrics"]["block_0_attn"].update(
        verdict="OUT_OF_RANGE", bracket="9 ± 0.1 · outside ladder range"
    )
    row = _row(
        _metric_table(render(d, plots_dir)), "residual_branch_norm.real.block_0_attn"
    )
    assert "OUT OF LADDER RANGE" in row


def test_real_vs_random_pairs_the_two_input_kinds(plots_dir):
    html = render(_diagnostics(), plots_dir)
    start = html.index("<h2>Real vs random</h2>")
    section = html[start : html.index("</table>", start)]
    row = _row(section, "attention_entropy_by_block.block0")
    assert "3 (unreplicated)" in row and "3.1 (unreplicated)" in row


def test_real_vs_random_omits_a_real_report_whose_control_failed(plots_dir):
    d = _diagnostics()
    d["reports"].append(_envelope("labelfree", {"rank": _metric(4.0)}))
    html = render(d, plots_dir)
    start = html.index("<h2>Real vs random</h2>")
    section = html[start : html.index("</table>", start)]
    assert "attention_entropy_by_block.block0" in section
    assert "labelfree" not in section


def test_rollout_caveat_accompanies_qualitative_reports(plots_dir):
    d = _diagnostics()
    assert ROLLOUT_CAVEAT not in render(d, plots_dir)
    d["reports"].append(_envelope("diag_qualitative", {}))
    assert ROLLOUT_CAVEAT in render(d, plots_dir)


def test_a_missing_png_renders_a_placeholder(tmp_path):
    html = render(_diagnostics(), tmp_path)  # no PNG written
    assert "plot missing: r1__cka_full.png" in html
    assert "<img" not in html


def test_footer_lists_errors_and_skips_with_reasons(plots_dir):
    html = render(_diagnostics(), plots_dir)
    footer = html[html.index("<footer") :]
    assert "VEATIC_ROOT unset" in _row(footer, "probe_by_depth")
    assert "no ref_bundle supplied" in _row(footer, "r2")


def test_what_to_do_next_only_for_red_cells(plots_dir):
    d = _diagnostics()
    assert "What to do next" not in render(d, plots_dir)
    d["reports"][0]["metrics"]["block_0_attn"]["red"] = True
    html = render(d, plots_dir)
    section = html[html.index("What to do next") :]
    assert WHAT_TO_DO_NEXT["residual_branch_norm"][1] in section
    assert WHAT_TO_DO_NEXT["branch_norm_ratio"][1] not in section


def test_a_red_check_without_authored_guidance_still_gets_a_row(plots_dir):
    d = _diagnostics()
    d["reports"][4]["metrics"]["block_0"]["red"] = True  # delta_from_init
    html = render(d, plots_dir)
    section = html[html.index("<h2>What to do next</h2>") :]
    row = _row(section, "delta_from_init red")
    assert "no authored guidance" in row


def test_all_five_panels_render_and_surface_caveats(plots_dir):
    d = _diagnostics()
    d["reports"].append(
        _envelope(
            "merge_regression", {}, panel="diloco", caveat="single-round, greedy sweep"
        )
    )
    html = render(d, plots_dir)
    titles = [
        "Depth panel",
        "CKA heatmaps",
        "Qualitative panels",
        "Round trajectories",
        "DiLoCo diagnostics",
    ]
    positions = [html.index(f"<h2>{t}</h2>") for t in titles]
    assert positions == sorted(positions)
    diloco = html[positions[4] :]
    assert "single-round, greedy sweep" in diloco[: diloco.index("<footer")]
    cka = html[positions[1] : positions[2]]
    assert "<img" in cka


def test_rendering_is_deterministic(plots_dir):
    d = _diagnostics()
    assert render(copy.deepcopy(d), plots_dir) == render(copy.deepcopy(d), plots_dir)


def test_importing_the_renderer_does_not_import_torch():
    code = (
        "import sys; import video_encoding.scripts.render_report; "
        "print('torch' in sys.modules, '_diag_common' in str(list(sys.modules)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False False"
