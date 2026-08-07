from video_encoding.scripts.audit_report import render_report


def test_render_report_includes_all_three_layers_and_sources():
    report = render_report(
        quorum_config=[
            {
                "run_id": "sw2-fish-lr050m090",
                "quorum_k": None,
                "grace_s": 1800.0,
                "quorum_enabled": False,
            }
        ],
        merge_summary={
            "sw2-fish-lr050m090": {
                "full": 3,
                "quorum": 0,
                "timeout": 1,
                "wait_s_between_merges": [1320.0],
            }
        },
        watcher_gaps=[
            {
                "gap_start": "2026-07-30T04:30:00+00:00",
                "gap_end": "2026-07-30T07:23:00+00:00",
                "gap_s": 10380.0,
            }
        ],
        pacing={"round_budget_s": None, "pacing_enabled": False},
        compile_status={"compile": False},
        quantize={"master_broadcast": True, "worker_push": True},
    )

    assert "## Watcher/scheduling" in report
    assert "## Mesh comm/sync" in report
    assert "## Model compute" in report
    assert "runs/state/mesh_run_configs.json" in report
    assert "sw2-fish-lr050m090" in report
    assert "quorum_enabled: False" in report
    assert "10380.0" in report  # cron-outage gap surfaced, not silently dropped
    assert "lower-confidence" in report  # compute recommendation flagged per design doc


def test_render_report_summarizes_many_gaps_with_a_method_caveat():
    many_gaps = [
        {
            "gap_start": f"2026-07-{22 + i:02d}T22:00:00+00:00",
            "gap_end": f"2026-07-{22 + i:02d}T22:{i:02d}:00+00:00",
            "gap_s": float(120 + i),
        }
        for i in range(8)
    ]
    outlier = {
        "gap_start": "2026-07-30T22:03:03+00:00",
        "gap_end": "2026-08-03T07:34:12+00:00",
        "gap_s": 293469.0,
    }
    report = render_report(
        quorum_config=[],
        merge_summary={},
        watcher_gaps=many_gaps + [outlier],
        pacing={"round_budget_s": None, "pacing_enabled": False},
        compile_status={"compile": False},
        quantize={"master_broadcast": True, "worker_push": True},
    )

    assert "9 gaps detected" in report
    assert "293469.0" in report  # the real outlier must survive summarization
    assert "most ticks are silent" in report.lower()  # method caveat present


def test_render_report_marks_unreachable_layers_explicitly():
    report = render_report(
        quorum_config=[],
        merge_summary={},
        watcher_gaps=[],
        pacing={"round_budget_s": None, "pacing_enabled": False},
        compile_status={"compile": False},
        quantize={"master_broadcast": True, "worker_push": True},
        unreachable_layers=["watcher/scheduling"],
    )

    assert "unverified" in report.lower()
    assert "watcher/scheduling" in report
