from stern.evaluation.throughput_audit.watcher_config import check_quorum_config


def test_flags_quorum_enabled_and_disabled_runs():
    run_configs = [
        {"run_id": "sw2-fish-lr050m090", "quorum_k": None, "grace_s": 1800.0},
        {"run_id": "sw3-birds-quorum12", "quorum_k": 12, "grace_s": 900.0},
    ]

    result = check_quorum_config(run_configs)

    assert result == [
        {
            "run_id": "sw2-fish-lr050m090",
            "quorum_k": None,
            "grace_s": 1800.0,
            "quorum_enabled": False,
        },
        {
            "run_id": "sw3-birds-quorum12",
            "quorum_k": 12,
            "grace_s": 900.0,
            "quorum_enabled": True,
        },
    ]


def test_missing_quorum_k_key_treated_as_disabled():
    run_configs = [{"run_id": "diloco-lit-30r", "grace_s": 1800.0}]

    result = check_quorum_config(run_configs)

    assert result == [
        {
            "run_id": "diloco-lit-30r",
            "quorum_k": None,
            "grace_s": 1800.0,
            "quorum_enabled": False,
        }
    ]
