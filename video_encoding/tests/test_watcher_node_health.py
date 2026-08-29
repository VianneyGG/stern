from __future__ import annotations

from unittest.mock import MagicMock, patch

from video_encoding.watcher.node_health import (
    NodeHealth,
    foreign_gpu_pids,
    query_node_health,
    truly_idle,
)

# The birds tier as measured on 2026-08-22, after factorial-tube-per_frame
# excluded 12 nodes. sinfo called every one of them "idle" while 7 still had
# ~22 of their 24.5 GiB GPU held by a foreign process and 5 held no shards.
# Replenishment picked 6 of these and all 6 crashed on submit -- CUDA OOM or
# FileNotFoundError on manifest.json. Values are the measured probe output;
# 0 shards is what the probe reports for an unseeded node (None is reserved
# for a probe that could not answer at all).
BIRDS_SNAPSHOT = [
    NodeHealth("node025", slurm_idle=True, gpu_free_mib=2453, seeded_shards=43),
    NodeHealth("node037", slurm_idle=True, gpu_free_mib=2369, seeded_shards=39),
    NodeHealth("node042", slurm_idle=True, gpu_free_mib=2449, seeded_shards=16),
    NodeHealth("node043", slurm_idle=True, gpu_free_mib=2039, seeded_shards=16),
    NodeHealth("node040", slurm_idle=True, gpu_free_mib=2443, seeded_shards=0),
    NodeHealth("node045", slurm_idle=True, gpu_free_mib=2073, seeded_shards=0),
    NodeHealth("node046", slurm_idle=True, gpu_free_mib=2198, seeded_shards=16),
    NodeHealth("node030", slurm_idle=True, gpu_free_mib=23954, seeded_shards=0),
    NodeHealth("node032", slurm_idle=True, gpu_free_mib=23954, seeded_shards=0),
    NodeHealth("node038", slurm_idle=True, gpu_free_mib=23954, seeded_shards=0),
    NodeHealth("node029", slurm_idle=True, gpu_free_mib=23951, seeded_shards=26),
    NodeHealth("node034", slurm_idle=True, gpu_free_mib=23954, seeded_shards=26),
]


# A node that must pass every check, paired with each rejection case below so
# the assertions bind a concrete accepted set rather than just emptiness -- a
# truly_idle that rejected everything would satisfy an `== set()` assertion.
CONTROL = NodeHealth("node029", slurm_idle=True, gpu_free_mib=23951, seeded_shards=26)


def test_truly_idle_rejects_every_node_replenishment_wrongly_picked():
    """The 6 nodes _replenish_pool submitted on 2026-08-21 all crashed.

    Guards the regression directly: each of the six, named, must be rejected
    while the tier's usable nodes are still accepted.
    """
    ready = truly_idle(BIRDS_SNAPSHOT)
    picked_live = ["node030", "node032", "node038", "node040", "node045", "node046"]
    assert {node: node in ready for node in picked_live} == {
        "node030": False,
        "node032": False,
        "node038": False,
        "node040": False,
        "node045": False,
        "node046": False,
    }
    assert ready == {"node029", "node034"}


def test_truly_idle_keeps_only_the_genuinely_usable_nodes():
    assert truly_idle(BIRDS_SNAPSHOT) == {"node029", "node034"}


def test_truly_idle_rejects_slurm_idle_node_whose_gpu_is_held():
    held = NodeHealth("node040", slurm_idle=True, gpu_free_mib=2443, seeded_shards=26)
    assert truly_idle([held, CONTROL]) == {"node029"}


def test_truly_idle_rejects_unseeded_node_with_a_free_gpu():
    unseeded = NodeHealth("node032", slurm_idle=True, gpu_free_mib=23954, seeded_shards=0)
    assert truly_idle([unseeded, CONTROL]) == {"node029"}


def test_truly_idle_rejects_node_slurm_does_not_call_idle():
    busy = NodeHealth("node026", slurm_idle=False, gpu_free_mib=23954, seeded_shards=43)
    assert truly_idle([busy, CONTROL]) == {"node029"}


def test_truly_idle_fails_closed_on_unreachable_node():
    """Unknown must never read as healthy -- an unreachable probe is exactly
    when submitting a doomed job is most likely."""
    unreachable = NodeHealth(
        "node034", slurm_idle=True, gpu_free_mib=None, seeded_shards=None
    )
    assert truly_idle([unreachable, CONTROL]) == {"node029"}


def test_truly_idle_threshold_is_configurable():
    modest = NodeHealth("x", slurm_idle=True, gpu_free_mib=8000, seeded_shards=4)
    assert truly_idle([modest, CONTROL]) == {"node029"}
    assert truly_idle([modest, CONTROL], min_gpu_free_mib=4000) == {"x", "node029"}


def test_truly_idle_can_skip_the_seeding_requirement():
    unseeded = NodeHealth("node032", slurm_idle=True, gpu_free_mib=23954, seeded_shards=0)
    assert truly_idle([unseeded, CONTROL], require_seeded=False) == {"node032", "node029"}


# --- query_node_health -------------------------------------------------


def _probe(gpu_free: str, shards: str) -> MagicMock:
    return MagicMock(returncode=0, stdout=f"{gpu_free}\n{shards}\n")


def test_query_node_health_parses_a_healthy_probe():
    with patch(
        "video_encoding.watcher.node_health.subprocess.run",
        return_value=_probe("24423", "26"),
    ):
        (health,) = query_node_health(["node029"], idle={"node029"})
    assert health == NodeHealth(
        "node029", slurm_idle=True, gpu_free_mib=24423, seeded_shards=26
    )


def test_query_node_health_marks_unreachable_node_unknown():
    with patch(
        "video_encoding.watcher.node_health.subprocess.run",
        side_effect=OSError("ssh: connect timed out"),
    ):
        (health,) = query_node_health(["node029"], idle={"node029"})
    assert health.gpu_free_mib is None
    assert health.seeded_shards is None
    assert truly_idle([health]) == set()


def test_query_node_health_marks_nonzero_exit_unknown():
    with patch(
        "video_encoding.watcher.node_health.subprocess.run",
        return_value=MagicMock(returncode=255, stdout=""),
    ):
        (health,) = query_node_health(["node029"], idle={"node029"})
    assert health.gpu_free_mib is None


def test_query_node_health_carries_slurm_idle_through():
    with patch(
        "video_encoding.watcher.node_health.subprocess.run",
        return_value=_probe("24423", "26"),
    ):
        healths = query_node_health(["node029", "node026"], idle={"node029"})
    assert {h.node: h.slurm_idle for h in healths} == {
        "node029": True,
        "node026": False,
    }


def test_query_node_health_survives_garbage_output():
    with patch(
        "video_encoding.watcher.node_health.subprocess.run",
        return_value=MagicMock(returncode=0, stdout="not-a-number\nalso-not\n"),
    ):
        (health,) = query_node_health(["node029"], idle={"node029"})
    assert health.gpu_free_mib is None
    assert health.seeded_shards is None


# --- the threshold must fit every tier's card, not just the 24 GiB ones -----


def _health(node, free):
    return NodeHealth(node=node, slurm_idle=True, gpu_free_mib=free, seeded_shards=42)


def test_idle_nodes_of_every_tier_are_healthy_at_the_default_threshold():
    """Free MiB on an idle node differs per tier (measured 2026-08-22).

    A 20_000 floor -- "most of a 24 GiB card" -- rejected every cars and
    countries node, which on a cross-tier roster halves the pool that
    replenishment can draw from.
    """
    idle = [
        _health("node016", 24080),  # fish
        _health("node024", 23954),  # birds
        _health("node048", 17560),  # cars
        _health("node088", 13392),  # countries -- the smallest card
    ]
    assert truly_idle(idle) == {"node016", "node024", "node048", "node088"}


def test_a_node_whose_gpu_is_held_by_a_foreign_process_is_still_rejected():
    """The gap the threshold must detect: 2453 occupied vs 13392 idle."""
    occupied = [
        _health("node037", 2067),
        _health("node046", 1495),
        _health("node025", 2453),
    ]
    assert truly_idle(occupied) == set()


def test_foreign_pids_excludes_job_pid_tree():
    gpu_pids = {111, 222, 333}
    job_pid_tree = {111, 222}
    assert foreign_gpu_pids(gpu_pids, job_pid_tree) == {333}


def test_foreign_pids_empty_when_all_pids_belong_to_job():
    assert foreign_gpu_pids({111, 222}, {111, 222}) == set()
