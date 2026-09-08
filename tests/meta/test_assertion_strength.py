"""Gate: every test in the repo binds a concrete expected value.

Runs the detector from tests/meta/assertion_strength.py over every
test_*.py file. A failure here is generated-code-proofing doing its job:
fix the test, or add a justified entry to ALLOWLIST.
"""

from __future__ import annotations

import pathlib

import pytest

from tests.meta.assertion_strength import scan_source

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# "file::test_name" -> reason. Only for tests that are deliberately
# assertion-free (e.g. "does not raise" smoke tests).
ALLOWLIST: frozenset[str] = frozenset(
    {
        "video_encoding/tests/test_wds_deps_smoke.py::test_decord_importable",
        "video_encoding/tests/test_wds_deps_smoke.py::test_webdataset_importable",
        "video_encoding/tests/test_mesh_shard_dataset.py::test_evict_is_idempotent_on_a_missing_file",
        "video_encoding/tests/test_worker_shard_failfast.py::test_require_local_shards_is_silent_when_every_shard_is_present",
        "video_encoding/tests/test_tb_logger.py::test_flush_final_tb_rounds_never_raises_on_a_bad_tb_dir",
        # Assertion is embedded in the monkeypatched time.sleep, which raises
        # AssertionError if called -- proves the flag-already-exists fast path
        # short-circuits the poll loop without a direct `assert`.
        "video_encoding/tests/test_mesh_round.py::test_push_checkpoint_and_wait_succeeds_when_flag_appears",
    }
)


def _iter_test_files():
    for path in sorted(REPO_ROOT.rglob("test_*.py")):
        rel = path.relative_to(REPO_ROOT)
        parts = rel.parts
        if ".venv" in parts or parts[0] in ("tests", "mutants"):
            continue
        yield path, rel


def _collect_findings():
    for path, rel in _iter_test_files():
        source = path.read_text()
        for finding in scan_source(source):
            key = f"{rel.as_posix()}::{finding.name}"
            if key in ALLOWLIST:
                continue
            yield key, finding


@pytest.mark.parametrize(
    "key,finding",
    list(_collect_findings()),
    ids=lambda v: v if isinstance(v, str) else v.kind.value,
)
def test_no_weak_assertions_outside_the_allowlist(key, finding):
    pytest.fail(
        f"{key} at line {finding.lineno}: {finding.kind.value} "
        "-- bind a concrete value, or add a justified ALLOWLIST entry"
    )
