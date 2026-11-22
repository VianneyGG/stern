from pathlib import Path

from stern.evaluation.diagnostics.conditioning import _run_id_and_shared


def test_run_id_and_shared_derives_three_parents_up():
    """CHECKPOINT_ROOT already ends in `checkpoints`, and load_bundle appends
    `checkpoints/<run_id>/master.pt` itself -- two `.parent`s would double the
    segment and 404."""
    assert _run_id_and_shared(Path("/s/checkpoints/rid/master.pt")) == (
        "rid",
        Path("/s"),
    )
