from pathlib import Path

from omegaconf import OmegaConf

from video_encoding.scripts.diagnose_conditioning import _run_id_and_shared


def _tiny_model_cfg():
    return OmegaConf.create(
        {
            "num_frames": 4,
            "image_size": 32,
            "in_channels": 3,
            "tubelet_size": 2,
            "patch_size": 16,
            "embed_dim": 16,
            "depth": 2,
            "num_heads": 2,
            "mlp_ratio": 2.0,
            "dropout": 0.0,
            "drop_path_rate": 0.0,
            "decoder_depth": 1,
            "decoder_embed_dim": 8,
            "decoder_num_heads": 2,
            "mask_ratio": 0.5,
            "decoder_mask_ratio": 0.0,
            "target_mode": "time_avg",
        }
    )


def test_run_id_and_shared_derives_three_parents_up():
    """CHECKPOINT_ROOT already ends in `checkpoints`, and load_bundle appends
    `checkpoints/<run_id>/master.pt` itself -- two `.parent`s would double the
    segment and 404."""
    assert _run_id_and_shared(Path("/s/checkpoints/rid/master.pt")) == (
        "rid",
        Path("/s"),
    )
