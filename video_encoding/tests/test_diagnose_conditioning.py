from pathlib import Path

import torch
from omegaconf import OmegaConf

from video_encoding.models.videomaev2.encoder import VideoMAEEncoder
from video_encoding.models.videomaev2.mae import VideoMAEForPreTraining
from video_encoding.scripts._diag_common import load_bundle
from video_encoding.scripts.diagnose_conditioning import _run_id_and_shared
from video_encoding.training.checkpoint import load_checkpoint_file, load_encoder_state


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


def test_load_encoder_state_swap_preserves_weights(tmp_path):
    """The same fixture checkpoint, loaded through the pre-swap path
    (load_checkpoint_file -> encoder.-prefix strip -> load_encoder_state) and
    through load_bundle(..., full_model=True), must produce the same encoder
    weights. This pins the swap, not load_bundle's self-consistency -- the two
    paths are independently implemented and could diverge."""
    cfg = _tiny_model_cfg()
    source_model = VideoMAEForPreTraining(cfg)
    ckpt_dir = tmp_path / "checkpoints" / "run_a"
    ckpt_dir.mkdir(parents=True)
    ckpt_path = ckpt_dir / "master.pt"
    torch.save({"model_state_dict": source_model.state_dict()}, ckpt_path)

    # pre-swap path
    payload = load_checkpoint_file(ckpt_path)
    state_dict = payload.get("model_state_dict", payload)
    encoder_state = {
        k[len("encoder.") :]: v
        for k, v in state_dict.items()
        if k.startswith("encoder.")
    }
    pre_swap_encoder = VideoMAEEncoder(cfg)
    missing, unexpected = load_encoder_state(pre_swap_encoder, encoder_state)
    assert missing == []
    assert unexpected == []

    # load_bundle path
    bundle = load_bundle("run_a", tmp_path, cfg, full_model=True)

    for name, p_pre_swap in pre_swap_encoder.state_dict().items():
        assert torch.allclose(p_pre_swap, bundle.encoder.state_dict()[name]), name
