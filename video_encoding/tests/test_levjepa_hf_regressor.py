"""HFLeVJEPAVARegressor against a tiny model built from the released remote code.

Needs the snapshot's `*.py` files (not its weights) at `$LEVJEPA_HF_PATH`
(default `~/models/levjepa-hf`); skipped where it is absent.
"""

import os
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from video_encoding.models.levjepa_hf import HFLeVJEPAVARegressor

HF_PATH = Path(os.environ.get("LEVJEPA_HF_PATH", Path.home() / "models/levjepa-hf"))
pytestmark = pytest.mark.skipif(
    not (HF_PATH / "modeling_levjepa.py").exists(),
    reason=f"no LeVJEPA HF snapshot at {HF_PATH}",
)

TINY = dict(
    img_size=32, patch_size=16, num_frames=4, embed_dim=32, depth=2, num_heads=2
)


def _tiny_backbone(seed: int = 0):
    from transformers import AutoConfig, AutoModel

    cfg = AutoConfig.from_pretrained(
        str(HF_PATH), trust_remote_code=True, local_files_only=True, **TINY
    )
    torch.manual_seed(seed)
    return AutoModel.from_config(cfg, trust_remote_code=True)


def _regressor(grad_checkpointing: bool = False, seed: int = 0):
    cfg = OmegaConf.create(
        {"embed_dim": TINY["embed_dim"], "grad_checkpointing": grad_checkpointing}
    )
    model = HFLeVJEPAVARegressor(cfg, backbone=_tiny_backbone(seed))
    torch.manual_seed(seed + 1)
    model.head.reset_parameters()
    return model


def _video(batch: int = 3):
    return torch.randn(
        batch, 3, TINY["num_frames"], 32, 32, generator=torch.Generator().manual_seed(7)
    )


def test_output_shape_and_gradient_reaches_head_and_last_block():
    model = _regressor().train()
    out = model(_video())
    assert out.shape == (3, 2)
    out.pow(2).sum().backward()
    last_block = model.backbone.encoder.blocks[-1]
    assert model.head.weight.grad.abs().sum() > 0
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in last_block.parameters()
    )


def test_output_is_unbounded_in_train_and_clamped_in_eval():
    # The rung-4 from-scratch run died because tanh saturated to exactly +-1
    # and zeroed the gradient; the training output must be able to leave [-1, 1].
    model = _regressor()
    # Zero weights isolate the bias: every output is exactly 5.0 in train, and
    # exactly the clamp bound 1.0 in eval.
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.fill_(5.0)
    x = _video()
    out_train = model.train()(x).tolist()
    assert out_train == [[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]]
    out_eval = model.eval()(x).tolist()
    assert out_eval == [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]


def test_grad_checkpointing_matches_plain_forward_and_backward():
    plain, ckpt = _regressor(False), _regressor(True)
    # Checkpointing must not rename a single key, or a checkpoint saved with it
    # on would not load into a model built with it off (the rung-4 loader).
    sd_p, sd_c = plain.state_dict(), ckpt.state_dict()
    assert sd_c.keys() == sd_p.keys()
    assert all(torch.equal(sd_c[k], sd_p[k]) for k in sd_p)
    x = _video()
    out_p, out_c = plain.train()(x), ckpt.train()(x)
    torch.testing.assert_close(out_c, out_p)
    out_p.sum().backward()
    out_c.sum().backward()
    torch.testing.assert_close(ckpt.head.weight.grad, plain.head.weight.grad)
    qkv_p = plain.backbone.encoder.blocks[0].attn.qkv.weight.grad
    qkv_c = ckpt.backbone.encoder.blocks[0].attn.qkv.weight.grad
    torch.testing.assert_close(qkv_c, qkv_p)
