"""Released LeVJEPA ViT-L (HuggingFace, `trust_remote_code`) + a linear VA head.

Ladder rung 4's supervised encoder: the public `galilai-group/LeVJEPA-VideoMix-
Large` checkpoint fine-tuned end-to-end on AFEW-VA. It is loaded through
`transformers` rather than into this repo's ViT because its geometry (24 blocks,
1024-d, tubelet 1, RoPE, block-causal attention) has no counterpart in
`models/ViT/` -- see the ssl-diag-03 design.md's rung-4 decision.

The head is a plain linear readout of the mean-pooled patch tokens with no
`tanh`: `LinearVAHead`'s `tanh` saturated within one epoch on the from-scratch
rung-4 run and zeroed every gradient after it. Outputs are clamped to `[-1, 1]`
only in eval mode, where no gradient flows.
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.checkpoint import checkpoint


def _checkpoint_block(block: nn.Module) -> None:
    """Recompute `block`'s activations in backward instead of storing them.

    The HF model declares `supports_gradient_checkpointing = False`, so this is
    the only way to fit a ViT-L fine-tune at 3137 tokens on a 20 GB card.
    `forward` is rebound in place rather than the block wrapped in a new module,
    so state-dict keys -- and therefore checkpoints -- are identical either way.
    """
    plain_forward = block.forward

    def forward(x, **kwargs):
        if not block.training:
            return plain_forward(x, **kwargs)
        return checkpoint(plain_forward, x, use_reentrant=False, **kwargs)

    block.forward = forward


def load_hf_levjepa(weights_path: str) -> nn.Module:
    """The released encoder from a local snapshot -- never the network."""
    from transformers import AutoModel

    return AutoModel.from_pretrained(
        str(weights_path), trust_remote_code=True, local_files_only=True
    )


class HFLeVJEPAVARegressor(nn.Module):
    def __init__(self, model_cfg: DictConfig, *, backbone: nn.Module | None = None):
        """`backbone` is injectable so tests can substitute a small model built
        from the same remote code; production loads `model_cfg.weights_path`."""
        super().__init__()
        self.backbone = (
            backbone
            if backbone is not None
            else load_hf_levjepa(model_cfg.weights_path)
        )
        if bool(model_cfg.get("grad_checkpointing", False)):
            for block in self.backbone.encoder.blocks:
                _checkpoint_block(block)
        self.head = nn.Linear(int(model_cfg.embed_dim), 2)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """`pixel_values`: `(B, C, T, H, W)` -> `(B, 2)`; clamped to `[-1, 1]` in eval."""
        tokens = self.backbone(pixel_values=pixel_values).last_hidden_state
        out = self.head(tokens[:, 1:].mean(dim=1))  # drop CLS, pool patches
        return out if self.training else out.clamp(-1.0, 1.0)
