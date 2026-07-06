"""VideoMAEv2 frozen inference encoder
(docs/superpowers/specs/2026-07-01-training-pipeline-design.md §4.1).

Not unit-tested per that spec's §5 "What does NOT get tests" -- downloads
real HF weights, so it's smoke-tested via scripts/verify_env.py-style checks
on the cluster instead of a mocked unit test.
"""

from __future__ import annotations

import numpy as np
import torch
from omegaconf import DictConfig
from transformers import VideoMAEImageProcessor, VideoMAEModel


class VideoMaeV2Encoder:
    def __init__(self, model_cfg: DictConfig):
        self.checkpoint = model_cfg.checkpoint
        self.num_frames = model_cfg.num_frames
        self.embed_dim = model_cfg.embed_dim
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = VideoMAEImageProcessor.from_pretrained(self.checkpoint)
        self.model = VideoMAEModel.from_pretrained(self.checkpoint).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, frames: list[np.ndarray]) -> np.ndarray:
        """frames: list of HxWx3 uint8, len must equal `num_frames`.

        Returns float32 ndarray of shape (num_tubelets, embed_dim), e.g. (8, 768):
        mean-pool the 196 spatial patches per tubelet out of
        last_hidden_state's (1, 1568, 768).
        """
        if len(frames) != self.num_frames:
            raise ValueError(
                f"expected {self.num_frames} frames, got {len(frames)}"
            )
        inputs = self.processor(list(frames), return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        hidden = outputs.last_hidden_state  # (1, num_tubelets * patches_per_tubelet, embed_dim)
        num_tubelets = self.num_frames // 2
        patches_per_tubelet = hidden.shape[1] // num_tubelets
        hidden = hidden.view(1, num_tubelets, patches_per_tubelet, self.embed_dim)
        pooled = hidden.mean(dim=2)  # (1, num_tubelets, embed_dim)
        return pooled.squeeze(0).cpu().numpy().astype(np.float32)

    def encode_clip(self, frames: list[np.ndarray]) -> np.ndarray:
        """Returns float32 ndarray of shape (embed_dim,) -- mean over time axis."""
        return self.encode(frames).mean(axis=0)
