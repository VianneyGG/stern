from __future__ import annotations

import torch
from omegaconf import OmegaConf

from video_encoding.training.pretrain_levjepa import run_levjepa_pretrain_stage


def test_run_levjepa_pretrain_stage_one_step_smoke(tmp_path, monkeypatch):
    """CPU smoke test on synthetic data: one optimizer step must not NaN and
    the SIGReg term must be strictly positive (regularizer actually engaged,
    not silently short-circuited)."""
    import video_encoding.training.pretrain_levjepa as mod

    class _FakeSSLDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, idx):
            return torch.randint(0, 256, (16, 32, 32, 3), dtype=torch.uint8)

    monkeypatch.setattr(
        mod,
        "_build_levjepa_loader",
        lambda cfg, rank, world_size: torch.utils.data.DataLoader(
            _FakeSSLDataset(), batch_size=2
        ),
    )

    cfg = OmegaConf.create(
        dict(
            experiment=dict(name="levjepa_smoke_test"),
            model=dict(
                encoder="levjepa",
                num_frames=16,
                image_size=32,
                local_image_size=16,
                in_channels=3,
                patch_size=16,
                embed_dim=12,
                depth=1,
                num_heads=2,
                mlp_ratio=2.0,
                dropout=0.0,
                drop_path_rate=0.0,
                drop_ratio=0.5,
                num_local_views=2,
                sigreg_lambda=0.02,
                sigreg_directions=32,
                projector_hidden=16,
                embed_dim_k=8,
                polyak_decay=0.999,
                polyak_update_every=2,
            ),
            train=dict(
                checkpoint_root=str(tmp_path),
                batch_size=2,
                seed=0,
                epochs=1,
                lr=1e-4,
                weight_decay=0.0,
                amp="off",
                resume=False,
            ),
        )
    )

    result = run_levjepa_pretrain_stage(
        cfg, rank=0, world_size=1, local_rank=0, device=torch.device("cpu")
    )
    assert "sigreg_loss" in result
    assert result["sigreg_loss"] > 0
    assert not torch.isnan(torch.tensor(result["total_loss"]))
