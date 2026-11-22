import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from stern.models.levjepa.encoder import LeVJEPAEncoder
from stern.evaluation.diagnostics.patch_pca import cosine_map, final_patch_grid, pca_rgb

_HF_SNAPSHOT = Path(os.environ.get("LEVJEPA_HF_PATH", Path.home() / "models/levjepa-hf"))


def _tiny_levjepa_cfg():
    return OmegaConf.create(
        {
            "encoder": "levjepa", "num_frames": 4, "image_size": 32, "in_channels": 3,
            "patch_size": 16, "embed_dim": 16, "depth": 3, "num_heads": 2,
            "mlp_ratio": 2.0, "dropout": 0.0, "drop_path_rate": 0.0, "drop_ratio": 0.5,
        }
    )


def _two_clusters(n_per: int = 50, dim: int = 32, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    direction = torch.zeros(dim)
    direction[3] = 5.0
    noise = 0.1 * torch.randn(2 * n_per, dim, generator=g)
    return torch.cat([direction.expand(n_per, dim), -direction.expand(n_per, dim)]) + noise


def test_pca_rgb_gives_separated_clusters_distinct_colours():
    rgb = pca_rgb(_two_clusters())

    assert rgb.shape == (100, 3)
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0
    gap = np.linalg.norm(rgb[:50].mean(axis=0) - rgb[50:].mean(axis=0))
    assert gap > 0.3
    # The first component carries the split, so its channel is near-constant within a cluster.
    assert rgb[:50, 0].std() < 0.05 and rgb[50:, 0].std() < 0.05


def test_pca_rgb_is_invariant_to_a_global_offset():
    feats = _two_clusters()
    np.testing.assert_allclose(pca_rgb(feats), pca_rgb(feats + 100.0), atol=1e-4)


class _NumberedTokens(torch.nn.Module):
    """Returns [CLS] + tokens whose every feature equals the token index."""

    def __init__(self, patch: int):
        super().__init__()
        self.patch = patch

    def forward(self, pixel_values):
        _, _, T, H, W = pixel_values.shape
        n = T * (H // self.patch) * (W // self.patch)
        ids = torch.cat([torch.tensor([-1.0]), torch.arange(n, dtype=torch.float32)])
        return ids.view(1, -1, 1).expand(1, -1, 4)


def test_final_patch_grid_drops_cls_and_is_frame_row_col_major_on_non_square_input():
    T, H, W, patch = 2, 32, 48, 16
    grid = final_patch_grid(_NumberedTokens(patch), torch.zeros(1, 3, T, H, W), patch_size=patch)

    h, w = H // patch, W // patch
    assert grid.shape == (T, h, w, 4)
    for t in range(T):
        for i in range(h):
            for j in range(w):
                assert grid[t, i, j, 0].item() == t * h * w + i * w + j
    assert (grid >= 0).all()  # CLS (-1) is gone


def test_cosine_map_peaks_at_query_and_stays_inside_the_object():
    T, h, w, D = 2, 4, 4, 8
    g = torch.Generator().manual_seed(0)
    grid = torch.randn(T, h, w, D, generator=g)
    obj = torch.randn(D, generator=g)
    grid[:, 1:3, 1:3] = obj + 0.05 * torch.randn(T, 2, 2, D, generator=g)

    sim = cosine_map(grid, (0, 1, 1))

    assert sim.shape == (T, h, w)
    assert np.isclose(sim[0, 1, 1], 1.0, atol=1e-5)
    assert np.unravel_index(sim.argmax(), sim.shape) == (0, 1, 1)
    inside = np.zeros((T, h, w), dtype=bool)
    inside[:, 1:3, 1:3] = True
    assert sim[inside].min() > 0.9  # the object, in both frames
    assert sim[~inside].max() < sim[inside].min()


def test_cosine_map_sees_through_a_large_shared_offset():
    """Real encoders put most of every token's norm in one shared vector; raw
    cosine would then be ~1 everywhere and hide the object."""
    T, h, w, D = 2, 4, 4, 8
    g = torch.Generator().manual_seed(1)
    grid = torch.randn(T, h, w, D, generator=g)
    grid[:, 1:3, 1:3] = torch.randn(D, generator=g) + 0.05 * torch.randn(T, 2, 2, D, generator=g)
    grid = grid + 50.0 * torch.randn(D, generator=g)

    sim = cosine_map(grid, (0, 1, 1))

    inside = np.zeros((T, h, w), dtype=bool)
    inside[:, 1:3, 1:3] = True
    assert sim[~inside].mean() < 0.5
    assert sim[~inside].max() < sim[inside].min()


def _causal_frame_check(encoder, num_frames: int, image_size: int, patch: int):
    """Block-causal attention: changing the last frame must leave every earlier
    frame's tokens untouched. That pins the grid's frame axis to real time,
    rather than only checking a shape."""
    torch.manual_seed(0)
    x = torch.randn(1, 3, num_frames, image_size, image_size)
    grid = final_patch_grid(encoder, x, patch_size=patch)
    side = image_size // patch
    assert grid.shape[:3] == (num_frames, side, side)

    x2 = x.clone()
    x2[:, :, -1] += 1.0
    grid2 = final_patch_grid(encoder, x2, patch_size=patch)
    torch.testing.assert_close(grid[:-1], grid2[:-1])
    assert not torch.allclose(grid[-1], grid2[-1])


def test_final_patch_grid_in_house_levjepa_keeps_time_axis():
    cfg = _tiny_levjepa_cfg()
    encoder = LeVJEPAEncoder(cfg).eval()
    _causal_frame_check(encoder, cfg.num_frames, cfg.image_size, cfg.patch_size)


@pytest.mark.skipif(
    not (_HF_SNAPSHOT / "modeling_levjepa.py").exists(),
    reason="released LeVJEPA remote code not found; set LEVJEPA_HF_PATH",
)
def test_final_patch_grid_hf_levjepa_keeps_time_axis(tmp_path):
    from transformers import AutoConfig, AutoModel

    # The released remote code with a tiny, weights-free config.
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    for name in ("configuration_levjepa.py", "modeling_levjepa.py"):
        shutil.copy(_HF_SNAPSHOT / name, snapshot / name)
    config = json.loads((_HF_SNAPSHOT / "config.json").read_text())
    config.update(img_size=32, num_frames=4, embed_dim=32, depth=2, num_heads=2)
    (snapshot / "config.json").write_text(json.dumps(config))
    hf_cfg = AutoConfig.from_pretrained(
        str(snapshot), trust_remote_code=True, local_files_only=True
    )
    encoder = AutoModel.from_config(hf_cfg, trust_remote_code=True).eval()
    _causal_frame_check(encoder, 4, 32, 16)
