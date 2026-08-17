import io
import pickle
import tarfile

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from video_encoding.models.videomaev2.encoder import preprocess_window
from video_encoding.models.videomaev2.mae import VideoMAEForPreTraining
from video_encoding.scripts.export_vit_internals import (
    attention_norm_density,
    consecutive_cosine_similarity,
    load_encoder,
    load_sample_windows,
    make_log_bin_edges,
    pca_rgb,
    received_attention_map,
    run_encoder_with_internals,
    sample_clip_keys,
)


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
        }
    )


def _jpeg_bytes(color: tuple[int, int, int], size: int = 8) -> bytes:
    img = Image.new("RGB", (size, size), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _write_shard(tmp_path, clip_frame_counts: dict[str, int]):
    shard_path = tmp_path / "shard-000000.tar"
    manifest = {}
    with tarfile.open(shard_path, "w") as tar:
        for clip_key, n_frames in clip_frame_counts.items():
            frames = [_jpeg_bytes((i, 0, 0)) for i in range(n_frames)]
            payload = pickle.dumps(frames)
            info = tarfile.TarInfo(name=f"{clip_key}.pickle")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
            manifest[clip_key] = {
                "shard": shard_path.name,
                "num_frames": n_frames,
                "size": len(payload),
            }
    (tmp_path / "manifest.json").write_text(__import__("json").dumps(manifest))
    return shard_path


# --- window sourcing ---------------------------------------------------


def test_sample_clip_keys_picks_requested_count_deterministically(tmp_path):
    shard_path = _write_shard(tmp_path, {f"clip{i}": 10 for i in range(10)})
    a = sample_clip_keys(shard_path, n_windows=3, seed=42)
    b = sample_clip_keys(shard_path, n_windows=3, seed=42)
    assert a == b
    assert len(a) == 3
    assert set(a) <= {f"clip{i}" for i in range(10)}


def test_sample_clip_keys_matches_random_sample_for_a_given_seed(tmp_path):
    clip_keys = sorted(f"clip{i}" for i in range(10))
    shard_path = _write_shard(tmp_path, {k: 10 for k in clip_keys})

    a = sample_clip_keys(shard_path, n_windows=3, seed=1)
    b = sample_clip_keys(shard_path, n_windows=3, seed=2)

    # Pinned: random.Random(1).sample(clip_keys, 3) / random.Random(2).sample(clip_keys, 3)
    assert a == ["clip2", "clip1", "clip4"]
    assert b == ["clip0", "clip1", "clip8"]


def test_sample_clip_keys_raises_when_shard_too_small(tmp_path):
    shard_path = _write_shard(tmp_path, {"clip0": 10})
    with pytest.raises(ValueError, match="only 1 eligible clips"):
        sample_clip_keys(shard_path, n_windows=3, seed=42)


def test_sample_clip_keys_excludes_clips_shorter_than_num_frames(tmp_path):
    shard_path = _write_shard(tmp_path, {"short": 2, "long0": 10, "long1": 10})
    shard_index = {
        "short": {"num_frames": 2},
        "long0": {"num_frames": 10},
        "long1": {"num_frames": 10},
    }
    picked = sample_clip_keys(
        shard_path, n_windows=2, seed=0, shard_index=shard_index, num_frames=4
    )
    assert "short" not in picked
    assert set(picked) == {"long0", "long1"}


def test_load_sample_windows_reads_only_sampled_clips(tmp_path):
    shard_path = _write_shard(tmp_path, {f"clip{i}": 12 for i in range(5)})
    windows = load_sample_windows(
        shard_path, n_windows=2, num_frames=4, image_size=8, seed=7
    )
    assert len(windows) == 2
    for clip_key, frames in windows:
        assert clip_key.startswith("clip")
        assert len(frames) == 4
        assert frames[0].shape == (8, 8, 3)


def test_load_sample_windows_raises_when_no_clip_is_long_enough(tmp_path):
    shard_path = _write_shard(tmp_path, {"clip0": 2})
    with pytest.raises(ValueError, match="eligible clips"):
        load_sample_windows(shard_path, n_windows=1, num_frames=4, image_size=8, seed=0)


def test_load_sample_windows_skips_clips_shorter_than_the_window(tmp_path):
    shard_path = _write_shard(tmp_path, {"short": 2, "long0": 12, "long1": 12})
    windows = load_sample_windows(
        shard_path, n_windows=2, num_frames=4, image_size=8, seed=3
    )
    assert {clip_key for clip_key, _ in windows} == {"long0", "long1"}


# --- checkpoint loading (the bug this redesign fixes) -------------------


def test_load_encoder_random_init_needs_no_checkpoint(tmp_path):
    cfg = _tiny_model_cfg()
    encoder = load_encoder("random_init", tmp_path / "does-not-exist", cfg)
    assert encoder.embed_dim == 16


def test_load_encoder_loads_correctly_prefixed_state_dict(tmp_path):
    cfg = _tiny_model_cfg()
    source_model = VideoMAEForPreTraining(cfg)
    ckpt_dir = tmp_path / "checkpoints" / "run_a"
    ckpt_dir.mkdir(parents=True)
    torch.save(
        {"model_state_dict": source_model.state_dict(), "master_version": 1},
        ckpt_dir / "master.pt",
    )

    loaded = load_encoder("run_a", tmp_path, cfg)

    for (name, p_expected), (_, p_loaded) in zip(
        source_model.encoder.state_dict().items(), loaded.state_dict().items()
    ):
        assert torch.equal(p_expected, p_loaded) is True, name


def test_load_encoder_raises_on_unprefixed_state_dict(tmp_path):
    """Regression guard: master.pt holds a VideoMAEForPreTraining state dict
    (keys prefixed `encoder.`), not a bare VideoMAEEncoder state dict. Loading
    the unprefixed form must fail loudly, not silently pass a random encoder."""
    cfg = _tiny_model_cfg()
    bare_encoder_state = VideoMAEForPreTraining(cfg).encoder.state_dict()
    ckpt_dir = tmp_path / "checkpoints" / "run_b"
    ckpt_dir.mkdir(parents=True)
    torch.save(
        {"model_state_dict": bare_encoder_state, "master_version": 1},
        ckpt_dir / "master.pt",
    )

    with pytest.raises(ValueError, match="key mismatch"):
        load_encoder("run_b", tmp_path, cfg)


def test_load_encoder_missing_checkpoint_raises_file_not_found(tmp_path):
    cfg = _tiny_model_cfg()
    with pytest.raises(FileNotFoundError):
        load_encoder("no_such_run", tmp_path, cfg)


# --- instrumented forward: must match the real encoder exactly ----------


@pytest.mark.xfail(
    reason=(
        "run_encoder_with_internals calls block.attn(h, h, h, need_weights=True, "
        "average_attn_weights=False), the old nn.MultiheadAttention calling "
        "convention. ViTBlock's attn is now SelfAttention (single-tensor input, "
        "no average_attn_weights kwarg) -- known, deferred to the follow-up "
        "change that updates this diagnostics script's block-internals access "
        "(see docs/superpowers/plans/2026-08-16-videomae-vit-block-rewrite.md)."
    ),
    strict=True,
)
def test_run_encoder_with_internals_matches_real_forward():
    cfg = _tiny_model_cfg()
    model = VideoMAEForPreTraining(cfg).eval()
    encoder = model.encoder

    frames = [np.zeros((32, 32, 3), dtype=np.uint8) + i * 20 for i in range(4)]
    pixel_values = preprocess_window(frames, num_frames=4, image_size=32)

    with torch.no_grad():
        real_output = encoder(pixel_values)[0]  # (D,) mean-pooled clip feature

    internals = run_encoder_with_internals(encoder, pixel_values)
    manual_output = torch.from_numpy(internals["final_tokens"]).mean(dim=0)

    assert torch.allclose(real_output, manual_output, atol=1e-5)
    assert len(internals["hidden"]) == cfg.depth
    assert len(internals["received_attn"]) == cfg.depth
    assert internals["received_attn"][0].shape == (encoder.pos_embed.shape[1],)


# --- pure reduction functions --------------------------------------------


def test_received_attention_map_uniform_attention_is_flat():
    n_tokens = 16
    token_grid = (1, 4, 4)  # T'=1 so temporal averaging is a no-op
    received = np.full((n_tokens,), 1.0 / n_tokens, dtype=np.float32)

    grid = received_attention_map(received, token_grid)

    assert grid.shape == (4, 4)
    assert np.allclose(grid, 1.0 / n_tokens)


def test_pca_rgb_recovers_dominant_axis_with_consistent_sign():
    token_grid = (1, 4, 4)
    n_tokens = 16
    rng = np.random.default_rng(0)
    tokens = rng.normal(scale=1e-3, size=(n_tokens, 6)).astype(np.float32)
    tokens[:, 0] = np.arange(n_tokens)  # dominant, monotonically increasing axis

    rgb = pca_rgb(tokens, token_grid)

    assert rgb.shape == (4, 4, 3)
    assert rgb.dtype == np.uint8
    channel0 = rgb[..., 0].astype(np.float64).flatten()
    order = np.arange(n_tokens)
    corr = np.corrcoef(channel0, order)[0, 1]
    assert corr > 0.9, (
        f"expected strong positive correlation with token order, got {corr}"
    )


def test_consecutive_cosine_similarity_orthogonal_layers_near_zero():
    n_tokens, dim = 8, 4
    layer_a = np.tile(np.eye(dim)[0], (n_tokens, 1)).astype(np.float32)
    layer_b = np.tile(np.eye(dim)[1], (n_tokens, 1)).astype(np.float32)

    sims = consecutive_cosine_similarity([layer_a, layer_b, layer_a])

    assert sims.shape == (2,)
    assert np.allclose(sims, 0.0, atol=1e-6)


def test_consecutive_cosine_similarity_identical_layers_is_one():
    n_tokens = 8
    layer = np.tile(np.array([1.0, 2.0, 0.0, 0.0]), (n_tokens, 1)).astype(np.float32)

    sims = consecutive_cosine_similarity([layer, layer])

    assert sims == pytest.approx(1.0, abs=1e-6)


def test_make_log_bin_edges_is_ascending_and_log_spaced():
    norms = np.array([0.1, 1.0, 10.0, 100.0])
    edges = make_log_bin_edges(norms, n_bins=4)

    assert edges.shape == (5,)
    assert np.all(np.diff(edges) > 0)
    ratios = edges[1:] / edges[:-1]
    assert np.allclose(ratios, ratios[0], rtol=1e-4)


def test_make_log_bin_edges_handles_degenerate_constant_input():
    norms = np.array([5.0, 5.0, 5.0])
    edges = make_log_bin_edges(norms, n_bins=4)
    assert edges.shape == (5,)
    assert np.all(np.diff(edges) > 0)


def test_attention_norm_density_rows_sum_to_one_and_bin_correctly():
    bin_edges = np.array([1.0, 2.0, 4.0, 8.0], dtype=np.float32)
    layer0 = np.array([1.5, 1.5, 1.5, 1.5])  # all in bin [1, 2)
    layer1 = np.array([6.0, 6.0])  # all in bin [4, 8)

    density = attention_norm_density([layer0, layer1], bin_edges)

    assert density.shape == (2, 3)
    assert np.allclose(density.sum(axis=1), 1.0)
    assert density[0, 0] == pytest.approx(1.0)
    assert density[1, 2] == pytest.approx(1.0)
