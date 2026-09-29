"""Tests for `video_encoding/scripts/diag_labelfree.py` -- pure-numpy
label-free representation-quality metrics (MST/IdEst, RankMe, alpha-ReQ,
LiDAR), plus their per-block/DiagReport wiring.

`reptrix` is explicitly rejected (design.md): every metric here is
hand-implemented against numpy only, so a cross-metric ground-truth
construction (a known `d`-dimensional linear subspace) is the load-bearing
check, not mere self-consistency.
"""

from __future__ import annotations

import io
import json
import pickle
import tarfile

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from video_encoding.models.videomaev2.encoder import VideoMAEEncoder
from video_encoding.scripts._diag_common import DiagBundle, forward_with_hidden_states
from video_encoding.scripts.diag_labelfree import (
    alpha_req,
    by_block,
    check,
    idest,
    lidar,
    mst_length,
    rankme,
)


def _tiny_model_cfg(**overrides):
    cfg = {
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
    }
    cfg.update(overrides)
    return OmegaConf.create(cfg)


def _jpeg_bytes(color: tuple[int, int, int], size: int = 32) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (size, size), color=color).save(buf, format="JPEG")
    return buf.getvalue()


def make_fake_shard(tmp_path, clip_frame_counts: dict[str, int]):
    shard_path = tmp_path / "shard-000000.tar"
    manifest = {}
    with tarfile.open(shard_path, "w") as tar:
        for clip_key, n_frames in clip_frame_counts.items():
            frames = [_jpeg_bytes((i * 7 % 256, 0, 0)) for i in range(n_frames)]
            payload = pickle.dumps(frames)
            info = tarfile.TarInfo(name=f"{clip_key}.pickle")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
            manifest[clip_key] = {
                "shard": shard_path.name,
                "num_frames": n_frames,
                "size": len(payload),
            }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return shard_path


def _subspace_points(
    *, d: int, ambient: int, n: int, noise: float = 0.01, seed: int = 0
):
    """`n` points near a `d`-dim linear subspace embedded in `R^ambient`."""
    rng = np.random.default_rng(seed)
    basis = rng.standard_normal((d, ambient))
    coords = rng.standard_normal((n, d))
    points = coords @ basis
    points += noise * rng.standard_normal((n, ambient))
    return points


# --- mst_length --------------------------------------------------------


def test_mst_length_known_geometry():
    # Unit square's MST is 3 unit edges (a path visiting all 4 corners).
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    assert mst_length(points) == pytest.approx(3.0, rel=1e-6)


def test_mst_length_single_point_is_zero():
    assert mst_length(np.array([[1.0, 2.0, 3.0]])) == pytest.approx(0.0)


def test_mst_length_invariant_to_point_order():
    rng = np.random.default_rng(0)
    points = rng.standard_normal((20, 5))
    shuffled = points[rng.permutation(20)]
    assert mst_length(points) - mst_length(shuffled) == pytest.approx(0.0, abs=1e-9)


# --- idest ---------------------------------------------------------------


def test_idest_finite_below_smallest_subsample_size():
    """A dataset smaller than the largest requested subsample size must still
    yield a finite estimate (clipped to the available point count), not a
    crash or a NaN."""
    points = _subspace_points(d=2, ambient=16, n=30, seed=1)
    value = idest(points, subsample_sizes=[10, 20, 1000], seed=0)
    assert value == pytest.approx(2.0, rel=0.5)


@pytest.mark.parametrize("d", [2, 8, 32])
def test_idest_recovers_known_subspace_dimension(d):
    points = _subspace_points(d=d, ambient=64, n=1500, noise=0.01, seed=d)
    value = idest(points, subsample_sizes=[100, 300, 600, 1000, 1500], seed=0)
    assert value == pytest.approx(d, rel=0.5)


def test_idest_is_nan_on_collapsed_input_not_ambient_dim():
    """Zero-variance input drives every MST length to 0, so no regression
    pair survives. The answer must be NaN -- returning the ambient dimension
    (32) would report collapse as maximally high-dimensional."""
    value = idest(np.zeros((100, 32)), subsample_sizes=[10, 50])
    assert value == pytest.approx(float("nan"), nan_ok=True)
    assert value != pytest.approx(32.0)


def test_idest_honours_seed_and_is_reproducible():
    points = _subspace_points(d=4, ambient=32, n=200, seed=2)
    a = idest(points, subsample_sizes=[50, 100, 150, 200], seed=5)
    b = idest(points, subsample_sizes=[50, 100, 150, 200], seed=5)
    c = idest(points, subsample_sizes=[50, 100, 150, 200], seed=6)
    assert a == pytest.approx(4.0, rel=0.5)
    assert a == pytest.approx(b)
    assert a != pytest.approx(c)


# --- rankme ----------------------------------------------------------------


def test_rankme_rank_one_scores_near_one():
    rng = np.random.default_rng(0)
    base = rng.standard_normal((1, 16))
    matrix = np.repeat(base, 50, axis=0) * rng.standard_normal((50, 1))
    assert rankme(matrix) == pytest.approx(1.0, abs=0.05)


def test_rankme_full_rank_isotropic_scores_near_ambient_dim():
    rng = np.random.default_rng(0)
    matrix = rng.standard_normal((500, 16))
    assert rankme(matrix) == pytest.approx(16.0, rel=0.15)


@pytest.mark.parametrize("d", [2, 8, 32])
def test_rankme_recovers_known_subspace_dimension(d):
    points = _subspace_points(d=d, ambient=64, n=1500, noise=0.01, seed=d)
    assert rankme(points) == pytest.approx(d, rel=0.5)


def test_rankme_monotonic_from_rank_one_to_full_rank():
    rng = np.random.default_rng(0)
    n, dim = 60, 10
    u, _, vt = np.linalg.svd(rng.standard_normal((n, dim)), full_matrices=False)
    values = []
    for k in range(1, dim + 1):
        s = np.zeros(dim)
        s[:k] = 1.0
        matrix = (u * s) @ vt
        values.append(rankme(matrix))
    # k equal singular values -> uniform spectrum -> exp(entropy) == k exactly.
    assert values == pytest.approx(list(range(1, dim + 1)), rel=1e-3)


def test_rankme_invariant_to_rotation_and_row_order():
    rng = np.random.default_rng(0)
    matrix = rng.standard_normal((40, 12))
    q, _ = np.linalg.qr(rng.standard_normal((12, 12)))
    rotated = matrix @ q
    permuted = matrix[rng.permutation(40)]
    base = rankme(matrix)
    # Isotropic Gaussian: near-full rank.
    assert base == pytest.approx(12.0, rel=0.15)
    assert rankme(rotated) == pytest.approx(base, rel=1e-3)
    assert rankme(permuted) == pytest.approx(base, rel=1e-9)


def test_rankme_invariant_to_isotropic_rescaling():
    rng = np.random.default_rng(0)
    matrix = rng.standard_normal((40, 12))
    assert rankme(matrix) == pytest.approx(12.0, rel=0.15)
    assert rankme(matrix * 5.0) == pytest.approx(rankme(matrix), rel=1e-3)


# --- alpha_req ---------------------------------------------------------


def test_alpha_req_scale_invariant():
    rng = np.random.default_rng(0)
    matrix = rng.standard_normal((200, 20))
    # Isotropic Gaussian: spectrum far flatter than the 1/k of alpha = 1.
    assert alpha_req(matrix) == pytest.approx(0.5, abs=0.5)  # in [0, 1]
    assert alpha_req(matrix * 3.0) == pytest.approx(alpha_req(matrix), rel=0.05)


def test_alpha_req_invariant_to_rotation():
    rng = np.random.default_rng(0)
    matrix = rng.standard_normal((200, 20))
    q, _ = np.linalg.qr(rng.standard_normal((20, 20)))
    assert alpha_req(matrix) == pytest.approx(0.5, abs=0.5)  # in [0, 1]
    assert alpha_req(matrix @ q) == pytest.approx(alpha_req(matrix), rel=0.05)


# --- lidar ---------------------------------------------------------------


def test_lidar_flags_identical_views_as_degenerate_not_high_rank():
    rng = np.random.default_rng(0)
    views = rng.standard_normal((30, 8))
    value = lidar(views, views.copy())
    assert value == pytest.approx(float("nan"), nan_ok=True)


def test_lidar_nondegenerate_pairing_returns_finite_rank():
    rng = np.random.default_rng(0)
    means = rng.standard_normal((30, 8)) * 3.0
    views_a = means + 0.1 * rng.standard_normal((30, 8))
    views_b = means + 0.1 * rng.standard_normal((30, 8))
    value = lidar(views_a, views_b)
    # RankMe-style entropy score over an 8-dim LDA spectrum: in [1, 8].
    assert value == pytest.approx(4.5, abs=3.5 + 1e-6)  # in [1, 8]


def test_lidar_views_sharing_no_signal_score_full_rank():
    """Independent views: each 2-point class's within scatter is diff diff^T/2
    over 2n samples, so S_w = S_b = Sigma/2 and every generalised eigenvalue
    is ~1 -> uniform spectrum -> rank ~ d. Overcounting S_w by 2x drives
    S_b = S_t - S_w to ~0 and the rank to noise."""
    rng = np.random.default_rng(0)
    views_a = rng.standard_normal((5000, 8))
    views_b = rng.standard_normal((5000, 8))
    assert lidar(views_a, views_b) == pytest.approx(8.0, rel=0.05)


def test_finiteness_sweep_every_metric_on_freshly_init_activations():
    cfg = _tiny_model_cfg()
    # Seeded: with only two subsample sizes, idest's log-log slope can reach
    # >= 1 on some inits and return its degenerate NaN (e.g. seed 1).
    torch.manual_seed(0)
    encoder = VideoMAEEncoder(cfg).eval()
    x = torch.randn(4, 3, 4, 32, 32)

    states = forward_with_hidden_states(encoder, x)
    tokens = states[-1].detach().numpy().reshape(-1, cfg.embed_dim)

    r = rankme(tokens)
    assert r == pytest.approx(cfg.embed_dim, rel=0.15)

    a = alpha_req(tokens)
    assert a == pytest.approx(1.5, abs=1.5)  # in [0, 3]

    i = idest(tokens, subsample_sizes=[tokens.shape[0] // 2, tokens.shape[0]])
    assert i == pytest.approx(cfg.embed_dim / 2, abs=cfg.embed_dim / 2)  # in [0, D]


def test_batch_split_invariance():
    """rankme/alpha_req computed on the full token matrix must not depend on
    an arbitrary split into two batches -- both read the same matrix either
    way, since neither metric processes batches separately."""
    rng = np.random.default_rng(0)
    matrix = rng.standard_normal((80, 16))
    half_a, half_b = matrix[:40], matrix[40:]
    combined = np.concatenate([half_a, half_b], axis=0)
    assert rankme(matrix) == pytest.approx(16.0, rel=0.15)
    assert rankme(combined) == pytest.approx(rankme(matrix))
    assert alpha_req(combined) == pytest.approx(alpha_req(matrix))


# --- by_block / check -------------------------------------------------


def test_by_block_returns_one_value_per_block():
    cfg = _tiny_model_cfg(depth=3)
    encoder = VideoMAEEncoder(cfg).eval()
    bundle = DiagBundle(encoder=encoder, model=None, run_id="r", round=None)
    pixel_values = torch.randn(4, 3, 4, 32, 32)

    result = by_block(bundle, pixel_values, "rankme")

    assert set(result.keys()) == set(range(3))
    for value in result.values():
        # in [1, D]
        assert value == pytest.approx(
            (1 + cfg.embed_dim) / 2, abs=(cfg.embed_dim - 1) / 2
        )


def test_by_block_scores_patch_tokens_only_not_registers():
    """Huge-norm register tokens would dominate the spectrum and read as
    collapse. by_block must score the register-stripped patch tokens."""
    cfg = _tiny_model_cfg(depth=2, num_registers=4)
    encoder = VideoMAEEncoder(cfg).eval()
    with torch.no_grad():
        encoder.registers.tokens.mul_(1000.0)
    bundle = DiagBundle(encoder=encoder, model=None, run_id="r", round=None)
    pixel_values = torch.randn(4, 3, 4, 32, 32)

    states = forward_with_hidden_states(encoder, pixel_values)
    patch = states[-1][:, 4:, :].detach().numpy().reshape(-1, cfg.embed_dim)
    with_registers = states[-1].detach().numpy().reshape(-1, cfg.embed_dim)
    assert rankme(with_registers) != pytest.approx(rankme(patch), rel=0.05)

    result = by_block(bundle, pixel_values, "rankme")

    assert result[1] == pytest.approx(rankme(patch), rel=1e-6)


def test_by_block_scores_only_requested_blocks():
    cfg = _tiny_model_cfg(depth=3)
    encoder = VideoMAEEncoder(cfg).eval()
    bundle = DiagBundle(encoder=encoder, model=None, run_id="r", round=None)
    pixel_values = torch.randn(4, 3, 4, 32, 32)

    full = by_block(bundle, pixel_values, "rankme")
    last = by_block(bundle, pixel_values, "rankme", blocks=[2])

    assert last.keys() == {2}
    assert last[2] == pytest.approx(full[2])


def test_check_scores_one_block_per_replicate(tmp_path, monkeypatch):
    """check reads only the final block, so each replicate must cost one
    metric call, not one per block (idest's MSTs are O(N^2 D) each)."""
    from video_encoding.scripts import diag_labelfree

    calls = []
    for name, fn in list(diag_labelfree._METRIC_REGISTRY.items()):
        monkeypatch.setitem(
            diag_labelfree._METRIC_REGISTRY,
            name,
            lambda x, _fn=fn, _name=name, **kw: calls.append(_name) or _fn(x, **kw),
        )
    shard = make_fake_shard(tmp_path, {f"clip{i}": 8 for i in range(10)})
    encoder = VideoMAEEncoder(_tiny_model_cfg(depth=3)).eval()
    bundle = DiagBundle(encoder=encoder, model=None, run_id="r", round=None)

    check(bundle, {"shard_path": shard, "n_windows": 8})

    # 2 input kinds x 3 metrics x 3 replicates.
    assert len(calls) == 18


def test_by_block_rejects_unknown_metric_name():
    cfg = _tiny_model_cfg()
    encoder = VideoMAEEncoder(cfg).eval()
    bundle = DiagBundle(encoder=encoder, model=None, run_id="r", round=None)
    with pytest.raises(ValueError, match="unknown"):
        by_block(bundle, torch.randn(1, 3, 4, 32, 32), "not_a_metric")


def test_check_draws_three_independent_windows_not_one_fixed_input(tmp_path):
    shard = make_fake_shard(tmp_path, {f"clip{i}": 8 for i in range(10)})
    cfg = _tiny_model_cfg()
    encoder = VideoMAEEncoder(cfg).eval()
    bundle = DiagBundle(encoder=encoder, model=None, run_id="r", round=None)

    # 8 windows -> 64 token rows: at 2 windows (16 rows) idest's regression is
    # underpowered and legitimately NaN on the random batch.
    reports = check(bundle, {"shard_path": shard, "n_windows": 8})

    assert reports.keys() == {"diag_labelfree_real", "diag_labelfree_random"}
    seen_seed_sets = set()
    for kind, report in reports.items():
        assert report.meta["input_kind"] == kind.removeprefix("diag_labelfree_")
        assert report.metrics.keys() == {"rankme", "alpha_req", "idest"}
        for metric_value in report.metrics.values():
            assert metric_value.n == 3
            assert len(set(metric_value.seeds)) == 3
            seen_seed_sets.add(metric_value.seeds)
    # Distinct metric/input-kind combinations must not all replay one fixed
    # window set under the hood -- at least some seed sets differ.
    assert len(seen_seed_sets) >= 2


def test_check_records_nan_metric_as_degenerate_instead_of_crashing(
    tmp_path, monkeypatch
):
    """A collapsed checkpoint makes idest NaN, and replicate cannot band a NaN
    (statistics.stdev raises). check must report that metric as degenerate in
    meta and still score the others -- not take the whole check down."""
    from video_encoding.scripts import diag_labelfree

    monkeypatch.setitem(
        diag_labelfree._METRIC_REGISTRY, "idest", lambda x, **_: float("nan")
    )
    shard = make_fake_shard(tmp_path, {f"clip{i}": 8 for i in range(10)})
    encoder = VideoMAEEncoder(_tiny_model_cfg()).eval()
    bundle = DiagBundle(encoder=encoder, model=None, run_id="r", round=None)

    reports = check(bundle, {"shard_path": shard, "n_windows": 2})

    for report in reports.values():
        assert report.metrics.keys() == {"rankme", "alpha_req"}
        assert report.meta["idest_degenerate"].startswith("NaN at window seed ")
