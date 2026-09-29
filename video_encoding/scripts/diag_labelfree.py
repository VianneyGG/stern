"""Label-free representation-quality metrics, hand-implemented in numpy.

`reptrix` is rejected (design.md): it transitively pulls scikit-learn and
torchvision into the single TF/PyTorch venv CLAUDE.md requires. Every metric
here is therefore pure numpy, checked against a cross-metric ground truth (a
known `d`-dimensional linear subspace embedded in a higher-dimensional
ambient space) rather than mere self-consistency.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch

from video_encoding.scripts._diag_common import (
    DiagBundle,
    DiagReport,
    LADDER_REVISION,
    forward_with_hidden_states,
    paired_inputs,
    replicate,
)


def mst_length(points) -> float:
    """Total edge length of the Euclidean minimum spanning tree over an
    `(N, D)` point set, via Prim's algorithm. Order-invariant: the MST's
    total weight does not depend on which node the tree is grown from."""
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    if n <= 1:
        return 0.0
    in_tree = np.zeros(n, dtype=bool)
    min_edge = np.full(n, np.inf)
    min_edge[0] = 0.0
    total = 0.0
    for _ in range(n):
        candidate = np.where(in_tree, np.inf, min_edge)
        u = int(np.argmin(candidate))
        total += float(candidate[u])
        in_tree[u] = True
        dists = np.linalg.norm(points - points[u], axis=1)
        min_edge = np.minimum(min_edge, dists)
    return float(total)


def idest(embeddings, *, subsample_sizes, seed: int = 0) -> float:
    """Log-log regression of MST length across subsamples of increasing
    size, estimating intrinsic dimension (Costa & Hero's MST-based
    estimator). `seed` is honoured -- it is the only source of randomness
    here (which points get drawn into each subsample)."""
    embeddings = np.asarray(embeddings, dtype=np.float64)
    n = embeddings.shape[0]
    rng = np.random.default_rng(seed)

    sizes = sorted({min(int(m), n) for m in subsample_sizes if int(m) >= 2})
    log_sizes, log_lengths = [], []
    for m in sizes:
        idx = rng.choice(n, size=m, replace=False)
        length = mst_length(embeddings[idx])
        if length <= 0.0:
            continue
        log_sizes.append(np.log(m))
        log_lengths.append(np.log(length))

    if len(log_sizes) < 2:
        # Collapsed input drives every MST length to 0. NaN, never the ambient
        # dimension: that would report collapse as maximally high-dimensional.
        return float("nan")
    slope, _ = np.polyfit(log_sizes, log_lengths, 1)
    if slope >= 1.0:
        # 1/(1-m) is undefined or negative -- degenerate, not "huge dimension".
        return float("nan")
    return float(1.0 / (1.0 - slope))


def rankme(embeddings, eps: float = 1e-7) -> float:
    """RankMe (Garrido et al. 2023): the exponentiated Shannon entropy of the
    normalised singular-value spectrum of an `(N, D)` token matrix."""
    embeddings = np.asarray(embeddings, dtype=np.float64)
    s = np.linalg.svd(embeddings, compute_uv=False)
    p = s / (s.sum() + eps) + eps
    entropy = float(-np.sum(p * np.log(p)))
    return float(np.exp(entropy))


def alpha_req(embeddings) -> float:
    """alpha-ReQ: the power-law decay exponent of the feature-covariance
    eigenspectrum. Scale-invariant (a uniform rescale of the covariance
    shifts every log-eigenvalue by the same constant, leaving the fitted
    slope unchanged) and rotation-invariant (eigenvalues of `X^T X` are
    unchanged by `X -> X @ Q` for orthogonal `Q`)."""
    embeddings = np.asarray(embeddings, dtype=np.float64)
    x = embeddings - embeddings.mean(axis=0, keepdims=True)
    cov = (x.T @ x) / max(x.shape[0] - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)[::-1]
    eigvals = np.clip(eigvals, 1e-12, None)
    k = np.arange(1, len(eigvals) + 1)
    slope, _ = np.polyfit(np.log(k), np.log(eigvals), 1)
    return float(-slope)


def lidar(views_a, views_b, eps: float = 1e-7) -> float:
    """LiDAR (Thilak et al. 2023): the effective rank (RankMe-style entropy)
    of the generalised eigenspectrum of a positive-pair LDA surrogate task.

    Each row-pair `(views_a[i], views_b[i])` is treated as a 2-point class.
    A (near-)zero within-class scatter -- e.g. identical view pairs -- makes
    the generalised eigenproblem degenerate; that is guarded explicitly
    *before* any regulariser is added, so a plain `eps * I` add cannot paper
    over it and report a spuriously maximal rank.
    """
    a = np.asarray(views_a, dtype=np.float64)
    b = np.asarray(views_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"views_a {a.shape} and views_b {b.shape} must match")
    n, d = a.shape

    # Each 2-point class contributes diff diff^T / 2 over 2n samples.
    diff = a - b
    s_w = (diff.T @ diff) / (4 * n)

    stacked = np.concatenate([a, b], axis=0)
    centered = stacked - stacked.mean(axis=0, keepdims=True)
    s_t = (centered.T @ centered) / (2 * n)

    trace_t = float(np.trace(s_t))
    trace_w = float(np.trace(s_w))
    if trace_t < eps or trace_w / (trace_t + eps) < 1e-6:
        # Degenerate: (near-)zero within-class scatter, e.g. identical view
        # pairs. Flagged before regularisation, per design.md.
        return float("nan")

    s_b = s_t - s_w
    s_w_reg = s_w + eps * np.eye(d)
    eigvals = np.linalg.eigvals(np.linalg.solve(s_w_reg, s_b))
    eigvals = np.clip(eigvals.real, 0.0, None)
    p = eigvals / (eigvals.sum() + eps) + eps
    entropy = float(-np.sum(p * np.log(p)))
    return float(np.exp(entropy))


_METRIC_REGISTRY = {"rankme": rankme, "alpha_req": alpha_req, "idest": idest}


def by_block(
    bundle: DiagBundle,
    pixel_values: torch.Tensor,
    metric_name: str,
    blocks: Iterable[int] | None = None,
) -> dict[int, float]:
    """`metric_name` at each of `blocks`' outputs (default: every block),
    dispatched against an explicit name registry (never `__name__`/identity)
    -- see design.md. Scored over patch tokens only: huge-norm register
    tokens would dominate the spectrum and read as collapse."""
    if metric_name not in _METRIC_REGISTRY:
        raise ValueError(
            f"unknown metric_name {metric_name!r}; choose from "
            f"{sorted(_METRIC_REGISTRY)}"
        )
    fn = _METRIC_REGISTRY[metric_name]
    states = forward_with_hidden_states(bundle.encoder, pixel_values)
    if blocks is None:
        blocks = range(len(bundle.encoder.blocks))

    result: dict[int, float] = {}
    for i in blocks:
        tokens = bundle.encoder.registers.strip(states[i + 1]).detach().cpu().numpy()
        flat = tokens.reshape(-1, tokens.shape[-1])
        if metric_name == "idest":
            n = flat.shape[0]
            sizes = sorted({s for s in (max(4, n // 4), max(6, n // 2), n) if s >= 2})
            result[i] = fn(flat, subsample_sizes=sizes)
        else:
            result[i] = fn(flat)
    return result


_CHECK_METRIC_NAMES = ("rankme", "alpha_req", "idest")


class _DegenerateSample(Exception):
    """A replicate drew a NaN. `replicate` cannot band a NaN (statistics.stdev
    raises on one), so `check` aborts that metric and records why."""

    def __init__(self, window_seed: int):
        super().__init__(window_seed)
        self.window_seed = window_seed


def check(bundle: DiagBundle, inputs: dict) -> dict[str, DiagReport]:
    """Scores `rankme`, `alpha_req`, and `idest` at the final block, each
    replicated over three independently-drawn windows, for both `"real"` and
    `"random"` input kinds -- one report per kind, keyed
    `diag_labelfree_<kind>`. Requires `inputs["shard_path"]` -- see design.md's
    documented follow-up for threading it through Plan 5's orchestrator.

    A metric whose replicate comes back NaN (`idest` on collapsed or
    underpowered tokens) is left out of `metrics` and recorded as
    `meta["<metric>_degenerate"]` -- collapse is a finding to report, not a
    reason to take the whole check down.

    Known gap (task 3.6): `lidar`'s `check()` wiring -- the two-masking
    positive pair it needs -- requires
    `models.videomaev2.masking.build_masker` to produce the second view.
    That is not wired here; a `lidar_pair` helper is a follow-up.
    """
    shard_path = inputs["shard_path"]
    n_windows = int(inputs.get("n_windows", 4))
    base_seed = int(inputs.get("seed", 0))
    num_frames = bundle.encoder.num_frames
    image_size = bundle.encoder.image_size
    final_block = len(bundle.encoder.blocks) - 1

    reports: dict[str, DiagReport] = {}
    for kind_idx, kind in enumerate(("real", "random")):
        metrics = {}
        meta = {"LADDER_REVISION": LADDER_REVISION, "input_kind": kind}
        for metric_idx, metric_name in enumerate(_CHECK_METRIC_NAMES):

            def _sample(
                window_seed: int, kind: str = kind, metric_name: str = metric_name
            ) -> float:
                real, random_batch = paired_inputs(
                    shard_path,
                    n_windows=n_windows,
                    num_frames=num_frames,
                    image_size=image_size,
                    seed=window_seed,
                )
                pixel_values = real if kind == "real" else random_batch
                value = by_block(
                    bundle, pixel_values, metric_name, blocks=[final_block]
                )[final_block]
                if np.isnan(value):
                    raise _DegenerateSample(window_seed)
                return value

            metric_seed = base_seed + kind_idx * 1000 + metric_idx * 100
            try:
                metrics[metric_name] = replicate(_sample, n=3, seed=metric_seed)
            except _DegenerateSample as exc:
                meta[f"{metric_name}_degenerate"] = (
                    f"NaN at window seed {exc.window_seed}"
                )
        reports[f"diag_labelfree_{kind}"] = DiagReport(
            metrics=metrics, arrays={}, meta=meta
        )
    return reports
