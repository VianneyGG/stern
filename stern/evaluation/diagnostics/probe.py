"""The attentive probe: what is actually recoverable for the downstream task.

Every per-depth diagnostic in `depth.py` reads weights only. This module is the
one check that says what is actually recoverable
for the downstream VA task -- a frozen-encoder probe fit at each requested
depth, read against a `random_init` capacity control so the reading is a
paired difference, never an unpaired absolute.

Every quantity that changes how well the probe fits is a pinned module
constant, not a per-call default: varying the fit budget across depth or rung
would manufacture the very profile this module exists to measure honestly.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from stern.data.manifest import ManifestRecord, read_manifest
from stern.evaluation.ccc import ccc_loss, concordance_correlation_coefficient
from stern.data.loaders import LabelledDataset, identity_disjoint_split
from stern.data.probe_windows import VAWindowDataset
from stern.models.vit.block import BlockConfig, ViTBlock
from stern.models.videomae.regressor import LinearVAHead
from stern.evaluation.diagnostics.common import (
    LADDER_REVISION,
    DiagBundle,
    DiagReport,
    MetricValue,
    encoder_geometry,
    forward_with_hidden_states,
    patch_tokens,
    replicate,
)

# Capacity constants. AttentiveProbe's parameter count depends only on these
# three -- never on which encoder block or ladder rung it is trained against.
PROBE_EMBED_DIM = 768
PROBE_NUM_HEADS = 16
PROBE_N_BLOCKS = 3

# Fit-budget constants. Every quantity that changes how well the probe fits is
# pinned here, not read from a per-dataset config or varied per call.
# 1e-5, not 1e-3: at 1e-3 the raw predictions blew up within one epoch on real
# AFEW-VA and ccc_loss's gradient vanished.
PROBE_FIT_LR = 1e-5
PROBE_FIT_EPOCHS = 20
# ccc_loss estimates two variances from the batch. A batch this small turns
# sampling noise into MetricValue.sd, which has nothing to do with the encoder.
PROBE_BATCH_SIZE = 16
PROBE_WINDOW_OVERLAP = 0.0
# AFEW-VA is ~30fps, but the SSL corpus is 8.33fps. At stride 1 a 16-frame
# window spans 0.53s, against 1.92s in pretraining, so per-frame motion is
# ~3.6x smaller than in training. Stride 4 matches the span but leaves too few
# clips to fit a probe. Stride 2 halves the mismatch and keeps 450 clips, the
# usable trade-off. Pinned like the fit budget: varying it per arm would turn
# the depth profile into a temporal-scale profile.
PROBE_FRAME_STRIDE = 2
PROBE_VA_RANGE = (-1.0, 1.0)
# 48 of AFEW-VA's 240 actors are held out. Fixed here, not tuned per checkpoint.
PROBE_SPLIT_FRACTION = 0.2
# Chunk size for the scoring forward pass. Not a fit-budget constant: the probe
# is in eval mode with no cross-window term, so chunking is numerically inert
# and exists only because the scored split is ~1.8GB of fp32 tokens at real
# scale and will not go through a GPU in one piece.
_SCORE_CHUNK = 32

# Default depth sweep for probe_by_depth, for a 12-block ViT-B encoder.
DEFAULT_PROBE_DEPTH_LAYERS = (1, 4, 8, 12)
# Rung 4 is the 24-block HF LeVJEPA ViT-L. Its layers are read at the same
# relative depth as the ViT-B rungs, so layer 2k there stands against layer k here.
RUNG4_PROBE_DEPTH_LAYERS = tuple(2 * layer for layer in DEFAULT_PROBE_DEPTH_LAYERS)

# The two arms' replicate seed bases for probe_by_depth: independent of each
# other and of layer, so a depth profile is never secretly a seed profile, and
# the two arms are not handed identical seeds when bundle is control_bundle.
_CHECKPOINT_SEED_BASE = 0
_CONTROL_SEED_BASE = 1

# The seed used only to order the (non-shuffled) materialization loaders.
# DataLoader always draws a base_seed from `generator or the global RNG`
# (torch/utils/data/dataloader.py) even with shuffle=False, so this is here to
# avoid silently consuming the caller's global RNG -- it plays no role in
# which windows land in which batch.
_MATERIALIZE_SEED = 0


class AttentiveProbe(torch.nn.Module):
    """`n_blocks` self-attention `ViTBlock`s feeding `LinearVAHead` as a
    cross-attention readout.

    Reuses `LinearVAHead` rather than re-implementing cross-attention pooling
    -- it is the shipped VA head `train.py` trains, and this probe being
    architecturally identical to the real head is worth more here than
    protocol fidelity to V-JEPA 2's four-block-with-residual probe.
    """

    def __init__(
        self,
        embed_dim: int = PROBE_EMBED_DIM,
        num_heads: int = PROBE_NUM_HEADS,
        n_blocks: int = PROBE_N_BLOCKS,
    ):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            ViTBlock(BlockConfig(embed_dim=embed_dim, num_heads=num_heads))
            for _ in range(n_blocks)
        )
        # num_heads passed explicitly: LinearVAHead's own default is 12, and a
        # readout running at a different head count from the blocks in front
        # of it is a silent construction bug, not a configuration choice.
        # bounded=False: a tanh output saturated to exactly +-1 within two fit
        # epochs on real AFEW-VA, which killed every fit.
        self.head = LinearVAHead(embed_dim, num_heads=num_heads, bounded=False)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def permute_labels(labels: "torch.Tensor", seed: int) -> "torch.Tensor":
    """`labels` with its rows shuffled: the protocol's negative control.

    Breaks the token/label pairing while leaving every other property of the
    fit untouched -- same split, same budget, same label distribution, and each
    (valence, arousal) row intact, so the label's own correlation structure
    survives and only its correspondence to the windows is destroyed. A
    materially non-zero CCC under this control means the protocol is reading
    something other than the encoder's features.

    Drawn from a local `torch.Generator`, never the global RNG: `probe_ccc` and
    `_fit_and_score_probe` both go out of their way to leave the caller's global
    RNG untouched, and a permutation drawn from it would make every probe fit
    depend on whether the control was requested.
    """
    generator = torch.Generator().manual_seed(seed)
    return labels[torch.randperm(labels.shape[0], generator=generator)]


def _probe_dataset(
    records: list[ManifestRecord], root: Path, *, num_frames: int, image_size: int
) -> VAWindowDataset:
    """Builds the dataset contract `VAWindowDataset` needs but a manifest path
    alone does not carry: `overlap`/`va_range` are pinned constants (varying
    them per dataset would let the fit budget vary per rung), `num_frames`/
    `image_size` come from the encoder being probed.
    """
    return VAWindowDataset(
        LabelledDataset(records=records, window_cfg=OmegaConf.create({}), root=root),
        num_frames=num_frames,
        image_size=image_size,
        overlap=PROBE_WINDOW_OVERLAP,
        va_range=PROBE_VA_RANGE,
        frame_stride=PROBE_FRAME_STRIDE,
    )


def _resolve_device() -> "torch.device":
    """Where the encoder forward and the probe fit run.

    Read from the machine rather than from a flag or a config key: every other
    quantity in this module is pinned because varying it would change what the
    probe measures, and the device is the one thing that does not -- the cached
    tokens and the fit are numerically the same protocol either way. On the
    32px fixtures CPU is the whole story; on AFEW-VA at 224px the fit is ~5.6k
    steps over 1568-token batches per layer and arm, which is why this exists.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _materialize_tokens(
    encoder, loader: DataLoader, layer: int, device: "torch.device | None" = None
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """One forward pass over `loader`, caching `layer`'s register-stripped,
    unnormed tokens (`forward_with_hidden_states`) and labels.

    The frozen encoder's per-layer tokens are a pure function of the input
    windows, so this is computed once and the probe is fit/scored against the
    cached result rather than re-running the encoder per epoch. Cached as fp16
    to halve resident memory at real-dataset scale; upcast to fp32 per-batch
    in `_fit_and_score_probe` just before the fp32-parameter probe consumes it.

    Writes into a preallocated tensor rather than list-append + `torch.cat`.
    `cat` holds every chunk and the result at once, which doubles peak memory.
    """
    device = _resolve_device() if device is None else device
    encoder.to(device)
    total = len(loader.dataset)
    tokens = None
    labels = []
    write_idx = 0
    for pixel_values, label in loader:
        hidden = forward_with_hidden_states(encoder, pixel_values.to(device))
        # Back to host memory immediately: the cache is several GB per layer at
        # real-dataset scale, far past any VRAM this runs on.
        batch_tokens = patch_tokens(encoder, hidden[layer]).half().cpu()
        if tokens is None:
            tokens = torch.empty((total, *batch_tokens.shape[1:]), dtype=torch.float16)
        tokens[write_idx : write_idx + batch_tokens.shape[0]] = batch_tokens
        write_idx += batch_tokens.shape[0]
        labels.append(label)
    return tokens[:write_idx], torch.cat(labels)


def _materialize_tokens_for_layers(
    encoder,
    loader: DataLoader,
    layers: set[int],
    device: "torch.device | None" = None,
) -> tuple[dict[int, "torch.Tensor"], "torch.Tensor"]:
    """One forward pass over `loader`, caching every requested layer's tokens.

    `forward_with_hidden_states` returns all `depth + 1` states from one pass,
    so `probe_by_depth` can sweep several layers without a forward pass per
    layer. Only the requested `layers` are kept, to limit memory. Cached as fp16
    to halve resident memory; upcast to fp32 per batch in `_fit_and_score_probe`.

    Writes into a preallocated tensor per layer rather than list-append +
    `torch.cat`, for the same peak-memory reason as `_materialize_tokens`.
    """
    device = _resolve_device() if device is None else device
    encoder.to(device)
    total = len(loader.dataset)
    tokens_by_layer: dict[int, "torch.Tensor"] = {}
    labels = []
    write_idx = 0
    for pixel_values, label in loader:
        hidden = forward_with_hidden_states(encoder, pixel_values.to(device))
        for layer in layers:
            # Back to host memory immediately: the cache is several GB per layer
            # at real-dataset scale, far past any VRAM this runs on.
            batch_tokens = patch_tokens(encoder, hidden[layer]).half().cpu()
            if layer not in tokens_by_layer:
                tokens_by_layer[layer] = torch.empty(
                    (total, *batch_tokens.shape[1:]), dtype=torch.float16
                )
            tokens_by_layer[layer][write_idx : write_idx + batch_tokens.shape[0]] = (
                batch_tokens
            )
        write_idx += pixel_values.shape[0]
        labels.append(label)
    return (
        {layer: chunk[:write_idx] for layer, chunk in tokens_by_layer.items()},
        torch.cat(labels),
    )


def standardise_on_fit(
    fit_tokens: "torch.Tensor",
    scored_tokens: "torch.Tensor",
    *,
    eps: float = 1e-6,
    chunk: int = 32,
) -> None:
    """Z-score both splits in place, per feature, using the FIT split's stats.

    Without this the probe reads the raw residual stream, whose norm in this
    workstream spans ~12 to ~450 across arms and layers (and 73k on one
    collapsed LeVJEPA arm). `LinearVAHead` ends in a tanh: fed activations of
    that size, a freshly-initialised head emits pre-tanh logits in the hundreds,
    saturates to exactly +-1, loses its gradient and predicts a constant
    forever. `ccc_loss` then returns exactly 0.0 for every seed. That looks like
    a precise null, not a dead fit, so every arm would silently report 0.0000.

    Standardising also removes stream norm as a confound. Without it, a ridge
    penalty penalises a norm-12 stream far more than a norm-450 one, so the
    comparison would partly rank arms by scale.

    `mu`/`sigma` come from the fit split alone -- taking them from the scored
    split, or from both, would leak it. Statistics accumulate in float64 over
    chunks because the cache is fp16 and a direct sum over ~10^6 rows would
    overflow it; the tokens themselves stay fp16, in place, since the cache is
    several GB per layer and a float32 copy would not fit alongside it.
    """
    dim = fit_tokens.shape[-1]
    total = 0
    sums = torch.zeros(dim, dtype=torch.float64)
    squares = torch.zeros(dim, dtype=torch.float64)
    for piece in fit_tokens.split(chunk):
        flat = piece.float().reshape(-1, dim)
        sums += flat.sum(dim=0).double()
        squares += (flat * flat).sum(dim=0).double()
        total += flat.shape[0]
    mu = sums / total
    sigma = (squares / total - mu * mu).clamp_min(0.0).sqrt()

    mu_cast = mu.to(fit_tokens.dtype)
    inv_sigma = (1.0 / (sigma + eps)).to(fit_tokens.dtype)
    for tokens in (fit_tokens, scored_tokens):
        for piece in tokens.split(chunk):
            piece.sub_(mu_cast).mul_(inv_sigma)


def _fit_and_score_probe(
    fit_tokens: "torch.Tensor",
    fit_labels: "torch.Tensor",
    scored_tokens: "torch.Tensor",
    scored_labels: "torch.Tensor",
    *,
    seed: int,
    epochs: int,
    embed_dim: int,
    device: "torch.device | None" = None,
) -> float:
    """Trains one `AttentiveProbe` on cached `(tokens, labels)` pairs and
    scores it under `train.py:_validate`'s CCC convention.

    Probe init is seeded via `torch.random.fork_rng` (restores the global RNG
    on exit) rather than `torch.manual_seed` directly -- standard `nn.init`
    calls (`xavier_uniform_`, `nn.Linear.reset_parameters`) read the global
    RNG and take no `generator=` argument, so this is what "seed the init
    without disturbing global state" requires in practice. `DataLoader`
    shuffling uses its own local `torch.Generator`, which needs no such trick.
    """
    loader_generator = torch.Generator().manual_seed(seed)
    fit_loader = DataLoader(
        TensorDataset(fit_tokens, fit_labels),
        batch_size=PROBE_BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        generator=loader_generator,
    )

    # Constructed on CPU and only then moved, so the seeded init draws the same
    # numbers on either device and a GPU run stays comparable to a CPU one.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        probe = AttentiveProbe(embed_dim=embed_dim)
    device = _resolve_device() if device is None else device
    probe.to(device)

    optimizer = torch.optim.AdamW(probe.parameters(), lr=PROBE_FIT_LR)
    probe.train()
    for _ in range(epochs):
        for batch_tokens, batch_labels in fit_loader:
            optimizer.zero_grad()
            pred = probe(batch_tokens.to(device).float())
            loss = ccc_loss(pred, batch_labels.to(device))
            loss.backward()
            optimizer.step()

    probe.eval()
    with torch.no_grad():
        pred = torch.cat(
            [
                probe(chunk.to(device).float()).cpu()
                for chunk in scored_tokens.split(_SCORE_CHUNK)
            ]
        ).numpy()
    target = scored_labels.numpy()
    ccc_v = concordance_correlation_coefficient(pred[:, 0], target[:, 0])
    ccc_a = concordance_correlation_coefficient(pred[:, 1], target[:, 1])
    return (ccc_v + ccc_a) / 2


def probe_ccc(
    bundle: DiagBundle,
    manifest_path: "Path | str",
    *,
    layer: int,
    seed: int,
    epochs: int = PROBE_FIT_EPOCHS,
) -> float:
    """One frozen-probe fit at `layer`, scored on an identity-disjoint split.

    `layer` indexes `forward_with_hidden_states`'s `depth + 1` states
    directly: `layer=12` on a depth-12 encoder is the final block's output.
    There is no `train.freeze_backbone` to reuse here -- the encoder is
    already frozen because `forward_with_hidden_states` runs under `no_grad`
    (`freeze_backbone` is a `VideoMAEVARegressor` method; this module never
    builds that wrapper).
    """
    manifest_path = Path(manifest_path)
    records = read_manifest(manifest_path, expected_role="probe")
    fit_ids, scored_ids = identity_disjoint_split(records, PROBE_SPLIT_FRACTION)
    fit_records = [r for r in records if r.identity_id in fit_ids]
    scored_records = [r for r in records if r.identity_id in scored_ids]

    encoder = bundle.encoder
    root = manifest_path.parent
    _, _, num_frames, image_size = encoder_geometry(encoder)
    fit_ds = _probe_dataset(
        fit_records, root, num_frames=num_frames, image_size=image_size
    )
    scored_ds = _probe_dataset(
        scored_records, root, num_frames=num_frames, image_size=image_size
    )
    # `generator=` here isn't for shuffling (shuffle=False) -- it's because
    # DataLoader always draws a base_seed from `generator or the global RNG`
    # (torch/utils/data/dataloader.py), even for a single-process, sequential
    # loader. Leaving it unset would silently consume the caller's global RNG.
    fit_loader = DataLoader(
        fit_ds,
        batch_size=PROBE_BATCH_SIZE,
        shuffle=False,
        generator=torch.Generator().manual_seed(seed),
    )
    scored_loader = DataLoader(
        scored_ds,
        batch_size=PROBE_BATCH_SIZE,
        shuffle=False,
        generator=torch.Generator().manual_seed(seed),
    )

    fit_tokens, fit_labels = _materialize_tokens(encoder, fit_loader, layer)
    scored_tokens, scored_labels = _materialize_tokens(encoder, scored_loader, layer)
    standardise_on_fit(fit_tokens, scored_tokens)

    return _fit_and_score_probe(
        fit_tokens,
        fit_labels,
        scored_tokens,
        scored_labels,
        seed=seed,
        epochs=epochs,
        embed_dim=encoder_geometry(encoder)[1],
    )


# --- probe_by_depth and check() -----------------------------------------------


def _bundle_layer_tokens(
    bundle: DiagBundle,
    fit_records: list[ManifestRecord],
    scored_records: list[ManifestRecord],
    root: Path,
    layers: set[int],
):
    """One forward pass each over the fit and scored splits, caching every
    requested layer's tokens for `bundle`'s (frozen) encoder."""
    encoder = bundle.encoder
    _, _, num_frames, image_size = encoder_geometry(encoder)
    fit_ds = _probe_dataset(
        fit_records, root, num_frames=num_frames, image_size=image_size
    )
    scored_ds = _probe_dataset(
        scored_records, root, num_frames=num_frames, image_size=image_size
    )
    fit_loader = DataLoader(
        fit_ds,
        batch_size=PROBE_BATCH_SIZE,
        shuffle=False,
        generator=torch.Generator().manual_seed(_MATERIALIZE_SEED),
    )
    scored_loader = DataLoader(
        scored_ds,
        batch_size=PROBE_BATCH_SIZE,
        shuffle=False,
        generator=torch.Generator().manual_seed(_MATERIALIZE_SEED),
    )
    fit_tokens, fit_labels = _materialize_tokens_for_layers(encoder, fit_loader, layers)
    scored_tokens, scored_labels = _materialize_tokens_for_layers(
        encoder, scored_loader, layers
    )
    return fit_tokens, fit_labels, scored_tokens, scored_labels


def probe_by_depth(
    bundle: DiagBundle,
    control_bundle: DiagBundle,
    manifest_path: "Path | str",
    layers: list[int],
    *,
    n_seeds: int = 3,
    epochs: int = PROBE_FIT_EPOCHS,
    permute_labels_seed: int | None = None,
) -> tuple[dict[int, tuple[MetricValue, MetricValue]], MetricValue]:
    """Paired checkpoint-vs-`random_init` probe CCC at every requested layer.

    Returns `(per_layer, final_block_reference)`: `per_layer` maps each entry
    of `layers` to a `(checkpoint_value, control_value)` pair of `MetricValue`,
    each aggregated over `n_seeds` independently-seeded probe fits.
    `final_block_reference` is the checkpoint's own CCC at its true final
    block, reported unconditionally regardless of what `layers` requested, so
    a "peak in the middle, lower at the end" profile is readable directly.

    The encoder runs exactly once per distinct bundle (twice total when
    `bundle is control_bundle`, matching the shared identity-disjoint split)
    -- see `_materialize_tokens_for_layers`. The absolute CCC is never the
    finding here; `check()` is what turns the paired values into an explicit
    difference metric.
    """
    manifest_path = Path(manifest_path)
    records = read_manifest(manifest_path, expected_role="probe")
    fit_ids, scored_ids = identity_disjoint_split(records, PROBE_SPLIT_FRACTION)
    fit_records = [r for r in records if r.identity_id in fit_ids]
    scored_records = [r for r in records if r.identity_id in scored_ids]
    root = manifest_path.parent

    final_layer, ckpt_dim = encoder_geometry(bundle.encoder)[:2]
    ctrl_dim = encoder_geometry(control_bundle.encoder)[1]
    all_layers = set(layers) | {final_layer}

    ckpt_fit_tokens, ckpt_fit_labels, ckpt_scored_tokens, ckpt_scored_labels = (
        _bundle_layer_tokens(bundle, fit_records, scored_records, root, all_layers)
    )
    if control_bundle is bundle:
        ctrl_fit_tokens, ctrl_fit_labels = ckpt_fit_tokens, ckpt_fit_labels
        ctrl_scored_tokens, ctrl_scored_labels = ckpt_scored_tokens, ckpt_scored_labels
    else:
        ctrl_fit_tokens, ctrl_fit_labels, ctrl_scored_tokens, ctrl_scored_labels = (
            _bundle_layer_tokens(
                control_bundle, fit_records, scored_records, root, all_layers
            )
        )

    # Once per (layer, arm), before any replicate: the cache is reused across
    # seeds and, when the control is the bundle itself, across arms, so
    # standardising inside the fit would apply it repeatedly to the same tensor.
    for layer in all_layers:
        standardise_on_fit(ckpt_fit_tokens[layer], ckpt_scored_tokens[layer])
    if ctrl_fit_tokens is not ckpt_fit_tokens:
        for layer in all_layers:
            standardise_on_fit(ctrl_fit_tokens[layer], ctrl_scored_tokens[layer])

    if permute_labels_seed is not None:
        # The same permutation for both arms, so the pairing that makes
        # checkpoint-minus-control a paired difference survives the control.
        ckpt_fit_labels = ctrl_fit_labels = permute_labels(
            ckpt_fit_labels, permute_labels_seed
        )
        ckpt_scored_labels = ctrl_scored_labels = permute_labels(
            ckpt_scored_labels, permute_labels_seed + 1
        )

    def _arm_metric(
        fit_tokens: "torch.Tensor",
        fit_labels: "torch.Tensor",
        scored_tokens: "torch.Tensor",
        scored_labels: "torch.Tensor",
        embed_dim: int,
        seed_base: int,
    ) -> MetricValue:
        return replicate(
            lambda s: _fit_and_score_probe(
                fit_tokens,
                fit_labels,
                scored_tokens,
                scored_labels,
                seed=s,
                epochs=epochs,
                embed_dim=embed_dim,
            ),
            n=n_seeds,
            seed=seed_base,
        )

    per_layer: dict[int, tuple[MetricValue, MetricValue]] = {}
    for layer in layers:
        checkpoint_value = _arm_metric(
            ckpt_fit_tokens[layer],
            ckpt_fit_labels,
            ckpt_scored_tokens[layer],
            ckpt_scored_labels,
            ckpt_dim,
            _CHECKPOINT_SEED_BASE,
        )
        control_value = _arm_metric(
            ctrl_fit_tokens[layer],
            ctrl_fit_labels,
            ctrl_scored_tokens[layer],
            ctrl_scored_labels,
            ctrl_dim,
            _CONTROL_SEED_BASE,
        )
        per_layer[layer] = (checkpoint_value, control_value)

    if final_layer in per_layer:
        # Same tokens, labels, and seeds: refitting would cost a third arm.
        return per_layer, per_layer[final_layer][0]
    final_block_reference = _arm_metric(
        ckpt_fit_tokens[final_layer],
        ckpt_fit_labels,
        ckpt_scored_tokens[final_layer],
        ckpt_scored_labels,
        ckpt_dim,
        _CHECKPOINT_SEED_BASE,
    )

    return per_layer, final_block_reference


def _skip_report(reason: str) -> DiagReport:
    return DiagReport(
        metrics={},
        arrays={},
        meta={
            "LADDER_REVISION": LADDER_REVISION,
            "input_kind": "real",
            "skipped": reason,
        },
    )


def check(bundle: DiagBundle, inputs: dict) -> dict[str, DiagReport]:
    """Orchestrator entry point: `probe_by_depth` against real
    AFEW-VA data, or a clean skip -- never a raise, never an empty mapping
    (indistinguishable from "this check does not exist" to a caller merging
    reports from every module).

    `inputs` supplies `control_bundle` (required -- `check` cannot build its
    own, since `random_init_bundle` needs a `model_cfg` and `DiagBundle`
    carries none) and, optionally, `layers`/`n_seeds`. It is never a source of
    fit-budget parameters: `epochs=PROBE_FIT_EPOCHS` unconditionally.
    """
    afewva_root = os.environ.get("AFEWVA_ROOT")
    if not afewva_root:
        return {"probe_by_depth": _skip_report("AFEWVA_ROOT unset")}

    manifest_path = Path(afewva_root) / "manifest.csv"
    if not manifest_path.exists():
        return {"probe_by_depth": _skip_report(f"no manifest.csv at {manifest_path}")}

    control_bundle = inputs.get("control_bundle")
    if control_bundle is None:
        return {"probe_by_depth": _skip_report("no control_bundle supplied")}

    layers = inputs.get("layers", DEFAULT_PROBE_DEPTH_LAYERS)
    n_seeds = inputs.get("n_seeds", 3)

    per_layer, final_block_reference = probe_by_depth(
        bundle,
        control_bundle,
        manifest_path,
        layers,
        n_seeds=n_seeds,
        epochs=PROBE_FIT_EPOCHS,
    )

    metrics: dict[str, MetricValue] = {}
    for layer, (checkpoint_value, control_value) in per_layer.items():
        metrics[f"layer_{layer}_checkpoint"] = checkpoint_value
        metrics[f"layer_{layer}_control"] = control_value
        # The absolute CCC is never the finding this module reports -- this
        # difference is, made explicit so no downstream reader has to
        # reconstruct it from the two arms above.
        metrics[f"layer_{layer}_diff"] = MetricValue.deterministic_value(
            checkpoint_value.mean - control_value.mean
        )
    metrics["final_block_reference"] = final_block_reference

    return {
        "probe_by_depth": DiagReport(
            metrics=metrics,
            arrays={},
            meta={"LADDER_REVISION": LADDER_REVISION, "input_kind": "real"},
        )
    }
