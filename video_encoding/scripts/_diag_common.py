"""Shared primitives for the SSL diagnostic checks.

Every diagnostic plan in the series imports this module rather than re-deriving
checkpoint loading, input sampling, or metric semantics -- the drift that left
`export_vit_internals.py` with a hand-mirrored `ViTBlock.forward` describing a
model the repo no longer had.

The load-bearing idea is that a diagnostic metric is a *distribution*, not a
number. A single reading cannot support "the model changed": that claim needs a
replication band, which is why `MetricValue` refuses to exist as a bare float
and `separated_from` refuses to compare things that have no band.
"""

from __future__ import annotations

import json
import math
import os
import random
import statistics
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from video_encoding.mesh.outer_step import to_fp32_state_dict
from video_encoding.mesh.pending_files import pending_filename
from video_encoding.models.videomaev2.encoder import (
    VideoMAEEncoder,
    preprocess_window,
)
from video_encoding.models.videomaev2.mae import VideoMAEForPreTraining
from video_encoding.scripts.export_vit_internals import load_sample_windows
from video_encoding.training import checkpoint as ckpt_io

# How many pooled standard deviations two means must differ by before the
# difference counts as real. 2.0 is roughly a 95% band for a normal difference;
# it is deliberately a module constant rather than a per-call default scattered
# across the later plans, so every check tiers its metrics on the same evidence
# bar and changing the bar is a one-line, reviewable edit.
DEFAULT_SEPARATION_K = 2.0


@dataclass(frozen=True)
class MetricValue:
    """One diagnostic metric: either sampled over seeds, or deterministic.

    The two kinds are kept apart on purpose. A sampled metric with `n=1` is a
    latent bug -- it *could* have been resampled and wasn't -- and is not the
    same thing as a metric with no sampling axis at all (`residual_branch_norm`
    and `delta_from_init` read weights only, so replication would return the
    identical number every time). Collapsing them would make `{"sd": null}`
    ambiguous to every reader of `diagnostics.json`.

    Construct deterministic values via `deterministic_value`; the invariant is
    enforced here rather than only at the orchestrator boundary, so an invalid
    metric cannot be built in the first place.
    """

    mean: float
    sd: float | None
    n: int | None
    seeds: tuple[int, ...]
    deterministic: bool

    def __post_init__(self) -> None:
        if self.deterministic:
            if self.sd is not None or self.n is not None or self.seeds != ():
                raise ValueError(
                    "a deterministic MetricValue carries no sampling fields: "
                    f"got sd={self.sd!r}, n={self.n!r}, seeds={self.seeds!r}"
                )
            return
        if self.n is None or self.n < 3:
            raise ValueError(
                f"a sampled MetricValue needs n>=3 replicates, got n={self.n!r}; "
                "use MetricValue.deterministic_value for a metric with no "
                "sampling axis"
            )
        if self.sd is None:
            raise ValueError(
                "a sampled MetricValue needs an sd -- without one it reaches "
                "separated_from and fails there, far from its cause"
            )

    @classmethod
    def deterministic_value(cls, mean: float) -> "MetricValue":
        """A metric with no sampling axis, e.g. one read straight off weights."""
        return cls(mean=mean, sd=None, n=None, seeds=(), deterministic=True)

    def separated_from(
        self, other: "MetricValue", k: float = DEFAULT_SEPARATION_K
    ) -> bool:
        """True when the two means differ by more than `k` pooled sds.

        Raises rather than answering when the question is ill-posed: a
        deterministic value has no band to pool, and a zero pooled band would
        make every non-zero difference significant. The latter is the failure
        mode that matters -- a sampled function that receives distinct seeds but
        ignores them reports sd=0 across the board, and silently accepting it is
        exactly how an unreplicated metric earns Tier A.
        """
        if self.deterministic or other.deterministic:
            raise ValueError(
                "separated_from needs a replication band on both sides, but a "
                "deterministic MetricValue has none; compare deterministic "
                "values exactly instead"
            )
        assert self.sd is not None and other.sd is not None  # __post_init__
        band = k * math.sqrt(self.sd**2 + other.sd**2)
        if band == 0.0:
            raise ValueError(
                "zero observed variance across distinct seeds (sd=0 on both "
                "sides): the sampled function ignored its seed, so no "
                "difference here is resolvable"
            )
        return abs(self.mean - other.mean) > band


def replicate(fn: Callable[[int], float], n: int = 3, seed: int = 0) -> MetricValue:
    """Sample `fn` over `n` seeds derived from `seed`, as one `MetricValue`.

    Seeds come from a local `Random` so the derivation neither depends on nor
    disturbs global RNG state, and the same base seed always yields the same
    sample seeds -- a diagnostic that cannot be re-run to the same numbers
    cannot be used to argue anything.

    Note the limit of what this can guarantee: distinct seeds are checked here,
    but a function that *receives* them and ignores them still passes. That
    residual case surfaces in `separated_from`'s zero-band check.
    """
    if n < 3:
        raise ValueError(f"replicate needs n>=3 to give a metric a band, got {n}")
    rng = random.Random(seed)
    seeds = tuple(rng.randrange(2**32) for _ in range(n))
    values = [float(fn(s)) for s in seeds]
    return MetricValue(
        mean=statistics.fmean(values),
        sd=statistics.stdev(values),
        n=n,
        seeds=seeds,
        deterministic=False,
    )


# The ladder version every DiagReport is stamped with. Tier A/B/C mean what the
# ladder of that revision made them mean, so a report whose tiers cannot be
# attributed to a revision cannot be compared with a later one. Bump this
# whenever LADDER's membership, order, or tier thresholds change, or a scored
# metric's definition does (2026-10-02.1: label-free metrics clip-pooled;
# 2026-10-03.1: direction-aware tau, values over three input draws).
LADDER_REVISION = "2026-10-03.1"


@dataclass(frozen=True)
class DiagBundle:
    """A loaded checkpoint, with the provenance needed to label a plot.

    `model` is populated only on the `full_model=True` path; the encoder is
    always present and is what every per-block diagnostic reads.
    """

    encoder: "VideoMAEEncoder"
    model: "VideoMAEForPreTraining | None"
    run_id: str
    round: int | None
    # The torch seed the run's weights were initialised from, when known: set
    # by random_init_bundle, and by the ladder from --ladder-init-seeds (mesh
    # workers seed init with train.seed). Lets delta_from_init accept the
    # rebuilt init as the run's own reference.
    init_seed: int | None = None


@dataclass
class DiagReport:
    """One check's output: typed metrics, bulk arrays, and ladder provenance.

    Metrics and arrays are written separately because they have different
    readers -- `diagnostics.json` is read by the orchestrator and by humans,
    while per-block/per-head arrays only ever feed plots.
    """

    metrics: dict[str, MetricValue]
    arrays: dict[str, "np.ndarray"]
    meta: dict

    def __post_init__(self) -> None:
        if "LADDER_REVISION" not in self.meta:
            raise ValueError(
                "DiagReport.meta must carry LADDER_REVISION -- a tier that "
                "cannot be attributed to a ladder version cannot be compared "
                "against a report written under a different one"
            )

    def to_json(self, path: "Path | str") -> None:
        payload = {
            "meta": self.meta,
            "metrics": {
                name: {
                    "mean": value.mean,
                    "sd": value.sd,
                    "n": value.n,
                    "seeds": list(value.seeds),
                    # Without this flag {"sd": null, "n": null} is
                    # indistinguishable from a malformed sampled metric.
                    "deterministic": value.deterministic,
                }
                for name, value in self.metrics.items()
            },
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    def write_npz(self, path: "Path | str") -> None:
        np.savez(Path(path), **self.arrays)

    @classmethod
    def from_json(cls, path: "Path | str") -> "DiagReport":
        """Re-checks the construction invariant, so a hand-edited or stale
        `diagnostics.json` fails on read rather than being tiered."""
        payload = json.loads(Path(path).read_text())
        metrics = {
            name: MetricValue(
                mean=entry["mean"],
                sd=entry["sd"],
                n=entry["n"],
                seeds=tuple(entry["seeds"]),
                deterministic=entry["deterministic"],
            )
            for name, entry in payload["metrics"].items()
        }
        return cls(metrics=metrics, arrays={}, meta=payload["meta"])


def _exempt_missing(missing: list[str], encoder: "VideoMAEEncoder") -> list[str]:
    """Drop the one benign missing key: a zero-element `registers.tokens`.

    Checkpoints predating RegisterTokens lack it, and at `num_registers=0` the
    parameter holds no values and is never used. Every other missing key is a
    real mismatch -- a silent strict=False pass here means reporting a random
    encoder's internals under a trained run's label.
    """
    return [
        key
        for key in missing
        if not (
            key.removeprefix("encoder.") == "registers.tokens"
            and encoder.registers.tokens.numel() == 0
        )
    ]


def _raise_on_real_mismatch(result, encoder: "VideoMAEEncoder", run_id: str) -> None:
    missing = _exempt_missing(list(result.missing_keys), encoder)
    if missing or result.unexpected_keys:
        raise ValueError(
            f"{run_id}: checkpoint/model key mismatch -- "
            f"missing={missing} unexpected={list(result.unexpected_keys)}"
        )


def _check_patch_embed_channels(
    state_dict: dict, encoder: "VideoMAEEncoder", run_id: str, prefix: str
) -> None:
    """Refuse a checkpoint whose first convolution has a different input depth.

    `training/checkpoint.py::load_encoder_state` zero-inflates or truncates this
    tensor and then reports missing=[] / unexpected=[]. That is right for
    fine-tuning -- a zero-init channel is a no-op that trains up -- and wrong
    here: the harness would measure a first convolution the checkpoint never
    contained while reporting a clean load, and both load paths would disagree
    (the full-model path's plain load_state_dict raises on the same mismatch).
    """
    key = f"{prefix}patch_embed.proj.weight"
    if key not in state_dict:
        return
    found = state_dict[key].shape[1]
    wanted = encoder.patch_embed.proj.weight.shape[1]
    if found != wanted:
        raise ValueError(
            f"{run_id}: checkpoint patch_embed has {found} input channels but "
            f"the config asks for {wanted}; a diagnostic load never inflates or "
            "truncates the first convolution"
        )


def load_bundle(
    run_id: str,
    mesh_shared: "Path | str",
    model_cfg,
    *,
    full_model: bool = False,
    round: int | None = None,
) -> DiagBundle:
    """Load `run_id`'s checkpoint into a `DiagBundle`.

    `run_id` names a checkpoint directory and nothing else -- there is no
    reserved sentinel, so a run genuinely called "random_init" loads its own
    weights. The untrained control is `random_init_bundle`.

    `model_cfg` is used as given: no run-id-derived overrides are applied here,
    so a caller comparing checkpoints with different decoder shapes or register
    counts passes the cfg appropriate to each (`export_vit_internals.
    model_cfg_for_run` is the existing helper for that). Keeping the derivation
    out of the harness is what makes "these two bundles were loaded the same
    way" checkable.

    Both paths raise on any key or shape mismatch except a zero-element
    `registers.tokens`, and both reject a differing patch-embed channel count.
    """
    filename = "master.pt" if round is None else f"master_round{round}.pt"
    ckpt_path = Path(mesh_shared) / "checkpoints" / run_id / filename
    return _bundle_from_checkpoint(
        ckpt_path, model_cfg, run_id=run_id, round=round, full_model=full_model
    )


def load_solo_bundle(
    run_id: str,
    mesh_shared: "Path | str",
    model_cfg,
    *,
    node: str,
    round: int,
    full_model: bool = False,
) -> DiagBundle:
    """Load a `mesh.merge=false` run's checkpoint -- rung 2's route.

    A solo run never writes `checkpoints/<run_id>/master.pt`: that file is
    produced by the merge, which `merge=false` skips by definition. What it
    does write is `checkpoints/pending/{run_id}_{round}_{node}.pt`
    (`mesh_round.push_checkpoint_and_wait`), so the overfit rung is loaded
    from there. `node` and `round` are required rather than discovered by
    globbing: a solo run has exactly one writer and the caller knows which
    round it wants, and picking "whatever pending file is newest" is how a
    straggler's or a sibling run's checkpoint would end up on the rung.

    Missing files raise. There is deliberately no fallback to `master.pt` --
    quietly reading some other, merged run's weights would put a different
    model on the overfit rung and report it as the control.
    """
    ckpt_path = (
        Path(mesh_shared)
        / "checkpoints"
        / "pending"
        / pending_filename(run_id, round, node)
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"{run_id}: no solo checkpoint at {ckpt_path} -- a merge=false run "
            f"writes pending/{{run_id}}_{{round}}_{{node}}.pt and never master.pt"
        )
    return _bundle_from_checkpoint(
        ckpt_path, model_cfg, run_id=run_id, round=round, full_model=full_model
    )


def load_snapshot_bundle(
    run_id: str,
    snapshot_dir: "Path | str",
    model_cfg,
    *,
    round: int,
    full_model: bool = False,
) -> DiagBundle:
    """Load a round master captured by `snapshot_mesh_checkpoints.sh` --
    `<snapshot_dir>/<run_id>/master_round<round>.pt` -- the round axis's route.

    A missing round raises. There is deliberately no fallback to `master.pt`:
    the final weights relabelled as round N would render one point as a
    trajectory.
    """
    ckpt_path = Path(snapshot_dir) / run_id / f"master_round{round}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"{run_id}: no captured snapshot at {ckpt_path} -- round {round} was "
            "not captured, and master.pt is never substituted for it"
        )
    return _bundle_from_checkpoint(
        ckpt_path, model_cfg, run_id=run_id, round=round, full_model=full_model
    )


def _bundle_from_checkpoint(
    ckpt_path: Path,
    model_cfg,
    *,
    run_id: str,
    round: int | None,
    full_model: bool,
) -> DiagBundle:
    """Shared by both entry points so they can differ only in which path they
    read -- the checking below is what makes two bundles comparable."""
    payload = ckpt_io.load_checkpoint_file(ckpt_path)
    state_dict = to_fp32_state_dict(payload["model_state_dict"])

    try:
        if full_model:
            model = VideoMAEForPreTraining(model_cfg)
            _check_patch_embed_channels(state_dict, model.encoder, run_id, "encoder.")
            result = model.load_state_dict(state_dict, strict=False)
            _raise_on_real_mismatch(result, model.encoder, run_id)
            return DiagBundle(
                encoder=model.encoder.eval(),
                model=model.eval(),
                run_id=run_id,
                round=round,
                init_seed=payload.get("init_seed"),
            )

        # The mesh saves a VideoMAEForPreTraining state dict (encoder+decoder),
        # so the encoder-only path selects its own subtree rather than loading
        # the whole thing and discarding the decoder.
        encoder = VideoMAEEncoder(model_cfg)
        encoder_sd = {
            key.removeprefix("encoder."): tensor
            for key, tensor in state_dict.items()
            if key.startswith("encoder.")
        }
        _check_patch_embed_channels(encoder_sd, encoder, run_id, "")
        result = encoder.load_state_dict(encoder_sd, strict=False)
        _raise_on_real_mismatch(result, encoder, run_id)
        return DiagBundle(
            encoder=encoder.eval(),
            model=None,
            run_id=run_id,
            round=round,
            init_seed=payload.get("init_seed"),
        )
    except RuntimeError as exc:
        # load_state_dict reports shape mismatches (e.g. a differing register
        # count, which changes the sequence every block attends over) as a
        # RuntimeError rather than in missing/unexpected.
        raise ValueError(f"{run_id}: checkpoint/model shape mismatch -- {exc}") from exc


def random_init_bundle(
    model_cfg, *, full_model: bool = False, seed: int | None = None
) -> DiagBundle:
    """The untrained control rung, touching no checkpoint path at all.

    `seed` pins the init without touching the caller's global RNG; unseeded,
    two controls built in two jobs are two different models. A `levjepa_hf`
    `model_cfg` gives rung 4's control: the same ViT-L, randomly initialised
    from its snapshot's `config.json`.
    """
    with torch.random.fork_rng(devices=[], enabled=seed is not None):
        if seed is not None:
            torch.manual_seed(seed)
        if str(model_cfg.get("encoder", "")) == "levjepa_hf":
            return DiagBundle(
                encoder=_hf_levjepa_from_config(model_cfg).eval(),
                model=None,
                run_id="",
                round=None,
                init_seed=seed,
            )
        if full_model:
            model = VideoMAEForPreTraining(model_cfg)
            return DiagBundle(
                encoder=model.encoder.eval(),
                model=model.eval(),
                run_id="",
                round=None,
                init_seed=seed,
            )
        return DiagBundle(
            encoder=VideoMAEEncoder(model_cfg).eval(),
            model=None,
            run_id="",
            round=None,
            init_seed=seed,
        )


def paired_inputs(
    shard_path: "Path | str",
    *,
    n_windows: int,
    num_frames: int,
    image_size: int,
    seed: int,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """A real batch and a shape-matched Gaussian batch, both seed-reproducible.

    The random half is the control that separates "this metric responds to
    video" from "this metric responds to any tensor of the right shape".
    """
    windows = load_sample_windows(
        Path(shard_path), n_windows, num_frames, image_size, seed
    )
    # Filled in place rather than torch.cat over a list: a 256-window draw is
    # ~2.5 GB, and the list plus its concatenation would hold it twice.
    first = preprocess_window(windows[0][1], num_frames, image_size)
    real = torch.empty((len(windows), *first.shape[1:]), dtype=first.dtype)
    real[0] = first[0]
    for i, (_, frames) in enumerate(windows[1:], start=1):
        real[i] = preprocess_window(frames, num_frames, image_size)[0]
    generator = torch.Generator().manual_seed(seed)
    random_batch = torch.randn(real.shape, generator=generator)
    return real, random_batch


def run_hyperparams(run_id: str, configs_path: "Path | str") -> dict:
    """`run_id`'s entry from `runs/state/mesh_run_configs.json`, or `{}`.

    Hyperparameters annotate a report; a missing, unreadable, or unknown entry
    must not take a diagnostic run down with it.
    """
    try:
        configs = json.loads(Path(configs_path).read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(configs, list):
        return {}
    for entry in configs:
        if isinstance(entry, dict) and entry.get("run_id") == run_id:
            return entry
    return {}


def forward_with_hidden_states(
    encoder: "VideoMAEEncoder", pixel_values: "torch.Tensor"
) -> list["torch.Tensor"]:
    """The residual stream before block 0 and after every block: `depth + 1`.

    This is the ONLY place `VideoMAEEncoder.forward`'s traversal is mirrored
    outside the model, and it is pinned bit-for-bit against the real forward by
    a parity test -- `export_vit_internals.py`'s hand-mirrored copy of
    `ViTBlock.forward` drifted from the model it described, and one traversal
    with a parity test is the fix.

    The stream is returned RAW: unnormed, registers still attached. That is what
    distinguishes it from `encoder.forward_tokens(..., return_layers=...)`,
    which takes pre-embedded tokens plus a `keep_indices` gather (the
    MAE-pretraining entry point, not raw pixels) and applies `self.norm` plus a
    register-strip to every intermediate -- destroying exactly the pre-norm
    magnitudes a branch-norm ratio is computed from.

    `no_grad`, deliberately not `inference_mode`: a caller holding `depth + 1`
    activations should not also hold the graph behind them, but inference-mode
    tensors can never re-enter autograd, which would break training an
    attentive probe on these features.
    """
    if is_hf_levjepa(encoder):
        return _hf_levjepa_hidden_states(encoder, pixel_values)
    with torch.no_grad():
        x = encoder.patch_embed(pixel_values)
        if x.shape[1] != encoder.pos_embed.shape[1]:
            raise ValueError(
                f"got {x.shape[1]} tokens but pos_embed has "
                f"{encoder.pos_embed.shape[1]}; check num_frames/image_size vs "
                "tubelet_size/patch_size"
            )
        x = x + encoder.pos_embed
        x = encoder.registers.prepend(x)
        states = [x]
        for block in encoder.blocks:
            x = block(x)
            states.append(x)
    return states


class Rung(Enum):
    """A calibration point on the measurement ladder.

    MUTILATION is deliberately unranked: it is a self-test control a check
    asserts against directly ("this metric must move when I break the model"),
    not a point a metric is scored for ordering correctly.
    """

    MUTILATION = -1
    RANDOM_INIT = 0
    ROUND = 1
    OVERFIT = 2
    PUBLIC_VIDEOMAEV2 = 3
    SUPERVISED_VA = 4


# Worst to best. The SIZE is load-bearing, not incidental: the Tier A/B
# thresholds below are derived from a five-rung ladder (tau = 1 - inv/5, so 0.8
# admits one inversion and 0.5 admits two). Changing LADDER's size without
# re-deriving TIER_A_TAU/TIER_B_TAU leaves both unjustified.
LADDER = (
    Rung.RANDOM_INIT,
    Rung.ROUND,
    Rung.OVERFIT,
    Rung.PUBLIC_VIDEOMAEV2,
    Rung.SUPERVISED_VA,
)

# ROUND's internal ordering (rounds 1 < 10 < 50 < 120 < final) is an assumption,
# not a finding: this project's standing evidence is that pretraining plateaus.
# A metric reporting the upper round checkpoints as tied agrees with a revised
# ROUND ordering; the tie-aware tau is what stops that reading being scored as
# an inversion.
ROUND_ORDERING_IS_PROVISIONAL = True

TIER_A_TAU = 0.8
TIER_B_TAU = 0.5
# Below four present rungs the rank half of the conjunctive rule carries no
# information -- on two rungs tau is +/-1 by construction, so every metric with a
# visible difference would collapse to Tier A.
MIN_RUNGS_FOR_TIER_AB = 4


class Tier(Enum):
    """How much confidence a metric's ladder behaviour earns it."""

    A = "A"
    B = "B"
    C = "C"


@dataclass(frozen=True)
class LadderResult:
    tau: float
    tier: Tier
    present_rungs: int
    # Ordered against its declared direction as firmly as Tier B would demand
    # in the declared one: the declaration, not the checkpoint, is suspect.
    inverted: bool = False


def _tied(a: MetricValue, b: MetricValue, k: float) -> bool:
    """Two rung values are tied when their difference is not resolvable.

    For sampled values that is the replication band `separated_from` already
    uses -- bitwise equality never occurs between real measurements, so an
    exact-equality rule would make tau-b's tie-awareness dead code and score a
    plateau as a clean ordering. A deterministic value has no sampling error, so
    for those any difference is real and equality is the right test.
    """
    if a.deterministic != b.deterministic:
        raise ValueError(
            "a metric cannot mix deterministic and sampled values across "
            "rungs: the two carry incomparable notions of a resolvable "
            "difference"
        )
    if a.deterministic:
        return a.mean == b.mean
    return not a.separated_from(b, k)


def _kendall_tau_b(
    values: list[MetricValue], tied: Callable[[int, int], bool]
) -> float:
    """Tie-aware Kendall tau-b of `values` against their own index order.

    `tied` is supplied as a predicate rather than inferred by comparing floats,
    which is the whole point: what counts as a tie is a question about
    measurement error, not about bit patterns.

    The reference order (the ladder) has no ties, so only the value side
    contributes `n2`. When every pair is tied the denominator vanishes and tau
    is 0.0 -- the metric resolved no ordering at all.
    """
    n = len(values)
    n0 = n * (n - 1) // 2
    concordant = discordant = ties = 0
    for i in range(n):
        for j in range(i + 1, n):
            if tied(i, j):
                ties += 1
            elif values[i].mean < values[j].mean:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(n0 * (n0 - ties))
    if denominator == 0.0:
        return 0.0
    return (concordant - discordant) / denominator


def ladder_tau(
    values: dict[Rung, MetricValue],
    k: float = DEFAULT_SEPARATION_K,
    *,
    direction: str = "rising",
) -> LadderResult:
    """Score a metric's values across ladder rungs, and tier its confidence.

    The tier is conjunctive, not merely rank-based: tau captures order, but Tier
    A additionally requires every adjacent gap to clear the replication band and
    Tier B a majority of them. A metric that ranks the ladder perfectly while
    every gap sits inside sampling noise is Tier C -- correct order on
    unresolvable differences is not evidence, and accepting it would let noise
    pass as signal.

    Rungs absent from `values` are simply absent; below four present rungs the
    result is Tier C regardless, and `present_rungs` records why.

    `direction` is how the metric moves as rungs improve; tau is reported in
    it, so +1 always means "ranks the ladder correctly". A metric whose every
    rung is deterministic is capped at Tier B: with no sampling band, any
    difference counts as separated, so the gap half of Tier A is free.
    """
    if direction not in ("rising", "falling"):
        raise ValueError(f"direction must be 'rising' or 'falling', got {direction!r}")
    present = [rung for rung in LADDER if rung in values]
    ordered = [values[rung] for rung in present]
    if len(ordered) < 2:
        return LadderResult(tau=0.0, tier=Tier.C, present_rungs=len(ordered))

    def tied(i: int, j: int) -> bool:
        return _tied(ordered[i], ordered[j], k)

    # Raise on a mixed-kind metric before scoring anything, so the error names
    # the metric rather than surfacing as a strange tau.
    for i in range(len(ordered) - 1):
        _tied(ordered[i], ordered[i + 1], k)

    tau = _kendall_tau_b(ordered, tied)
    if direction == "falling":
        tau = -tau
    separated = [not tied(i, i + 1) for i in range(len(ordered) - 1)]
    all_separated = all(separated)
    majority_separated = sum(separated) * 2 > len(separated)

    enough_rungs = len(ordered) >= MIN_RUNGS_FOR_TIER_AB
    if not enough_rungs:
        tier = Tier.C
    elif tau >= TIER_A_TAU and all_separated:
        tier = Tier.A
    elif tau >= TIER_B_TAU and majority_separated:
        tier = Tier.B
    else:
        tier = Tier.C
    if tier is Tier.A and all(v.deterministic for v in ordered):
        tier = Tier.B
    inverted = enough_rungs and tau <= -TIER_B_TAU and majority_separated
    return LadderResult(
        tau=tau, tier=tier, present_rungs=len(ordered), inverted=inverted
    )


class RungUnavailable(Exception):
    """A ladder rung has no valid checkpoint to load.

    Distinguishable from a bare exception so `load_ladder_rung` callers can
    react to "this rung is absent" without string-matching an error message.
    Raised before any tensor value is copied into a model -- a rung reported
    unavailable must never leave a partially-loaded bundle behind.
    """


# Rung 3's weights. Plan 3 surveyed the published release for a checkpoint
# matching this repo's ViT-Base geometry (depth=12, embed_dim=768) rather than
# assuming the first downloaded file (vit_g_hybrid_pt_1200e, ViT-giant,
# depth=40 -- geometry-incompatible) was the only option. Two ViT-B-geometry
# candidates exist in OpenGVLab/VideoMAE2's file listing:
#   mae-b/pytorch_model.bin (691MB) -- adopted below.
#   distill/vit_b_k710_dl_from_giant.pth (174MB) -- K710-supervised, not chosen.
#
# Neither is the bare SSL-pretrain checkpoint the design doc's "public SSL"
# framing assumed: mae-b carries a `cls_head.fc_cls` (101-way -- UCF101
# fine-tune), confirmed by inspecting its state_dict, not by its filename.
# Rung 3 is therefore recorded as a *supervised* public ceiling. The HF repo's
# file listing was checked and carries no other ViT-B-geometry file.
#
# mae-b's checkpoint also has no `pos_embed` key at all: the original
# VideoMAEv2 ViT-Base uses `use_learnable_pos_emb=False`, i.e. a fixed sin-cos
# table computed at forward time and never saved to state_dict. This repo's
# `VideoMAEEncoder.pos_embed` is a required learnable parameter, so the loader
# below computes that same deterministic table
# (`_public_videomae_sinusoid_pos_embed`, ported from `modeling_finetune.py`'s
# `get_sinusoid_encoding_table`) rather than leaving it at random init -- a
# faithful reproduction of what the checkpoint's own model does, not an
# approximation.
#
# Stored on node090's local NVMe (not the shared home -- keeps large weights
# off the NFS quota); a local path (this constant is NOT one -- see below) or
# `PUBLIC_VIDEOMAEV2_LOCAL_PATH` must be supplied by the caller/host.
PUBLIC_VIDEOMAEV2_WEIGHTS = (
    "node090:/Data/$USER/ladder_weights/mae-b_pytorch_model.bin"
)
PUBLIC_VIDEOMAEV2_SOURCE = (
    "https://huggingface.co/OpenGVLab/VideoMAE2/resolve/main/mae-b/pytorch_model.bin"
)

# Rung -1 (MUTILATION) target: Plan 2's self-test control zeroes this block's
# residual-branch weight and asserts the check under test actually moves.
# Block 0 is the target because it is what every later diagnostic reads first
# (branch-norm ratios, attention entropy) -- a check insensitive to a mutilation
# this early would be insensitive to everything downstream of it too. Recorded
# here, not in Plan 2, so the constant is defined once and every check that
# builds rung -1 mutilates the same block the same way.
MUTILATION_BLOCK = 0
# GELUMlp has no `.fc2` attribute -- its down-projection is `net[3]`, a
# Sequential index, which attrgetter cannot traverse (dotted paths only, no
# subscripting). The key below is a named_parameters()/state_dict() key, not
# an attribute path: resolve it via `dict(encoder.named_parameters())[key]`,
# not `operator.attrgetter`.
MUTILATION_TARGET_KEY = f"blocks.{MUTILATION_BLOCK}.mlp.net.3.weight"

# mae-b's block count, pinned to this repo's own ViT-Base depth (both are 12 --
# verified in task 1.0). _KEY_MAP is built for exactly this many blocks; a
# checkpoint with a different block count fails the geometry gate in
# `_load_public_videomaev2_rung` before `_KEY_MAP` is ever consulted.
_PUBLIC_VIDEOMAEV2_DEPTH = 12


def _build_public_videomaev2_key_map(depth: int) -> dict[str, str]:
    """`mae-b/pytorch_model.bin`'s key names -> `VideoMAEEncoder`'s, post
    `backbone.` strip and q_bias/v_bias merge (`_prepare_public_videomae_state_dict`).

    `cls_head.*` and `pos_embed` are deliberately absent: the head is dropped
    (a `DiagBundle` carries only the encoder), and the checkpoint has no
    `pos_embed` value at all -- see `_public_videomae_sinusoid_pos_embed`.
    """
    key_map = {
        "patch_embed.proj.weight": "patch_embed.proj.weight",
        "patch_embed.proj.bias": "patch_embed.proj.bias",
        "fc_norm.weight": "norm.weight",
        "fc_norm.bias": "norm.bias",
    }
    for i in range(depth):
        key_map.update(
            {
                f"blocks.{i}.norm1.weight": f"blocks.{i}.norm1.weight",
                f"blocks.{i}.norm1.bias": f"blocks.{i}.norm1.bias",
                f"blocks.{i}.attn.qkv.weight": f"blocks.{i}.attn.qkv.weight",
                f"blocks.{i}.attn.qkv.bias": f"blocks.{i}.attn.qkv.bias",
                f"blocks.{i}.attn.proj.weight": f"blocks.{i}.attn.proj.weight",
                f"blocks.{i}.attn.proj.bias": f"blocks.{i}.attn.proj.bias",
                f"blocks.{i}.norm2.weight": f"blocks.{i}.norm2.weight",
                f"blocks.{i}.norm2.bias": f"blocks.{i}.norm2.bias",
                f"blocks.{i}.mlp.fc1.weight": f"blocks.{i}.mlp.net.0.weight",
                f"blocks.{i}.mlp.fc1.bias": f"blocks.{i}.mlp.net.0.bias",
                f"blocks.{i}.mlp.fc2.weight": f"blocks.{i}.mlp.net.3.weight",
                f"blocks.{i}.mlp.fc2.bias": f"blocks.{i}.mlp.net.3.bias",
            }
        )
    return key_map


# Populated against the real downloaded mae-b/pytorch_model.bin checkpoint
# (task 1.0-1.5): its `backbone.*`-prefixed keys, post q_bias/v_bias merge,
# rename 1:1 onto `VideoMAEEncoder.state_dict()`'s keys except the MLP's
# `fc1`/`fc2` (-> `mlp.net.0`/`mlp.net.3`, this repo's `Sequential` indices)
# and `fc_norm` (-> `norm`, the only normalisation layer this repo's encoder
# has at that position). Empty for any block count other than 12 -- a rung
# built against the wrong depth is a config bug the geometry gate below
# catches before this dict is ever indexed.
_KEY_MAP: dict[str, str] = _build_public_videomaev2_key_map(_PUBLIC_VIDEOMAEV2_DEPTH)


def remap_public_videomae(state_dict: dict, target_state_dict: dict) -> dict:
    """Rename a published VideoMAEv2 state dict onto this repo's parameter names.

    Raises on any key it has no mapping for and on any shape mismatch. A
    `strict=False` pass here would mean tiering metrics against a partially
    random model while reporting it as the public checkpoint.
    """
    remapped = {}
    for key, tensor in state_dict.items():
        if key not in _KEY_MAP:
            raise KeyError(
                f"no mapping for published key {key!r}; _KEY_MAP is populated "
                "against the real mae-b/pytorch_model.bin checkpoint (Plan 3)"
            )
        target_key = _KEY_MAP[key]
        target = target_state_dict.get(target_key)
        if target is not None and tuple(target.shape) != tuple(tensor.shape):
            raise ValueError(
                f"{key} -> {target_key}: published shape {tuple(tensor.shape)} "
                f"does not match target {tuple(target.shape)}"
            )
        remapped[target_key] = tensor
    return remapped


def _unwrap_public_videomae_checkpoint(raw) -> dict:
    """Published checkpoints wrap their state dict under one of a few keys."""
    for wrap_key in ("model", "state_dict", "module"):
        if isinstance(raw, dict) and isinstance(raw.get(wrap_key), dict):
            return raw[wrap_key]
    if isinstance(raw, dict):
        return raw
    raise RungUnavailable(
        f"public_videomaev2 checkpoint has no dict-like state_dict container "
        f"(got {type(raw).__name__})"
    )


def _prepare_public_videomae_state_dict(raw_state_dict: dict) -> dict:
    """Merge `q_bias`/`v_bias` into `qkv.bias`, strip `backbone.`, drop `cls_head.*`.

    The published checkpoint's `Attention` module stores `qkv.bias` as two
    separate parameters and concatenates them at forward time --
    `cat([q_bias, zeros_like(v_bias), v_bias])` (the k-bias is always exactly
    zero) -- rather than storing one fused bias tensor the way this repo's
    `SelfAttention.qkv` does. Materialising that same concatenation here is a
    faithful reproduction of the checkpoint's own forward pass, not a
    reinterpretation of it.
    """
    q_biases: dict[str, "torch.Tensor"] = {}
    v_biases: dict[str, "torch.Tensor"] = {}
    rest: dict[str, "torch.Tensor"] = {}
    for key, tensor in raw_state_dict.items():
        if key.startswith("cls_head."):
            continue
        if key.endswith(".attn.q_bias"):
            q_biases[key[: -len(".q_bias")]] = tensor
        elif key.endswith(".attn.v_bias"):
            v_biases[key[: -len(".v_bias")]] = tensor
        else:
            rest[key] = tensor

    merged = dict(rest)
    for prefix, q_bias in q_biases.items():
        v_bias = v_biases[prefix]
        merged[f"{prefix}.qkv.bias"] = torch.cat(
            (q_bias, torch.zeros_like(v_bias), v_bias)
        )

    return {key.removeprefix("backbone."): tensor for key, tensor in merged.items()}


def _public_videomae_geometry(raw_state_dict: dict) -> tuple[int, int | None]:
    """Depth and patch-embed output width, read from key names/shapes only.

    No tensor *value* is read -- this is metadata inspection, run before any
    remap is attempted, so a mismatched checkpoint is rejected before a single
    weight is copied anywhere.
    """
    block_ids: set[int] = set()
    embed_dim: int | None = None
    for key, tensor in raw_state_dict.items():
        parts = key.split(".")
        if "blocks" in parts:
            idx = parts.index("blocks")
            if idx + 1 < len(parts) and parts[idx + 1].isdigit():
                block_ids.add(int(parts[idx + 1]))
        if key.endswith("patch_embed.proj.weight"):
            embed_dim = int(tensor.shape[0])
    depth = max(block_ids) + 1 if block_ids else 0
    return depth, embed_dim


def _public_videomae_sinusoid_pos_embed(n_position: int, d_hid: int) -> "torch.Tensor":
    """Ports `OpenGVLab/VideoMAEv2`'s `get_sinusoid_encoding_table` exactly.

    `mae-b/pytorch_model.bin` was trained with `use_learnable_pos_emb=False`:
    a fixed sin-cos position table computed at forward time and never saved to
    `state_dict`. This repo's `VideoMAEEncoder.pos_embed` is a required
    learnable parameter, so there is no checkpoint value to remap onto it.
    Computing the same deterministic table the original model computes is a
    faithful reproduction of that model's behaviour, not an approximation --
    see modeling_finetune.py#L302 in the published repo.
    """

    def angle_vec(position: int) -> list[float]:
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    table = np.array([angle_vec(pos) for pos in range(n_position)])
    table[:, 0::2] = np.sin(table[:, 0::2])
    table[:, 1::2] = np.cos(table[:, 1::2])
    return torch.tensor(table, dtype=torch.float32).unsqueeze(0)


def _load_public_videomaev2_rung(
    model_cfg, ckpt_path: "Path | str | None" = None
) -> DiagBundle:
    """Load rung 3 from a local copy of `mae-b/pytorch_model.bin`, or raise
    `RungUnavailable` naming why -- never a partial load.

    `PUBLIC_VIDEOMAEV2_WEIGHTS` is a `host:path` string, not a local path, so
    it is never passed to `torch.load`/`Path.exists` directly: a local path
    must be supplied via `ckpt_path` or the `PUBLIC_VIDEOMAEV2_LOCAL_PATH` env
    var (the file must be copied to the running host first).
    """
    resolved = ckpt_path or os.environ.get("PUBLIC_VIDEOMAEV2_LOCAL_PATH")
    if not resolved:
        raise RungUnavailable(
            "public_videomaev2 checkpoint not available locally -- "
            f"{PUBLIC_VIDEOMAEV2_WEIGHTS} must be copied to this host and its "
            "local path passed as ckpt_path or PUBLIC_VIDEOMAEV2_LOCAL_PATH"
        )
    if not Path(resolved).exists():
        raise RungUnavailable(f"public_videomaev2 checkpoint not found at {resolved}")

    raw = torch.load(str(resolved), map_location="cpu", weights_only=False)
    raw_state_dict = _unwrap_public_videomae_checkpoint(raw)

    depth, embed_dim = _public_videomae_geometry(raw_state_dict)
    target_depth = int(model_cfg.depth)
    target_embed_dim = int(model_cfg.embed_dim)
    if depth != target_depth or embed_dim != target_embed_dim:
        raise RungUnavailable(
            f"public_videomaev2 checkpoint geometry (depth={depth}, "
            f"embed_dim={embed_dim}) does not match model_cfg "
            f"(depth={target_depth}, embed_dim={target_embed_dim}); reported "
            "unavailable rather than approximated"
        )

    prepared = _prepare_public_videomae_state_dict(raw_state_dict)
    target = VideoMAEEncoder(model_cfg)
    try:
        remapped = remap_public_videomae(prepared, target.state_dict())
    except (KeyError, ValueError) as exc:
        raise RungUnavailable(f"public_videomaev2 remap failed: {exc}") from exc

    remapped["pos_embed"] = _public_videomae_sinusoid_pos_embed(
        int(target.pos_embed.shape[1]), target.embed_dim
    )
    result = target.load_state_dict(remapped, strict=False)
    _raise_on_real_mismatch(result, target, "public_videomaev2")
    return DiagBundle(
        encoder=target.eval(), model=None, run_id="public_videomaev2", round=None
    )


def is_hf_levjepa(encoder) -> bool:
    """Rung 4's encoder: the released LeVJEPA ViT-L, loaded through transformers."""
    return getattr(getattr(encoder, "config", None), "model_type", None) == "levjepa"


def _hf_levjepa_hidden_states(encoder, pixel_values: "torch.Tensor") -> list:
    """`forward_with_hidden_states` for the HF LeVJEPA encoder: `depth + 1` raw
    (pre-norm, CLS attached) streams, read by hooks off the model's own forward
    rather than a mirrored traversal -- its RoPE and block-causal mask are built
    inside `VisionTransformer.forward`, and a copy of that is exactly the drift
    the VideoMAE path needs a parity test to guard against."""
    blocks = encoder.encoder.blocks
    states: list = []
    handles = [
        blocks[0].register_forward_pre_hook(lambda _m, args: states.append(args[0]))
    ] + [
        b.register_forward_hook(lambda _m, _a, out: states.append(out)) for b in blocks
    ]
    try:
        with torch.no_grad():
            encoder(pixel_values=pixel_values)
    finally:
        for handle in handles:
            handle.remove()
    return states


def _hf_levjepa_from_config(model_cfg):
    """A randomly initialised LeVJEPA from `model_cfg.weights_path`'s `config.json`."""
    from transformers import AutoConfig, AutoModel

    hf_cfg = AutoConfig.from_pretrained(
        str(model_cfg.weights_path), trust_remote_code=True, local_files_only=True
    )
    return AutoModel.from_config(hf_cfg, trust_remote_code=True)


def encoder_geometry(encoder) -> tuple[int, int, int, int]:
    """`(depth, embed_dim, num_frames, image_size)` for either encoder family."""
    if is_hf_levjepa(encoder):
        c = encoder.config
        return c.depth, c.embed_dim, c.num_frames, c.img_size
    return (
        len(encoder.blocks),
        encoder.embed_dim,
        encoder.num_frames,
        encoder.image_size,
    )


def patch_tokens(encoder, hidden: "torch.Tensor") -> "torch.Tensor":
    """`hidden` with the non-patch tokens removed: registers for VideoMAE, the
    CLS token for HF LeVJEPA."""
    if is_hf_levjepa(encoder):
        return hidden[:, 1:]
    return encoder.registers.strip(hidden)


def _load_supervised_va_rung(
    model_cfg, ckpt_path: "Path | str | None" = None
) -> DiagBundle:
    """Load rung 4 -- a Stage-2 VA regressor fine-tuned end-to-end on AFEW-VA
    -- as its encoder alone, or raise.

    The checkpoint's `model_name` picks the architecture, and `model_cfg` must
    agree with it: a `videomae_va_regressor` (rung 3 fine-tuned,
    `experiment=ssl_diag_rung4_public_videomae`) takes the ladder's ViT-B
    config; a `levjepa_hf_va_regressor` takes `conf/model/levjepa_hf_large.yaml`
    (its tensors overwrite an architecture read from `weights_path`'s
    `config.json`). The tensors come from the `backbone.*` subtree. The
    `head.*` subtree is dropped by design; any other key, a missing key, or a
    shape mismatch raises before anything is loaded -- never a partial load,
    never a fallback path.
    """
    if ckpt_path is None or not Path(ckpt_path).is_file():
        raise RungUnavailable(f"supervised_va checkpoint not found at {ckpt_path}")
    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model_name = payload.get("model_name")
    expected_encoder = {
        "videomae_va_regressor": "videomae",
        "levjepa_hf_va_regressor": "levjepa_hf",
    }.get(model_name)
    if expected_encoder is None:
        raise RungUnavailable(
            f"{ckpt_path} is a {model_name!r} checkpoint, not a videomae or "
            "levjepa_hf VA regressor"
        )
    encoder_kind = str(model_cfg.get("encoder", "videomae"))
    if encoder_kind != expected_encoder:
        raise ValueError(
            f"rung 4 checkpoint is a {model_name}; pass a model config with "
            f"encoder={expected_encoder}, got encoder={encoder_kind!r}"
        )
    state = {
        k.removeprefix("module.").removeprefix("_orig_mod."): v
        for k, v in payload["model_state_dict"].items()
    }
    stray = sorted(k for k in state if not k.startswith(("backbone.", "head.")))
    if stray:
        raise RungUnavailable(f"supervised_va checkpoint has unexpected keys {stray}")
    backbone_state = {
        k.removeprefix("backbone."): v
        for k, v in state.items()
        if k.startswith("backbone.")
    }

    if expected_encoder == "levjepa_hf":
        encoder = _hf_levjepa_from_config(model_cfg)
    else:
        encoder = VideoMAEEncoder(model_cfg)
    target = encoder.state_dict()
    missing = sorted(set(target) - set(backbone_state))
    unexpected = sorted(set(backbone_state) - set(target))
    mismatched = sorted(
        k
        for k in set(target) & set(backbone_state)
        if target[k].shape != backbone_state[k].shape
    )
    if missing or unexpected or mismatched:
        raise RungUnavailable(
            f"supervised_va backbone does not match {model_cfg.get('name')}: "
            f"missing={missing} unexpected={unexpected} shape_mismatch={mismatched}"
        )
    encoder.load_state_dict(backbone_state, strict=True)
    return DiagBundle(
        encoder=encoder.eval(),
        model=None,
        run_id=f"supervised_va:{Path(ckpt_path).parent.name}",
        round=None,
    )


def load_ladder_rung(
    rung: Rung,
    model_cfg,
    *,
    mesh_shared: "Path | str | None" = None,
    ladder_run_ids: dict[Rung, "str | dict"] | None = None,
    full_model: bool = False,
    public_videomaev2_ckpt_path: "Path | str | None" = None,
    supervised_va_ckpt_path: "Path | str | None" = None,
    supervised_va_model_cfg=None,
    seed: int | None = None,
) -> DiagBundle:
    """Build one ladder rung, or raise naming what is missing. `seed` pins
    `RANDOM_INIT`'s weights, so the floor is the same network in every job.

    Never falls back to a default run: a rung silently built from the wrong
    checkpoint is worse than an absent rung, which `ladder_tau` already handles
    by scoring the ladder as incomplete.
    """
    if rung is Rung.MUTILATION:
        raise ValueError(
            "MUTILATION is a self-test control asserted directly by the check "
            "that builds it, not a ladder rung to load"
        )
    if rung is Rung.RANDOM_INIT:
        return random_init_bundle(model_cfg, full_model=full_model, seed=seed)
    if rung is Rung.PUBLIC_VIDEOMAEV2:
        return _load_public_videomaev2_rung(model_cfg, public_videomaev2_ckpt_path)
    if rung is Rung.SUPERVISED_VA:
        # Rung 4's checkpoint names its architecture; a levjepa_hf one takes its
        # own model config, and a config that disagrees is rejected, loudly.
        return _load_supervised_va_rung(
            supervised_va_model_cfg or model_cfg, supervised_va_ckpt_path
        )

    entry = (ladder_run_ids or {}).get(rung)
    if entry is None:
        raise ValueError(
            f"{rung.name} needs an explicit entry in ladder_run_ids; there is "
            "no default run to fall back to"
        )

    # An entry is either a run id (a merged run, loaded from master.pt) or a
    # solo route -- {"run_id", "node", "round"} -- for a mesh.merge=false run,
    # which writes no master.pt at all and must be read from its pending file.
    if isinstance(entry, dict):
        missing = [k for k in ("run_id", "node", "round") if entry.get(k) is None]
        if missing:
            raise ValueError(
                f"{rung.name}'s solo route is missing {', '.join(missing)}; a "
                "pending checkpoint is keyed by run_id, round and node, so a "
                "partial route cannot name one"
            )
        if mesh_shared is None:
            raise ValueError(
                f"{rung.name} needs mesh_shared to locate {entry['run_id']}"
            )
        return load_solo_bundle(
            entry["run_id"],
            mesh_shared,
            model_cfg,
            node=entry["node"],
            round=entry["round"],
            full_model=full_model,
        )

    if mesh_shared is None:
        raise ValueError(f"{rung.name} needs mesh_shared to locate {entry}")
    return load_bundle(entry, mesh_shared, model_cfg, full_model=full_model)


# The rung-2 submission as actually run, not a description of one: these
# flags are operational and drift, so the next person reproduces the rung by
# re-running this, not by re-deriving it.
#
# The 50-clip set was carved from the built 8fps corpus rather than encoded
# fresh -- a straight WebDataset member copy (one `.pickle` per clip) out of
# /Data/voxceleb2_shards_8fps/shard-000000.tar on `node114`, so rung 2's frames
# are bit-identical to what every other rung trained on. Re-encoding would
# have made the rung differ from rung 1 in the data as well as the training,
# which is exactly the confound the ladder exists to avoid.
#
# The first build (ssl-diag-overfit50, 2026-09-08, node030, 4 rounds) was not
# an overfit: a round on 50 clips is ~331 micro-batches, ~5 optimizer steps,
# so it ended at train loss 0.95 and read as random init on every check. The
# rebuild runs 300 rounds from rung 1's init seed (train.seed=0), so rungs 0-2
# share one initialisation. It runs on node114, where the subset lives (node-to-
# node rsync to node030 is refused since the key-auth change), with
# MESH_ROOT pointed at the subset. A merge=false run pushes a 183 MB pending
# file per round and never prunes them -- 300 rounds overran the 30 GB NFS
# quota once -- so the wrapper keeps only the newest two.
LADDER_RUNG2_ROUTE = {
    "run_id": "ssl-diag-overfit50-v2",
    "node": "node114",
    "slurm_job_id": 1064,
    "submitted_from": "login-node:~/stern-ladder",
    "shard_subset": "node114:/Data/$USER/overfit50_shards (50 train + 10 val clips)",
    # Per-run files, written by hand: shard_assignments_<run>.json is
    # {"node114": ["shard-000000.tar"]}; shard_map_<run>.json is the 2026-09-08
    # map with node030 replaced by node114. The global shard_map.json is shared
    # by every run and must not be repointed.
    "sbatch_cmd": (
        "sbatch rung2_overfit.sbatch  # --nodelist=node114; exports "
        "MESH_SHARD_MAP=$HOME/stern_mesh/metadata/"
        "shard_map_ssl-diag-overfit50-v2.json "
        "MESH_ROOT=/Data/$USER/overfit50_shards; runs "
        "video_encoding/scripts/run_mesh_worker.sh videomae_mesh_ssl "
        "ssl-diag-overfit50-v2 +run_tag=ab-single_node ++mesh.merge=false "
        "++mesh.rounds=300 ++train.seed=0, pruning pending files every 20 s"
    ),
    "init_seed": 0,
}

# Rung 2 is loaded through the solo route: a merge=false run writes no
# master.pt (see load_solo_bundle). Round 299 is the last of the 300 (0-based).
LADDER_RUN_IDS = {
    Rung.OVERFIT: {
        "run_id": LADDER_RUNG2_ROUTE["run_id"],
        "node": LADDER_RUNG2_ROUTE["node"],
        "round": 299,
    },
}
