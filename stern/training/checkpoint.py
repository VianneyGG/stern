"""Checkpoint save/load with NFS-safe atomic writes and all-time-best guard.

The cluster's shared filesystem (login-node NFS, T1 findings) makes naive
per-epoch `torch.save` risky on two axes: partial writes if the job is
killed mid-write, and stalls on rank 0 during a large load. This module
addresses both:

- Atomic writes: everything goes through `save()`'s tmp→rename pattern so
  a crash mid-write leaves the previous good file intact.
- All-time-best guard: `save_best()` avoids the "worse-than-disk" overwrite
  bug where a resumed run's first-epoch val loss races past a better older
  checkpoint. The guard uses a *cached* `_disk_best` float instead of
  loading the full checkpoint payload every epoch (a 1+ GB NFS read blocks
  rank 0 and can trip the NCCL watchdog).

Load-side mismatch guards (`model_name`, `optimizer_name`) copy the WHN
pattern: fail closed rather than silently loading state into a differently-
shaped model. A run that resumes with an incompatible optimizer starts the
optimizer fresh but keeps the model weights.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import io
import logging
import os
import time
import zipfile

import torch

logger = logging.getLogger(__name__)


def save(path: Path | str, payload: Mapping[str, Any]) -> None:
    """Atomic torch.save: write to a `.tmp` sibling then rename.

    Rename is atomic on POSIX filesystems (including our NFS mount when
    src and dst are on the same mount, which they are since both are
    inside the run's checkpoint dir). A SIGKILL between torch.save and
    rename leaves the previous good file intact, worst case.
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), tmp)
    tmp.replace(path)


def try_save(
    path: Path | str,
    payload: Mapping[str, Any],
    *,
    attempts: int = 3,
    backoff_s: float = 30.0,
) -> bool:
    """Best-effort `save()`: retries on failure, returns False instead of
    raising once `attempts` is exhausted.

    The observed production failure (NFS write errors during finalization)
    clusters into cluster-wide events lasting minutes, not per-node blips --
    so a caller that needs to keep training rather than die should get a
    bool back, not an exception.
    """
    for attempt in range(1, attempts + 1):
        try:
            save(path, payload)
            return True
        except Exception:
            if attempt == attempts:
                return False
            time.sleep(backoff_s)
    return False


def save_best(
    path: Path | str,
    payload: Mapping[str, Any],
    current_best: float,
    *,
    is_ssl: bool,
    cached_disk_best: float | None,
) -> tuple[bool, float | None]:
    """Save `payload` to `path` iff `current_best` beats both the running
    best AND the on-disk all-time best.

    Returns `(did_save, new_disk_best)`. Caller must persist the returned
    `new_disk_best` for the next epoch's call — we deliberately don't
    re-read the checkpoint to avoid an NFS round-trip per epoch.

    "Better" is objective-specific: for SSL it's a smaller reconstruction
    loss; for probe it's a larger CCC / accuracy. `is_ssl` selects the
    comparator once at call time.
    """
    if cached_disk_best is not None:
        beats_disk = (
            (current_best < cached_disk_best)
            if is_ssl
            else (current_best > cached_disk_best)
        )
        if not beats_disk:
            return False, cached_disk_best
    save(path, payload)
    return True, current_best


def load(
    path: Path | str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device | str,
    expected_model_name: str,
    expected_optimizer_name: str | None,
) -> tuple[int, float | None]:
    """Load a checkpoint into `model`/`optimizer`/`scheduler` in place.

    Returns `(start_epoch, best_metric)`. Mismatch guards:

    - If `checkpoint['model_name'] != expected_model_name`, skip the load
      entirely and return `(0, None)` (fresh start — do not silently mix
      state dicts across architectures).
    - If `checkpoint['optimizer_name'] != expected_optimizer_name`, keep
      the model weights but start the optimizer from scratch (common on
      Adam→AdamW switches).

    `strict=False` on model load: LoRA-wrapped runs vs. non-LoRA share
    most keys but not all, and we want the union to succeed with a warning
    rather than a hard failure.
    """
    ckpt: MutableMapping[str, Any] = torch.load(
        path, map_location=device, weights_only=False
    )

    ckpt_model_name = ckpt.get("model_name")
    if ckpt_model_name != expected_model_name:
        # Fail closed — a wrong-architecture load is worse than no load.
        return 0, None

    raw_model = unwrap(model)
    state_dict = ckpt["model_state_dict"]
    state_dict = _strip_orig_mod_prefix(state_dict)
    load_result = raw_model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        # strict=False intentionally tolerates key differences (e.g. LoRA vs.
        # non-LoRA runs sharing most keys) -- but a silent tolerance is
        # indistinguishable from a real architecture-drift bug (e.g. a block
        # internals rewrite renaming attention/MLP parameters) without this
        # log line. Mirrors the equivalent guard in
        # training/diloco/mesh_round.py's master-checkpoint load.
        logger.warning(
            "checkpoint.load: state_dict key mismatch loading %s -- "
            "missing=%s unexpected=%s",
            path,
            load_result.missing_keys,
            load_result.unexpected_keys,
        )

    start_epoch = int(ckpt.get("epoch", -1)) + 1
    best_metric = ckpt.get("best_val_metric", ckpt.get("best_val_accuracy"))

    if optimizer is not None:
        ckpt_opt_name = ckpt.get("optimizer_name")
        opt_state = ckpt.get("optimizer_state_dict")
        if opt_state is not None and ckpt_opt_name == expected_optimizer_name:
            try:
                optimizer.load_state_dict(opt_state)
            except (ValueError, KeyError):
                # Shapes changed (e.g. LoRA rank bumped); silently drop optimizer state.
                pass

    if scheduler is not None:
        sched_state = ckpt.get("scheduler_state_dict")
        if sched_state is not None:
            try:
                scheduler.load_state_dict(sched_state)
            except (ValueError, KeyError):
                pass

    return start_epoch, (float(best_metric) if best_metric is not None else None)


def peek_best(path: Path | str) -> float | None:
    """Read only the `best_val_metric` field from a checkpoint on disk.

    Called once at startup to seed the `save_best` cache without loading
    the full payload every epoch. Falls back to legacy field names for
    checkpoints written before this module existed.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        meta = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError):
        return None
    value = meta.get("best_val_metric", meta.get("best_val_accuracy"))
    return float(value) if value is not None else None


def _strip_orig_mod_prefix(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Undo the `_orig_mod.` prefix `torch.compile` prepends.

    Checkpoints written from a compiled model include this prefix; loads
    into an uncompiled model would silently miss every param without this
    strip. Idempotent — safe when no prefix is present.
    """
    if any(k.startswith("_orig_mod.") for k in state_dict):
        return {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    return dict(state_dict)


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Strip DDP + torch.compile wrappers to expose the raw nn.Module.

    Public: shared by every module that needs to inspect/save the raw
    model (this file, `stern/training/pretrain.py`,
    `stern/training/finetune.py`) instead of each re-implementing the
    same two-step unwrap.
    """
    raw = model.module if hasattr(model, "module") else model
    if hasattr(raw, "_orig_mod"):
        raw = raw._orig_mod
    return raw


def base_payload(
    *,
    model,
    optimizer,
    epoch: int,
    best_val_metric: float,
    model_name: str,
    optimizer_name: str,
) -> dict[str, Any]:
    """Common 4-key checkpoint payload spine shared by every training stage.

    Each stage's own `_payload()` calls this for the shared fields and adds
    whatever else it needs on top (e.g. `scheduler_state_dict`) -- the exact
    `model_name`/`optimizer_name` strings passed in are load-bearing:
    `load()`'s `expected_model_name`/`expected_optimizer_name` checks them
    verbatim to decide whether a checkpoint is compatible.
    """
    return {
        "model_state_dict": unwrap(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer_name": optimizer_name,
        "epoch": epoch,
        "best_val_metric": best_val_metric,
        "model_name": model_name,
    }


_LEGACY_RENAMES = (
    (".attn.in_proj_", ".attn.qkv."),
    (".attn.out_proj.", ".attn.proj."),
)
_LEGACY_MLP = re.compile(r"\.mlp\.(\d+)\.")


def remap_legacy_attention_keys(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Rename pre-ViT-refactor parameter keys onto the current module layout.

    The refactor replaced nn.MultiheadAttention with ViTAttention and wrapped
    the MLP's nn.Sequential in a `net` attribute. Every checkpoint written
    before it -- including pool-cars-best-120r, the run the motion probes score
    -- shares only 77 of 205 keys with the current model and would otherwise
    load as 128 unexpected + 129 missing keys.

    This is a pure rename. ViTAttention's (qkv, proj) parameters mirror
    nn.MultiheadAttention's (in_proj, out_proj) shapes and ordering exactly, so
    the tensors transfer 1:1 with no reshaping -- the numerical guarantee is
    test_vit_attention.py's parity test, which copies weights across by these
    same names.

    Idempotent, so callers may apply it unconditionally rather than sniffing
    the checkpoint's vintage.
    """
    remapped = {}
    for key, value in state_dict.items():
        for old, new in _LEGACY_RENAMES:
            key = key.replace(old, new)
        key = _LEGACY_MLP.sub(r".mlp.net.\1.", key)
        remapped[key] = value
    return remapped


class ArchMismatchError(RuntimeError):
    """A checkpoint's encoder geometry disagrees with the model loading it."""


_BLOCK_INDEX = re.compile(r"(?:^|\.)blocks\.(\d+)\.")


def encoder_arch_fingerprint(state_dict: Mapping[str, Any]) -> dict[str, int]:
    """Derive encoder geometry from tensor shapes alone.

    Shape-derived rather than read from a recorded metadata field, so it works
    on every checkpoint ever written by this repo without a migration.

    `num_heads` is deliberately absent: `qkv` is (3*D, D) for any head count, so
    a head-count change is invisible in the shapes and cannot be caught here.
    `in_channels` is absent too -- 3ch<->6ch inflation is a *supported* load
    (see `load_encoder_state`), so it must not trip the guard.
    """
    fp: dict[str, int] = {}
    depth = -1
    for key, value in state_dict.items():
        m = _BLOCK_INDEX.search(key)
        if m:
            depth = max(depth, int(m.group(1)))
        if key.endswith("pos_embed") and value.ndim == 3:
            fp["num_tokens"] = int(value.shape[1])
            fp["embed_dim"] = int(value.shape[2])
        if key.endswith("blocks.0.mlp.net.0.weight") and value.ndim == 2:
            fp["mlp_hidden"] = int(value.shape[0])
    if depth >= 0:
        fp["depth"] = depth + 1
    return fp


def check_encoder_arch(encoder: torch.nn.Module, state_dict: Mapping[str, Any]) -> None:
    """Raise if the checkpoint's geometry differs from the target encoder's.

    Without this, `load_state_dict(strict=False)` turns every geometry change
    into a silent partial load: mismatched blocks land in `missing_keys`, the
    run proceeds on randomly-initialised weights, and the loss curve is the
    only hint. Fails closed, matching the module's `model_name` guard.
    """
    want = encoder_arch_fingerprint(encoder.state_dict())
    got = encoder_arch_fingerprint(state_dict)
    diffs = [
        f"{k}: checkpoint={got[k]} model={want[k]}"
        for k in sorted(want.keys() & got.keys())
        if want[k] != got[k]
    ]
    if diffs:
        raise ArchMismatchError(
            "encoder geometry differs from the checkpoint: "
            + "; ".join(diffs)
            + ". Pass allow_arch_mismatch=True only if a partial load is intended."
        )


def load_encoder_state(
    encoder: torch.nn.Module,
    state_dict: Mapping[str, Any],
    *,
    allow_arch_mismatch: bool = False,
) -> tuple[list[str], list[str]]:
    """Load an encoder state dict, inflating/truncating patch_embed input
    channels when they differ from the target encoder's configured
    in_channels, instead of failing on a shape mismatch.

    Extra channels are zero-initialized: a zero-init channel starts as a
    no-op, so the pretrained RGB weights are preserved exactly and gradient
    updates "activate" the new channel(s) during fine-tuning rather than
    corrupting what was already learned. Returns the `(missing, unexpected)`
    keys from the underlying `load_state_dict(strict=False)` call.

    Raises `ArchMismatchError` when the checkpoint's transformer geometry
    differs from the target's; set `allow_arch_mismatch` for deliberate
    partial loads.
    """
    if not allow_arch_mismatch:
        check_encoder_arch(encoder, state_dict)
    state_dict = dict(state_dict)
    w = state_dict.get("patch_embed.proj.weight")
    # Some encoders (LeVJEPAEncoder) have no top-level patch_embed attribute
    # at all -- only reach for it once w confirms there's actually something
    # to inflate/truncate.
    target_w = encoder.patch_embed.proj.weight if w is not None else None
    if w is not None and target_w.shape[1] != w.shape[1]:
        if target_w.shape[1] > w.shape[1]:
            extra = torch.zeros(
                w.shape[0],
                target_w.shape[1] - w.shape[1],
                *w.shape[2:],
                dtype=w.dtype,
            )
            state_dict["patch_embed.proj.weight"] = torch.cat([w, extra], dim=1)
        else:
            state_dict["patch_embed.proj.weight"] = w[
                :, : target_w.shape[1]
            ].contiguous()
    result = encoder.load_state_dict(state_dict, strict=False)
    return list(result.missing_keys), list(result.unexpected_keys)


def _remap_payload(payload: dict) -> dict:
    """Apply the legacy key rename to a loaded checkpoint's model weights.

    Both payload shapes in circulation are handled: a dict carrying a
    "model_state_dict" entry, and a bare state dict -- train.py and
    dump_probe_predictions.py both fall back to the payload itself via
    .get("model_state_dict", ckpt). Sibling entries (optimizer state, round,
    epoch) are left untouched.
    """
    if not isinstance(payload, dict):
        return payload
    if "model_state_dict" in payload:
        payload["model_state_dict"] = remap_legacy_attention_keys(
            payload["model_state_dict"]
        )
        return payload
    return remap_legacy_attention_keys(payload)


def load_checkpoint_file(path: Path | str) -> dict:
    """torch.load a checkpoint, transparently supporting an unzipped PyTorch
    archive directory (data.pkl, data/, version, ...) alongside a normal
    single-file checkpoint.

    A directory is repacked into an in-memory ZIP (ZIP_STORED, no compression)
    so torch.load can consume it -- useful for checkpoints synced off the
    cluster as an extracted directory rather than the single .pt file.

    Pre-ViT-refactor parameter names are remapped on the way out, so every
    consumer reads one key layout regardless of when the checkpoint was
    written. See remap_legacy_attention_keys.
    """
    p = Path(path)
    if not p.is_dir():
        return _remap_payload(torch.load(p, map_location="cpu", weights_only=False))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        for root, _, files in os.walk(str(p)):
            for fname in files:
                fpath = os.path.join(root, fname)
                arcname = "archive/" + os.path.relpath(fpath, str(p))
                # Files may carry pre-1980 timestamps; ZipInfo requires >= 1980.
                zi = zipfile.ZipInfo(arcname, date_time=(1980, 1, 1, 0, 0, 0))
                with open(fpath, "rb") as fh:
                    zf.writestr(zi, fh.read())
    buf.seek(0)
    return _remap_payload(torch.load(buf, map_location="cpu", weights_only=False))
