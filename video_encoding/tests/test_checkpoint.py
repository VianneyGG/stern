"""Unit tests for the checkpoint-loading robustness helpers."""

from __future__ import annotations

import zipfile

import pytest
import torch
from omegaconf import OmegaConf

from video_encoding.models.videomaev2.encoder import VideoMAEEncoder
from video_encoding.models.videomaev2.mae import VideoMAEForPreTraining
from video_encoding.training.checkpoint import (
    load,
    load_checkpoint_file,
    load_encoder_state,
    remap_legacy_attention_keys,
    save,
    try_save,
)

pytestmark = pytest.mark.integration


def _tiny_cfg(in_channels: int):
    return OmegaConf.create(
        {
            "num_frames": 4,
            "image_size": 16,
            "in_channels": in_channels,
            "tubelet_size": 2,
            "patch_size": 8,
            "embed_dim": 32,
            "depth": 2,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "dropout": 0.0,
        }
    )


def test_load_encoder_state_inflates_extra_channels():
    """A 3-channel donor checkpoint loaded into a 6-channel encoder must zero-pad
    the extra channels (a zero-init channel starts as a no-op) rather than fail."""
    encoder = VideoMAEEncoder(_tiny_cfg(in_channels=6))
    donor = VideoMAEEncoder(_tiny_cfg(in_channels=3))

    missing, unexpected = load_encoder_state(encoder, donor.state_dict())

    assert missing == []
    assert unexpected == []
    w = encoder.patch_embed.proj.weight
    assert torch.equal(w[:, 0:3], donor.patch_embed.proj.weight)
    assert torch.all(w[:, 3:6] == 0.0)


def test_load_encoder_state_truncates_extra_channels():
    """A 6-channel donor checkpoint loaded into a 3-channel encoder must keep
    only the first 3 channels rather than fail on a shape mismatch."""
    encoder = VideoMAEEncoder(_tiny_cfg(in_channels=3))
    donor = VideoMAEEncoder(_tiny_cfg(in_channels=6))

    load_encoder_state(encoder, donor.state_dict())

    assert torch.equal(
        encoder.patch_embed.proj.weight, donor.patch_embed.proj.weight[:, :3]
    )


def test_load_encoder_state_same_channels_is_a_plain_load():
    encoder = VideoMAEEncoder(_tiny_cfg(in_channels=3))
    donor = VideoMAEEncoder(_tiny_cfg(in_channels=3))

    load_encoder_state(encoder, donor.state_dict())

    assert torch.equal(encoder.patch_embed.proj.weight, donor.patch_embed.proj.weight)


def test_load_checkpoint_file_from_plain_file(tmp_path):
    payload = {"epoch": 3, "model_state_dict": {"x": torch.tensor([1.0, 2.0])}}
    path = tmp_path / "ref.pt"
    torch.save(payload, path)

    loaded = load_checkpoint_file(path)

    assert loaded["epoch"] == 3
    assert torch.equal(
        loaded["model_state_dict"]["x"], payload["model_state_dict"]["x"]
    )


def test_load_checkpoint_file_from_unzipped_directory(tmp_path):
    """Some checkpoints get synced off the cluster as an extracted PyTorch
    archive directory instead of the single .pt file; load_checkpoint_file
    must transparently repack and read those too."""
    payload = {"epoch": 3, "model_state_dict": {"x": torch.tensor([1.0, 2.0])}}
    ref_path = tmp_path / "ref.pt"
    torch.save(payload, ref_path)

    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with zipfile.ZipFile(ref_path) as zf:
        zf.extractall(extracted)
    # torch's own zip nests everything one level under an archive-name folder;
    # flatten that so archive_dir directly contains data.pkl/data/version.
    inner = next(extracted.iterdir())
    archive_dir = tmp_path / "archive_dir"
    inner.rename(archive_dir)

    loaded = load_checkpoint_file(archive_dir)

    assert loaded["epoch"] == 3
    assert torch.equal(
        loaded["model_state_dict"]["x"], payload["model_state_dict"]["x"]
    )


def test_try_save_writes_payload_and_reports_success(tmp_path):
    """The happy path must still produce a real, loadable file -- try_save is a
    wrapper around save(), not a separate write path."""
    path = tmp_path / "ckpt.pt"
    payload = {"epoch": 1, "model_state_dict": {"w": torch.tensor([1.0, 2.0])}}

    assert try_save(path, payload, backoff_s=0.0) is True
    assert path.exists()
    assert torch.equal(
        torch.load(path, weights_only=False)["model_state_dict"]["w"],
        payload["model_state_dict"]["w"],
    )


def test_try_save_recovers_when_a_write_fails_once(tmp_path, monkeypatch):
    """The 74 production crashes were a transient-looking NFS write failure at
    torch.save. One failure must not lose the checkpoint."""
    path = tmp_path / "ckpt.pt"
    real_save = torch.save
    calls = []

    def flaky_save(obj, f, *a, **kw):
        calls.append(f)
        if len(calls) == 1:
            raise RuntimeError("basic_ios::clear: iostream error")
        return real_save(obj, f, *a, **kw)

    monkeypatch.setattr(torch, "save", flaky_save)

    assert try_save(path, {"epoch": 1}, backoff_s=0.0) is True
    assert len(calls) == 2
    assert path.exists()


def test_try_save_returns_false_instead_of_raising_when_all_attempts_fail(
    tmp_path, monkeypatch
):
    """Degrade, don't die: 68 of 74 production failures were one correlated
    cluster-wide event, so exhausting retries must let the worker skip the round
    and keep training rather than propagate and kill the node."""
    path = tmp_path / "ckpt.pt"
    calls = []

    def always_fails(obj, f, *a, **kw):
        calls.append(f)
        raise RuntimeError("basic_ios::clear: iostream error")

    monkeypatch.setattr(torch, "save", always_fails)

    assert try_save(path, {"epoch": 1}, attempts=3, backoff_s=0.0) is False
    assert len(calls) == 3
    assert not path.exists()


def test_load_warns_on_state_dict_key_mismatch(tmp_path, caplog):
    """strict=False on the model load must not be silent: a key mismatch
    (e.g. an architecture rewrite renaming parameters) should log a warning
    identifying exactly which keys were missing/unexpected, mirroring the
    equivalent guard in training/diloco/mesh_round.py's master-checkpoint
    load -- without it, a resumed run silently reinitializes the mismatched
    submodules with no signal that anything went wrong."""
    donor = VideoMAEEncoder(OmegaConf.merge(_tiny_cfg(in_channels=3), {"depth": 3}))
    target = VideoMAEEncoder(
        _tiny_cfg(in_channels=3)
    )  # depth=2, per _tiny_cfg's default

    path = tmp_path / "ckpt.pt"
    save(
        path,
        {
            "model_name": "videomae_encoder",
            "model_state_dict": donor.state_dict(),
            "epoch": 0,
        },
    )

    with caplog.at_level("WARNING"):
        load(
            path,
            model=target,
            optimizer=None,
            scheduler=None,
            device="cpu",
            expected_model_name="videomae_encoder",
            expected_optimizer_name=None,
        )

    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "blocks.2." in message  # depth-3 donor's extra block is unexpected


def test_load_does_not_warn_when_state_dicts_match(tmp_path, caplog):
    donor = VideoMAEEncoder(_tiny_cfg(in_channels=3))
    target = VideoMAEEncoder(_tiny_cfg(in_channels=3))

    path = tmp_path / "ckpt.pt"
    save(
        path,
        {
            "model_name": "videomae_encoder",
            "model_state_dict": donor.state_dict(),
            "epoch": 0,
        },
    )

    with caplog.at_level("WARNING"):
        load(
            path,
            model=target,
            optimizer=None,
            scheduler=None,
            device="cpu",
            expected_model_name="videomae_encoder",
            expected_optimizer_name=None,
        )

    assert len(caplog.records) == 0


def test_load_with_completely_disjoint_state_dict_warns_and_leaves_model_untouched(
    tmp_path, caplog
):
    """Pins current behavior for the extreme end of the key-mismatch guard:
    a checkpoint sharing zero keys with the target (e.g. `expected_model_name`
    collided across two unrelated architectures). `load_state_dict(strict=False)`
    treats every key as missing/unexpected -- it warns but does not raise, and
    the target model's own random-init params are left exactly as constructed.
    If this ever changes to raise instead, update this test deliberately."""
    target = VideoMAEEncoder(_tiny_cfg(in_channels=3))
    before = {k: v.clone() for k, v in target.state_dict().items()}

    path = tmp_path / "ckpt.pt"
    save(
        path,
        {
            "model_name": "videomae_encoder",
            "model_state_dict": {"totally_unrelated_param": torch.zeros(3)},
            "epoch": 0,
        },
    )

    with caplog.at_level("WARNING"):
        load(
            path,
            model=target,
            optimizer=None,
            scheduler=None,
            device="cpu",
            expected_model_name="videomae_encoder",
            expected_optimizer_name=None,
        )

    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "totally_unrelated_param" in message
    after = target.state_dict()
    for key, value in before.items():
        assert torch.equal(after[key], value), key


class TestRemapLegacyAttentionKeys:
    """The ViT refactor renamed attention and MLP parameters. Checkpoints
    trained before it (every run through pool-cars-best-120r) share only 77 of
    205 keys with the current model, so probing them needs a rename shim.

    The key names below are copied verbatim from
    ~/stern_mesh/checkpoints/pool-cars-best-120r/master.pt -- ground truth from
    a real checkpoint, not from inverting the mapping under test, which would
    make these assertions tautological.
    """

    def test_renames_the_fused_qkv_projection(self):
        out = remap_legacy_attention_keys(
            {
                "encoder.blocks.0.attn.in_proj_weight": torch.zeros(3, 1),
                "encoder.blocks.0.attn.in_proj_bias": torch.zeros(3),
            }
        )
        assert set(out) == {
            "encoder.blocks.0.attn.qkv.weight",
            "encoder.blocks.0.attn.qkv.bias",
        }

    def test_weight_and_bias_do_not_get_crossed(self):
        """A rename table that mapped in_proj_* onto the wrong target would
        still produce the right key set, so bind each tensor to its value."""
        out = remap_legacy_attention_keys(
            {
                "encoder.blocks.0.attn.in_proj_weight": torch.full((3, 1), 7.0),
                "encoder.blocks.0.attn.in_proj_bias": torch.full((3,), 9.0),
            }
        )
        assert out["encoder.blocks.0.attn.qkv.weight"].flatten()[0] == 7.0
        assert out["encoder.blocks.0.attn.qkv.bias"].flatten()[0] == 9.0

    def test_renames_the_output_projection(self):
        out = remap_legacy_attention_keys(
            {"decoder_blocks.0.attn.out_proj.weight": torch.zeros(1)}
        )
        assert list(out) == ["decoder_blocks.0.attn.proj.weight"]

    def test_renames_the_sequential_mlp_into_its_net_submodule(self):
        out = remap_legacy_attention_keys(
            {
                "decoder_blocks.0.mlp.0.weight": torch.zeros(1),
                "decoder_blocks.0.mlp.3.bias": torch.zeros(1),
            }
        )
        assert set(out) == {
            "decoder_blocks.0.mlp.net.0.weight",
            "decoder_blocks.0.mlp.net.3.bias",
        }

    def test_leaves_unrelated_keys_untouched(self):
        keys = {
            "encoder.pos_embed": torch.zeros(1),
            "encoder.patch_embed.proj.weight": torch.zeros(1),
            "encoder.blocks.0.norm1.weight": torch.zeros(1),
            "mask_token": torch.zeros(1),
        }
        assert set(remap_legacy_attention_keys(keys)) == set(keys)

    def test_is_idempotent_on_already_current_keys(self):
        """Re-running the shim on a current checkpoint must be a no-op, so the
        loader can apply it unconditionally without sniffing the format."""
        current = {
            "encoder.blocks.0.attn.qkv.weight": torch.zeros(1),
            "encoder.blocks.0.attn.proj.bias": torch.zeros(1),
            "encoder.blocks.0.mlp.net.0.weight": torch.zeros(1),
        }
        once = remap_legacy_attention_keys(current)
        assert set(once) == set(current)
        assert set(remap_legacy_attention_keys(once)) == set(current)

    def test_a_remapped_legacy_checkpoint_loads_into_the_current_model(self):
        """End to end: the only key the shim cannot supply is the registers
        tensor, which the refactor added. At num_registers=0 it is an empty
        (1, 0, D) parameter that prepend/strip treat as a no-op, so leaving it
        at init changes no computation."""
        cfg = OmegaConf.load("video_encoding/conf/model/videomae_base.yaml")
        model = VideoMAEForPreTraining(cfg)

        inverse = {
            ".attn.qkv.": ".attn.in_proj_",
            ".attn.proj.": ".attn.out_proj.",
            ".mlp.net.": ".mlp.",
        }
        legacy = {}
        for key, value in model.state_dict().items():
            if key == "encoder.registers.tokens":
                continue
            for new, old in inverse.items():
                key = key.replace(new, old)
            legacy[key] = value

        result = model.load_state_dict(
            remap_legacy_attention_keys(legacy), strict=False
        )
        assert result.unexpected_keys == []
        assert result.missing_keys == ["encoder.registers.tokens"]
        assert model.encoder.registers.tokens.numel() == 0


class TestLoadCheckpointFileRemapsLegacyKeys:
    """load_checkpoint_file is the single door every consumer reads a
    checkpoint through (train.py, dump_probe_predictions.py,
    probe_motion_decodability.py, diag_reconstruction.py,
    export_vit_internals.py), so the pre-ViT-refactor rename belongs here
    rather than repeated at each call site.

    Two payload shapes are in circulation and both must be covered: a dict with
    a "model_state_dict" entry, and a bare state dict -- train.py:86 and
    dump_probe_predictions.py:62 both fall back to the payload itself via
    .get("model_state_dict", ckpt).
    """

    def test_remaps_keys_nested_under_model_state_dict(self, tmp_path):
        payload = {
            "epoch": 3,
            "model_state_dict": {
                "encoder.blocks.0.attn.in_proj_weight": torch.tensor([1.0]),
                "encoder.blocks.0.mlp.0.weight": torch.tensor([2.0]),
            },
        }
        path = tmp_path / "legacy.pt"
        torch.save(payload, path)

        loaded = load_checkpoint_file(path)

        assert set(loaded["model_state_dict"]) == {
            "encoder.blocks.0.attn.qkv.weight",
            "encoder.blocks.0.mlp.net.0.weight",
        }
        assert loaded["epoch"] == 3

    def test_remaps_a_bare_state_dict_payload(self, tmp_path):
        payload = {"encoder.blocks.0.attn.out_proj.bias": torch.tensor([5.0])}
        path = tmp_path / "bare.pt"
        torch.save(payload, path)

        loaded = load_checkpoint_file(path)

        assert list(loaded) == ["encoder.blocks.0.attn.proj.bias"]
        assert torch.equal(
            loaded["encoder.blocks.0.attn.proj.bias"], torch.tensor([5.0])
        )

    def test_leaves_sibling_payload_entries_alone(self, tmp_path):
        """Only the model weights are renamed; optimizer state and metadata
        must survive byte-for-byte, since the outer-step code reads them."""
        payload = {
            "model_state_dict": {
                "encoder.blocks.0.attn.in_proj_bias": torch.tensor([1.0])
            },
            "optimizer_state_dict": {"state": {0: {"exp_avg": torch.tensor([4.0])}}},
            "round": 120,
        }
        path = tmp_path / "full.pt"
        torch.save(payload, path)

        loaded = load_checkpoint_file(path)

        assert loaded["round"] == 120
        assert torch.equal(
            loaded["optimizer_state_dict"]["state"][0]["exp_avg"], torch.tensor([4.0])
        )


def test_load_encoder_state_works_on_a_target_without_patch_embed():
    """Some encoders (LeVJEPAEncoder) have no top-level patch_embed attribute
    at all (they use tubelet_embed instead) -- load_encoder_state must not
    unconditionally reach for encoder.patch_embed.proj.weight before checking
    whether the state dict even has a patch_embed-shaped key to inflate."""
    import torch.nn as nn

    class _NoPatchEmbedEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.tubelet_embed = nn.Module()
            self.tubelet_embed.proj = nn.Conv3d(3, 4, kernel_size=1)

    encoder = _NoPatchEmbedEncoder()
    donor_state = {
        "tubelet_embed.proj.weight": torch.zeros(4, 3, 1, 1, 1),
        "tubelet_embed.proj.bias": torch.zeros(4),
    }

    missing, unexpected = load_encoder_state(encoder, donor_state)

    assert missing == []
    assert unexpected == []
