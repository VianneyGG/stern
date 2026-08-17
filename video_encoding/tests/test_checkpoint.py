"""Unit tests for the checkpoint-loading robustness helpers."""

from __future__ import annotations

import zipfile

import torch
from omegaconf import OmegaConf

from video_encoding.models.videomaev2.encoder import VideoMAEEncoder
from video_encoding.training.checkpoint import (
    load,
    load_checkpoint_file,
    load_encoder_state,
    save,
    try_save,
)


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
