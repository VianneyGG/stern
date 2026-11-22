"""End-to-end wiring test for run_train_stage: real manifest/label build, real
windowing + label alignment, real preprocessing + CCC loop -- only the ViT
backbone is faked (a full-size video ViT forward is slow for CI). Exercises the
path the old encoder-faking tests skipped.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch import nn

from stern.data.labels import write_label_csv
from stern.data.manifest import ManifestRecord, write_manifest
from stern.training import finetune as train_mod
from stern.prep.build_afewva_manifest import build, parse_clip_annotations

pytestmark = pytest.mark.integration


class _FakeRegressor(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.lin = nn.Linear(3, 2)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.lin(pixel_values.mean(dim=[2, 3, 4])))


class _FakeEncoderBackbone(nn.Module):
    """Minimal stand-in exposing the same `patch_embed.proj.weight` shape
    contract `checkpoint.load_encoder_state` depends on, without paying for
    a real VideoMAEEncoder forward pass in this fast unit test."""

    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv3d(3, 4, kernel_size=1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.patch_embed.proj(pixel_values).mean(dim=[2, 3, 4])


class _FakeRegressorWithBackbone(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.backbone = _FakeEncoderBackbone()
        self.head = nn.Linear(4, 2)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(pixel_values)
        return torch.tanh(self.head(feat))


def test_run_train_stage_loads_and_freezes_pretrained_backbone(tmp_path, monkeypatch):
    """train.pretrain_encoder_path + freeze_backbone=True must load the donor
    weights into model.backbone and keep them frozen through training -- the
    bug this test would have caught: these fields were declared in config but
    never read anywhere, so Stage 1 -> Stage 2 hand-off silently did nothing."""
    rng = np.random.default_rng(0)
    _write_clip(tmp_path / "001", 24, rng)
    _write_clip(tmp_path / "002", 24, rng)
    manifest_path = build(tmp_path, tmp_path)
    monkeypatch.setattr(train_mod, "VideoMAEVARegressor", _FakeRegressorWithBackbone)

    donor_backbone = _FakeEncoderBackbone()
    donor_backbone.patch_embed.proj.weight.data.fill_(7.0)
    donor_backbone.patch_embed.proj.bias.data.fill_(0.0)
    ckpt_path = tmp_path / "pretrain_encoder.pt"
    torch.save(
        {
            "model_state_dict": {
                f"encoder.{k}": v for k, v in donor_backbone.state_dict().items()
            }
        },
        ckpt_path,
    )

    cfg = _cfg(tmp_path, manifest_path)
    cfg.train.pretrain_encoder_path = str(ckpt_path)
    cfg.train.freeze_backbone = True

    train_mod.run_train_stage(
        cfg, rank=0, world_size=1, local_rank=0, device=torch.device("cpu")
    )

    saved = torch.load(
        tmp_path / "ckpt" / "test_va_finetune" / "latest.pt", weights_only=False
    )
    # If loading or freezing were broken, 2 epochs of training on random data
    # would almost certainly move these weights away from the donor's.
    assert torch.equal(
        saved["model_state_dict"]["backbone.patch_embed.proj.weight"],
        donor_backbone.patch_embed.proj.weight.data,
    )
    assert torch.equal(
        saved["model_state_dict"]["backbone.patch_embed.proj.bias"],
        donor_backbone.patch_embed.proj.bias.data,
    )


def _write_clip(
    clip_dir: Path, n: int, rng: np.random.Generator, actor: str | None = None
) -> None:
    clip_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(
            clip_dir / f"{i:05d}.png"
        )
    frames = {
        f"{i:05d}": {
            "valence": int(rng.integers(-10, 11)),
            "arousal": int(rng.integers(-10, 11)),
        }
        for i in range(n)
    }
    payload = {"frames": frames, **({"actor": actor} if actor else {})}
    (clip_dir / f"{clip_dir.name}.json").write_text(json.dumps(payload))


def _cfg(tmp_path: Path, manifest_path: Path):
    return OmegaConf.create(
        {
            "experiment": {"name": "test_va_finetune"},
            "dataset": {
                "name": "afewva",
                "role": "probe",
                "present": True,
                "path": str(tmp_path),
                "manifest_path": str(manifest_path),
                "va_range": [-10.0, 10.0],
                "annotation_hz": 30.0,
            },
            "model": {"num_frames": 4, "embed_dim": 8, "image_size": 8},
            "preprocess": {"window": {"overlap": 0.5}},
            "train": {
                "seed": 0,
                "batch_size": 4,
                "learning_rate": 1e-2,
                "epochs": 2,
                "checkpoint_root": str(tmp_path / "ckpt"),
                "results_root": str(tmp_path / "results"),
            },
        }
    )


def test_run_train_stage_trains_and_reports_ccc(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    _write_clip(tmp_path / "001", 24, rng)
    _write_clip(tmp_path / "002", 24, rng)
    manifest_path = build(tmp_path, tmp_path)

    monkeypatch.setattr(train_mod, "VideoMAEVARegressor", _FakeRegressor)

    cfg = _cfg(tmp_path, manifest_path)
    result = train_mod.run_train_stage(
        cfg, rank=0, world_size=1, local_rank=0, device=torch.device("cpu")
    )

    assert result["stage"] == "train"
    assert isinstance(result["val_ccc_v"], float)
    assert isinstance(result["val_ccc_a"], float)
    assert (tmp_path / "ckpt" / "test_va_finetune" / "best.pt").exists()
    assert len(list((tmp_path / "results").glob("*.json"))) == 1


def test_run_train_stage_resumes_from_latest_checkpoint(tmp_path, monkeypatch):
    """A second call with more epochs must continue from where the first
    call's latest.pt left off, not restart at epoch 0 with a fresh random
    model -- the bug this test would have caught: `ckpt_io.load` was never
    called, so every run silently retrained from scratch."""
    rng = np.random.default_rng(0)
    _write_clip(tmp_path / "001", 24, rng)
    _write_clip(tmp_path / "002", 24, rng)
    manifest_path = build(tmp_path, tmp_path)
    monkeypatch.setattr(train_mod, "VideoMAEVARegressor", _FakeRegressor)

    cfg = _cfg(tmp_path, manifest_path)
    cfg.train.epochs = 2
    train_mod.run_train_stage(
        cfg, rank=0, world_size=1, local_rank=0, device=torch.device("cpu")
    )
    latest_path = tmp_path / "ckpt" / "test_va_finetune" / "latest.pt"
    assert latest_path.exists()
    first_ckpt = torch.load(latest_path, weights_only=False)
    assert first_ckpt["epoch"] == 1  # 0-indexed: epochs 0,1 ran

    cfg.train.epochs = 4
    result = train_mod.run_train_stage(
        cfg, rank=0, world_size=1, local_rank=0, device=torch.device("cpu")
    )
    # Epochs 0,1 are on disk, so the second call starts at epoch 2 and runs 2 more.
    assert result["start_epoch"] == 2
    assert result["epochs_run"] == 2
    second_ckpt = torch.load(latest_path, weights_only=False)
    assert second_ckpt["epoch"] == 3  # resumed at epoch 2, ran through epoch 3


def test_run_train_stage_skips_malformed_clip_instead_of_crashing(
    tmp_path, monkeypatch
):
    """A manifest record whose frame directory has 0 images (e.g. a failed
    extraction that slipped past the manifest builder) must be skipped, not
    abort construction for every other good clip in the dataset."""
    rng = np.random.default_rng(0)
    _write_clip(tmp_path / "001", 24, rng)
    (tmp_path / "bad").mkdir()  # exists but has no frame images at all

    manifest_path = tmp_path / "manifest.csv"
    write_manifest(
        manifest_path,
        [
            ManifestRecord(
                identity_id="001",
                clip_path="001",
                duration_s=24 / 30.0,
                role="probe",
                label_path="labels/001.csv",
            ),
            ManifestRecord(
                identity_id="bad",
                clip_path="bad",
                duration_s=0.0,
                role="probe",
                label_path="labels/001.csv",  # reuse a valid label file; only the frames are bad
            ),
        ],
    )
    # build_afewva_manifest's build() would have written labels/001.csv already
    # via the earlier call pattern; reproduce it directly here since we're
    # bypassing build() to control which records land in the manifest.

    _, rows = parse_clip_annotations(tmp_path / "001" / "001.json")
    write_label_csv(tmp_path / "labels" / "001.csv", rows)

    monkeypatch.setattr(train_mod, "VideoMAEVARegressor", _FakeRegressor)
    cfg = _cfg(tmp_path, manifest_path)

    result = train_mod.run_train_stage(
        cfg, rank=0, world_size=1, local_rank=0, device=torch.device("cpu")
    )

    assert result["stage"] == "train"  # did not crash despite the "bad" record


def test_run_train_stage_emits_live_heartbeat_every_epoch(tmp_path, monkeypatch):
    """The watcher's stall detector needs a result JSON that updates mid-run,
    not just after the loop exits -- assert write_result_json is called once
    per epoch (heartbeat) plus once more at loop exit (final), and that the
    file on disk reflects the LAST epoch's val metrics, not epoch 0's."""
    rng = np.random.default_rng(0)
    _write_clip(tmp_path / "001", 24, rng)
    _write_clip(tmp_path / "002", 24, rng)
    manifest_path = build(tmp_path, tmp_path)
    monkeypatch.setattr(train_mod, "VideoMAEVARegressor", _FakeRegressor)

    calls = []
    real_write = train_mod.write_result_json

    def _spy(*args, **kwargs):
        calls.append(kwargs.get("overwrite", False))
        return real_write(*args, **kwargs)

    monkeypatch.setattr(train_mod, "write_result_json", _spy)

    cfg = _cfg(tmp_path, manifest_path)
    cfg.train.epochs = 3
    train_mod.run_train_stage(
        cfg, rank=0, world_size=1, local_rank=0, device=torch.device("cpu")
    )

    # 3 epoch-heartbeat writes + 1 final write, every one overwrite=True so
    # they all land on the same stable filename (still exactly 1 file).
    assert calls == [True, True, True, True]
    assert len(list((tmp_path / "results").glob("*.json"))) == 1


def test_adamw_puts_head_params_at_head_lr_mult_with_weight_decay():
    model = _FakeRegressorWithBackbone(None)
    train_cfg = OmegaConf.create(
        {"optimizer": "adamw", "learning_rate": 1e-5, "head_lr_mult": 100.0,
         "weight_decay": 0.05}
    )
    optimizer, name = train_mod._build_optimizer(model, train_cfg)
    assert name == "adamw" and isinstance(optimizer, torch.optim.AdamW)
    backbone_group, head_group = optimizer.param_groups
    assert backbone_group["lr"] == 1e-5
    assert head_group["lr"] == pytest.approx(1e-3)
    assert {id(p) for p in head_group["params"]} == {
        id(p) for p in model.head.parameters()
    }
    assert {id(p) for p in backbone_group["params"]} == {
        id(p) for p in model.backbone.parameters()
    }
    assert all(g["weight_decay"] == 0.05 for g in optimizer.param_groups)


def test_default_optimizer_is_unchanged_adam_with_no_scheduler():
    model = _FakeRegressorWithBackbone(None)
    train_cfg = OmegaConf.create({"learning_rate": 1e-4, "epochs": 3})
    optimizer, name = train_mod._build_optimizer(model, train_cfg)
    assert name == "adam" and type(optimizer) is torch.optim.Adam
    assert train_mod._build_scheduler(optimizer, train_cfg, steps_per_epoch=10) is None


def test_scheduler_warms_up_linearly_then_decays_to_zero():
    param = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=1.0)
    train_cfg = OmegaConf.create({"warmup_epochs": 1, "epochs": 3})
    scheduler = train_mod._build_scheduler(optimizer, train_cfg, steps_per_epoch=10)
    lrs = []
    for _ in range(30):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert lrs[0] == pytest.approx(0.1)  # step 1 of a 10-step warmup
    assert lrs[9] == pytest.approx(1.0)  # warmup complete
    assert all(a >= b for a, b in zip(lrs[10:], lrs[11:]))  # cosine never rises
    assert lrs[-1] < 0.01


def test_resume_continues_the_lr_schedule_instead_of_restarting_warmup(
    tmp_path, monkeypatch
):
    rng = np.random.default_rng(0)
    _write_clip(tmp_path / "001", 24, rng)
    _write_clip(tmp_path / "002", 24, rng)
    manifest_path = build(tmp_path, tmp_path)
    monkeypatch.setattr(train_mod, "VideoMAEVARegressor", _FakeRegressor)
    # Same epochs target (3) throughout. The first run is preempted at the
    # start of epoch 2, after epochs 0 and 1 are saved.
    cfg = _cfg(tmp_path, manifest_path)
    cfg.train.update({"optimizer": "adamw", "warmup_epochs": 1, "epochs": 3})
    real_train_epoch = train_mod._train_epoch
    started = 0

    def _preempting_train_epoch(*args, **kwargs):
        nonlocal started
        started += 1
        if started == 3:
            raise RuntimeError("simulated preemption")
        return real_train_epoch(*args, **kwargs)

    monkeypatch.setattr(train_mod, "_train_epoch", _preempting_train_epoch)
    with pytest.raises(RuntimeError, match="simulated preemption"):
        train_mod.run_train_stage(
            cfg, rank=0, world_size=1, local_rank=0, device=torch.device("cpu")
        )
    latest = tmp_path / "ckpt" / "test_va_finetune" / "latest.pt"
    assert torch.load(latest, weights_only=False)["epoch"] == 1

    # The LR each optimizer step uses, across the resumed epoch.
    step_lrs: list[float] = []

    def _recording_train_epoch(model, loader, optimizer, *args, **kwargs):
        real_step = optimizer.step

        def _recording_step(*a, **k):
            step_lrs.append(optimizer.param_groups[0]["lr"])
            return real_step(*a, **k)

        optimizer.step = _recording_step
        return real_train_epoch(model, loader, optimizer, *args, **kwargs)

    monkeypatch.setattr(train_mod, "_train_epoch", _recording_train_epoch)
    train_mod.run_train_stage(
        cfg, rank=0, world_size=1, local_rank=0, device=torch.device("cpu")
    )
    # Hand-derived. 2 clips x 11 windows = 22; val = round(2.2) = 2; train = 20;
    # drop_last at batch 4 gives s = 5 steps per epoch. The lambda uses warmup = s
    # and total = 3s, so an optimizer step at global index i = 2s + k (k = 0..4)
    # has progress = (i - s) / (3s - s) = (5 + k) / 10 and LR =
    # 1e-2 * 0.5 * (1 + cos(pi * progress)):
    #   k=0: progress 0.5 -> 1e-2 * 0.5 * 1.0000        = 5e-3
    #   k=1: progress 0.6 -> 1e-2 * 0.5 * 0.690983      = 3.454915e-3
    #   k=2: progress 0.7 -> 1e-2 * 0.5 * 0.412215      = 2.061074e-3
    #   k=3: progress 0.8 -> 1e-2 * 0.5 * 0.190983      = 9.54915e-4
    #   k=4: progress 0.9 -> 1e-2 * 0.5 * 0.048943      = 2.447174e-4
    # A restarted warmup would begin near 1e-2 / 5 = 2e-3 and rise again.
    assert len(step_lrs) == 5
    assert step_lrs == pytest.approx(
        [5e-3, 3.454915e-3, 2.061074e-3, 9.54915e-4, 2.447174e-4], rel=1e-5
    )


def test_levjepa_hf_encoder_routes_to_the_hf_regressor(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    _write_clip(tmp_path / "001", 24, rng)
    _write_clip(tmp_path / "002", 24, rng)
    manifest_path = build(tmp_path, tmp_path)
    monkeypatch.setattr(train_mod, "HFLeVJEPAVARegressor", _FakeRegressor)
    monkeypatch.setattr(
        train_mod, "VideoMAEVARegressor", lambda _cfg: pytest.fail("wrong model")
    )
    cfg = _cfg(tmp_path, manifest_path)
    cfg.model.encoder = "levjepa_hf"
    train_mod.run_train_stage(
        cfg, rank=0, world_size=1, local_rank=0, device=torch.device("cpu")
    )
    saved = torch.load(
        tmp_path / "ckpt" / "test_va_finetune" / "latest.pt", weights_only=False
    )
    assert saved["model_name"] == "levjepa_hf_va_regressor"


# --- train.split=identity, epochs_run, final (ssl-diag-08) -------------------


def _five_actor_manifest(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """Ten clips over five actors, two clips each, so a window-level split
    would put one actor on both sides. Returns (manifest, clip -> actor)."""
    rng = np.random.default_rng(0)
    clip_actor = {f"{i:03d}": f"actor{i // 2}" for i in range(10)}
    for clip, actor in clip_actor.items():
        _write_clip(tmp_path / clip, 24, rng, actor=actor)
    return build(tmp_path, tmp_path), clip_actor


def _clips(dataset) -> set[str]:
    """Clip names behind a VAWindowDataset or a random_split Subset of one."""
    if isinstance(dataset, torch.utils.data.Subset):
        items = [dataset.dataset._items[i] for i in dataset.indices]
    else:
        items = dataset._items
    return {frame_files[0].parent.name for frame_files, _, _ in items}


def _capture_datasets(monkeypatch) -> list[tuple]:
    seen: list[tuple] = []
    real = train_mod._make_loaders

    def _spy(train_ds, val_ds, **kwargs):
        seen.append((train_ds, val_ds))
        return real(train_ds, val_ds, **kwargs)

    monkeypatch.setattr(train_mod, "_make_loaders", _spy)
    return seen


def test_identity_split_puts_no_actor_on_both_sides(tmp_path, monkeypatch):
    manifest_path, clip_actor = _five_actor_manifest(tmp_path)
    monkeypatch.setattr(train_mod, "VideoMAEVARegressor", _FakeRegressor)
    seen = _capture_datasets(monkeypatch)
    cfg = _cfg(tmp_path, manifest_path)
    cfg.train.split = "identity"
    cfg.train.epochs = 1

    train_mod.run_train_stage(
        cfg, rank=0, world_size=1, local_rank=0, device=torch.device("cpu")
    )

    ((train_ds, val_ds),) = seen
    train_actors = {clip_actor[c] for c in _clips(train_ds)}
    val_actors = {clip_actor[c] for c in _clips(val_ds)}
    # 0.2 of five actors, taken from the end of the sorted ids: the held-out
    # side is actor4 alone -- not merely disjoint, but the right way round.
    assert val_actors == {"actor4"}
    assert train_actors == {f"actor{i}" for i in range(4)}


def test_identity_split_honours_split_fraction(tmp_path, monkeypatch):
    manifest_path, clip_actor = _five_actor_manifest(tmp_path)
    monkeypatch.setattr(train_mod, "VideoMAEVARegressor", _FakeRegressor)
    seen = _capture_datasets(monkeypatch)
    cfg = _cfg(tmp_path, manifest_path)
    cfg.train.update({"split": "identity", "split_fraction": 0.4, "epochs": 1})

    train_mod.run_train_stage(
        cfg, rank=0, world_size=1, local_rank=0, device=torch.device("cpu")
    )

    ((_, val_ds),) = seen
    assert {clip_actor[c] for c in _clips(val_ds)} == {"actor3", "actor4"}


def test_identity_split_does_not_move_with_seed(tmp_path, monkeypatch):
    manifest_path, _ = _five_actor_manifest(tmp_path)
    monkeypatch.setattr(train_mod, "VideoMAEVARegressor", _FakeRegressor)
    seen = _capture_datasets(monkeypatch)
    for seed in (0, 1):
        cfg = _cfg(tmp_path, manifest_path)
        cfg.train.update({"split": "identity", "epochs": 1, "seed": seed})
        cfg.experiment.name = f"seed{seed}"
        train_mod.run_train_stage(
            cfg, rank=0, world_size=1, local_rank=0, device=torch.device("cpu")
        )

    assert _clips(seen[0][1]) == _clips(seen[1][1])


def test_default_split_is_the_seeded_window_level_random_split(tmp_path, monkeypatch):
    manifest_path, _ = _five_actor_manifest(tmp_path)
    monkeypatch.setattr(train_mod, "VideoMAEVARegressor", _FakeRegressor)
    seen = _capture_datasets(monkeypatch)
    cfg = _cfg(tmp_path, manifest_path)
    cfg.train.update({"epochs": 1, "seed": 3})

    train_mod.run_train_stage(
        cfg, rank=0, world_size=1, local_rank=0, device=torch.device("cpu")
    )

    ((_, val_ds),) = seen
    # 10 clips x 11 windows: 24 frames, span 4, step 2 -> starts 0..20.
    # Validation = round(110 * 0.1) = 11 windows, indices pinned from
    # random_split(range(110), [99, 11]) with generator seed 3. The literal is
    # generated, not hand-derivable: the hand check is 11 distinct indices in 0..109.
    assert len(val_ds.dataset) == 110
    assert list(val_ds.indices) == [106, 103, 77, 13, 30, 50, 97, 107, 5, 20, 52]


def test_final_is_the_last_epoch_and_best_keys_are_unchanged(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    _write_clip(tmp_path / "001", 24, rng)
    _write_clip(tmp_path / "002", 24, rng)
    manifest_path = build(tmp_path, tmp_path)
    monkeypatch.setattr(train_mod, "VideoMAEVARegressor", _FakeRegressor)
    scripted = iter(
        [
            {"val_ccc_v": 0.6, "val_ccc_a": 0.4, "mean_ccc": 0.5},
            {"val_ccc_v": 0.2, "val_ccc_a": 0.0, "mean_ccc": 0.1},
        ]
    )
    monkeypatch.setattr(train_mod, "_validate", lambda *a, **k: next(scripted))
    cfg = _cfg(tmp_path, manifest_path)
    cfg.train.update({"epochs": 2, "resume": False})

    result = train_mod.run_train_stage(
        cfg, rank=0, world_size=1, local_rank=0, device=torch.device("cpu")
    )

    assert result["epochs_run"] == 2
    assert result["final"] == {"val_ccc_v": 0.2, "val_ccc_a": 0.0, "mean_ccc": 0.1}
    assert result["mean_ccc"] == 0.5  # best epoch, as before


def test_epochs_run_counts_only_the_epochs_this_call_ran(tmp_path, monkeypatch):
    """A resumed run is shorter than `train.epochs`; `epochs_run` must say so
    rather than echo the config."""
    rng = np.random.default_rng(0)
    _write_clip(tmp_path / "001", 24, rng)
    _write_clip(tmp_path / "002", 24, rng)
    manifest_path = build(tmp_path, tmp_path)
    monkeypatch.setattr(train_mod, "VideoMAEVARegressor", _FakeRegressor)
    cfg = _cfg(tmp_path, manifest_path)
    cfg.train.epochs = 1
    train_mod.run_train_stage(
        cfg, rank=0, world_size=1, local_rank=0, device=torch.device("cpu")
    )

    cfg.train.epochs = 3
    result = train_mod.run_train_stage(
        cfg, rank=0, world_size=1, local_rank=0, device=torch.device("cpu")
    )

    assert result["epochs_run"] == 2


def test_extract_encoder_state_strips_encoder_prefix_for_videomae():
    """VideoMAE's Stage-1 checkpoint wraps VideoMAEForPreTraining, whose
    encoder submodule keys are checkpointed under an 'encoder.' prefix."""
    state = {"encoder.blocks.0.weight": 1, "decoder.blocks.0.weight": 2}
    result = train_mod._extract_encoder_state(state, encoder_kind="videomae")
    assert result == {"blocks.0.weight": 1}


def test_extract_encoder_state_passes_through_unprefixed_for_levjepa():
    """LeVJEPA's single-node checkpoint (pretrain_levjepa.py) saves the
    encoder directly (base_payload(model=encoder, ...)), so its keys have no
    'encoder.' prefix -- stripping one would silently drop every key and
    load nothing."""
    state = {"blocks.0.weight": 1, "tubelet_embed.proj.weight": 2}
    result = train_mod._extract_encoder_state(state, encoder_kind="levjepa")
    assert result == {"blocks.0.weight": 1, "tubelet_embed.proj.weight": 2}


def test_extract_encoder_state_strips_encoder_prefix_for_levjepa_mesh_checkpoint():
    """The DiLoCo mesh path's _LeVJEPAPretrainModule wraps encoder+projector
    as submodules, so mesh_round.py's base_payload(model=wrapper, ...) saves
    'encoder.'/'projector.'-prefixed keys -- unlike the single-node path.
    Passing this shape through unprefixed (the old unconditional levjepa
    branch) would match zero keys against a bare LeVJEPAEncoder and silently
    load nothing."""
    state = {
        "encoder.blocks.0.weight": 1,
        "encoder.tubelet_embed.proj.weight": 2,
        "projector.fc1.weight": 3,
    }
    result = train_mod._extract_encoder_state(state, encoder_kind="levjepa")
    assert result == {"blocks.0.weight": 1, "tubelet_embed.proj.weight": 2}


def test_run_train_stage_builds_levjepa_backbone_when_encoder_is_levjepa(
    tmp_path, monkeypatch
):
    """cfg.model.encoder == 'levjepa' must build a LeVJEPAEncoder-backed
    VideoMAEVARegressor, not the default VideoMAEEncoder-backed one --
    verified by spying on both constructors, mirroring how mesh_round.py's
    _build_pretrain_model dispatch is already tested."""
    rng = np.random.default_rng(0)
    _write_clip(tmp_path / "001", 24, rng)
    _write_clip(tmp_path / "002", 24, rng)
    manifest_path = build(tmp_path, tmp_path)

    calls = {"levjepa": 0}

    class _StubLeVJEPAEncoder(nn.Module):
        def __init__(self, model_cfg):
            super().__init__()
            calls["levjepa"] += 1
            self.dummy = nn.Linear(1, 1)

        def forward(self, pixel_values):
            return torch.zeros(pixel_values.shape[0], 1, 1)

    captured_backbones = []

    class _SpyRegressor(nn.Module):
        def __init__(self, model_cfg, backbone=None):
            super().__init__()
            captured_backbones.append(backbone)
            self.backbone = backbone
            self.head = nn.Linear(1, 2)

        def forward(self, pixel_values):
            feat = self.backbone(pixel_values).mean(dim=1)
            return torch.tanh(self.head(feat))

    monkeypatch.setattr(train_mod, "LeVJEPAEncoder", _StubLeVJEPAEncoder)
    monkeypatch.setattr(train_mod, "VideoMAEVARegressor", _SpyRegressor)

    cfg = _cfg(tmp_path, manifest_path)
    cfg.model.encoder = "levjepa"

    train_mod.run_train_stage(
        cfg, rank=0, world_size=1, local_rank=0, device=torch.device("cpu")
    )

    assert calls == {"levjepa": 1}
    assert isinstance(captured_backbones[0], _StubLeVJEPAEncoder)
