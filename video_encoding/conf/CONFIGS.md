# Config registry

Living index of every Hydra config file under `conf/`. Update this table
whenever a config is added, its status changes, or it's deprecated.

## Composition order (`conf/config.yaml`)

```
defaults:
  - dataset: afewva
  - model: videomae_base
  - preprocess
  - train: default
  - experiment: <name>
  - _self_
```

`experiment` composes **after** `train` deliberately: an experiment's own
`train:`/`dataset:` overrides (see below) must win over the plain group
selection, and Hydra composes top-to-bottom.

## Experiment pattern

`conf/experiment/*.yaml` are self-contained, not name-only stubs. Each one
uses Hydra's `# @package _global_` + `defaults: [override /group: x]`
pattern:

```yaml
# @package _global_
defaults:
  - override /dataset: veatic   # re-target a config group this experiment owns

experiment:
  name: <must equal the file stem>
  description: "..."
  stage: pretrain | va_finetune   # routes scripts/run_experiment.sh

model:
  drop_path_rate: 0.2   # only if this experiment needs an architecture delta

train:
  pretrain_encoder_path: ${train.checkpoint_root}/<ssl-experiment-name>/best.pt
  freeze_backbone: true
```

`model:`/`train:`/`dataset:` blocks are **deltas only** — they override
specific fields on top of `conf/model/videomae_base.yaml` and
`conf/train/{default,ssl}.yaml`, which remain the source of truth for the
full recipe. Comment every override with the rationale, not just the value.

`override /dataset: x` / `override /train: x` replace shell-level CLI flags
(`scripts/run_experiment.sh` no longer hardcodes `dataset=voxmini train=ssl`
for pretrain-stage runs — see the experiment file itself).

Naming convention (unchanged): `<encoder>_<strategy>[_<variant>]_<dataset>.yaml`,
`name:` inside the file MUST equal the file stem (checkpoints/result JSONs key
off it).

## `model/`

One shared architecture (`videomae_base`) — Stage 1 (SSL) and Stage 2
(probe/finetune) use the same `VideoMAEEncoder`, so there is nothing to
select between; per-stage/per-experiment knobs live as fields on this one
file, overridden per-experiment where needed.

| Field | Default | Purpose |
|---|---|---|
| `num_frames`, `image_size`, `in_channels`, `tubelet_size`, `patch_size`, `embed_dim`, `depth`, `num_heads`, `mlp_ratio` | ViT-B geometry | Architecture, shared by both stages |
| `dropout` | `0.0` | Standard dropout inside attention/MLP |
| `drop_path_rate` | `0.0` | Stochastic depth, linearly scaled across blocks; override per-experiment for finetune-style runs |
| `decoder_depth`, `decoder_embed_dim`, `decoder_num_heads` | `4`/`256`/`8` | MAE decoder (Stage 1 only; unused by the probe/finetune path) |
| `mask_ratio` | `0.9` | Fraction of tubelets masked during SSL pretraining |
| `motion_mask` | `false` | `true` -> mask highest-motion tubelets (`MotionGuidedTubeMasking`) instead of random ones |
| `decoder_mask_ratio` | `0.0` | V2 dual masking: fraction of masked tokens the decoder skips reconstructing |

## `train/`

| File | Used by | Notes |
|---|---|---|
| `default.yaml` | Stage 2 (`training/train.py`) | Adam, flat LR, CCC-loss batch sizing |
| `ssl.yaml` | Stage 1 (`training/pretrain.py`) | AdamW, linear-scaling-rule LR (`base_lr`/`lr_mult`), warmup+cosine, throughput/VRAM instrumentation fields |

Both share the checkpoint-hand-off fields (`default.yaml` only, since
Stage 1 never loads a downstream checkpoint):

| Field | Default | Purpose |
|---|---|---|
| `pretrain_encoder_path` | `null` | Path to a Stage 1 SSL `best.pt`; `null` = train the encoder from scratch |
| `freeze_backbone` | `false` | `true` = frozen-encoder probe (only the VA head trains); `false` = end-to-end fine-tune |

See inline comments in each file for the rest of the hyperparameter surface
(`accum_steps`, `ddp.*`, `warmup_epochs`, etc.) — those are unchanged by this
registry rewrite.

## `dataset/`

| Name | Role | Present | Notes |
|---|---|---|---|
| `afewva` | probe | `false` | Flip to `true` after the ibug EULA download + `scripts/build_afewva_manifest.py` |
| `veatic` | probe | env-gated (`VEATIC_PRESENT`) | Never hardcode `true` in-repo — cluster-only |
| `voxmini` | ssl | env-gated (`VOXMINI_PRESENT`) | Never hardcode `true` in-repo — cluster-only |
| `affwild2` | probe | `false` | Blocked on signed academic application |
| `talkingcelebs` | ssl | `false` | Scaffolded, download/ingest scripts real |
| `talkingcelebs_mini` | ssl | `true` | Real acquired mini slice (25 clips / 5 identities) |

## `experiment/`

| Name | Stage | Dataset (self-selected) | Purpose |
|---|---|---|---|
| `videomae_ssl_pretrain_voxmini` | `pretrain` | `voxmini` | Stage 1 MAE pretraining |
| `videomae_probe_veatic` | `va_finetune` | `veatic` | Stage 2 frozen-encoder probe, loads Stage 1's `best.pt` |
| `videomae_va_finetune_afewva` | `va_finetune` | `afewva` | End-to-end from-scratch fine-tune, no freezing |
| `videomae_va_finetune_veatic` | `va_finetune` | `veatic` | End-to-end from-scratch fine-tune, no freezing |

## Training watcher (cron)

`scripts/cron/poll_squeue.sh` / `scripts/cron/aggregate_results.sh` were never
installed and are superseded by a single script:
`video_encoding/scripts/watch_training.py`
(docs/superpowers/specs/2026-07-15-slurm-training-watcher-design.md). Install
on login-node via `crontab -e`:

    * * * * *  cd ~/stern && ~/.local/bin/uv run --no-sync python -m video_encoding.scripts.watch_training

Three gotchas this line must keep, all already learned the hard way elsewhere
in this codebase (`mesh_tick.py`'s `_do_merging`, `run_mesh_worker.sh`,
`launch_mesh_pool.sh`, `submit_login-node.sh`, and `watch_training.py`'s own
module docstring) but previously missing from -- or wrong in -- this doc,
causing the same class of failure to recur across three separate incidents
(2026-07-27, 2026-07-29 x2):
- cron's minimal PATH does not include `~/.local/bin`, so a bare `uv` here
  resolves fine when installed interactively but fails with `uv: command not
  found` on every cron tick (silently, into `.claude/state/cron_stderr.log`).
  Always use the absolute/`~` path to the `uv` binary itself.
- `-m video_encoding.scripts.watch_training` is required, NOT a direct file
  path (`python video_encoding/scripts/watch_training.py`). Python sets
  `sys.path[0]` to the *script's own directory* when given a file path, but
  to the current directory (repo root) when given `-m` -- and this project
  has no editable install of `video_encoding` to compensate. The file-path
  form fails with `ModuleNotFoundError: No module named 'video_encoding'`
  every single tick, identically whether run from cron or an interactive
  shell -- this is a Python invocation bug, not a `uv`/cron/NFS issue, and
  no amount of PATH or `--no-sync` fiddling touches it.
- Separately, without `--no-sync`, every tick re-resolves/reinstalls the
  9.4GB NFS-hosted venv from the lockfile -- wasted work against a venv
  shared with concurrent worker/merge processes, though not itself the
  source of the `ModuleNotFoundError` above (that persists even with
  `--no-sync` if the file-path form is used).

State lives in `.claude/state/` (`jobs_registry.json`, `watcher_state.json`,
`squeue.json`, `watcher.log`, `escalations/*.md`) — gitignored, inspect with
`cat`/`jq` on login-node directly.
