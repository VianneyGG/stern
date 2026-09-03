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

## Training watcher (systemd --user timer)

`scripts/cron/poll_squeue.sh` / `scripts/cron/aggregate_results.sh` were never
installed. The watcher itself is a single script,
`video_encoding/scripts/watch_training.py`
(docs/superpowers/specs/2026-07-15-slurm-training-watcher-design.md), but as
of 2026-08-12 it is no longer installed via cron — the cron spool was found
wiped 5 separate times with no confirmed root cause. It now runs as a
systemd `--user` timer, converged onto exactly one designated host via
`video_encoding/scripts/install_watcher.sh`:

    video_encoding/scripts/install_watcher.sh <host>   # designate <host>, converge this machine
    video_encoding/scripts/install_watcher.sh          # converge this machine against the existing designation

Designation lives in `runs/state/watcher_host` (shared NFS home) and is read
by `watch_training.py` on every tick, so a host whose timer survives a
decommission still refuses to touch shared state. `submit_login-node.sh` calls
the no-arg form on every job submission, but only against `login-node` itself
(its ssh heredoc runs there) — it converges login-node to `stop` when login-node
isn't the designated host, and does **not** reach out to re-converge the
designated host if that host's timer dies. There is currently no automated
self-healing for the designated host's own timer going down (confirmed gap,
2026-09-03) — if `systemctl --user status stern-watcher.timer` on the
designated host shows anything other than `enabled`/`active (waiting)`,
re-run `install_watcher.sh` there manually.

Do not reinstall the old cron line from pre-2026-08-12 history — it will
work briefly but gets no ongoing supervision and nothing will notice if it
silently stops ticking again.

State lives in `runs/state/` (`jobs_registry.json`, `watcher_state.json`,
`squeue.json`, `watcher.log`, `watcher_systemd.log`, `watcher_host`,
`escalations/*.md`) — gitignored, inspect with `cat`/`jq` directly.

## Watcher self-check (cron)

The watcher now runs as a systemd `--user` timer on one designated host
(`runs/state/watcher_host`), converged via
`video_encoding/scripts/install_watcher.sh`. Nothing re-converges that host's
own timer if it goes down — a dead/restarted user manager is invisible to
anything that itself depends on that same manager. `video_encoding/scripts/
check_watcher_health.sh` closes that gap: it checks `systemctl --user is-active
stern-watcher.timer` on the designated host and re-runs `install_watcher.sh` if
it isn't active. It must run from system cron, not another systemd `--user`
unit, to be an independent failure domain from the thing it's checking.
Install on the designated host only (currently node089) via `crontab -e`:

    */15 * * * * XDG_RUNTIME_DIR=/run/user/$(id -u) $HOME/stern/video_encoding/scripts/check_watcher_health.sh >> $HOME/stern/runs/state/watcher_health_cron.log 2>&1

Requires `loginctl enable-linger $USER` (already on) so `XDG_RUNTIME_DIR`
exists outside a login session. This is a mitigation, not a full fix — it
narrows outage windows to ≤15 minutes instead of indefinite, and it does not
itself have a supervisor; deliberately not adding one, since chaining another
watched process just relocates the same single point of failure.
