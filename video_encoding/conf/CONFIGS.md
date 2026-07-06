# Config registry

Living index of every Hydra config file under `conf/`. Update this table
whenever a config is added, its status changes, or it's deprecated — this
replaces "grep the yaml files to remember what they mean" with one file to
read. Experiments are no longer a fixed enum of 3; see the naming convention
below before adding a new one.

## Experiment naming convention

Follows the What_Happens_Next repo's `configs/experiment/` scheme (chosen for
cross-project consistency — same repos, same cluster, same team). File name:

    <encoder>_<strategy>[_<variant>]_<dataset>.yaml

- `<encoder>`: `videomae`, `vjepa2`, `marlin` (matches `model/` file stems)
- `<strategy>`: `frozen_probe`, `lora_ssl`, `lora_supervised`
- `<variant>` (optional): distinguishing knob, e.g. `s2_lowlr`, `perceiver`
- `<dataset>`: dataset the experiment is bound to at its defaults

Examples: `videomae_frozen_probe_affwild2.yaml`,
`videomae_lora_ssl_voxmini.yaml`, `videomae_lora_supervised_affwild2.yaml`.

Inside the YAML, `name:` MUST equal the file stem — checkpoints and result
JSONs key off it. Only add a sibling file for a new
dataset/objective/encoder combination — never overload an existing
experiment's meaning by silently changing what it points to.

## `dataset/`

| Name | Role | Present | Purpose | Status | Introduced by |
|---|---|---|---|---|---|
| `affwild2` | probe | false | Labelled VA, per-frame, 30Hz | stub | video-arm-scaffold |
| `voxmini` | ssl | false | Unlabelled SSL adaptation, 218h target | stub | video-arm-scaffold |
| `talkingcelebs` | ssl | false | Full-corpus unlabelled SSL set (HF `samarrik/talkingcelebs`) | scaffolded (download/ingest scripts real) | video-arm-scaffold |
| `talkingcelebs_mini` | ssl | **true** | Real acquired mini slice: 25 clips / 5 identities / ~0.2h (the whole corpus is only 5 identities, not the ~40 assumed) | **acquired**, on login-node `/Data/$USER/talkingcelebs_mini` | T5/HPO plan Phase 2 |
| `sewa` | eval | false | Domain-shift eval, VA range **[0,1]** (not [-1,1] — harmonise before use) | stub | video-arm-scaffold |
| `afew_va` | eval | false | Domain-shift eval, native68 landmarks | stub | video-arm-scaffold |

Known fix: `manifest_path: ${path}/manifest.csv` doesn't resolve once composed
under the full Hydra tree (`path` isn't a root-level key) -- all three
`manifest_path` fields (`talkingcelebs`, `talkingcelebs_mini`, `voxmini`) now
use the relative-reference form `${.path}/manifest.csv` instead.

## `model/`

| Name | Encoder | Embed dim | Purpose | Status | Introduced by |
|---|---|---|---|---|---|
| `videomae_v2_base` | videomae | 768 | **Working encoder** for the approved pipeline (2026-07-01-training-pipeline-design.md) | implemented (`src/encoders/videomae.py`) | T5/HPO plan Phase 1 |
| `vjepa2_vitl` | vjepa2 | 1024 | Target long-term encoder (D-MODEL-1) | deferred (encoder stub, not built against) | video-arm-scaffold |
| `vjepa2_vitg` | vjepa2 | 1408 | Ablation-only per D-MODEL-1 | deferred | video-arm-scaffold |
| `marlin` | marlin | 768 | Clip-level alternative encoder | deferred | video-arm-scaffold |

Note: `conf/config.yaml`'s default `model:` is still `vjepa2_vitl` (D-MODEL-1
remains formally open) — all real runs pass `model=videomae_v2_base`
explicitly rather than changing the shared default.

## `experiment/`

| Name | Purpose | Status | Introduced by |
|---|---|---|---|
| `videomae_frozen_probe_affwild2` | Frozen encoder, cache once, train VA head only | probe stage implemented + tested against synthetic fixtures; a real run is blocked on Aff-Wild2 acquisition | video-arm-scaffold |
| `videomae_lora_ssl_voxmini` | LoRA + SSL objective on VoxMini, then freeze + probe | SSL stage implemented; probe tail implemented (frozen-cache path) but a real run is blocked on Aff-Wild2 | video-arm-scaffold |
| `videomae_lora_supervised_affwild2` | LoRA + VA head jointly trained, supervised | approved; blocked on Aff-Wild2 acquisition | video-arm-scaffold |
| `videomae_lora_ssl_talkingcelebs_mini` | LoRA + SSL on the mini TalkingCelebs slice | ready — talkingcelebs_mini is the only `present: true` dataset, though its data only exists on the acquiring cluster machine (no `TALKINGCELEBS_MINI_ROOT`/local dir in this worktree) | T5/HPO plan Phase 4 |

The old letter-based files (`a_frozen`, `b_lora_ssl`, `c_lora_sup`) have been
renamed to the WHN-style names above; the letter shorthand no longer exists.

## `train/`

Phase 1 fields (`conf/train/default.yaml`): `seed`, `lora_rank`,
`batch_size`, `grad_accum_steps` (legacy alias — `accum_steps` below is
what new code reads), `checkpoint_root`, `results_root`, plus per-stage
groups `ssl.{learning_rate,epochs,mask_ratio}` and
`probe.{learning_rate,epochs}` mirroring `ssl_adapt()`/`train_probe()`'s
own function defaults.

Phase 2 fields (WHN-style training pipeline enhancement,
2026-07-02):

| Field | Default | Purpose |
|---|---|---|
| `accum_steps` | `1` | Gradient accumulation micro-steps per optimizer step. Uses DDP `no_sync()` on non-boundary micro-steps to avoid useless allreduces. |
| `use_amp` | `false` | Enable `torch.amp.autocast('cuda')` + `GradScaler`. Off by default because AMP masks numerical bugs; opt in per experiment. |
| `grad_clip` | `1.0` | `clip_grad_norm_` threshold. `0.0` disables. |
| `warmup_epochs` | `1` | Linear LR warmup epochs before cosine anneal. |
| `lr_scheduler` | `cosine` | Only option currently — cosine anneal with warmup. |
| `resume` | `true` | Auto-resume from `checkpoint_root/<experiment>/{ssl_best,ssl_best_latest}.pt` if present. |
| `save_latest` | `true` | Also save a per-epoch snapshot for crash recovery. |
| `ddp.bucket_cap_mb` | `25` | DDP allreduce bucket size in MB — see rationale below. |
| `ddp.grad_compress` | `""` | `""` or `"fp16"`. fp16 halves NCCL bytes at negligible convergence cost. |
| `ddp.nccl_timeout_sec` | `300` | Passed as `timedelta` to `init_process_group`. |
| `ddp.find_unused_parameters` | `false` | Set `true` for LoRA experiments whose forward path leaves some LoRA branches cold. |
| `ddp.backend` | `nccl` | Override to `gloo` only for CPU-only debug runs. |
| `ddp.nccl_env` | `{}` | Kwargs like `{NCCL_ALGO: Ring, NCCL_PROTO: Simple}` propagated to `os.environ` via `setdefault`. |

**Why `ddp.bucket_cap_mb=25` and `grad_compress=fp16` are cluster-specific
defaults.** SallesInfo is 1GbE end to end; T2/T3 probes measured NCCL
`busBW` saturating at ~0.115 GB/s for messages ≥16 MB. Two consequences:

1. Smaller allreduce buckets (25 MB vs. WHN's 200 MB default) let more of
   the allreduce overlap with the next micro-step's compute. On a fat
   interconnect the larger bucket wins by amortising latency; on 1GbE
   bandwidth dominates and pipelining wins.
2. `grad_compress='fp16'` halves the bytes-per-allreduce. On this fabric
   that translates ~1:1 into halved comm time for any run where allreduce
   is >5% of step time (i.e. anything with a non-trivial gradient volume).
   Off by default because it can degrade convergence on toy models;
   safe-and-recommended for the current VideoMAEv2 + LoRA runs.

## Logger + checkpoint contract

`src/training/logger.py` is two plain functions, not a class — no
wrapper object to keep in sync with `SummaryWriter`'s own API.
`make_writer(log_dir, rank=)` returns a `SummaryWriter` on rank 0 and
`None` everywhere else; call sites gate every write with `if writer:`.
TB scalars land under `checkpoint_root/<experiment>/tb_ssl/`.
`write_result_json(metrics, cfg, results_root=, flops=)` is rank-gated the
same way and writes the final result JSON under `train.results_root`
(default `.claude/results/`) with WHN-compatible schema so cross-project
isoFLOP tables keep working. Per-rank status prints use plain `print()`
gated on `rank == 0` at the call site — no logging abstraction.

`src/training/checkpoint.py` writes atomically (tmp→rename) and guards
against overwriting a better all-time-best on resume via a cached
`_disk_best` float — deliberately avoiding a full `torch.load` per epoch
on NFS, which stalls rank 0 and can trip the NCCL watchdog.

## VA label contract (`common/data/labels.py`)

`ManifestRecord.label_path` points to a CSV with two columns
`valence,arousal`, one row per annotated frame at the dataset's
`annotation_hz` (matches Aff-Wild2's actual per-frame annotation export --
no explicit timestamp column, row index implies frame number).
`load_clip_va_label(label_path, annotation_hz, va_range)` harmonises each
column to `[-1, 1]` via `common/data/harmonise.py:harmonise_va`, then
mean-pools over the clip's frames to one `(valence, arousal)` pair --
`train_probe`/`LinearVAHead` operate at one embedding per clip (matching
`VideoMaeV2Encoder.encode_clip`'s single mean-pooled-over-time output), not
per-frame.

## Fixed: `run_experiment.sh` / `train.sbatch` never actually ran

`python -m video_encoding.probe.train` initializes the `video_encoding.probe` package first, whose
`__init__.py` imports `common.eval.ccc` as a side effect — that import only
resolved under pytest (`pythonpath=["."]` + pytest's own rootdir insertion),
never under a plain invocation. `scripts/run_experiment.sh` only `cd`s into
`video_encoding/`, so `common/` (one level up) was never on
`sys.path`; every experiment (SSL included, not just probe) failed with
`ModuleNotFoundError: No module named 'common'` before reaching any real
logic. This matches the earlier finding that `train.sbatch` has no evidence
of a completed real run. Fixed by exporting `PYTHONPATH` to include the repo
root in `run_experiment.sh` (inherited by `train.sbatch`, which only wraps
it) — verified end-to-end: `videomae_frozen_probe_affwild2` now reaches and
fails at the correct place (`DatasetNotAcquiredError` on Aff-Wild2), not an
import error.

## Known gaps (not implemented, not faked)

- **`video_encoding/probe/train.py:run_probe_stage`**: now code-complete (caches
  frozen-encoder embeddings via `common/cache/store.py`, loads labels via
  `common/data/labels.py`, trains via `video_encoding/probe/train_probe`) and covered
  by `video_encoding/tests/test_probe_stage.py` against synthetic
  fixtures. Still can't produce a real trained probe until Aff-Wild2 is
  actually acquired (`conf/dataset/affwild2.yaml: present: false`) — that
  part remains gated on the signed academic access application, not code.
  Recipe C (`videomae_lora_supervised_affwild2`: `use_lora: true`,
  `use_ssl: false`) currently falls through to this same frozen-cache path
  via `run_training`'s `use_ssl` branch, which is **wrong** for it: LoRA
  adapter params need gradients from the joint CCC loss flowing back
  through the encoder, so embeddings can't be cached once the way Recipe
  A's truly-frozen encoder allows. `run_probe_stage` doesn't check
  `cfg.experiment.use_lora` at all yet — needs its own LoRA-attached,
  no-caching training path before Recipe C can actually run correctly.
- **`common/eval/baselines.py`, `src/eval/*`**: unchanged stubs, out of scope
  for this plan (per the approved spec's own §7).

## Superseded / conflicting docs (do not build against)

`docs/superpowers/specs/2026-06-30-cluster-deep-network-protocol.md`,
`2026-06-30-cluster-tier1-findings.md`, and
`2026-07-01-t5-real-recipe-c-benchmark-design.md` exist only on `master` and
describe a V-JEPA2 + RetinaFace scaffold (`bench_recipe_c.py`,
`vjepa2_vitl.yaml` as the working encoder) that was never actually built on
this branch. The cluster/network/SLURM *infrastructure* facts in those docs
are still valid (architecture-independent); the encoder/pipeline design in
them is not — this branch's approved architecture is
`2026-07-01-training-pipeline-design.md` (VideoMAEv2).
