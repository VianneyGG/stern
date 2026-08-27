# EXPERIMENTS.md — the experiment ledger

**Git-tracked on purpose.** `runs/state/` and `runs/results/` are gitignored and
live only on this machine and on login-node; `.claude/` is gitignored by policy.
Everything recorded there dies on a fresh clone. This file is the durable record
of *what was run, why, and what it concluded* — the part that must outlive both
the checkpoints and the code that produced them.

Companion to `SCAFFOLD_ARCHIVE.md`, which records deleted *code*. This file
records finished *experiments*.

## The rule

> **Nothing that an experiment depended on gets deleted until that experiment is
> recorded here.**

That includes config knobs, `conf/experiment/*.yaml` files, model branches,
scripts, and checkpoints. Record first, delete second, in that order and in
separate commits. A knob whose only justification was a run nobody wrote down is
exactly the knob that gets deleted and then re-derived from scratch six months
later.

## How to add an entry

Append a row to the ledger, then a subsection under **Campaigns** if the run is
part of one. Minimum viable row: run ID, date, the experiment yaml plus any
`EXTRA_OVERRIDES`, the pool, and **the conclusion in one sentence**. A row with
no conclusion is not a record, it is a filename.

When you delete code, add it to **Retired code and configs** below with the last
experiment that used it — that back-reference is the whole safety mechanism.

---

## Ledger

Seeded 2026-08-27 from `runs/results/mesh/*_rounds.jsonl`, which is local-only
and gitignored. Those files carry merge bookkeeping (`round`, `n_contributors`,
`alphas`, `timeout_forced`) but **no loss** — per-round loss lives in the metrics
JSONL (2026-08-04 onward) or, for older runs, only in SLURM stdout. Columns
marked *backfill* need a pass over
`login-node:~/stern/runs/state/mesh_run_configs.json`; see **Backfill gaps**.

`Contributors min–max` is worth reading: a minimum well below the roster size
means merge quorum collapsed during the run and those rounds averaged fewer
workers than intended.

| Run ID | Last merge | Rounds logged | Last round | Contributors min–max | Timeout-forced | Config / arm | Outcome |
|---|---|---|---|---|---|---|---|
| `factorial-mixed-per_frame` | 2026-08-22 | 57 | 39 | 12–16 | 12 | *backfill* | *backfill* |
| `factorial-mixed-time_avg` | 2026-08-22 | 55 | 39 | 9–16 | 24 | *backfill* | *backfill* |
| `factorial-tube-per_frame` | 2026-08-22 | 54 | 39 | 4–11 | 10 | *backfill* | *backfill* |
| `factorial-tube-time_avg` | 2026-08-22 | 64 | 39 | 1–16 | 19 | *backfill* | *backfill* |
| `pool-cars-2r` | 2026-08-04 | 2 | 1 | 7–13 | 2 | *backfill* | *backfill* |
| `pool-cars-best-120r` | 2026-08-17 | 231 | 230 | 8–15 | 231 | *backfill* | *backfill* |
| `pool-cars-full-100r` | 2026-08-09 | 40 | 39 | 9–13 | 40 | *backfill* | *backfill* |
| `pool-cars-full-50r` | 2026-08-09 | 50 | 49 | 9–13 | 50 | *backfill* | *backfill* |
| `pool-cars-phase3-smoke` | 2026-08-07 | 3 | 2 | 1–10 | 3 | *backfill* | *backfill* |
| `pool-cars-phase4-8r` | 2026-08-08 | 8 | 7 | 9–13 | 8 | *backfill* | *backfill* |
| `pool-cars-phase4-fix2` | 2026-08-08 | 3 | 2 | 10–13 | 3 | *backfill* | *backfill* |
| `pool-cars-phase4-fix` | 2026-08-08 | 2 | 1 | 9–13 | 2 | *backfill* | *backfill* |
| `step0-smoke-fish-1r` | 2026-07-28 | 1 | 0 | 7–7 | 1 | *backfill* | *backfill* |
| `sw1-birds-lr070m000` | 2026-07-29 | 7 | 6 | 1–16 | 7 | *backfill* | *backfill* |
| `sw1-birds-lr070m080` | 2026-07-29 | 7 | 6 | 15–16 | 1 | *backfill* | *backfill* |
| `sw1-birds-lr070m090` | 2026-07-28 | 7 | 6 | 16–16 | 0 | *backfill* | *backfill* |
| `sw1-birds-lr070m095` | 2026-07-29 | 7 | 6 | 16–16 | 0 | *backfill* | *backfill* |
| `sw1-fish-lr050m090` | 2026-07-29 | 7 | 6 | 1–1 | 0 | *backfill* | *backfill* |
| `sw1-fish-lr090m090` | 2026-07-28 | 7 | 6 | 16–16 | 0 | *backfill* | *backfill* |
| `sw1-fish-lr110m090` | 2026-07-28 | 7 | 6 | 15–16 | 1 | *backfill* | *backfill* |
| `sw1-fish-lr130m090` | 2026-07-29 | 7 | 6 | 8–16 | 6 | *backfill* | *backfill* |
| `sw2-birds-lr070m080` | 2026-07-30 | 11 | 10 | 12–12 | 0 | *backfill* | *backfill* |
| `sw2-fish-lr050m090` | 2026-07-30 | 7 | 6 | 1–13 | 3 | *backfill* | *backfill* |
| `sw2-fish-lr090m090` | 2026-07-30 | 6 | 5 | 2–12 | 6 | *backfill* | *backfill* |
| `sw3-cars-accum128` | 2026-08-12 | 30 | 29 | 8–13 | 30 | *backfill* | *backfill* |
| `sw3-cars-lr060m090` | 2026-08-11 | 40 | 39 | 8–16 | 39 | *backfill* | *backfill* |
| `sw3-cars-lr080m090` | 2026-08-12 | 40 | 39 | 8–13 | 40 | *backfill* | *backfill* |
| `sw3-cars-lrmult15` | 2026-08-13 | 23 | 22 | 8–16 | 22 | *backfill* | *backfill* |
| `sw3-cars-wd010` | 2026-08-13 | 20 | 19 | 8–13 | 20 | *backfill* | *backfill* |
### `recipe2x2` runs — recovered from login-node 2026-08-27

These have no local `_rounds.jsonl`. Recovered from
`login-node:~/stern/runs/results/mesh/` and cross-checked against
`login-node:~/stern/runs/state/mesh_run_configs.json`. This **supersedes** the
earlier two-row entry, whose provenance was a deleted scratchpad file: the
scratchpad listed 2 runs, the authoritative sources show **7**, and both
recovered rows differ from what the scratchpad claimed.

| Run ID | Config | Pool (tier) | Rounds logged | Last round | Contributors min–max | Timeout-forced |
|---|---|---|---|---|---|---|
| `recipe2x2-legacy-per_frame-20260823` | `videomae_mesh_ssl` + `steps_per_round=10240 round_budget_s=null target_mode=per_frame` | birds, 10 | 20 | 19 | 7–10 | 3/20 |
| `recipe2x2-legacy-time_avg-20260823` | `videomae_mesh_ssl` + `steps_per_round=10240 round_budget_s=null target_mode=time_avg` | fish, 14 | 22 | 21 | 11–14 | 18/22 |
| `recipe2x2-paper-per_frame-20260823` | `videomae_mesh_paper2fps_per_frame` | countries, 16 | 24 | 23 | 13–16 | 8/24 |
| `recipe2x2-paper-time_avg-20260823` | `videomae_mesh_paper2fps_time_avg` | cars, 16 | 25 | 24 | 14–16 | 14/25 |
| `recipe2x2-carscmp-paper-per_frame-20260825` | `videomae_mesh_paper2fps_per_frame` | cars, 16 | 100 | 99 | 12–16 | 90/100 |
| `recipe2x2-fishcmp-paper-time_avg-20260825` | `videomae_mesh_paper2fps_time_avg` | fish, 8 | 100 | 99 | 6–8 | 37/100 |
| `recipe2x2-cars3rd-paper-stride4perframe-20260825` | `videomae_mesh_paper_d_stride4_perframe` | cars, 16 | 120 | 119 | 12–16 | 40/120 |

**`depthprobe-lr025-20260826`** is registered in `mesh_run_configs.json`
(`videomae_mesh_paper2fps_per_frame`, 60 rounds, 8 nodes) but has **no
`_rounds.jsonl` on login-node** — it never completed a merge round. Treat it as
launched-but-no-data, not as a result. Note also that its config records
`outer_lr 0.7` despite the `lr025` in its name; the name is not trustworthy.

**Validity caveat — arm is confounded with tier.** In the 2026-08-23 2×2, all
four cells ran on **four different tiers**: paper/time_avg on cars,
paper/per_frame on countries, legacy/time_avg on fish, legacy/per_frame on
birds. The 2026-08-25 pair is confounded too — carscmp is cars/16 while fishcmp
is fish/8, differing in both tier and node count. Per `MEMORY.md` →
tier-throughput-vs-vram there is a 1.47× throughput spread across tiers, so
none of these comparisons isolates the arm. This is precisely the invariant
`docs/research/2026-08-27-depth-collapse-3pool-prompt.md` fixes by pinning
identical compute per round across its three pools.

**Merge-quality caveat.** `recipe2x2-carscmp-paper-per_frame-20260825` merged on
**timeout rather than quorum in 90 of 100 rounds**, and `fishcmp` fell to 6
contributors against an 8-node roster. Neither run averaged the intended number
of workers.

---

## Campaigns

### Single-node profiling (2026-07-10 → 2026-07-11)
~40 runs in `runs/results/*.json`, named
`<ts>-videomae_ssl_pretrain_voxmini-p1-<knob>`. Swept nodes × batch × accum ×
AMP dtype × DDP bucket size × dataloader workers.
**Conclusion:** single-node beats multi-node DDP on this cluster; all five tiers
are 1 GbE, so no higher-bandwidth tier exists to escape to. This is what
motivated DiLoCo over DDP.

### `sw1` / `sw2` — DiLoCo outer-optimizer sweep (2026-07-28 → 2026-07-30)
Outer LR × momentum on fish and birds, ~7 rounds each. Short by design.
**Conclusion:** *backfill.* Related: `MEMORY.md` → diloco-30r-results records
that lr 0.7 / m 0.9 beat 0.45 / 0.8 per round, but an outage collapsed merge
quorum and left the grid open.

### `pool-cars-*` — pool bring-up and scale-up (2026-08-04 → 2026-08-17)
`2r` → `phase3-smoke` → `phase4-8r` → `phase4-fix`/`fix2` → `full-50r` →
`full-100r` → `best-120r`. Infrastructure bring-up, not science, except
`best-120r` (231 rounds logged, the longest run in the repo) which is the
reference checkpoint for `runs/results/vit_internals/`.
**Conclusion:** `best-120r` is the incumbent baseline encoder. `MEMORY.md` →
motion-decodability-saturates: 231 rounds buys the same `motion_r2` as 3000
steps, so training budget is not the explanation for weak motion features.

### `factorial-*` — mask_type × target_mode 2×2 (2026-08-22)
`{tube,mixed}` × `{time_avg,per_frame}`, 40 rounds each, one tier per arm.
**Conclusion:** mask_type is a small effect. **Caveat that must not be lost:**
`mesh_round` initially ignored `mask_type` entirely (`MEMORY.md` →
factorial-mask-type-trap), so early rounds of this campaign are suspect. Note
also `factorial-tube-time_avg` dropped to **1 contributor** at its minimum —
merge quorum collapsed.

### `recipe2x2-*` — paper-faithful recipe (2026-08-23 → 2026-08-26)
Seven runs in two waves: a short {legacy, paper} × {per_frame, time_avg} 2×2 on
2026-08-23 (20–25 rounds each), then three longer confirmation runs on
2026-08-25 (100, 100, 120 rounds), one of which (`cars3rd`) switched to the
`stride4_perframe` config.

**Conclusion:** the new recipe fixed positional collapse, but introduced a new
failure — encoder layer 1 overwrites the residual stream and layers 8–12 go
dead, invisible in `val_loss`. This is the finding that
`docs/research/2026-08-27-depth-collapse-3pool-prompt.md` (E2b) exists to chase.

**How much weight this conclusion carries:** the depth-collapse observation
comes from probing the merged checkpoints and is not in dispute. The *ranking*
between arms is, because every cell sat on a different tier and most rounds
merged on timeout — see the two caveats under the recovered-runs table. Read
these runs as "the new recipe produces depth collapse", not as
"per_frame beats time_avg" or vice versa.

### `xtier2-*` — cross-tier repeat of the factorial (2026-08-22)
Four runs, missing from the ledger above and recovered from
`mesh_run_configs.json` on 2026-08-27:
`xtier2-{tube,mixed}-{time_avg,per_frame}-20260822`, all
`videomae_mesh_ssl` plus
`mesh.round_budget_s=null model.mask_type=<m> model.target_mode=<t>`.
The `round_budget_s=null` override is the tell: this wave removed the per-round
wall-clock budget so that a round ends on step count rather than on time, which
is what makes a cross-tier comparison meaningful given the 1.47× throughput
spread (`MEMORY.md` → tier-throughput-vs-vram).
**Conclusion:** *backfill* — round logs are on login-node, not local.

---

## How knobs actually reach a run

Recorded 2026-08-27 because it invalidates an obvious-looking cleanup heuristic.

Model knobs are **not** set in `conf/experiment/*.yaml`. They are passed as
Hydra tokens through `EXTRA_OVERRIDES` →
`launch_mesh_pool.sh` → `register_mesh_run.sh`, and stored per run in the
`extra_overrides` field of `runs/state/mesh_run_configs.json` (14 of 46 records
carry them). Only `target_mode` appears in any experiment yaml.

So **"no experiment yaml sets this knob" is not evidence the knob is dead.**
Applied literally it would flag `model.mask_type` as clutter, when eight runs
(the four `factorial-*` and the four `xtier2-*`) set it via overrides — and
`mask_type` is precisely the knob this repo already lost data to when
`mesh_round` silently ignored it (`MEMORY.md` → factorial-mask-type-trap).

To check whether a knob is live, grep `extra_overrides` in
`mesh_run_configs.json`, not the yaml directory.

---

## Retired code and configs

Append here **before** deleting. One row per removal.

| Removed | Commit | Last experiment that used it | Why it is safe to remove |
|---|---|---|---|
| `video_encoding/mesh/fetch.py` + `video_encoding/tests/test_mesh_fetch.py` | *(the commit that follows this one)* | **None — never wired.** Its replacement was motivated by `factorial-tube-per_frame` (2026-08-21) | Imported only by its own test; no `.sh`/`.sbatch`/`.yaml`/`.json` reference. `watcher/shard_transfer.py` states in its module docstring that "the P2P fetch path is gone" as settled fact, and exists precisely because it was: that run dropped 46 of 260 shards (17.7%) because orphaned shards could not be fetched. `shard_transfer.py` moves shards by rsync pull instead. |
| `visible_distance_weights` / `model.visible_loss_distance_weighting` — arm B weighted by `exp(-d)` to the nearest **masked** token | `392ebc4` | **None — never launched.** Written and deleted the same day (2026-08-27) | Provably a no-op at every ratio we train at, for two independent reasons, measured on the production (8, 14, 14) grid. (a) At `mask_ratio: 0.9` only 10% of tokens survive, so *every* visible token sits at d=1.0 from a hole — one distinct weight, which normalises away entirely; still one value at 0.75; spreads only at 0.5. (b) Tube masking keeps a whole (h, w) column across all of T', so a token's own tube is always present at d=1 along the temporal axis and any 3D metric is constant regardless of ratio. Replaced by `visible_isolation_weights`, which measures distance in the (H', W') plane only, to the nearest visible token outside the tube — six distinct weights at 0.9, and it degenerates as masking *falls*, the opposite regime. Had this shipped, arm B's weighted pool would have been a bit-identical duplicate of its unweighted one for 120 rounds. |

### Kept, with reasons — evaluated 2026-08-27 and *not* deleted

The Stage 0 sweep of `docs/research/2026-08-27-depth-collapse-3pool-prompt.md`
proposed five candidates. Only one passed all five criteria. The rest, and why
each survived — recorded so the next pass does not re-litigate them:

- **`conf/experiment/videomae_mesh_paper2fps_{per_frame,time_avg}.yaml`** —
  **criterion 4 fails.** Three runs referencing them are *non-terminal* in
  `login-node:~/stern/runs/state/watcher_state.json`:
  `recipe2x2-paper-per_frame-20260823` (RUNNING),
  `recipe2x2-paper-time_avg-20260823` (DEGRADED) and `depthprobe-lr025-20260826`
  (DEGRADED, 0 rounds). No SLURM job holds them now, but per `MEMORY.md` →
  scancel-triggers-chunk-relaunch an abandoned run can be resurrected by chunk
  relaunch, and it would then fail on a missing config. Re-evaluate only after
  those three are driven to a terminal state.
- **`cache/store.py`** — kept because the prompt's own gate ("confirm nothing in
  the VA-probe path is about to need it") **does not hold**. E2b's primary metric
  is a per-layer attentive probe over 12 layers × 3 merged checkpoints, and this
  module's docstring describes exactly that workflow: "compute frozen embeddings
  once, write here, all probe training reads tensors from this cache." It is 40
  lines and still unwired; the right resolution is for the E2b probe work to
  either wire it or delete it, not for a cleanup pass to remove it a fortnight
  before it is needed.
- **`landmark_encoding/`** — **criterion 3 fails.** Named in both `CLAUDE.md`
  and `README.md` as the deliberate placeholder for the landmark arm. Kept
  *as* a placeholder; this note is what makes it unambiguous rather than
  forgotten.
- **`data/loaders.py`'s `NotImplementedError` stubs** — **already gone.** No
  `NotImplementedError` remains anywhere in non-test code; the only surviving
  match is a docstring that mentions the name. Nothing to do.

### Readability pass on the E2b path — 2026-08-27, zero deletions

The second half of Stage 0: audit the seven files E2b is about to edit
(`models/videomaev2/{mae,encoder,masking}.py`, `training/diloco/mesh_round.py`,
`training/pretrain.py`, `data/{mesh_shard_dataset,loaders}.py`) for config
branches and dead parameters kept alive only by a superseded experiment.

**Result: nothing qualified.** All eleven `conf/model/videomae_base.yaml` knobs
are read behind a `model_cfg.get(...)` default and **every one is covered by at
least two test files** — `dropout` 12, `target_mode` 6, `mask_type` 5,
`num_registers` 4, `motion_mask`/`qk_norm`/`mlp_type`/`norm_type` 3,
`layer_scale_init`/`decoder_mask_ratio`/`drop_path_rate` 2. The prompt's
criterion is "no live experiment sets it **and** no test covers it"; the second
clause fails for all eleven, so the pass removed nothing.

One genuine dead *value*, recorded rather than deleted:

- **`mask_type="frame"` → `RandomFrameMasking`** (`masking.py:292`). Across all
  46 records in `mesh_run_configs.json` only `tube` (4 runs) and `mixed` (4 runs)
  were ever set; `frame` has never run on the mesh. It stays because it is
  test-covered and is one arm of a three-way dispatch whose other two arms are
  live — deleting it would make `mask_type` a boolean in fact but not in name.
  If the factorial is never revisited, retire the value and the branch together.

`motion_mask` reads as dead from the yamls alone (no experiment sets it) but is
**live** — it is what the motion-decodability probe campaign exercised, and
`select_masking` documents it as authoritative over `mask_type` precisely so old
configs do not change meaning. Another instance of the yaml-only heuristic being
wrong; see *How knobs actually reach a run* above.

Per the prompt's "do not improve anything else while you are in there", no
renames, reformatting or comment rewrites were made to these files.

---

### `e2b-*` — depth-collapse ablation, **pre-registered 2026-08-27, NOT yet launched**

Written before launch deliberately: the arm→pool mapping and the two decisions
below are the parts a post-hoc reading of the results could not recover.

Base recipe `conf/experiment/videomae_mesh_paper_a_stride1_timeavg.yaml`
unchanged for all four arms — 120 rounds × 2400 steps, eff batch 256, bf16,
`target_mode: time_avg`, `frame_stride: 1`, augment on, `quorum_k=15`,
`chunk_rounds=4`. Everything varies through `EXTRA_OVERRIDES` only.

| Arm | Pool | Override | Question |
|---|---|---|---|
| A | fish | *(none)* | control |
| B | orchids | `model.visible_loss_weight=0.25` | does reconstructing visible patches keep the stream alive? |
| C | cars | `model.deep_supervision_layers=[4,8,12]` | do layers 4/8/12 carry reconstructable information? |
| D | countries | `model.visible_loss_weight=1.0` | is arm B's effect W-sensitive or robust? |

**Why one tier per arm is safe here, though it was not in the 2×2.** With one
tier per arm, "arm" and "tier" are the same variable, and the tiers differ by
1.47× in throughput (`MEMORY.md` → tier-throughput-vs-vram). That is only a
confound while a round ends on *wall-clock*: `videomae_mesh_ssl.yaml` sets
`mesh.round_budget_s: 750.0`, so a faster tier would take more steps per round
and the arms would be trained for different amounts. Every `paper_*` recipe
already sets `round_budget_s: null`, so a round ends on step count and tier
speed changes only how long the pool takes in hours. This is the same fix the
`xtier2-*` wave applied for the same reason; it is a property of the base
config, so **re-check it if the base recipe ever changes.**

Pool size is held at 16 active nodes for all four arms, not at each tier's
roster size (fish 23, birds 23, cars 23, countries 19, orchids 25). Node count
sets the DiLoCo averaging width, so an unequal roster would be a real confound
even with `round_budget_s: null`. `birds` is skipped as the slowest tier — it
would add roughly a third to the critical path for no scientific gain; it is the
fallback if `orchids` turns out not to be seeded.

**Two decisions worth recording, both of which could otherwise look arbitrary:**

1. **Pool D brackets `W`, it does not test the isolation weighting.** `W` is the
   only parameter that decides whether arm B does anything at all, so a single
   untested value risks a null result that cannot be distinguished from "the
   weight was wrong". Two points 4× apart answer that; the weighting can be
   revisited only if plain arm B moves the probe curve.
2. **`visible_loss_isolation_weighting` is implemented, tested and left OFF.**
   It is a hypothesis formed on 2026-08-27, not something from the literature.
   Turning it on in both arm-B pools would confound every arm-B result with an
   untested metric. See the *Retired code* note below for why the definition it
   replaced could never have worked.

**Result:** *not yet run.*

---

## Backfill gaps

Known-incomplete, with the exact source for each:

- **Per-run config and arm** — `login-node:~/stern/runs/state/mesh_run_configs.json`
  is the authoritative record of experiment yaml, overrides, roster and quorum
  per run. Not mirrored locally. *Partially closed 2026-08-27:* pulled and used
  to recover the `recipe2x2` runs above. It holds **46 records**, of which 40 are
  `videomae_mesh_ssl`; the `*backfill*` cells in the main ledger can be filled
  from it by matching `run_id`. Note it carries the launch config, **not** the
  outcome — the conclusion column still has to be written by hand.
  Note also login-node has **40** `_rounds.jsonl` files against **29** local ones,
  so the main ledger is missing roughly 11 runs entirely.
- **Per-round loss** — metrics JSONL for runs from 2026-08-04 onward; SLURM
  stdout only for anything earlier (`MEMORY.md` → mesh-run-loss-extraction).
  Aggregate on a fixed node cohort, since roster size varies within a run.
- **`sw1`/`sw2` conclusions** — never written down beyond a memory entry.
- ~~**The two `recipe2x2` rows** — provenance is a deleted scratchpad file.~~
  *Closed 2026-08-27:* recovered from login-node; there were 7 runs, not 2.
- **Archived checkpoints** — completed runs are archived to node089
  `/Data/...stern_mesh_results/`, which is node-local, not NFS. Record the node,
  or the checkpoint is unfindable.
