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

## Evaluation basis — read this before any row below

**Every row in this ledger dated on or before 2026-08-28 was scored on
`val_loss` (MAE reconstruction) plus qualitative internals plots, and on nothing
else.** Not by choice: all three labelled VA sets were `present: false`, so no
downstream number existed to rank an arm by. Read every "Outcome" with that
limitation attached.

That is a real limitation, not a formality. The `recipe2x2-*` depth-collapse
finding below is the proof: layer 1 overwriting the residual stream and layers
8–12 going dead was **invisible in `val_loss`**. Reconstruction loss does not
rank encoders for VA regression.

### VEATIC is available from 2026-08-28 — encoder comparison is now possible

```
.data/veatic/   (13 GB, workstation)
clips: 124   labelled frames: 257,601
observed VA range (post-harmonise): [-0.9146, 0.8893]   # matches conf/dataset/veatic.yaml exactly
clips where #frames != #labels: 0 of 124
```

```bash
export VEATIC_ROOT=~/Projects/STERN/.data/veatic
export VEATIC_PRESENT=true
```

Verified through the real contract, not just on disk: `read_manifest(..., "probe")`
parses all 124 records, `load_per_frame_va` reads every label file through
`harmonise_va`, Hydra composes with `present: True` as a real bool, and 57 tests
pass (`test_build_veatic_manifest`, `test_probe`, `test_probe_windows`,
`data/tests`).

So arms can be ranked on **CCC** for the first time: freeze the encoder →
`LinearVAHead` probe on VEATIC → `eval/ccc.py`. All the machinery already exists
(`models/videomaev2/regressor.py`, `training/train.py`,
`scripts/dump_probe_predictions.py`). Any arm comparison from here on should
report CCC, and `val_loss`-only comparisons should be labelled as provisional.

Two caveats that must not get lost:

- The previous VEATIC copy at `login-node:/Data/$USER/veatic` (built
  2026-07-07) was **wiped by the 2026-08-22 SallesInfo rebuild** — verified gone
  2026-08-28, that directory now holds only `uv-cache/` and `voxceleb2/`.
  `VEATIC.zip` has **no scriptable download URL** (hand-download from Google
  Drive), so there is no re-fetch step to automate: any node that needs it gets
  the extracted tree copied. Check `$VEATIC_ROOT/manifest.csv` exists on a host
  before flipping `VEATIC_PRESENT=true` there.
- 124 clips is enough to **rank arms**, not enough to be a headline result.
  `affwild2` (signed academic application) and `afewva` (ibug EULA) remain
  `present: false` and unrequested; both are pure queue latency.

---

## Ledger

Seeded 2026-08-27 from `runs/results/mesh/*_rounds.jsonl`, which is local-only
and gitignored. Those files carry merge bookkeeping (`round`, `n_contributors`,
`alphas`, `timeout_forced`) but **no loss** — per-round loss lives in the metrics
JSONL. *Backfilled 2026-08-28:* the `Config / arm` and `Outcome` columns were
filled by joining `run_id` against `~/stern/runs/state/mesh_run_configs.json`
(arm) and `~/stern_mesh/metrics/<run_id>/<node>.jsonl` (per-round loss), both
read via `node089`. Every Outcome below is a `val_loss` statement and nothing
more — see **Evaluation basis** above for why that is a weak ranking signal,
and **Backfill gaps** for what is still missing.

`Contributors min–max` is worth reading: a minimum well below the roster size
means merge quorum collapsed during the run and those rounds averaged fewer
workers than intended.

| Run ID | Last merge | Rounds logged | Last round | Contributors min–max | Timeout-forced | Config / arm | Outcome |
|---|---|---|---|---|---|---|---|
| `factorial-mixed-per_frame` | 2026-08-22 | 57 | 39 | 12–16 | 12 | `videomae_mesh_ssl` + `mask_type=mixed target_mode=per_frame` | val_loss 0.685→0.463 over 40 rounds; plateauing (last-5 mean improves only 0.007 over the preceding 5). |
| `factorial-mixed-time_avg` | 2026-08-22 | 55 | 39 | 9–16 | 24 | `videomae_mesh_ssl` + `mask_type=mixed target_mode=time_avg` | val_loss 0.608→0.387 over 40 rounds, plateauing (last-5 gain 0.007); timeout-forced on 24/55 rounds, the worst of the four cells. |
| `factorial-tube-per_frame` | 2026-08-22 | 54 | 39 | 4–11 | 10 | `videomae_mesh_ssl` + `mask_type=tube target_mode=per_frame` | val_loss 0.693→0.494 over 40 rounds, plateauing (last-5 gain 0.009); quorum collapsed to 4 contributors at minimum and the cohort never exceeded 11, well under the ~16-node roster of the other cells — the least-comparable cell. |
| `factorial-tube-time_avg` | 2026-08-22 | 64 | 39 | 1–16 | 19 | `videomae_mesh_ssl` + `mask_type=tube target_mode=time_avg` | val_loss 0.613→0.391 over 40 rounds, plateauing (last-5 gain 0.007) — best final loss of the four cells, but also the cell that dropped to **1 contributor** at its minimum, so read it with that caveat. |
| `pool-cars-2r` | 2026-08-04 | 2 | 1 | 7–13 | 2 | `videomae_mesh_ssl` (data weighting) | val_loss 0.666→0.641 over 2 rounds; bring-up smoke, confirms the merge loop runs, not a result. |
| `pool-cars-best-120r` | 2026-08-17 | 231 | 230 | 8–15 | 231 | `videomae_mesh_ssl`, cars tier — **not present in `mesh_run_configs.json`**; arm inferred from metrics JSONL only (inner lr 3.75e-05, 16–17 shards) | val_loss 0.615→0.252 over 231 rounds (the `120r` in the name undercounts what actually ran, same naming trap as `depthprobe-lr025`). Incumbent baseline checkpoint, referenced by `runs/results/vit_internals/`; `MEMORY.md` → motion-decodability-saturates notes 231 rounds buys no more `motion_r2` than 3000 steps. |
| `pool-cars-full-100r` | 2026-08-09 | 40 | 39 | 9–13 | 40 | `videomae_mesh_ssl`, token weighting, `quorum_k=8` | val_loss 0.449→0.404 over 40 rounds (the `100r` name is wrong — `mesh_run_configs.json` and the round log both say 40). |
| `pool-cars-full-50r` | 2026-08-09 | 50 | 49 | 9–13 | 50 | `videomae_mesh_ssl`, token weighting, `quorum_k=8` | val_loss 0.613→0.451 over 50 rounds. |
| `pool-cars-phase3-smoke` | 2026-08-07 | 3 | 2 | 1–10 | 3 | `videomae_mesh_ssl` (data weighting) | val_loss 0.624→0.580 over 3 rounds; bring-up smoke. |
| `pool-cars-phase4-8r` | 2026-08-08 | 8 | 7 | 9–13 | 8 | `videomae_mesh_ssl`, token weighting, `quorum_k=8` | val_loss 0.616→0.556 over 8 rounds; bring-up. |
| `pool-cars-phase4-fix2` | 2026-08-08 | 3 | 2 | 10–13 | 3 | `videomae_mesh_ssl`, token weighting, `quorum_k=8` | val_loss 0.614→0.579 over 3 rounds; bring-up iteration after `phase4-fix`. |
| `pool-cars-phase4-fix` | 2026-08-08 | 2 | 1 | 9–13 | 2 | `videomae_mesh_ssl`, cars tier — **not present in `mesh_run_configs.json`** | val_loss 0.613→0.605 over 2 rounds; bring-up fix, too short to read anything into. |
| `step0-smoke-fish-1r` | 2026-07-28 | 1 | 0 | 7–7 | 1 | `videomae_mesh_ssl` (data weighting) | val_loss 0.614 at the only round logged — first end-to-end mesh smoke test, not a result. |
| `sw1-birds-lr070m000` | 2026-07-29 | 7 | 6 | 1–16 | 7 | `videomae_mesh_ssl`, `outer_lr=0.7 outer_momentum=0.0` | val_loss 0.624→0.578 over 7 rounds. |
| `sw1-birds-lr070m080` | 2026-07-29 | 7 | 6 | 15–16 | 1 | `videomae_mesh_ssl`, `outer_lr=0.7 outer_momentum=0.8` | val_loss 0.622→0.563 — best final loss of the `sw1` grid among runs with a real (>1) contributor cohort. |
| `sw1-birds-lr070m090` | 2026-07-28 | 7 | 6 | 16–16 | 0 | `videomae_mesh_ssl`, `outer_lr=0.7 outer_momentum=0.9` | val_loss 0.628→0.567 over 7 rounds, full 16-node quorum every round. |
| `sw1-birds-lr070m095` | 2026-07-29 | 7 | 6 | 16–16 | 0 | `videomae_mesh_ssl`, `outer_lr=0.7 outer_momentum=0.95` | val_loss 0.622→0.569 over 7 rounds, full quorum every round. |
| `sw1-fish-lr050m090` | 2026-07-29 | 7 | 6 | 1–1 | 0 | `videomae_mesh_ssl`, `outer_lr=0.5 outer_momentum=0.9` | val_loss 0.619→0.551 — nominally the best number in the grid, but only **1 node ever contributed a merge round**, so this is not a real multi-worker DiLoCo result. |
| `sw1-fish-lr090m090` | 2026-07-28 | 7 | 6 | 16–16 | 0 | `videomae_mesh_ssl`, `outer_lr=0.9 outer_momentum=0.9` | val_loss 0.614→0.566 over 7 rounds, full quorum every round. |
| `sw1-fish-lr110m090` | 2026-07-28 | 7 | 6 | 15–16 | 1 | `videomae_mesh_ssl`, `outer_lr=1.1 outer_momentum=0.9` | val_loss 0.612→0.568 over 7 rounds. |
| `sw1-fish-lr130m090` | 2026-07-29 | 7 | 6 | 8–16 | 6 | `videomae_mesh_ssl`, `outer_lr=1.3 outer_momentum=0.9` | val_loss 0.615→0.578, worst final loss among the fish arms and highest lr tried — consistent with the grid's outer_lr not benefiting past ~1.1. |
| `sw2-birds-lr070m080` | 2026-07-30 | 11 | 10 | 12–12 | 0 | `videomae_mesh_ssl`, `outer_lr=0.7 outer_momentum=0.8` | Configured for 15 rounds, reached only 11 before quorum collapsed (contributors stuck at 12/16 roster); val_loss 0.613→0.536 over the rounds it did complete — not comparable to the full-length `sw1` runs. |
| `sw2-fish-lr050m090` | 2026-07-30 | 7 | 6 | 1–13 | 3 | `videomae_mesh_ssl`, `outer_lr=0.5 outer_momentum=0.9` | val_loss 0.604→0.554 over 7 rounds (configured length matched, unlike the other two `sw2` arms). |
| `sw2-fish-lr090m090` | 2026-07-30 | 6 | 5 | 2–12 | 6 | `videomae_mesh_ssl`, `outer_lr=0.9 outer_momentum=0.9` | Configured for 15 rounds, reached only 5 before quorum collapsed (13 nodes at round 1 down to **2** at round 5); val_loss 0.583→0.574 — too short, and too cohort-unstable, to compare against `sw1-fish-lr090m090` at the same lr/momentum. |
| `sw3-cars-accum128` | 2026-08-12 | 30 | 29 | 8–13 | 30 | `videomae_mesh_ssl`, token weighting, `quorum_k=8`, + `train.accum_steps=128` | val_loss 0.631→0.437 over 30 rounds (shorter horizon than the lr arms below, not directly comparable on final loss). |
| `sw3-cars-lr060m090` | 2026-08-11 | 40 | 39 | 8–16 | 39 | `videomae_mesh_ssl`, token weighting, `quorum_k=8`, `outer_lr=0.6 outer_momentum=0.9` | val_loss 0.616→0.403 over 40 rounds. |
| `sw3-cars-lr080m090` | 2026-08-12 | 40 | 39 | 8–13 | 40 | `videomae_mesh_ssl`, token weighting, `quorum_k=8`, `outer_lr=0.8 outer_momentum=0.9` | val_loss 0.611→0.395 over 40 rounds — best final loss in the `sw3-cars` group at matched round count, slightly ahead of `lr060m090`. |
| `sw3-cars-lrmult15` | 2026-08-13 | 23 | 22 | 8–16 | 22 | `videomae_mesh_ssl`, token weighting, `quorum_k=8`, + `train.lr_mult=1.5` | val_loss 0.596→0.445 over 22 rounds; shorter horizon than the lr arms, not directly comparable. |
| `sw3-cars-wd010` | 2026-08-13 | 20 | 19 | 8–13 | 20 | `videomae_mesh_ssl`, token weighting, `quorum_k=8`, + `train.weight_decay=0.1` | val_loss 0.652→0.455 over 20 rounds — worst starting *and* final loss in the group, but also the shortest run, so weight decay is not isolated from round count here. |
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
**Conclusion:** within the `sw1` grid, final val_loss is flattest across
momentum (0.0 through 0.95 land within ~0.02 of each other at lr=0.7) and
mildly worse at the highest lr tried (1.3). No arm here isolates a clear
winner — the spread across the whole grid is small relative to a single run's
own round-to-round noise. `sw2` extended two of the three carried-forward arms
to 15 rounds but both **collapsed via quorum before reaching it** (11/15 and
5/15), consistent with `MEMORY.md` → diloco-30r-results: lr 0.7 / m 0.9 beat
0.45 / 0.8 per round at 30 rounds elsewhere, but an outage collapsed merge
quorum here and left the grid open. One `sw1` row
(`sw1-fish-lr050m090`) never had more than 1 real contributor and should be
read as a broken single-node run, not a data point in the grid.

### `sw3-cars-*` — outer-LR and regularization follow-up on cars tier (2026-08-11 → 2026-08-13)
Five runs, all cars tier, token weighting, `quorum_k=8`: two 40-round outer-LR
arms (`lr060m090`, `lr080m090`), plus three shorter arms each isolating one
knob against the same base recipe (`accum128`: `accum_steps=128`,
`lrmult15`: `lr_mult=1.5`, `wd010`: `weight_decay=0.1`).
**Conclusion:** at matched round count (40), `outer_lr=0.8` edges out `0.6`
(final val_loss 0.395 vs 0.403) — a small effect, same direction as the `sw1`
grid's lean toward higher lr up to a point. The three single-knob arms ran at
20–30 rounds, not 40, so their final losses are **not comparable** to the lr
arms or to each other on round count; none of the three isolates its knob
cleanly enough to conclude anything beyond "training still descends with this
override on."

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

### Conditioning/rank diagnostic on the `recipe2x2-*` checkpoints (2026-08-28)

Run before launching the pre-registered `e2b-*` ablation below, to see whether
a direct read on the checkpoints already on disk could localize the collapse
more cheaply than four new 120-round pool runs. Script:
`video_encoding/scripts/diagnose_conditioning.py`, run once on an orchids node
(job 7059) against all four `recipe2x2-{legacy,paper}-{per_frame,time_avg}`
`master.pt` files. Per encoder block it reports the singular-value spectrum
(condition number, spectral-entropy effective rank) of the qkv/proj weight
matrices, of the block's output activations on one fixed-seed random input
batch, and of the qkv weight gradient from one MAE-loss backward pass at that
checkpoint. Full log: `runs/results/diagnostics/conditioning_recipe2x2_20260828.txt`.

**Caveat carried over from the run itself:** input is fixed-seed random noise,
not real video data — no manifest/shard path was wired for this script. This
isolates architecture/weight-space rank collapse; it cannot rule out
additional data-dependent collapse a real-data pass might show. Also a single
checkpoint-time snapshot, not a trajectory across training rounds — no
intermediate-round checkpoints were pulled for `recipe2x2`, only the final
`master.pt` per cell.

**Findings, and why they revise the "layers 8–12 go dead" framing above:**

1. **Weight matrices are not where this lives.** qkv/proj effective rank stays
   high (640–725 of a 768-dim max) at every depth in *both* recipes. Whatever
   collapses, it is not the weight matrices going low-rank.
2. **Activation rank collapse starts at block 1 in the paper recipe, not
   block 8.** legacy activations hold 220 (per_frame) / 219 (time_avg)
   effective dimensions at block 1, settling to 41 / 51 by block 12. paper
   activations are already down to 13 (per_frame) / 18 (time_avg) at block 1,
   asymptoting to 2.4 / 8.0 by block 12 — condition numbers in the tens of
   thousands vs. hundreds for legacy. The residual stream is compounding an
   existing collapse through depth, not developing one partway through.
3. **Gradients confirm the deep layers are stuck, not merely quiet.** legacy
   gradient effective rank plateaus at 9–12 in deep blocks (cond ~1e8–1e9).
   paper drops to 1.9–2.5 (per_frame) / 4–8 (time_avg) with condition numbers
   reaching 1e10 — deep-layer gradients are nearly rank-1.
4. **`per_frame` is consistently worse than `time_avg`** on every metric at
   every depth within the paper recipe — a finer split than the ranking
   caveat above could previously support, though it is still one checkpoint
   per cell (point 2 of "how much weight this carries," above, still applies).

**Implication for `e2b-*` below:** arm C's `deep_supervision_layers=[4,8,12]`
targets a "does layer 4/8/12 carry information" question that assumes
collapse is a mid/late-depth phenomenon. This diagnostic says it is not — it
is present at block 1 and compounds. Worth deciding whether arm C's layer
choice should move earlier (e.g. `[1,4,8]`) before that pool launches, if it
has not already.

### PCA / attention-map visualisation of the same `recipe2x2-*` checkpoints (2026-08-28)

Follow-up to the spectrum-stats diagnostic above, same day, same checkpoints,
same fixed-seed random input — this time making the collapse visible rather
than just numeric. Script: `video_encoding/scripts/diagnose_pca_attn.py`, run
once on an orchids node (job 7076). Per encoder block: PCA of the block's
token-activation matrix (# principal components needed for 90% cumulative
variance, plus a PC1–PC2 scatter at blocks 1/6/12), and the block's raw
attention matrix (query-averaged, head-averaged) at blocks 1/6/12. Plots:
`runs/results/diagnostics/pca_attn/{pca_dim_vs_depth,pca_scatter_grid,attention_maps_grid}.png`;
full log `runs/results/diagnostics/pca_attn_20260828.txt`. Same caveats as
above (random input, single snapshot, no parity test against `block.py` — see
the script's module docstring for the mirror-drift risk this carries).

**Findings, and how they sharpen point 2 above:**

1. **`pca_dim_vs_depth` shows the paper recipe is already collapsed at block 1,
   not "starting normal and compounding."** legacy needs 24–27 PCs for 90%
   variance at block 1, dropping to a 5–6 floor by block 3 and holding flat
   through block 12. paper needs only 3–5 PCs at block 1 and is already flat
   there — there is no observable "normal" starting point to compound from;
   whatever collapses the representation has already acted by patch_embed +
   block 1.
2. **The PC1–PC2 scatter shows *what* legacy's higher dimensionality buys it**:
   at blocks 6/12, legacy's token cloud traces a clear ring/annulus — a
   structured low-dimensional manifold, not just "more variance." paper's
   scatter at every depth is a single unstructured blob with no visible
   geometry. This is a qualitative difference, not just a magnitude one.
3. **Attention maps show both recipes have sink tokens (bright vertical
   stripes = a few tokens every query attends to), but paper's sink is far
   more extreme**: paper's stripes are brighter (peak attention mass ~0.25–0.35
   vs legacy's ~0.10–0.14), narrower (fewer distinct tokens), and this
   sharpens with depth — by block 12, paper's attention is dominated by 3-4
   near-saturated columns. This is consistent with (arguably a visible cause
   of) the activation/gradient rank collapse: if attention output is
   dominated by a handful of value vectors, the block's output activation
   cannot have effective rank much above that handful.

**Implication for `e2b-*`:** this adds a candidate mechanism the ledger didn't
have before — attention-sink collapse, not just "deep layers go dead." If arm
C's `deep_supervision_layers` gets revised per the note above, it's also worth
asking whether any arm should touch attention entropy/sink behavior directly
(e.g. an entmax/sink-token-aware attention variant) rather than only patching
the loss side (arms B/D) or supervision depth (arm C) — open question, not
actioned here.

### Real-video confirmation of the same collapse: `export_vit_internals.py` on `recipe2x2-*` (2026-08-28)

The two entries above both use fixed-seed random noise as input. This checks
whether the collapse shows on real video — reusing the existing
`export_vit_internals.py` / `plot_vit_internals.py` pair (built 2026-08-14 for
the `sw3-cars-*` sweep, see the `export_vit_internals mirror drift` note) with
the four `recipe2x2-*` checkpoints plus a `random_init` control, run on
`node047` (a `cars`-tier pool node with the VoxCeleb2 shard corpus already
node-local from the recipe2x2 pool run) against real clips from
`shard-000000.tar`, 3 fixed-seed sampled windows. Output:
`runs/results/diagnostics/vit_internals_recipe2x2/{contact_sheet,layer_diagnostics}.png`,
raw arrays in `runs/results/diagnostics/recipe2x2_internals.npz`.

**Bug fixed to make this run:** `export_vit_internals.py`'s `load_encoder`
assumed one `model_cfg` shape for every `--run-ids` entry, but recipe2x2 mixes
per_frame/time_avg (`predict_head` width 1536 vs 768) and legacy/paper
(decoder width 256/8 vs 384/6) checkpoints in a single call — the exact same
shape-mismatch class as the conditioning-diagnostic bugs above. Added
`model_cfg_for_run()` (clone + override per run_id from the run_id string,
same convention as `diagnose_conditioning.py`) so `--run-ids` can mix recipes.
Lint-clean, existing 19 `test_export_vit_internals.py` cases still pass.

**Findings:**

1. **PCA-RGB contact-sheet panels show the same split visually.** legacy rows
   (`recipe2x2-legacy-{per_frame,time_avg}`) paint smooth, spatially coherent
   color gradients across the face in every window — recognizable structure
   tracking real image content. paper rows (`recipe2x2-paper-*`) are visibly
   speckled/noisy with no coherent spatial gradient, consistent with "there
   isn't enough effective rank left to paint a smooth 3-component PCA."
2. **`layer_diagnostics.png`'s attn-branch-output-norm panel is the sharpest
   real-data signal.** legacy's norm density sits entirely in the 8.8–43 band
   through all 12 layers. paper's has a bright band at 209–1275 — two orders
   of magnitude higher output norms — which is the real-data fingerprint of
   the attention-sink collapse the synthetic-input attention maps showed
   above: a few tokens dominating the attention output drives that output's
   norm far outside the rest of the token population.
3. **`cos_sim` (consecutive-layer cosine similarity) does not distinguish the
   recipes** — all four checkpoints plus `random_init` converge to ~0.99+ by
   layer 3–4. This metric doesn't carry the signal the others do; don't read
   anything into it being flat across recipes.

No new implication beyond confirming 1–2 above hold on real data, not just
synthetic noise — the "isolates architecture collapse, doesn't rule out
data-dependent collapse" caveat on both earlier diagnostics is now closed.

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
**Conclusion:** with `round_budget_s=null` fixing the tier confound, the
`factorial-*` finding survives on a different tier: `target_mode` is the
dominant axis (time_avg final val_loss 0.38–0.41 vs. per_frame 0.47–0.50),
`mask_type` remains a small effect within each target_mode (tube vs. mixed
differ by ≤0.03 either way). val_loss: mixed/per_frame 0.69→0.47 (40 rounds),
mixed/time_avg 0.61→0.38 (39/40), tube/per_frame 0.69→0.50 (40),
tube/time_avg 0.61→0.41 (40).

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

### `e2b-*` — depth-collapse ablation, **pre-registered 2026-08-27, launched 2026-08-28, STOPPED INCOMPLETE 2026-08-29**

Written before launch deliberately: the arm→pool mapping and the two decisions
below are the parts a post-hoc reading of the results could not recover. The
pre-registration is preserved verbatim below; **what actually launched differs
from it on four axes** — see *As-launched divergences* after the table.

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

### As-launched divergences from the pre-registration (recorded 2026-08-28)

All four arms **are** registered in `mesh_run_configs.json` and all four have
per-round metrics — an earlier session's conclusion that B and D never launched
was wrong. Live state, read from `~/stern_mesh/metrics/<run>/` and
`mesh_run_configs.json` via `node089`:

| Arm | Run ID | Pool as launched | Roster (cfg / reporting) | `chunk_rounds` | Round where it **stalled** |
|---|---|---|---|---|---|
| A | `e2b-a-fish` | fish | 18 / 16 | 4 | 26 |
| B | `e2b-b-birds` | **birds**, not orchids | 22 / 18 | 4 | 13 |
| C | `e2b-c-cars` | cars | 16 / 16 | 4 | 27 |
| D | `e2b-d-countries` | countries | 16 / 16 | **5** | 42 |

That last column is **not progress — it is where each arm died.** All four were
halted as of 2026-08-28 21:14 UTC; see *All four arms are stalled* below.

Four deviations, each of which weakens the comparison the pre-registration was
designed to make:

1. **Arm B ran on `birds`, not `orchids`.** The documented fallback fired
   ("`birds` … is the fallback if `orchids` turns out not to be seeded"). Since
   birds is the slowest tier, this is also the direct cause of divergence 4.
2. **Rounds registered as 50, not the 120 the recipe above specifies.** All four
   arms, so at least this one is applied uniformly.
3. **Roster sizes are not held at 16.** The pre-registration calls equal node
   count a real confound because it sets DiLoCo averaging width — yet A
   registered 18 nodes and B registered 22, against 16 for C and D. Reporting
   rosters are 16/18/16/16. `quorum_k=15` against a 16-node roster is also very
   tight (`MEMORY.md` → quorum-excludes-slowest-tier).
4. **Stall points are wildly unequal: 13 to 42 rounds.** The arms are nowhere
   near matched compute. Any comparison drawn today would be reading round
   count, not arm. D also uses `chunk_rounds=5` where the others use 4.

**Result:** *stalled, no outcome.* Divergences 1 and 3 mean this will not be
the clean one-variable ablation it was pre-registered as, even if the arms are
restarted and run to completion; decide before then whether to re-run B on a
matched roster or to reframe the comparison. Do **not** rank arms on the
current partial curves.

### All four arms are stalled — two distinct failures (2026-08-28 21:14 UTC)

**The watcher is not the cause and was never down.** `crontab -l` on `node089`
is `* * * * *`, the log was written 6 s before this check, and the watcher was
concurrently relaunching chunks for other runs
(`recipe2x2-paper-per_frame: relaunched 16/16 from round 24`). An earlier draft
of this section blamed the missing overnight-survival flags for the uneven
progress; **that attribution was wrong** and is corrected here. The
overnight-survival gap below is real but is not what stopped these runs.

**C and D — `run_outer_step` crash loop.** Merging fails every tick:

| Arm | Failing since (UTC) | Failed attempts | Round | Pending ckpts |
|---|---|---|---|---|
| `e2b-c-cars` | 15:48 | **327** | 27 | 16 |
| `e2b-d-countries` | 14:37 | **393** | 42 | **0** |

The failure *changes shape*, which is the diagnostic tell. The first failure on
both arms was `Command [...run_outer_step...] timed out after 300 seconds`.
Every failure since is `token_weights: total weight is zero (stats={})` (C) /
`data_weights: total sample count is zero (n_samples={})` (D) — empty dicts.
Most likely reading, **not yet confirmed**: the initial 300 s timeout killed
`run_outer_step` after it had consumed or moved the pending checkpoints, so
each retry finds nothing left to weight. If so the loop destroyed its own
input and cannot recover by retrying, which is what it has done ~700 times
between the two arms. Last successful merges: C 17:46, D 16:35 (local).

**A and B — relaunching without advancing.** Not a merge failure; the chunk
relaunch fires and re-submits from the *same* `start_round`:

```
20:18  e2b-a-fish:  chunk done -- relaunched 16/16 worker(s) from round 26
21:06  e2b-a-fish:  chunk done -- relaunched 15/15 worker(s) from round 26
14:06  e2b-b-birds: chunk done -- relaunched 22/22 worker(s) from round 13
15:04  e2b-b-birds: chunk done -- relaunched 22/22 worker(s) from round 13
```

B has been pinned at round 13 for over nine hours (last successful merge
13:03 local; 5 pending checkpoints against a 22-node roster, far under
`quorum_k=15`). A last merged 15:30 with 14 pending.

**Not yet diagnosed.** Both failures need a real debugging pass —
`scripts/run_outer_step.py` for the weighting crash, and the
`chunk_relaunch.py` / `mesh_tick.py` `start_round` advance for A and B. Do not
`scancel` the stale `PENDING` workers while investigating (`MEMORY.md` →
scancel-triggers-chunk-relaunch).

**Separately, the overnight-survival prerequisite is still open** (see
`docs/superpowers/plans/2026-08-28-overnight-survival-handoff.md`):
`video_encoding/slurm/mesh_train.sbatch` carries no `--signal`, `--requeue` or
`--dependency` as of 2026-08-28, so every chunk boundary still depends on the
watcher cron firing. That remains a single point of failure worth removing —
it just is not the explanation for this particular stall.

### Resolved, then stopped — what the stall actually was (2026-08-29)

The section above recorded the symptoms correctly and **the mechanism wrong on
three counts**. All three are corrected here; the run was then stopped
deliberately, mid-flight, with the arms unequal.

**1. The retry loop did not destroy its own input.** The hypothesis above —
"the timeout killed `run_outer_step` after it had consumed or moved the pending
checkpoints" — is false. `mesh_tick.py` unlinks `pending/` only on
`returncode == 0`, so all 16 of C's checkpoints sat on disk untouched through
all 327 failures. The section even notes this ("nothing deletes pending files on
failure") without connecting it.

**What actually happened.** The 300 s timeout was not an unrelated first
failure — it is the whole cause. `subprocess.run(..., timeout=300)` SIGKILLed
the merge **~90 s after every durable write had already landed**: on
`e2b-c-cars`, `broadcast_residual.pt` 15:46:54, `master.pt` 15:46:56 and
`outer_momentum.pt` 15:46:56 (UTC, same second) — the durable phase finished at
~210 s and the kill at 300 s caught `main()` in its trailing diagnostics. So the
merge *succeeded* and `master.pt` advanced to `master_version=27`, while the
watcher saw a non-zero exit, kept `pending/` and never advanced the round.

Every retry then re-loaded the same round-27 deltas, each stamped
`master_version=26` because they were diffed against the master that round 27
had just replaced. `filter_stale_deltas` **correctly** dropped all 16 as stale,
and the empty dict fell through to the weighting function with no guard. The
weighting crash was a true statement about a stale set, not a bug in weighting.
That also explains what the section called a second mystery — D's *zero* pending
checkpoints with the same error: a consequence of the same loop, not a
second defect.

The two arms failed through different weighting modes (`token` for C, `data` for
D) for the same reason, which is why "one shared step upstream of the mode split"
was the right instinct even though the located step was wrong.

**2. The watcher *did* die, ~4 h after this section was written.** "Not the
cause and was never down" was true at 21:14 UTC and false by 23:50 CEST: the
`node089` crontab was found **completely empty**, with the last tick at
21:18:15 UTC. Cause unknown — it was reinstalled and verified firing. Note the
consequence for diagnosis: a dead watcher and a crash-looping watcher produce
*the same* observable ("nothing has merged since 13:54"), and only the log's
**mtime** separates them. Check liveness against `runs/state/squeue.json`, which
every tick rewrites, not against `watcher.log`, whose ticks are mostly silent.

**3. A and B were never a separate bug.** "Relaunching without advancing" is the
correct behaviour of a run that is below quorum: the round cannot advance, so
each chunk boundary legitimately relaunches the same `start_round`. The cause
was **cluster oversubscription** — 9 live runs claiming ~126 roster slots against
~62 schedulable slots. B held a 22-node roster and could reach 3 of the 15
workers it needed. No `chunk_relaunch.py` / `start_round` defect exists.

**The fix** — `5addb16`, `Tests: 1187 passed`:

- `merge_round` recognises an already-applied round (`master_version == round_num`)
  and returns a no-op success, so the caller clears `pending/` and advances
  instead of retrying forever.
- Bootstrap sentinel `0 → BOOTSTRAP_MASTER_VERSION = -1` in `mesh/outer_step.py`,
  so a seeded master can never be mistaken for round 0's own merge.
- A fail-closed guard on any *other* all-stale set, naming both versions.
- `--tb-dir` removed from the merge subprocess. This was the budget: 
  `write_mesh_round_scalars` re-reads and `json.loads`-es **every line of every
  node's** metrics JSONL on every merge, so its cost grows with round count
  against a fixed timeout. `flush_final_tb_rounds` still writes the scalars at
  DONE, so nothing is lost.
- `MERGE_TIMEOUT_S = 900` (was 300, against a measured 210 s durable phase).

**Verified live, not just in tests:** 21:55 UTC, under cron,
`e2b-c-cars: MERGING -> SYNCING`, `SYNCING -> RUNNING (round 28/50)` — a real
merge, after 722 consecutive failures across the two arms.

**Then stopped on purpose (2026-08-29 ~00:10 UTC).** All runs marked `DONE` and
73 SLURM jobs cancelled. Final state of the four arms, all **incomplete and
unequal**:

| Arm | Pool | Round reached | Of | Status at stop |
|---|---|---|---|---|
| A | fish | 26 | 50 | DEGRADED (below quorum) |
| B | birds | 13 | 50 | DEGRADED (roster 22, ~3 alive) |
| C | cars | 28 | 50 | RUNNING, merging normally again |
| D | countries | 43 | 50 | RUNNING, merging normally again |

**These four arms are not comparable to each other as they stand** — 13 vs 43
rounds is not an ablation. Any encoder ranking from these checkpoints must
either re-run the arms to a common round or state the round each was frozen at.

**Stopping procedure, for reuse.** `scancel` alone resurrects runs:
`advance_all_runs` re-creates a fresh `RunState` (default `ASSIGNING`) for any
config lacking state, and all 50 configs stay in `mesh_run_configs.json`
permanently. Mark each run `DONE` in `watcher_state.json` *first* — that is the
proven terminal marker (11 runs have held it for days without restarting) and it
blocks both paths: `advance_all_runs`'s `while status != "DONE"` and
`chunk_relaunch.RELAUNCHABLE`. Pause cron while writing the state file, since a
tick rewrites it wholesale.

**Nine zombie runs surfaced during the stop** — carrying live state and
re-entering the state machine every tick, long after anyone was watching:
`videomae_mesh_ssl-pool-cars` / `-birds` / `-fish` (stuck `MERGING` at round 0),
`pool-cars-best-120r` (`RUNNING` at round 231), `pool-cars-phase4-fix`,
`diloco-lit-8h`, `cand2a-8h`, `videomae_mesh_ssl-diloco-cand2c` and `-cand2e`.
A run that ends by any path other than `SYNCING -> DONE` leaves its state
behind; nothing sweeps it.

---

## Backfill gaps

Known-incomplete, with the exact source for each:

- **Per-run config and arm** — `~/stern/runs/state/mesh_run_configs.json`
  is the authoritative record of experiment yaml, overrides, roster and quorum
  per run. Not mirrored locally. *Partially closed 2026-08-27:* pulled and used
  to recover the `recipe2x2` runs above. *Re-checked 2026-08-28:* it now holds
  **50 records** (a JSON **list**, not a dict), fields `run_id, experiment,
  nodes, n_shards, outer_lr, outer_momentum, weighting, rounds, merge` plus
  `quorum_k` (29), `extra_overrides` (17), `grace_s` (8), `chunk_rounds` (16).
  The `*backfill*` cells in the main ledger were filled from it by matching
  `run_id` (2026-08-28). Note it carries the launch config, **not** the
  outcome — the conclusion column was written by hand from the metrics JSONL.
  **Access:** `login-node` was refusing SSH connections on 2026-08-28. Home is
  shared NFS across hosts, so `ssh node089` reaches the identical `~/stern/` and
  `~/stern_mesh/` trees — do not block on login-node being up.
  All four E2b arms (`e2b-a-fish`, `e2b-b-birds`, `e2b-c-cars`,
  `e2b-d-countries`) **are** registered here; an earlier session's conclusion
  that B and D never launched was wrong.
  Two backfilled run IDs (`pool-cars-best-120r`, `pool-cars-phase4-fix`) are
  **not present in this file at all** — their `Config / arm` cells above are
  inferred from the metrics JSONL only (inner lr, shard count), not from the
  authoritative registry.
- ~~**Per-round loss** — metrics JSONL for runs from 2026-08-04 onward; SLURM
  stdout only for anything earlier.~~ *Corrected 2026-08-28:* the stdout-only
  caveat is obsolete. `~/stern_mesh/metrics/<run_id>/<node>.jsonl` holds **45 run
  directories** with per-round `train_loss`/`val_loss`. The main ledger's
  `*backfill*` cells covered **33** run IDs, not 29 — the 4 `xtier2-*` runs sat
  in prose form and weren't counted the first time. All 33 were recoverable
  from the metrics JSONL, zero requiring stdout archaeology. Of the 45 metrics
  dirs, exactly **one** (`smoke-paper`) has no mention anywhere in this ledger
  and no entry in `mesh_run_configs.json`: 1 node, 2 round-1 attempts, no
  merge — a throwaway smoke test, not given a row.
  **Aggregation method, and a trap that bit this backfill.** Aggregate on a
  fixed node cohort, since roster size varies within a run (`MEMORY.md` →
  mesh-run-loss-extraction) — though on these runs the fixed-cohort mean and
  the all-reporting-nodes mean differ by <0.0002, so cohort drift turned out
  **not** to be the real hazard. The real hazard is **duplicate `(node, round)`
  records**: a relaunched chunk appends a second entry for the same round, and
  averaging raw records instead of deduplicating per `(node, round)` mixes
  abandoned partial rounds into the mean. This inflated four rows above by up
  to **0.13 val_loss** before correction (`factorial-mixed-per_frame` alone had
  352 duplicate records). Dedupe last-write-wins per `(node, round)` first;
  the `factorial-*` cells are the ones to sanity-check if these numbers are
  ever recomputed.
- **`sw1`/`sw2` conclusions** — never written down beyond a memory entry.
- ~~**The two `recipe2x2` rows** — provenance is a deleted scratchpad file.~~
  *Closed 2026-08-27:* recovered from login-node; there were 7 runs, not 2.
- **Archived checkpoints** — completed runs are archived to node089
  `/Data/...stern_mesh_results/`, which is node-local, not NFS. Record the node,
  or the checkpoint is unfindable.
