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
| `recipe2x2-legacy-per_frame-20260823` | *(pre-registration)* | birds, 10 | 20 | 19 | 7–10 | 3/20 |
| `recipe2x2-legacy-time_avg-20260823` | *(pre-registration)* | fish, 14 | 22 | 21 | 11–14 | 18/22 |
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

---

## Retired code and configs

Append here **before** deleting. One row per removal.

| Removed | Commit | Last experiment that used it | Why it is safe to remove |
|---|---|---|---|
| *(none yet — the Stage 0 cleanup pass populates this)* | | | |

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
