# Experiment log

Condensed record of what was run, when, and what it showed. Numbers and
interpretation are in `results.md`. Dates are run or commit dates. Campaigns
are listed in the order they were run.

## Evaluation basis

**Read this before any outcome below.** Every mesh run dated on or before
2026-08-28 was scored on VoxCeleb2 reconstruction `val_loss` and on qualitative
internals plots, and on nothing else. No downstream VA label set was available
to those runs, so no encoder was ranked on VA during that period.

From 2026-08-28, VEATIC was staged and used for arm ranking only. Its 124 clips
and 257,601 labelled frames were checked for frame-label agreement (0 mismatches). No VEATIC
result is a headline result, and none is reported in `results.md`.

From 2026-09-10, AFEW-VA (600 clips, 240 actors) became the probe dataset. Every
probe and fine-tune result below uses AFEW-VA, scored on a held-out actor split.

Reconstruction loss did not rank depth. The depth-collapse finding was invisible
in it, so any `val_loss` ranking should be read as provisional.

## Campaigns

| Campaign | Dates | Question | Outcome |
|---|---|---|---|
| Single-node and small-pool profiling (24 rows in `docs/data/throughput_sweep.csv`) | 2026-07-10 to 07-11 | Batch, accumulation, precision, and node count vs throughput | Single node beats 8-node DDP: 13.4 samples/s at batch 1 and 24.0 at batch 4 (bf16). 8-node scaling efficiency 0.027. Motivated DiLoCo. |
| Mesh bring-up (`step0-*`, `pool-cars-2r` to `phase4-fix2`) | 2026-07-28 to 08-08 | Does the merge loop run end to end? | Infrastructure check. No science. |
| `sw1` outer-optimizer grid (fish, birds) | 2026-07-28 to 07-29 | Outer lr and momentum, 7 rounds, no quorum | Final val_loss within about 0.02 across momentum. No winner. `sw1-fish-lr050m090` had one contributor and is excluded. |
| `sw2` extension | 2026-07-30 | Same arms at 15 rounds | Two arms collapsed on quorum at 11 and 5 rounds. Not comparable. |
| Quorum audit | 2026-08-07 | Where does mesh time go? | Quorum off for all 21 registered runs. 19.1% of mesh wall-clock was idle, per the `pacing.py` note. A possible 3.4-day watcher gap was flagged and not confirmed. |
| `pool-cars-*` scale-up (full-50r, full-100r, best-120r) | 2026-08-04 to 08-17 | Scale-up on the cars tier (full runs: token weighting, `quorum_k=8`) | `pool-cars-best-120r` ran 231 rounds (0.615 to 0.252), the longest run. It is the reference checkpoint. Its name undercounts rounds. `full-100r` ran 40, not 100. |
| `sw3-cars` | 2026-08-11 to 08-13 | Outer lr (0.6, 0.8) and single-knob arms, 20 to 40 rounds | At 40 rounds, lr 0.8 final 0.395 vs lr 0.6 final 0.403. Single-knob arms ran too few rounds to compare. |
| Motion budget and motion-blindness program | 2026-08-17 to 08-19 | Does time-averaging remove target motion? Is the best checkpoint motion-blind? | Time-averaging removes 0.178 / 0.230 / 0.263 of target variance at strides 1 / 2 / 4. `pool-cars-best-120r` is motion-blind on intrinsic measures. No downstream probe run. |
| `factorial` (mask × target) and `xtier2` repeat | 2026-08-22 | Which mesh axis matters? | `time_avg` beats `per_frame` (0.38 to 0.41 vs 0.47 to 0.50 in xtier2). Mask type is a small effect. Early rounds ignored mask type. One arm fell to one contributor. |
| `recipe2x2` (legacy and paper recipes, 2 targets) | 2026-08-23 to 08-26 | Does the paper-faithful recipe help? | The paper recipe fixed positional collapse and introduced depth collapse. The four cells ran on four tiers, and most rounds merged on timeout. The ranking between arms is not reliable. |
| `recipe2x2` conditioning, PCA, and real-video diagnostics | 2026-08-28 | Where does the collapse live? | Weights stay high-rank. Activations and gradients collapse, starting at block 1. Attention sinks sharpen with depth. Real-video norms confirm it. Single checkpoints only. |
| `e2b` depth-collapse ablation (pre-registered 2026-08-27) | launched 2026-08-28, stopped 2026-08-29 | Does a visible-loss weight, or deep supervision at layers 4, 8, 12, keep the stream alive? | Stopped at rounds 26 (fish), 13 (birds), 28 (cars), 43 (countries) of 50. Rosters were unequal and cluster oversubscribed. No outcome. |
| `e2b` merge fault | 2026-08-28 | Why did merges fail? | A 300 s subprocess timeout killed each merge about 90 s after its writes finished. The retry loop found stale deltas and failed 722 times. Fixed by a no-op for already-applied rounds and a 900 s timeout. |
| LeVJEPA objective and encoder | 2026-09-01 to 09-30 | Does a second SSL objective avoid collapse? | In-house checkpoints collapsed more. The mesh loss path still called the VideoMAE step at archive time (2026-09-30). |
| Attentive-probe ladder (AFEW-VA) | 2026-09-26 to 09-27 | CCC of frozen probes vs random init | Rung 1 separates from its control at layer 6 by a ± sd band (+0.123). Rung 2 does not (superseded run). Rungs 3 and 4 use pretrained weights and are reference rungs. |
| Dose-response, Phase 3b | 2026-09-21 | Does effective update size drive motion R²? | Motion R² falls with the multiplier. Low doses avoid damage but match random init. The pre-registered test was not run. |
| `p3v` seven-arm batch | launched 2026-09-22 | Pre-registered claims C1 and C7 | Training finished. The readout pass was not run, so no claim was tested. |
| AFEW-VA fine-tune gate | 2026-09-30 to 10-02 | End-to-end check of the from-scratch checkpoint vs random init | Early runs read exactly 0 CCC from a saturated head (fixed with an unbounded head). Final verdict INCONCLUSIVE: −0.056 vs −0.030, CI95 [−0.277, 0.225]. Revised 1-epoch, 10-seed gate: not yet run. |
| Ladder rebuild | 2026-10-03 to 10-04 | Make rung 2 and rung 4 comparable | Rung 2 rebuilt as 300 rounds from seed 0, as rung 1. Rung 4 re-pointed at the in-house lean LeVJEPA. Neither has a probe result. |
| Depth-collapse repair | 2026-10-07 | Fix blocks 4 to 12 | LayerNorm scaling, depth-aware init, and a fail-closed loader were committed. Not yet validated. |

## Known data problems

- **Duplicate records.** A relaunched chunk writes a second record for the same
  round. Averaging raw records mixed partial rounds into some means and inflated
  four rows by up to 0.13 val_loss. Recomputed values must dedupe per (node,
  round), last write wins.
- **Quorum collapse.** Several merges used fewer contributors than the roster
  (`factorial-tube-time_avg` fell to one, `recipe2x2-carscmp` merged on timeout in 90
  of 100 rounds). Their per-round averages are not full-cohort values.
- **Tier confounds.** Tiers differ by 1.47× in throughput. Arms on separate tiers
  without a round budget do not isolate the arm. Paper recipes use
  `round_budget_s: null` so that step count, not time, ends a round.
- **Run names.** `depthprobe-lr025` was registered at outer lr 0.7 and never
  completed a merge round. It has no data.
- **Checkpoints.** Completed runs keep only their final master. Per-round
  metrics exist, but no checkpoint trajectory does.

## Not in this log

Parameter counts are not recorded anywhere in the repository. The node-scaling
sweep was designed but never run.
