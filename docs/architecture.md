# Architecture

This project pretrains a video encoder on face clips without labels, then
measures how well the encoder predicts continuous valence and arousal (VA),
on a pool of GPU workstations.

Three things make it unusual:

- **Training is decentralized.** Each worker is a single-GPU process. Workers
  train on local shards and merge through a DiLoCo-style outer step once per
  round.
- **A watcher keeps runs alive.** It runs once a minute, resubmits dead jobs,
  and advances each run's round state machine.
- **Every dataset passes through shared contracts:** one manifest schema, one
  VA range, one CCC implementation.

## Pipeline

```
 raw corpora ──► manifests (CSV, one schema) ──► WebDataset shards (.tar)
                                                        │
                    Stage 1: SSL pretraining ◄──────────┘
                    (single run, or DiLoCo mesh rounds)
                                 │
                                 ▼
                    encoder checkpoint (master.pt / best.pt)
                                 │
                    Stage 2: frozen-probe or fine-tune on AFEW-VA
                                 │
                                 ▼
                    evaluation: CCC on identity-disjoint held-out actors
```

| Stage | Component | Output |
|---|---|---|
| Manifests | dataset builder scripts | one role-tagged CSV per dataset |
| Shards | `build_wds_shards.py` | `shard-*.tar` plus `manifest.json` |
| Stage 1 | `training/pretrain.py` (`run_pretrain_stage`); mesh path in `training/diloco/mesh_round.py` | encoder weights |
| Mesh merge | `scripts/run_outer_step.py` (`merge_round`) | new `master.pt` |
| Stage 2 | `training/train.py` (`run_train_stage`) | VA head, or fine-tuned encoder |
| Diagnostics | `scripts/run_diagnostics.py`, `diag_*` checks | `diagnostics.json`, report |

## Data and shared contracts

- **Manifest.** `stern/data/manifest.py` defines one CSV schema for every dataset:
  `identity_id`, `clip_path`, `duration_s`, `role`, `label_path`. The role is
  `ssl`, `probe`, or `eval`. `read_manifest` checks the role it is given, so a
  probe loader cannot silently read an SSL manifest.
- **Role separation.** SSL pretraining uses an unlabelled VoxCeleb2 subset
  (VoxMini, 218 hours). VA probing and fine-tuning use AFEW-VA (600 clips,
  240 actors). Loaders are split by role (`build_ssl_loader`,
  `build_probe_loader`, `build_eval_loader`).
- **VA harmonisation.** `stern/data/harmonise.py` maps every VA source onto
  `[-1, 1]` (`harmonise_va`) and resamples annotations to one rate
  (`harmonise_rate`). `stern/data/labels.py` reads per-frame labels through
  `harmonise_va`, so no caller sees raw ranges.
- **CCC.** `stern/evaluation/ccc.py` holds the Lin (1989) concordance correlation
  coefficient as a NumPy metric (`concordance_correlation_coefficient`) and as
  a differentiable loss (`ccc_loss`).

## Stage 1: SSL pretraining

Two objectives are implemented. They share the ViT building blocks in
`models/ViT/`.

- **Masked reconstruction (VideoMAE-style, from scratch).** 16 frames at
  224 px, tubelet 2, patch 16, mask ratio 0.9. Encoder geometry is ViT-B
  (embed 768, depth 12, 12 heads). The decoder is shallow (depth 4).
- **LeVJEPA (from scratch).** Invariance loss across global and local views,
  plus SIGReg (weight 0.02), with RoPE and block-causal attention on the shared
  blocks. SIGReg runs on each worker's local batch, not all-reduced across the
  mesh. This departs from the paper's guarantee on purpose, to avoid a per-step
  all-reduce.

The notebook path reads shards through `StreamingShardDataset`
(`data/shard_stream.py`), which fetches them from a shard server on demand.

## DiLoCo mesh

**Roster.** `stern/mesh/tiers.tsv` lists 113 nodes in five tiers:
fish 23, birds 23, cars 23, countries 19, orchids 25. Each node is one GPU with
no `torch.distributed`. The network is 1 GbE, which is why multi-node DDP was
dropped (see `results.md`).

**Placement and assignment.** Each shard is held by exactly one node. The
watcher reads holdings from `verified_holdings.json`, written only by
`validate_seeding.py` after a successful scan. `assign_static`
(`mesh/assignment.py`) rejects a shard held twice or holdings that do not cover
the expected set. The P2P fetch path that once covered missing shards never
succeeded on the cluster, and it is gone.

**Round.** Each worker trains for `steps_per_round` steps, writes a bf16
checkpoint and a metrics line. The base recipe adds a time budget
(`round_budget_s`, 750 s); paper recipes set it to null, so every node runs
the same steps.

**Merge (outer step).** `scripts/run_outer_step.py` (`merge_round`) takes each
worker's pseudo-gradient (master minus worker), averages with weights, and
applies a Nesterov step (`mesh/outer_step.py`, torch-SGD convention). Weighting
modes:

| Weighting | Weight per worker |
|---|---|
| `data` | proportional to samples trained |
| `token` | `c_tokens * (c_tokens / c_steps)`, normalised (`mesh/quorum.py`) |
| `inverse_loss` | inverse of validation loss |

Stale deltas (wrong master version) are dropped by `filter_stale_deltas`;
non-finite ones by `filter_nonfinite`. The sw3 and pool-cars-full runs used `token`
weighting. Default outer lr and momentum are 0.7 and 0.9.

**Quorum and timeouts.** With `quorum_k` set, a merge happens once that many
nodes check in. `grace_s` is the fallback deadline. `effective_quorum_k` lowers
the quorum for missing nodes, never below 60% of the pool (`mesh_tick.py`).
Round and merge timeouts are both 900 s. The merge timeout was 300 s until a
merge killed at 300 s was found to have finished its writes (`5addb16`).

**Checkpoint flow.** Workers push to `pending/`, which a merge clears only on
success. `master.pt` is overwritten each merge, so a finished run keeps only
its final master. On DONE, checkpoints move to a storage node.

**Round state machine** (`watcher/mesh_tick.py`): ASSIGNING -> RUNNING ->
AWAITING_CHECKPOINTS -> MERGING -> SYNCING, then back to RUNNING for the next
round or to DONE. A run that ends by any other path leaves its state behind.

## Watcher

- **Cadence.** `scripts/watch_training.py` runs once a minute. Most ticks exit
  early; a full check runs when the backoff timer is due.
- **Detectors** (`watcher/detectors.py`, pure): dead, stalled, foreign GPU
  process, quota, drained node, NaN.
- **Remediation.** A dead job is resubmitted once, on the same node list. A
  second death or a stalled job goes to escalation: a headless model call with
  a cooldown and a daily budget, whose transcript is written before it acts.
  Quota pressure cleans up files from DONE runs (`watcher/quota_cleanup.py`).
- **Backoff.** An action resets the interval to 60 s; otherwise it grows 1.5x
  per tick, to 1800 s.
- **Mesh runs.** `mesh_tick.py` owns mesh-job recovery. Only the host in
  `runs/state/watcher_host` may tick.

## Class-hours gate

No cluster job may run Monday to Friday, 08:00 to 20:00, Europe/Paris; the
administrators of these shared machines cancel jobs during teaching hours.
`scripts/class_hours.py` refuses a job if any part of `[now, now + walltime]`
overlaps those hours. With the 2-hour walltime cap,
weekday launches are allowed from 20:00 to 06:00; weekends are open. There is
no override. Enforcement sits in three places: the job scripts, a shim in
front of `sbatch`/`srun`/`salloc` on the shared home, and the watcher, which
suppresses resubmission during class hours. Added 2026-10-07 (`8882f2a`).
