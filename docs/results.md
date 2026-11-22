# Results

## Where things stand

- **Update size.** DiLoCo's default outer step (0.7 / 0.9) is an effective 7x
  update. At 7x the stream norm grows and motion information falls below its
  random-init level. At 1x it does not (section 4).
- **Depth collapse.** In from-scratch ViT-B pretraining, block 1 carries nearly
  all of the function, at every update size. Reconstruction loss cannot see
  this (section 3). Still open.
- **Fine-tune gate.** A from-scratch lean LeVJEPA encoder trained at 1x beats
  random init by +0.031 [+0.005, +0.057] at round 48. At round 78 the
  difference is unresolved. One training run. The ViT-B encoders have only the
  earlier 3-seed gate, which was inconclusive (section 5).
- **Continued pretraining** of the released LeVJEPA ViT-L raises fine-tuned CCC
  from 0.287 to 0.331 over 30 seeds (section 7).
- **The project's own encoders are trained from scratch.** Probe rungs 3 and 4
  and the continued-pretraining arms start from released weights and are labelled.
- Probe and fine-tune numbers are on **AFEW-VA**; VEATIC was used only for early arm ranking.

**Sources.** Each number below names its table in `docs/data/` where one
exists. The sweep table in section 2, the spectra and attention numbers in
section 3, the depth-3 comparison, the September LeVJEPA numbers in section 6
and the motion-blindness numbers in section 8 come from project notes, not in
the repository.

## 1. Why `val_loss` does not rank encoders

- Every sweep outcome in section 2 is a `val_loss` statement. Qualitative
  internals plots were the only other evidence.
- `val_loss` does not follow what the encoder keeps. Among five 20-round arms,
  the lowest loss (3x, 0.514) goes with the lowest clip-split motion R² (−0.13)
  (`loss_vs_motion.csv`). Five points: a description, not a test.
- Layer 1 overwrote the residual stream and layers 8 to 12 went dead, with no
  visible change in loss (`recipe2x2`).

## 2. DiLoCo sweeps (VoxCeleb2 reconstruction `val_loss` only)

| Campaign | Setup | Result |
|---|---|---|
| sw1 (fish, birds) | 7 rounds, lr 0.7, momentum 0.0 to 0.95, no quorum (audit) | Within about 0.02 across momentum. No winner. |
| sw2 | 15-round extension | Two arms collapsed on quorum at 11 and 5 rounds. |
| sw3 (cars) | 40 rounds, `quorum_k=8`, token weighting | lr 0.8 final 0.395; lr 0.6 final 0.403. Small effect. |
| pool-cars-best-120r | 16 nodes, 230 rounds logged | 0.615 to 0.252. Longest run (`loss_curves.csv`). |
| factorial, xtier2 | mask × target, 40 rounds; repeat with no round budget | `time_avg` beats `per_frame`: 0.38 to 0.41 vs 0.47 to 0.50 in xtier2. Mask type differs by at most 0.03. |
| recipe2x2 | paper vs legacy, 2 targets | Confounded with tier. Most rounds merged on timeout. |
| e2b | 4 arms, 50 rounds each | Stopped at rounds 26, 13, 28, 43. No outcome. |

**Best settings recorded:** outer lr 0.8, momentum 0.9 (matched 40 rounds),
token weighting, `quorum_k=8`, `time_avg` targets. These are the best measured
values, not proven optima.

## 3. Depth collapse

**Block-skip** (`depth_profile.csv`, column `skip_cost_mse`). Replace one
block with the identity and measure the rise in held-out reconstruction error.
Ten encoders, 20 rounds on one node, 0.7x to 7x: block 1 costs 0.26 to 1.12,
blocks 4 to 12 at most 0.017 each. Random init varies within ±0.013. After 100
rounds on 16 nodes at 7x, block 2 contributes too (block 1 0.42, block 2 0.16)
and blocks 4 to 12 still do not (at most 0.014).

**Spectra** (single checkpoints, fixed-seed synthetic input; `recipe2x2`
conditioning):

| Quantity | legacy (per_frame / time_avg) | paper (per_frame / time_avg) |
|---|---|---|
| Activation effective rank, block 1 | 220 / 219 | 13 / 18 |
| Activation effective rank, block 12 | 41 / 51 | 2.4 / 8.0 |
| Deep-layer gradient effective rank | 9 to 12 | 1.9 to 2.5 / 4 to 8 |
qkv and proj weights keep 640 to 725 of 768 dimensions: the weights are not
low-rank, the activations and gradients are.

**Attention and real video.** Legacy needs 24 to 27 principal
components for 90% variance at block 1; paper needs 3 to 5. Sink tokens take
0.25 to 0.35 of attention mass in paper, 0.10 to 0.14 in legacy. On VoxCeleb2
clips, attention-branch output norms span 209 to 1275 (paper) vs 8.8 to 43 (legacy).

**Depth-3 matches depth-12** on val loss and holdout MSE, and beats it on motion
R² (0.58 vs 0.53).

**Recipe variants that did not work** (`depth_profile.csv`, group `fix`; 7x
recipe, 20 rounds, one node, one seed each). LayerNorm scaling, embedding
LayerNorm, both together, 4 register tokens, qk norm, decoder depth 1, mask
0.75, lr_mult 0.5 and deep supervision all left blocks 4 to 12 unused (at most
0.019 per block). `layer_scale_init=1e-4` flattened the profile only by
stopping writes to the stream: norm 20 after block 12, and token-split motion
R² 0.45 against 0.53 to 0.54 for the plain 7x recipe (`motion_by_layer.csv`).

## 4. Effective-update size dose-response

Effective multiplier = `outer_lr / (1 - momentum) × lr_mult`. Phase 3b
(2026-09-21), 20 rounds on one node. Motion R² is a linear readout from frozen
block-12 tokens to per-patch frame change, with train and test split by token
(`dose_response.csv`). Stream norm is after block 12 (`depth_profile.csv`).

| Multiplier | Route | L12 motion R² (token split) | Stream norm |
|---|---|---|---|
| 0.7× | 0.7 / 0.0 | 0.64 | 91 |
| 1× | plain AdamW (2 seeds) | 0.65, 0.66 | 92, 95 |
| 2× | 0.2 / 0.9 | 0.62 | 193 |
| 2.25× | 0.45 / 0.8 | 0.63 | 162 |
| 3× | plain, lr_mult 3 | 0.56 | 137 |
| 7× | DiLoCo 0.7 / 0.9 (2 seeds) | 0.53, 0.54 | 373, 375 |
| 7× | plain, lr_mult 7 (2 seeds) | 0.41, 0.43 | 459, 406 |
| random init | | 0.64 | 45 |

Motion R² falls and the norm grows as the multiplier rises. Plain AdamW at 7x
is hit harder than DiLoCo at 7x, so the cause is update size, not outer
momentum. Low-dose arms avoid damage but add no motion information over random
init. Budget is not the explanation: the 230-round run gives the same motion
R² as 3000-step runs.

**Clip-split readout** (`motion_by_layer.csv`, readout `window_grouped`;
`loss_vs_motion.csv`). The token split puts tokens of one clip on both sides.
A stricter readout splits by clip window. It was run on a second batch of
20-round encoders, one seed each, 48 windows. Block-12 motion R²: 0.33 at 0.7x,
0.24 at 1x, 0.21 and 0.14 at 2x, −0.13 at 3x. Random init: 0.55. In every
trained encoder motion R² is highest at block 1 (0.46 to 0.59) and falls with
depth. The 9-node, 60-round 1x encoder (rung 1 of section 5) holds 0.65 at
block 1 and 0.49 at block 12.

**Pre-registered test C1** (Spearman ρ between multiplier and block-12
clip-split motion R² at most −0.8, with p < 0.05). Run on the saved
out-of-fold arrays with the project's claims script (not in the repository).
Doses [0.7, 1, 2, 3]: ρ = −0.80, p = 0.17. **FAIL.** Four doses are too few
for the threshold. The other pre-registered claims are incomplete: C2 to C6
because their checkpoints were deleted, C7 because only one of its pairs
survives.

## 5. Probe ladder against random init (AFEW-VA)

Frozen attentive probe, layer 6, 3 seeds, CCC, each against a control from the same job.

| Rung | Checkpoint | Layer 6 CCC | Control | Diff | Separated |
|---|---|---|---|---|---|
| 1 | from-scratch, 9-node DiLoCo ViT-B (`p3v-9n-60r-lr1-20260922`) | 0.189 ± 0.009 | 0.066 ± 0.014 | +0.123 | yes |
| 2 | overfit-50, seed-42 run (superseded) | 0.095 ± 0.038 | 0.090 ± 0.025 | +0.004 | no |
| 3 | public VideoMAE V2 ViT-B, MAE-pretrained then fine-tuned on UCF101 (`mae-b`) | 0.200 ± 0.026 | 0.097 ± 0.026 | +0.103 | yes |
| 4 | released LeVJEPA ViT-L, fine-tuned (layer 12 of 24) | 0.209 ± 0.014 | 0.070 ± 0.017 | +0.140 | yes |

- **Rungs 3 and 4 use pretrained weights** and are reference encoders only.
- Rung 4 was first defined as the released ViT-L; on 2026-10-03 it was
  re-pointed at the in-house lean LeVJEPA, which has no recorded probe.
- Rung 2's "no separation" was an artifact of its seed and short training;
  its 300-round rebuild has no probe result.
- The control is unseeded (0.066 in one job, 0.097 in another). Compare only within a job.
- Separation is a ± sd band, not a paired interval.
- Final block: rung 1 0.213 ± 0.026; rung 3 0.271 ± 0.029; rung 4 0.252 ± 0.018.
  Rung 4's own head, 102 clips from 48 held-out actors: CCC 0.218 (valence 0.073, arousal 0.363).

### First fine-tune gate (AFEW-VA, ViT-B, n=3 per arm, last epoch): INCONCLUSIVE

| Arm | CCC |
|---|---|
| Checkpoint (`p3v-9n-60r-lr1-20260922`) | −0.056 ± 0.045 |
| Random-init control | −0.030 ± 0.067 |
| Public VideoMAEv2 ceiling | 0.376 ± 0.006 |

Difference −0.026, 95% CI [−0.277, 0.225], minimum detectable effect 0.251.
The interval is too wide to call either direction.

Earlier runs read exactly 0 CCC from a saturated tanh head (fixed with an unbounded head).
The gate scored the last, overfit epoch; best-epoch CCC was 0.05 to 0.13.
This led to the revised gate below.

### Paired gate (AFEW-VA, 1 epoch × 10 seeds)

Fine-tune the encoder and a random init with the same 10 seeds and report the
paired CCC difference with a 95% interval
(`stern/evaluation/probes/finetune_gate.py`). The encoder is a from-scratch
lean LeVJEPA (depth 7), 14 nodes, DiLoCo at 0.1 / 0.9 (1x). All rows are in
`gate.csv`.

| Round | Encoder CCC | Random init | Difference [95% CI] | Verdict |
|---|---|---|---|---|
| 48 | 0.089 | 0.058 | +0.031 [+0.005, +0.057] | PASS |
| 78 | 0.069 | 0.058 | +0.011 [−0.026, +0.048] | UNRESOLVED |

- **Seeding fix (2026-11-01).** Before it, the gate did not apply
  `train.seed`, so seeds within one job could repeat each other and intervals
  were too narrow. The two rows above were re-run after the fix.
- **Before the fix** (`seeding_fix = before`; point estimates only): the
  difference rises from +0.002 at round 1 to +0.033 to +0.041 at rounds 18 to
  48, and is +0.024 at round 78. Controls: random init against random init
  +0.004; released ViT-L +0.26.
- It is one training run. The ViT-B encoders have not been through this gate.

## 6. LeVJEPA

- In September, in-house LeVJEPA checkpoints collapsed too, and worse: motion
  R² 0.27 (random init 0.64); one arm fell to 0.01.
- **Rank collapse and its fix** (`collapse_fix.csv`, final token, 30 rounds).
  With SIGReg on batches of 16, effective rank falls from 2.7 at init to 1.2
  after one round and stays at 1.2 to 1.6. A speaker-identity probe falls from
  0.168 to 0.12. With SIGReg pooled over a batch of 256, rank reaches 10.4 and
  speaker accuracy 0.21, on both seeds.
- The rank fix alone did not help VA: on the gate these encoders scored about
  0.03 below random init (before the seeding fix, point estimate, project notes).
- The gated lean LeVJEPA run is in section 5. It has no frozen-probe result.

## 7. Continued pretraining of a released model

Start from the released LeVJEPA ViT-L, continue SSL pretraining on the same
unlabelled face video, then fine-tune on AFEW-VA. 30 seeds per arm, one job per
seed (`continued_pretraining.csv`).

| Arm (n = 30) | CCC | Valence | Arousal |
|---|---|---|---|
| Random init | 0.054 | 0.090 | 0.018 |
| Released ViT-L | 0.287 | 0.090 | 0.483 |
| Released ViT-L + continued pretraining | 0.331 | 0.176 | 0.487 |

Paired gain +0.045, 95% CI [+0.036, +0.053]; 29 of 30 seeds improve. The gain
is in valence. One job per seed, so the gate's seeding bug does not apply.
Scope: 48 held-out actors, 102 clips, one identity split.

## 8. Motion budget and motion blindness

**Motion budget** (`stern/evaluation/diagnostics/motion_budget.py`). Target variance
removed by time-averaging, VoxCeleb2 shards 000007/000029/000051: stride 1
**0.178**, stride 2 **0.230**, stride 4 **0.263**. The targets contain real
inter-frame change.

**Motion blindness** (intrinsic only, no downstream probe). On `pool-cars-best-120r`: temporal variance ratio 0.32
(trajectory 0.001, 0.218, 0.321); at a 50% budget, tube masking scores 0.184
and temporal hold-out 0.443 (2.4×). Reconstruction is 25.26 dB PSNR.

## 9. Open measurements

- **Deep blocks.** Blocks 4 to 12 of the from-scratch ViT-B stay unused at
  every update size and under every variant in section 3.
- **Depth-collapse fixes** (2026-10-07: LayerNorm scaling, depth-aware
  init, fail-closed loader): implemented, not validated. LayerNorm scaling was
  already run once as a variant of the 7x recipe and did not revive blocks 4 to
  12 (section 3). It stays in the code as an option (`layernorm_scaling`, off
  by default). No run has used the 2026-10-07 changes together. Encoders loaded
  before 2026-10-07 need re-checking, since a non-strict load can silently
  train half-random encoders (`check_encoder_arch` in
  `stern/training/checkpoint.py` now refuses a geometry mismatch).
- **Paired gate.** One training run. Unresolved at round 78. Not run on ViT-B.
- **Pre-registered claims C2 to C7.** Incomplete (section 4). C1 failed.
- **Rung 2 (300-round rebuild) and rung 4 (in-house lean LeVJEPA).** No
  frozen-probe result.
- **Parameter counts** are not recorded in the repository.
