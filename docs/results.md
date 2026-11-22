# Results

## Where things stand

- **Main finding: depth collapse.** In from-scratch ViT pretraining, block 1
  carries nearly all of the function. Reconstruction loss cannot see this.
- **No project encoder uses pretrained weights.** Probe rungs 3 and 4 are
  labelled reference encoders that do.
- **No from-scratch encoder has cleared the fine-tune gate** (inconclusive).
  Probe and fine-tune numbers are on **AFEW-VA**; VEATIC was used only for early arm ranking.

## 1. Why `val_loss` does not rank encoders

- Every ledger outcome is a `val_loss` statement. Qualitative internals plots
  were the only other evidence (`EXPERIMENTS.md`, "Evaluation basis").
- `val_loss` is blind to depth. Arms with very different depth profiles land
  within about 0.005 of each other (status note 2026-09-23, §1).
- Layer 1 overwrote the residual stream and layers 8 to 12 went dead, with no
  visible change in loss (`EXPERIMENTS.md`, `recipe2x2`).

## 2. DiLoCo sweeps (VoxCeleb2 reconstruction `val_loss` only)

| Campaign | Setup | Result |
|---|---|---|
| sw1 (fish, birds) | 7 rounds, lr 0.7, momentum 0.0 to 0.95, no quorum (audit) | Within about 0.02 across momentum. No winner. |
| sw2 | 15-round extension | Two arms collapsed on quorum at 11 and 5 rounds. |
| sw3 (cars) | 40 rounds, `quorum_k=8`, token weighting | lr 0.8 final 0.395; lr 0.6 final 0.403. Small effect. |
| pool-cars-best-120r | 231 rounds logged | 0.615 to 0.252. Longest run. |
| factorial, xtier2 | mask × target, 40 rounds; repeat with no round budget | `time_avg` beats `per_frame`: 0.38 to 0.41 vs 0.47 to 0.50 in xtier2. Mask type differs by at most 0.03. |
| recipe2x2 | paper vs legacy, 2 targets | Confounded with tier. Most rounds merged on timeout. |
| e2b | 4 arms, 50 rounds each | Stopped at rounds 26, 13, 28, 43. No outcome. |

**Best settings recorded:** outer lr 0.8, momentum 0.9 (matched 40 rounds),
token weighting, `quorum_k=8`, `time_avg` targets. These are the best measured
values, not proven optima.

## 3. Depth collapse

**Block-skip** (2026-09-23 status note; readout script not in this repo).
Replacing one block at a time, blocks 4 to 12 each cost at most 0.01, against a
random-init noise floor of about 0.013. Block 1 carries almost all of it.

**Spectra** (single checkpoints, fixed-seed synthetic input; `EXPERIMENTS.md`,
`recipe2x2` conditioning):

| Quantity | legacy (per_frame / time_avg) | paper (per_frame / time_avg) |
|---|---|---|
| Activation effective rank, block 1 | 220 / 219 | 13 / 18 |
| Activation effective rank, block 12 | 41 / 51 | 2.4 / 8.0 |
| Deep-layer gradient effective rank | 9 to 12 | 1.9 to 2.5 / 4 to 8 |
qkv and proj weights keep 640 to 725 of 768 dimensions: the weights are not
low-rank, the activations and gradients are.

**Attention and real video** (`EXPERIMENTS.md`). Legacy needs 24 to 27 principal
components for 90% variance at block 1; paper needs 3 to 5. Sink tokens take
0.25 to 0.35 of attention mass in paper, 0.10 to 0.14 in legacy. On VoxCeleb2
clips, attention-branch output norms span 209 to 1275 (paper) vs 8.8 to 43 (legacy).

**Depth-3 matches depth-12** on val loss and holdout MSE, and beats it on motion
R² (0.58 vs 0.53). **Fixes that did not work** (2026-09-23 note, §2): LayerNorm
Scaling, embed LN, registers, qk_norm, decoder depth 1, mask 0.75 and lr_mult 0.5
all left blocks 4 to 12 dead. `layer_scale_init=1e-4` flattened the profile only
by stopping writes to the stream (motion R² 0.53 to 0.40).

## 4. Effective-update size dose-response

Effective multiplier = `outer_lr / (1 - momentum) × lr_mult`. Phase 3b
(2026-09-21), final-block motion R²:

| Multiplier | Route | L12 motion R² |
|---|---|---|
| 0.7× | 0.7 / 0.0 | 0.64 |
| 1× | plain AdamW | 0.65 to 0.66 |
| 2× | 0.2 / 0.9 | 0.62 |
| 2.25× | 0.45 / 0.8 | 0.63 |
| 3× | plain, lr_mult 3 | 0.56 |
| 7× | DiLoCo 0.7 / 0.9 | 0.53 to 0.54 |
| 7× | plain, lr_mult 7 | 0.41 to 0.43 |
| random init | | 0.64 |

Motion R² falls as the multiplier rises. The note concludes that effective
update size, not outer momentum, is the cause. Low-dose arms avoid damage but
add no motion information over random init. Pre-registered test C1
(Spearman ≤ −0.8) has not been run. Budget is not the explanation: 231 rounds
gives the same motion R² as 3000-step runs.

## 5. Probe ladder against random init (AFEW-VA)

Frozen attentive probe, layer 6, 3 seeds, CCC, each against a control from the same job.

| Rung | Checkpoint | Layer 6 CCC | Control | Diff | Separated |
|---|---|---|---|---|---|
| 1 | from-scratch, 9-node DiLoCo ViT-B (`p3v-9n-60r-lr1-20260922`) | 0.189 ± 0.009 | 0.066 ± 0.014 | +0.123 | yes |
| 2 | overfit-50, seed-42 run (superseded) | 0.095 ± 0.038 | 0.090 ± 0.025 | +0.004 | no |
| 3 | public ViT-B, supervised UCF101 weights (`mae-b`) | 0.200 ± 0.026 | 0.097 ± 0.026 | +0.103 | yes |
| 4 | released LeVJEPA ViT-L, fine-tuned (layer 12 of 24) | 0.209 ± 0.014 | 0.070 ± 0.017 | +0.140 | yes |

- **Rungs 3 and 4 use pretrained weights** and are reference encoders only.
- Rung 4 was first defined as the released ViT-L; on 2026-10-03 it was
  re-pointed at the in-house lean LeVJEPA, which has no recorded probe.
- Rung 2's "no separation" was an artifact of its seed and short training;
  its 300-round rebuild has no probe result.
- The control is unseeded (0.066 in one job, 0.097 in another). Compare only within a job.
- Separation is a ± sd band. No checkpoint meets the paired-CI bar.
- Final block: rung 1 0.213 ± 0.026; rung 3 0.271 ± 0.029; rung 4 0.252 ± 0.018.
  Rung 4's own head, 102 clips from 48 held-out actors: CCC 0.218 (valence 0.073, arousal 0.363).

### Fine-tune gate (AFEW-VA, n=3 per arm, last epoch): INCONCLUSIVE

| Arm | CCC |
|---|---|
| Checkpoint (`p3v-9n-60r-lr1-20260922`) | −0.056 ± 0.045 |
| Random-init control | −0.030 ± 0.067 |
| Public VideoMAEv2 ceiling | 0.376 ± 0.006 |

Difference −0.026, 95% CI [−0.277, 0.225], minimum detectable effect 0.251.
The interval is too wide to call either direction.

Earlier runs read exactly 0 CCC from a saturated tanh head (fixed with an unbounded head).
The gate scored the last, overfit epoch; best-epoch CCC was 0.05 to 0.13.
**Revised gate (1 epoch × 10 seeds): not yet measured.**

## 6. LeVJEPA

- In-house LeVJEPA checkpoints collapse too, and worse: motion R² 0.27 (random
  init 0.64); one arm fell to 0.01 (2026-09-23 note, §2). No final LeVJEPA probe or gate exists.

## 7. Motion budget and motion blindness

**Motion budget** (`stern/evaluation/diagnostics/motion_budget.py`). Target variance
removed by time-averaging, VoxCeleb2 shards 000007/000029/000051: stride 1
**0.178**, stride 2 **0.230**, stride 4 **0.263**. The targets contain real
inter-frame change.

**Motion blindness** (2026-08-17 write-up; intrinsic only, no
downstream probe). On `pool-cars-best-120r`: temporal variance ratio 0.32
(trajectory 0.001, 0.218, 0.321); at a 50% budget, tube masking scores 0.184
and temporal hold-out 0.443 (2.4×). Reconstruction is 25.26 dB PSNR.

## 8. Open measurements

- **Revised fine-tune gate** (1 epoch × 10 seeds). Not yet measured.
- **Depth-collapse fixes** (2026-10-07: LayerNorm scaling, depth-aware
  init, fail-closed loader): implemented, not validated. Encoders loaded before
  2026-10-07 need re-checking, since a non-strict load can silently train
  half-random encoders.
- **Pre-registered dose-response test C1.** Not run.
- **Rung 2 (300-round rebuild), rung 4 (in-house lean LeVJEPA), and the
  2026-09-22 arm batch.** No probe result; the batch's readout has not run.
- **Parameter counts** are not recorded in the repository.
