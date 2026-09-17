# Bounded falsification results

**MEASURED: local CPU, real ACDC, compact actor. No CUDA or full-training claim.** Source/GT/gradient software checks passed, but the scientific promotion gate failed. Stage D was not launched.

## Data and fixed protocol

The `vast-gpu` machine initially exposed `/root/Self-Audit/data/ACDC/training` and `splits/acdc_patient_split_seed42.json`. The existing manifest assigns 80 training and 20 validation patients; its SHA256 is `bc42574017276a6defc668bf533330fd41a64a5e90a75ea1b8958fc9e469265a`. The local read-only copy is `/tmp/astra_event_acdc_training`. It contains the paired annotated ED/ES frames (200 image volumes and 200 GT volumes), not the 4D cine or official test set. Each image/mask hash and patient assignment is preserved in `evidence/acdc_data_identity.json`. There are **1,526 training slices and 376 validation slices from 40 complete volumes / 20 patients**.

The initially observed RTX 4060 Ti was running an existing REW job, so all new execution used local CPU, four Torch threads, `num_workers=0`. The final SSH recheck returned connection refused; current remote availability is UNKNOWN. No remote job was stopped or changed.

The locked ACDC configuration was H=W=64, physical/effective B=8/8, accumulation=1, C=16 feature channels, K=8 samples, one optional refinement, Adam LR=0.001, seeds 0/1/2. All policies share the same per-seed actor weights, actor Adam moments and example stream after 40 supervised warmup updates. RF warmup has 16 updates; the learned value head has 16 RF-twin updates. There are then **80 actor updates per policy**, making **120 total supervised optimizer updates / 960 supervised examples**. This is less than one pass through the training slices, deliberately bounded, and strongly underfit.

RF and Trigger update every four actor steps, from detached replay no older than eight steps. Uniform twin exploration probability is 0.25 per training state. Policy cap is 0.5 per batch; net-value cost is a fixed 0.001 in intrinsic score units. No validation-based threshold change, split change, M&Ms calibration, pretrained-backbone substitution or 50-epoch run occurred.

## Stage A: mechanism fixture

The phantom fixture used B=4, 32x32, seed 0 and 12 actor warmup + 12 post-warmup updates. It is explicitly synthetic. Actor and window gradients were nonzero; losses stayed finite. The learned policy made six policy audit calls over 48 post-warmup examples, with eleven explored transitions: five positive and six negative raw RF contrasts. Its final validation policy nevertheless selected zero audits. Thus exploration provided both signs without guaranteeing a stable learned trigger.

Phantom final Dice: no audit 0.6516, always 0.7313, random 0.7386, entropy 0.7269, learned 0.7092. These numbers are mechanism checks, not medical accuracy. Stage A exposed collapse, so Stage B/C was limited to a small falsification comparison to see whether the same failure persisted on actual images; it was not a gate pass authorizing a long run.

## Stage B/C: real images

All 21 bounded runs completed (seven policies x three seeds). Every final evaluation checked full volume coverage and 20 patient IDs. There were no nonfinite training losses and all recorded actor/window gradient norms were positive. Root's 16-test suite verifies gradient and GT boundaries independently. Full comparisons are in 11 and convergence limits in 12.

Final learned policy calls were **0, 4, 0 out of 376 validation slices**, respectively. The mean of patient-average audit fractions across seeds was 0.003883; it differs from the pooled slice fraction because patients are weighted equally. Both raw RF signs occurred in exploration, but the delayed value head became negatively biased relative to current counterfactual targets. Exploration alone did not prevent near-NEVER-AUDIT collapse.

The first 16 validation slices were also used only for a fixed intrinsic distillation diagnostic, with no optimization/calibration. RF MSE before/after warmup was 2.0363/1.7343, 1.9521/1.6496 and 2.0445/1.6931. Pair ranking against the intrinsic teacher was 0.683/0.692, 0.675/0.783 and 0.767/0.500. This small, patient-clustered prefix does not establish general ranking quality, and none of those targets is true Dice.

## Frozen natural transitions

These diagnostics keep actor, Auditor, starting state and random state fixed. All 376 validation slice states are processed. A few both-empty foreground states have undefined slice Dice; 375/373/373 finite contrasts remain for descriptive GT diagnostics. They are not treated as independent patients for confidence intervals.

| Learned-run seed | Patient-level RF/true-DeltaDice Spearman, n=20 | Slice-state Spearman | RF AUROC for beneficial transition | Trigger AUROC | RF-positive / true-positive state fraction |
|---|---:|---:|---:|---:|---:|
| 0 | 0.2496 | 0.2374 | 0.6494 | 0.4996 | 0.6853 / 0.3920 |
| 1 | 0.4331 | 0.2182 | 0.6440 | 0.5363 | 0.7212 / 0.4155 |
| 2 | -0.3323 | -0.0793 | 0.5087 | 0.6932 | 0.4933 / 0.2869 |

RF/Trigger AUPRC, Pearson correlation, signs, value bins and per-state/patient records are retained in JSON. RF contrast is **not DeltaDice**. Value bins compare quantities in different units only as diagnostics; no calibration slope, temperature or probability interpretation was fitted to validation GT. Natural-transition association is weak/inconsistent across seeds and does not justify a reliable relative-utility claim.

## Causal evidence acquisition tests

Entries below are means over finite slice-state contrasts, without independent-slice CIs. Delta means variant minus ordinary guided refinement; logit/coordinate effects are also stored.

| Frozen intervention | Seed 0 DeltaDice | Seed 1 | Seed 2 | Interpretation |
|---|---:|---:|---:|---|
| Zero sampled K/V source; fixed queries, coordinates and local writer path | -0.04353 | -0.13366 | -0.13990 | Reader uses sampled evidence |
| Swap sampled K/V source across batch; same query/geometry | -0.00667 | -0.01536 | -0.00968 | Evidence identity matters |
| Remove audit guidance | +0.00014 | +0.00540 | -0.00019 | No consistent useful contribution from current guide |
| Shuffle geometry across batch | -0.00123 | -0.00358 | -0.00265 | Geometry has some consequence |
| Same coordinates, different guidance | exactly 0 | exactly 0 | exactly 0 | Guidance affects output through support only |

Removing guidance changes mean normalized coordinates by 0.02289/0.02334/0.01699 and changes only 0.0855%/0.1384%/0.1303% of hard predictions. Auditor image swapping and guidance swapping have small effects (full receipts included). Batch swaps often exchange neighboring slices from the same patient; they are controlled source interventions, not a cross-patient generalization test. This proves neither that guidance is ignored in every state nor that the model learns useful audit-driven where-to-look behavior. The latter claim has failed this first test.

## One diagnostic to separate collapse from proxy validity

After observing collapse, Astra fixed one additional diagnostic in `evidence/frozen_value_probe_plan.json`: freeze seed-0 actor and Auditor, fit the **same** value head for 256 updates on 256 randomly chosen training-image twin states, retaining cost 0.001 and the same cap. GT was not passed to fitting. It had 130 positive and 126 negative net targets. Training MSE fell from 0.0001627 to 0.00001755; held-out current-target MSE fell from 0.0001868 to 0.00001460. Validation audit calls rose from 0 to 146/376.

This shows that negative bias/insufficient tracking of changing targets contributes to collapse; it does **not** isolate replay staleness from optimization count or establish a convergence fix. Extra 256 value updates are fully separate from the original 21-run table.

At this frozen realized budget, learned / random / entropy each use exactly 146 calls. Mean true slice-DeltaDice per call is **0.004741 / 0.004598 / 0.002030**, respectively. Learned false-intervention rate is 56.85%, beneficial recall 42.86%, and trigger AUROC 0.5559. The offline oracle uses at most the same per-batch cap and skips negative actions (106 calls, 0.02172 per call). These are one-seed descriptive results, not a learned-versus-random equivalence test. Removing collapse alone did not reveal a compelling learned-policy advantage.

## Reference-free score counterexamples

Semantic permutation keeps the fixed score **exactly unchanged**, while an evaluation-only oracle foreground mask loses one full unit of foreground Dice. GT constructs this counterexample only; it never trains the score or action policy.

A separate, fixed one-pixel 4-connected erosion/dilation test used oracle myocardium masks on 364 nonempty validation slices. Erosion was correctly penalized on all slices (mean myocardium Dice 0.1993), whereas **67.31% of dilated, incorrect masks received a higher RF score** than the oracle mask (mean myocardium Dice 0.6526). Fifteen of twenty patients had positive mean RF preference for dilation. Both results are reported; no corruption severity was selected to make the score fail. At 64x64, one pixel is a substantial perturbation, so this is a stress test, not a model of subtle clinical contour error.

The proxy can prefer wrong extent even with fixed image evidence. This is a concrete bottleneck for the selected energy, not proof that every possible reference-free audit signal must fail.

## What was not run

Strong pretrained A0, original-resolution accuracy/surface measures, full convergence, CUDA/4090/late-training VRAM, label-noise robustness, temporal or cross-slice objectives, external M&Ms, supervised-REW matched retraining and full Candidate-C combinations are **NOT RUN**. The observed REW job and its reported early KEEP behavior are not experimental results for this new method. No stage-D run was justified.
