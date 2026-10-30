# Scalar Cross-Patient Transfer Utility — runnable protocol

Source of truth: `configs/transfer_utility/protocol.json`, `docs/COMPACT_STUDENT.md`,
`docs/METHOD.md`. This is a separate scalar pilot. The historical trainer/default
selector remain `original`; no joint search, prototype, student EMA or TEST tuning.

## Locked experiment

Four arms: plain KD; scalar original; scalar larger-tie; scalar cross-patient
utility. They use the same 603,956-parameter compact residual U-Net, seed 42,
widths 16/32/64/128, supervised AdamW LR 3e-4 and decay 1e-4, additive U SGD
LR 1e-3, temperature/coefficient 1, raw-teacher confidence .7, L/U batch 8/8,
crop 160. The existing compact recipe samples crops; it does not perform the
native EMA recipe's additional flips/intensity augmentation. Input is prepared
TZYX with spatial Z context, canonical BG/RV/MYO/LV, and validity masks.

Budget: 2,000 total optimizer steps, including 1,000 supervised warm-up steps.
Evaluate at 1,000/1,500/2,000; **final step 2,000 is primary**, without VAL-best
selection. Each arm has a 7,200-second watchdog including segment startup,
checkpoint export and evaluation. A timeout is INCOMPLETE; no automatic restart
or extension. The separate resource profile is 32 steps with 8 warm-up steps.
Its weights are not used to initialize pilot arms. Profile cost is reported.

Original objective is `1 - FG_Dice_support + ridge * sum(alpha**2)/2`, with
ridge 1e-6. This penalizes coefficients, not the parameter displacement norm.
The scalar vector has one identical entry per active class direction. Its grid
is 1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128. Original selects the strict minimum
among feasible grid trials, without adding zero as an objective competitor.
Larger-tie runs original first and then selects the largest feasible scale with
exactly equal FG Dice (tolerance 0). Neither modifies the guard or ridge.

Utility selects maximal KD reduction on a fixed, independent TRAIN-U probe;
probe seed 42017, batch 8, patient different from every proposal patient.
Raw-teacher confidence, accepted-pixel denominator and temperature match proposal
KD. Gain and tie tolerances are both 1e-7; zero is a competitor and wins ties by
smaller displacement. Empty/nonfinite/unavailable probes produce zero. The scalar
trial budget remains eight; the separate support and probe forward counts are
logged. The raw student is committed once, without a second optimizer step.
The frozen teacher is represented by its verified immutable soft-target cache.

Hard-Dice guard compares each represented foreground class and patient to the
post-supervised reference, and checks absent-frame FP with fixed valid-voxel
denominators. Dice budget and absent-FP budget are 0; numeric tolerance is 1e-6.
This guards only the extra U step on labeled audit support. It does not certify
query Dice, supervised updates, generalization, or teacher correctness.

## Pinned evidence and assets

| Asset | SHA-256 |
|---|---|
| TRAIN L8/U72 manifest | `e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4` |
| VAL18 manifest | `cac4426393ca11959f56df2876869e825443916cdec02c6d95ec8da3fac0ca50` |
| Recovered B8 teacher manifest | `ae36d026ce26ca188627e6ea4699695e9eb99501312ce05d5324b81e44d3ddaa` |
| Recovered B8 teacher payload | `0ae593ecb6b6f9dcb88d0a41d4a77d7b3b3a3edd87396624c3648afb1281f1b6` |
| Teacher cache manifest | `783624211e760fbeb2c3a80ded7b8dacb9d1407f0dcc1e484010cda2c343897b` |

Labeled patient union: patient007,010,011,024,026,041,077,099 (all use `patient`
prefix). Teacher inference uses the **raw student of recovered full-EMA B8**.
It is not the original historical payload, nor an EMA-state deployment.
The archive reproduces 761 of 7,028 accepted scalar updates with a larger
feasible scale and exactly equal logged support FG Dice. This is a stored-log
finding, not evidence of equal predictions or a better training trajectory.
See `reports/transfer_utility/preflight/EVIDENCE_REPORT.md` for sources and limits.

All physical source, checkpoint, receipt and cache identities must verify before
real execution. No U mask is permitted. VAL masks are opened only by the frozen
prediction evaluator. Historical TEST50 is already observed development evidence;
it is never used by this pilot selector or runner.

## Local software verification

```bash
python scripts/check_repo.py
python -m pytest -q
python scripts/verify_compact_original_parity.py --out /tmp/cptu-original-parity.json
python scripts/smoke_compact_pipeline.py --transfer-utility --device cpu --out /tmp/cptu-smoke-new
```

The parity script requires Git object `d93c4dea4f42895297ec5941bb938b47140aa9e4`.
The smoke creates only manufactured data, trains all four arms, exports/reloads,
checks inference after hiding owned teacher/cache/TRAIN assets, evaluates native
geometry, and benchmarks CPU inference. This cannot prove Dice87 or GPU speed.

## Server preparation and launch

Use an isolated checkout of this delivered branch and an idle PRO4000 24GB.
Do not alter running jobs. Keep a separate verified historical source checkout
for recovered B8 validation/benchmark. Use the actual paths of the pinned files;
the following shell variables are inputs, not guessed asset locations:

```bash
git fetch origin
git worktree add -b run/cptu-pilot-20261030 ../Self-Audit-cptu-20261030 \
  origin/QuocKhanhLuong/cptu-scalar-20261030
cd ../Self-Audit-cptu-20261030

export CPTU_TRAIN=/absolute/path/to/pinned/TRAIN_manifest.json
export CPTU_VAL=/absolute/path/to/pinned/VAL18_manifest.json
export CPTU_TEACHER=/workspace/compact-acdc-l8/20261028/b8_recovery/full/final
export CPTU_CACHE=/dev/shm/compact_teacher_b8/CACHE_MANIFEST.json
export CPTU_OUT=/workspace/cptu-pilot-new
```

Run from this repository with the environment/dependencies recorded. Preflight
is read-only: it prints JSON and does not create the proposed output directory.
Save stdout outside that directory if a receipt is desired.

```bash
python scripts/run_transfer_utility_pilot.py --mode preflight \
  --data "$CPTU_TRAIN" --data-sha256 e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4 \
  --val-manifest "$CPTU_VAL" --val-manifest-sha256 cac4426393ca11959f56df2876869e825443916cdec02c6d95ec8da3fac0ca50 \
  --teacher-checkpoint "$CPTU_TEACHER" --teacher-checkpoint-sha256 ae36d026ce26ca188627e6ea4699695e9eb99501312ce05d5324b81e44d3ddaa \
  --teacher-payload-sha256 0ae593ecb6b6f9dcb88d0a41d4a77d7b3b3a3edd87396624c3648afb1281f1b6 \
  --teacher-cache "$CPTU_CACHE" --teacher-cache-sha256 783624211e760fbeb2c3a80ded7b8dacb9d1407f0dcc1e484010cda2c343897b \
  --device cuda:0 --out "$CPTU_OUT"
```

For the authorized four-arm pilot, use the same command with `--mode all` in a
new tmux session. `all` repeats preflight, performs the bounded resource profile,
then launches the four arms sequentially, exports and evaluates each boundary.
No `--skip-preflight` or `--skip-jobs-check` bypass is allowed for real data.
Any existing output directory is refused. A process failure or missing metric
stops subsequent launches and preserves all logs/checkpoints.

## Train/resume/export/infer/evaluate

The runner materializes `runs/ARM/ARM_CONFIG.json` from the chosen config and the
actual cache's teacher identity. Config templates intentionally lack a fabricated
verified identity. It logs exact subprocess commands under each arm's `logs/`.
Each stop segment uses a fresh output directory; checkpoints store optimizer,
Python/NumPy/torch/data/probe RNG, recipe, source, teacher identity, cumulative
counters and a verified log prefix. Legacy checkpoints still export but cannot
be claimed to support exact resume when state is missing.

Examples using an already materialized config (replace paths and hashes with
values printed in that run's receipts):

```bash
python train.py --config "$CPTU_OUT/runs/scalar_guard_cross_patient_utility/ARM_CONFIG.json" \
  --data "$CPTU_TRAIN" --data-sha256 e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4 \
  --selector cross_patient_utility --steps 2000 --stop-after-step 1000 \
  --allow-full-run --device cuda:0 --out /workspace/owned-fresh-segment1

python train.py --config "$CPTU_OUT/runs/scalar_guard_cross_patient_utility/ARM_CONFIG.json" \
  --data "$CPTU_TRAIN" --data-sha256 e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4 \
  --steps 2000 --stop-after-step 1500 --allow-full-run --device cuda:0 \
  --resume /workspace/owned-fresh-segment1/final --resume-sha256 "$CPTU_CHECKPOINT_SHA" \
  --out /workspace/owned-fresh-segment2

python scripts/export_compact_student.py --checkpoint "$CPTU_CHECKPOINT" \
  --checkpoint-sha256 "$CPTU_CHECKPOINT_SHA" --out "$CPTU_EXPORT"
python scripts/infer_compact_student.py --export "$CPTU_EXPORT" --export-sha256 "$CPTU_EXPORT_SHA" \
  --manifest "$CPTU_VAL" --manifest-sha256 cac4426393ca11959f56df2876869e825443916cdec02c6d95ec8da3fac0ca50 \
  --device cuda:0 --batch-size 16 --normalization prepared --out /workspace/predictions-new
python scripts/evaluate_compact_student.py --export "$CPTU_EXPORT" --export-sha256 "$CPTU_EXPORT_SHA" \
  --manifest "$CPTU_VAL" --manifest-sha256 cac4426393ca11959f56df2876869e825443916cdec02c6d95ec8da3fac0ca50 \
  --train-manifest "$CPTU_TRAIN" --train-manifest-sha256 e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4 \
  --device cuda:0 --batch-size 16 --normalization prepared --out /workspace/evaluation-new
```

These are continuation examples, not authorization to restart a stopped server
job or run additional arms. Do not increase the immutable total budget.

## Inference benchmark and analysis

Use the exact exported artifact that was evaluated. The isolated-process
benchmark shares cases, native geometry, precision FP32, batch16, preprocessing,
warmup and I/O policy. It measures model-only and E2E latency, median/p95,
throughput, GPU allocated/reserved memory, process-tree host RAM, parameters and
weights-only bytes. It separates cold load/startup from steady-state.

```bash
python scripts/benchmark_compact_student.py \
  --compact-export "$CPTU_EXPORT" --compact-export-sha256 "$CPTU_EXPORT_SHA" \
  --manifest "$CPTU_VAL" --manifest-sha256 cac4426393ca11959f56df2876869e825443916cdec02c6d95ec8da3fac0ca50 \
  --baseline-checkpoint "$CPTU_TEACHER" --baseline-checkpoint-sha256 ae36d026ce26ca188627e6ea4699695e9eb99501312ce05d5324b81e44d3ddaa \
  --baseline-state student --historical-source "$CPTU_HISTORICAL_SOURCE" \
  --baseline-teacher-identity "$CPTU_TEACHER_IDENTITY_JSON" --baseline-teacher-identity-sha256 "$CPTU_TEACHER_IDENTITY_SHA" \
  --device cuda:0 --batch-size 16 --warmup 1 --repeats 3 --normalization prepared \
  --out /workspace/cptu-benchmark-new.json
python scripts/analyze_transfer_utility.py --pilot-dir "$CPTU_OUT"
```

The analysis writes per-class/patient metrics, pooled RV recall/precision,
alpha histogram, effective displacement, accept/reject/zero rates, KD gains,
VAL curves, costs and plots. Undefined results remain null/NOT_RUN. Equal-time
analysis uses the last measured checkpoint no later than the common training-time
horizon, without interpolation. Cold teacher training/cache generation costs
remain separately identified archive evidence; student/profile/audit costs are
measured on the new run.

## Decision and delivery status

Screening GO requires utility final VAL18 Macro4 to exceed **each** of plain KD,
original and larger-tie by **strictly more than .002**, with complete matching
cohort, source, initialization, TRAIN manifest, cache and sampling schedule.
Completed valid comparisons failing this criterion are NO_GO. Missing/incomplete
runs are BLOCKED/INCOMPLETE, not empirical NO_GO. Synthetic results cannot trigger
GO or additional patient arms.

If utility exceeds larger-tie by >.002, `all` runs the same-batch utility control
with the same 2,000-step budget. A strong claim still requires matched-displacement
controls; this runner does not launch those or joint experiments automatically.
Deployment is achieved only after the evaluated export reaches historical
Macro4 >=.87 and is demonstrably faster/smaller in all requested inference
resources than B8 on the same device/workload. Pilot/software success alone is
insufficient. See the delivery verification receipt for what actually ran.
