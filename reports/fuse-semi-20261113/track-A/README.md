# Track A — semi K8 results

## Run and provenance

Track A is the semi-supervised K8 recipe: only the locked eight ACDC TRAIN
patients (16 ED/ES masks) were labeled; the remaining TRAIN cine frames were
image-only U. It is distinct from the full-label comparison, which used all
TRAIN80 labels. VAL18 and TEST50 were unchanged and disjoint.

- Server: `vast-gpu`
- Run: `/root/mount_data/runs/fuse-acdc-k8-20261113`
- Pinned source commit: `32e528e39b79493b8e2251bd884d7a2ed4583748` (remote
  checkout verified clean at completion)
- Recipe SHA-256: `d7af9576751eb7409093a8d7a5685d65c50baf088b168b289c83cf4b4d1bf8a8`
- Completion: 2026-11-13 17:53:56 UTC; T, J, VAL3, EXPORT, TEST_ONCE completed
- Training: T 8,000 labeled-frame exposures / 4,000 optimizer steps; J 100
  epochs / 8,000 updates
- TEST ledger: `COMMITTED`, attempt 1. TEST was not rerun.
- Final run state: `COMPLETED`, `error: null`; both named A supervisors exited,
  and the GPU was idle at inspection.

The standalone selected J checkpoint was saved at epoch 69 / update 5,520 with
SHA-256 `2e942ef3cbd7a8d5f55b37ec077d839518926769f25a65b5c7b38cc9676bd378`.
The VAL18 deploy rule was `student_VAL_FG >= fusion_VAL_FG - 0.01`:
student FG was 0.824066 and fusion FG was 0.825663, so deployment selected the
student export. Its SHA-256 is
`f7ae3a1473c83db4e09b473db1ea5e743a368c6cce422e6dbbbe248581a00beb`.
Teacher and fusion export SHA-256 values are recorded in their fetched export
manifests. No weights were fetched.

## VAL18 comparison

Values are patient-macro mean Dice; HD95 is in millimeters. Rows are from the
completed Track A receipts and the existing all-TRAIN80 full-label report.

| Recipe | Mode | N | FG Dice | RV Dice | RV HD95 mm | MYO Dice | MYO HD95 mm | LV Dice | LV HD95 mm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full-label allTRAIN80 | teacher | 18 | 0.906321 | 0.904167 | 4.015716 | 0.884625 | 3.175795 | 0.930170 | 3.535632 |
| Full-label allTRAIN80 | student | 18 | 0.897585 | 0.891818 | 6.326360 | 0.878702 | 3.847752 | 0.922236 | 4.068039 |
| Full-label allTRAIN80 | fusion | 18 | 0.909016 | 0.906451 | 3.981750 | 0.887407 | 3.032847 | 0.933190 | 3.256051 |
| Track A semi K8 | teacher | 18 | 0.803583 | 0.718061 | 74.501291 | 0.810031 | 9.450931 | 0.882658 | 10.132171 |
| Track A semi K8 | student | 18 | 0.824066 | 0.772582 | 28.112602 | 0.811491 | 10.751369 | 0.888126 | 6.279868 |
| Track A semi K8 | fusion | 18 | 0.825663 | 0.758075 | 41.590135 | 0.822111 | 9.229105 | 0.896802 | 4.503012 |

## TEST50 comparison

TEST was opened only by the locked one-shot transaction after training,
checkpoint selection, and VAL-based deployment choice. All TEST metrics and
statistics here are **exploratory post-test descriptions**; they were not used
for tuning or model selection.

| Recipe | Mode | N | FG Dice | RV Dice | RV HD95 mm | MYO Dice | MYO HD95 mm | LV Dice | LV HD95 mm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full-label allTRAIN80 | teacher | 50 | 0.906247 | 0.898215 | 7.613027 | 0.887557 | 3.030311 | 0.932970 | 3.596920 |
| Full-label allTRAIN80 | student | 50 | 0.907701 | 0.900933 | 4.924102 | 0.887883 | 3.057719 | 0.934288 | 3.676788 |
| Full-label allTRAIN80 | fusion | 50 | 0.908257 | 0.899690 | 7.459073 | 0.892713 | 2.935183 | 0.932367 | 3.647657 |
| Track A semi K8 | teacher | 50 | 0.738143 | 0.670056 | 68.330658 | 0.732728 | 22.221243 | 0.811645 | 21.488436 |
| Track A semi K8 | student | 50 | 0.772627 | 0.720166 | 38.904382 | 0.752598 | 22.471659 | 0.845116 | 17.244338 |
| Track A semi K8 | fusion | 50 | 0.764073 | 0.707491 | 59.868191 | 0.757616 | 19.202278 | 0.827112 | 19.203290 |

## Patient-paired analysis

`scripts/fuse_compare_results.py` analyzed stored per-patient rows for FG,
RV/MYO/LV Dice, and RV/MYO/LV HD95, using paired two-sided Wilcoxon signed-rank
tests and 10,000 paired patient bootstrap resamples (seed 42). The complete
results for both VAL18 and TEST50 are in `ANALYSIS.json`; the actual source
rows are in `PATIENT_METRICS.json`. The analysis status is
`EXPLORATORY_POST_TEST_OPEN`.

For foreground Dice, fusion-minus-model mean differences are in percentage
points below. Confidence intervals are paired bootstrap 95% percentile CIs.

| Split | Contrast | Mean difference, pp | 95% CI, pp | Wilcoxon p | N |
|---|---|---:|---:|---:|---:|
| VAL18 | fusion − teacher | +2.208 | [+1.238, +3.261] | 0.000328 | 18 |
| VAL18 | fusion − student | +0.160 | [−1.264, +1.537] | 0.324730 | 18 |
| TEST50 | fusion − teacher | +2.593 | [+1.942, +3.302] | 7.656e-12 | 50 |
| TEST50 | fusion − student | −0.855 | [−1.979, +0.230] | 0.204476 | 50 |

The test and bootstrap CI summarize different features of the paired
distribution. The TEST50 fusion-minus-student mean CI includes zero. No
multiplicity adjustment was made. The previously opened full-label TEST
analysis remains in `../full-label/ANALYSIS.json` and is a separate
all-TRAIN80 result.

These are descriptive comparisons of two training recipes on the same cohort;
they do not establish clinical efficacy or causal benefit from a single
component.

## Track B status in transfer receipts

The pinned M&Ms geometry policy `mnms_open345_mm_polar_v1` passed and target L8
is locked. The current transfer PLAN (`observed_utc` 2026-11-13 18:03:00 UTC)
records B as `FAILED` during `PROFILE_TJ`, before a profile receipt was
produced. Its exact error is `ValueError: source mask contains fractional
labels`, raised at `self_audit/candidate_fuse/mnms_data.py:525` during zero-shot
teacher evaluation on development VAL data. B target training and TEST are
`NOT_RUN`; B0 is optional and `NOT_RUN`. The 15:42:23 UTC launch receipt is an
older snapshot showing the queue waiting for A. This A-only report did not
launch or modify B.

## Fetched receipts

`RESULT.json`, `RUN_STATE.json`, `STATE.json`, the committed
`TEST_ONCE/TEST_TRANSACTION.json`, `joint_best.sha256.json`, and all three
export manifests were copied from the pinned run. Their downloaded bytes match
the server `sha256sum` output in `RECEIPTS.remote.sha256`. The report contains
no model weights or new large uploads.
