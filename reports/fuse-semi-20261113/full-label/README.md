# FUSE full-label patient results

## Scope and evidence

This package documents the completed FUSE GPU campaign on the complete TRAIN80 labeled split: 80 training patients, all 160 original ED/ES labeled frames, and 1,988 image-only non-ED/ES cine frames. VAL18 (18 patients) and TEST50 (50 patients) are disjoint. The test transaction was committed once, on attempt 1; these metrics are already-opened TEST outcomes.

No model weights were copied or loaded for this report. The server guide `/etc/vast-agents-guide.md` and all `/etc/vast_agents/*.md` were read first. Only small JSON receipts and export manifests were fetched. Their local SHA-256 values match server `sha256sum` output; the full model artifact hashes are recorded in the receipts and verified remotely using `sha256sum` (not loaded).

## Locked selection and deployment

J selection was by best VAL18 fusion foreground Dice, ties earliest. The selected J snapshot is epoch 83 / update 6,640, with `joint_best.pt` SHA-256 `3367f4206f4d46c46f510a56b2f1bf2674347737e4018fcf06e1adf6999ae797`. (The selected epoch is in the campaign's final evidence record; per-run state records the final J epoch 100 / update 8,000.) The deployment rule was fixed at `student_VAL_FG >= fusion_VAL_FG - 0.01`. Since student = 0.897585 and fusion = 0.909016, the threshold was 0.899016 and the selected mode is **fusion**. Selected export SHA-256: `086c80a0eb6162444199b368862db2a7ca45112f9ecd51e4f3ef6f15f0365869`. The teacher inference export SHA-256 is `c55a526288192a04a7a816dce4f3f2588d812489830456b6c511e3acfbe89c07`.

The selected training teacher artifact `teacher.pt` (the source for any future Track B) has SHA-256 `bf2de8287a3fb6a5df7c490b0e081d39e58911a99cfa081243015a2c6ca894ec`; this differs from the inference export serialization.

The campaign completed T50 (8,000 labeled-frame exposures; 4,558 optimizer steps, physical batch 2 after exposure 1,100) and J100 (8,000 updates). TEST was not consulted for epoch selection, deploy selection, or any other decision. No TEST evaluation was rerun.

## Metrics

Patient-macro values are mean Dice and mean HD95 in mm from the receipts. FG is foreground mean Dice. The complete patient rows are preserved in `PATIENT_METRICS.json`; `ANALYSIS.json` carries patient-level paired results.

| Split / mode | N | FG Dice | RV Dice | RV HD95 mm | MYO Dice | MYO HD95 mm | LV Dice | LV HD95 mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VAL18 teacher | 18 | 0.906321 | 0.904167 | 4.015716 | 0.884625 | 3.175795 | 0.930170 | 3.535632 |
| VAL18 student | 18 | 0.897585 | 0.891818 | 6.326360 | 0.878702 | 3.847752 | 0.922236 | 4.068039 |
| VAL18 fusion | 18 | 0.909016 | 0.906451 | 3.981750 | 0.887407 | 3.032847 | 0.933190 | 3.256051 |
| TEST50 teacher | 50 | 0.906247 | 0.898215 | 7.613027 | 0.887557 | 3.030311 | 0.932970 | 3.596920 |
| TEST50 student | 50 | 0.907701 | 0.900933 | 4.924102 | 0.887883 | 3.057719 | 0.934288 | 3.676788 |
| TEST50 fusion | 50 | 0.908257 | 0.899690 | 7.459073 | 0.892713 | 2.935183 | 0.932367 | 3.647657 |

## Paired analysis

`ANALYSIS.json` was generated from one patient-level row per patient and mode, not frame/voxel observations. For fusion-minus-teacher and fusion-minus-student, each metric uses a two-sided paired Wilcoxon signed-rank test and a paired bootstrap percentile 95% CI for the mean patient difference (10,000 resamples, seed 42). All inference preserves patient pairing and uses only matched patients. No multiplicity adjustment was made.

The TEST50 portion is explicitly `EXPLORATORY_POST_TEST_OPEN`: it is a post hoc statistical description after the sole locked TEST transaction, not an independent confirmatory analysis. It had no role in training, checkpoint selection, deployment selection, or rerun. The VAL18 analysis is exploratory as well; the deployment rule remains the predeclared threshold above.

## Track status and limitations

New Track A, Track B, and Track B0 results are **NOT_RUN**. These cohort metrics do not establish clinical efficacy or external generalization. Patient inference timing was NOT_RUN. GPU determinism status is `BEST_EFFORT_GPU_WARN_ONLY`.

## Files

- `RESULTS.json`: portable aggregate results, provenance, selection, artifact identity, scope, and limitations.
- `PATIENT_METRICS.json`: portable patient-level VAL18 and TEST50 mode metrics with source receipt hashes.
- `ANALYSIS.json`: reusable script output with patient-level descriptive statistics and paired inference.
- `RESULT.remote.json`, `TEST_TRANSACTION.remote.json`, `STATE.remote.json`, `RUN_STATE.remote.json`: fetched small receipts.
- `teacher.pt.meta.json`, `teacher.pt.manifest.json`, `student.pt.manifest.json`, `fusion.pt.manifest.json`: provenance / export receipts only; no weights. `metrics.remote.jsonl` is the hash-verified validation history used to verify selected epochs; it contains metrics only.

Reusable future comparison command (requires the same JSON schema):

```bash
python scripts/fuse_compare_results.py --input PATIENT_METRICS.json \
  --output ANALYSIS.json --models teacher student fusion \
  --metrics FG RV.dice MYO.dice LV.dice RV.hd95_mm MYO.hd95_mm LV.hd95_mm
```
