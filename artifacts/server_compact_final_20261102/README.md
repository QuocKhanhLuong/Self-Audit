# Final S0 / S1 / S2 evaluation — 2026-11-02

Evaluation completed at 2026-11-02T15:45:12.431224+00:00. All three arms had already trained 10,000 steps. This isolated job added **zero training steps**, completed step-10,000 VAL18, selected the highest VAL class-first Macro4 across all 10 checkpoints (earliest exact tie), then froze and scored the selected raw student on TEST50. Native evaluation used batch 8 and the locked manifests.

## Results

Dice values below are percentages; exact values and all patient/class/frame records are in the JSON receipts. Foreground is the mean of RV, MYO and LV.

| Arm | Selected step | VAL18 Macro4 | VAL18 FG | TEST50 Macro4 | TEST50 FG |
|---|---:|---:|---:|---:|---:|
| S0 | 8000 | 76.5211 | 68.9852 | 73.8318 | 65.3983 |
| S1 | 9000 | 77.0113 | 69.6376 | 74.6701 | 66.5200 |
| S2 | 10000 | 72.2806 | 63.3761 | 72.6647 | 63.8553 |

| Arm / cohort | BG | RV | MYO | LV |
|---|---:|---:|---:|---:|
| S0 / VAL18 | 99.1290 | 58.3262 | 69.5427 | 79.0866 |
| S0 / TEST50 | 99.1323 | 55.7343 | 65.6835 | 74.7772 |
| S1 / VAL18 | 99.1326 | 60.9534 | 68.3395 | 79.6197 |
| S1 / TEST50 | 99.1204 | 59.9157 | 64.6357 | 75.0086 |
| S2 / VAL18 | 98.9942 | 54.0377 | 61.6084 | 74.4823 |
| S2 / TEST50 | 99.0931 | 54.7805 | 61.3936 | 75.3917 |

All three students remain below Macro4 0.87. The deployment quality gate is **NO-GO**. No matched-device efficiency benchmark was run for this publication, so it makes no faster/lighter claim. S0 is supervised crop 160; S1 and S2 are plain KD with crops 160 and 256. All use the declared ACDC L8 budget and training batch 32. These are different experimental factors, not a causal ranking.

VAL is development selection. TEST50 was already exposed in prior teacher eligibility evidence; these scores do not establish untouched confirmation. The completed evaluation did not use TEST to select checkpoints. ACDC+M&Ms joint training remains unfinished and outside this delivery.

## Selected actual weights

Each `best.pt` is an ordinary Git binary containing only the raw student, 9,662,432 bytes. The same bytes are included as `best_export/weights.pt`, next to the original pinned export manifest; no teacher or external training assets are required for reload.

- [S0/best.pt](S0/best.pt), [export manifest](S0/best_export/MANIFEST.json): SHA256 `80614aba6d36910df2f6fb9801444e94e2cf64623d6be7d9911f77fa8f02283b`.
- [S1/best.pt](S1/best.pt), [export manifest](S1/best_export/MANIFEST.json): SHA256 `41727ae03d263b223ed8e18876776e681062714a1c58dd90bc01278f0afb46b9`.
- [S2/best.pt](S2/best.pt), [export manifest](S2/best_export/MANIFEST.json): SHA256 `692ffb76b3e0fe82872aa56c38a2c876c0f4d40e24e4dcc5ac19fe5f54c675fd`.

S0 and S1 retain steps 8000 and 9000. S2 changes from the provisional step 5000 to final step 10000. The old [provisional bundle](../server_compact_20261102/) and its training logs remain unchanged. Full optimizer/resume checkpoints remain on the server at the exact paths in each `FULL_RESUME_CHECKPOINT.json`; this bundle contains deployment weights only.

## Evidence and reproduction

- [PUBLICATION.json](PUBLICATION.json): source/config/data/checkpoint hashes, exact aggregate and per-class results, and retained checkpoint paths.
- [PLAN.json](PLAN.json), [LAUNCH.json](LAUNCH.json), [STATUS.json](STATUS.json), [live.log](live.log), and [complete_evaluation.py](complete_evaluation.py): exact execution snapshot. LAUNCH is the original submission receipt, and STATUS retains `git_publication=PENDING` from before this publication. Neither receipt has been rewritten. The execution helper is archived for audit; do not rerun it.
- Per arm: `RESULT.json`, `BEST.json`, `SELECTION_LOCK_BEFORE_TEST.json`, `VALIDATION_LOG.jsonl`, `RUN_LOCK.json`, `LOCKED_CONFIG.json`, `PROVENANCE.json`, `test50/`, and `validation_history/step_*/` preserve all 10 VAL metric/freeze/recipe receipts. Selected VAL receipts may point into the original run; their exact bytes are included under `validation_history/`.
- [REMOTE_VERIFICATION.json](REMOTE_VERIFICATION.json): original receipt pins unchanged, independent highest-VAL/earliest-tie selection, exact selected/final checkpoint hashes, metric/recipe/freeze links, recomputed class and macro means, complete 18/50 patient coverage, and full SHA256 of 204 selected-VAL/TEST prediction arrays on the server. File timestamps also place selection before TEST recipe and freeze before score. This is file/receipt verification, not a fresh evaluation.
- [TRANSFER_MANIFEST.json](TRANSFER_MANIFEST.json): source paths, exact sizes and SHA256 of every transferred file. Only weights, metadata, source and logs were transferred. MRI, reference masks and prediction arrays stayed on the server.
- `runtime_source/` retains the exact five evaluation source files pinned in PLAN. S0 required the documented metadata-only geometry-provenance compatibility change; numerical prediction/metric code was unchanged.
- [LOCAL_RELOAD.json](LOCAL_RELOAD.json): all transferred files verified; three real exports strictly loaded and produced exact parity with direct `best.pt` loading on synthetic CPU input. This check is software evidence, separate from the actual server evaluation.
- [CHECKS.json](CHECKS.json), [check_repo.log](check_repo.log), [pytest.log](pytest.log): required checks using `/Users/alvinluong/miniforge3/envs/selfaudit-modal/bin/python`; **1186 passed, 2 warnings**. Warnings concern a deliberate duplicate-optimizer-parameter test and deprecated `torch.jit.script`.

Run the portable local integrity/reload check from the repository root:

```sh
python artifacts/server_compact_final_20261102/verify_exports.py
```

**Drive remains `UPLOAD_PENDING`, `remote_verified=false`.** The user authorized backup deferral. Git publication/readback is separate from Drive backup. Original run and failed-backup receipts are preserved.
