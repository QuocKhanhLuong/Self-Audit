# ACDC measured results — stop after EMA-block

EMA-block completed and scored; remaining matrix runs cancelled by user.

Protocol: dense labeled TRAIN budget8 (audit included), subset seed42, initialization seed42, random crop256, ordinary batch52, FP32, 10,000 steps, fixed final student checkpoint. EMA is not independent truth.

All scores below are patient-macro Dice on development-exposed VAL18 (%). No independent confirmation or clinical efficacy claim.

| Method | RV | MYO | LV | Timed hours | Accepted pseudo updates |
|---|---:|---:|---:|---:|---:|
| supervised | 60.75 | 77.13 | 88.54 | 2.58 | 0 |
| supervised_extra | 64.19 | 77.17 | 87.46 | 4.03 | 0 |
| ema_full | 73.62 | 80.67 | 91.42 | 4.39 | 9000 |
| ema_block | 52.26 | 80.29 | 87.83 | 13.47 | 9000 |

Completed and scored: 4/4 retained methods. Each result uses one label-subset seed and one initialization seed; across-seed standard deviations are unavailable. Timed hours cover the training loop/final save and exclude startup/data hashing; they are not billed reservation cost or optimized latency. EMA-block includes diagnostic audit/JVP work and is not an optimized baseline latency measurement.

## Scope and missing evidence

The five allocation/guard arms (joint_guard, independent_guard, sign, joint_no_class, joint_no_guard), other label budgets/subset seeds and TEST inference are cancelled by the user. No result establishes superiority of Self-Audit v2: its full guarded experiments were not executed. Native batch16 was previously stopped at step1515; its checkpoint/logs remain preserved and were not exactly resumed.

Frozen checkpoint, payload, pretrained weight, executed-source, training-data and metric hashes are recorded per completed run in logs/*_MEASURED.json. CSVs contain classwise confusion, IoU, precision, recall and patient metrics. Patient Dice averages available reference-frame Dice; pooled-frame statistics are named explicitly. Undefined absent-reference Dice/IoU remain blank with explicit FP counts.

No TEST Dice, HD95, multi-seed uncertainty, patient bootstrap, modern external SSL baseline or independent-confirmation result exists. HD95 geometry/implementation remain uncertified. VAL18 is exposed development data. The original TRAIN80/VAL20 split has a duplicate TRAIN/VAL cine; reviewed TRAIN80/VAL18 metadata was reused as directed by the runbook, leaving all raw files and TRAIN IDs unchanged.

## Data, initialization and paths

Drive: KhanhDrive:Dataset/ACDC; raw local data: /workspace/self-audit-v2/data/raw/ACDC. All 1,053 files (7.269 GiB) matched remote checksums. All nine 4/8/16 × 42/123/2026 preparation caches completed, with zero U-mask reads. DATA_PREPARATION.csv records actual labeled frames/slices: L8/S42 has 16 frames and 154 slices. Crops are geometry-only and aligned; padding is ignored by supervised, audit and pseudo losses. VAL scoring uses full native cines. TEST GT filenames on Drive were never used for training or scoring.

Reviewed split SHA256: b826eaf494760079aa559b0c0c12f7444951d779ebbb5ed291c0398f20bf835f. Pretrained weight SHA256: 21b726bb286e037f00a23fb4699fa9bda9c75b6a615bd57ce3013cec1b528d54, from the user-authorized jaychempan Hugging Face mirror. Mirror LFS and official short hash match; full independent official upstream certification is unavailable. Official source is clean at 6876159a11b4df116f30f667f8c9888617df0751. No random fallback, external license acceptance or historical trainer/weights were used. Reviewed split metadata alone came from the archive.

Source worktree: /root/Self-Audit-vram24. Executed training-source SHA256: 6e12885e44b725c132a01437dc6179cbf524b94c51dd5a5f5254701eb78cf442. Runs, checkpoints, raw logs, full frozen probabilities and patient overlays remain under /workspace/self-audit-v2/vram24 or local ignored files. Only compact tables/receipts and aggregate figures are committed.

## Verification and figures

python scripts/check_repo.py and python -m pytest -q are required by the finalization script before the final Git push. Last completed CPU-isolated checks (CUDA_VISIBLE_DEVICES empty): PASS, 58 passed, one TorchScript deprecation warning; final receipts are recorded in logs/FINAL_CHECKS.json when available. Manufactured tests are software evidence, not patient accuracy evidence.

Aggregate confusion matrices, patient Dice boxplots, loss/acceptance/VRAM curves, ablation and accuracy/cost summaries are retained. Patient image overlays/probability maps stay local and are excluded from Git. No VAL learning curve exists because only final checkpoints were scored. Full source/config/command receipts and prior real pilot capacity measurements remain in this branch.

ACDC: O. Bernard et al., IEEE TMI 37(11), 2514–2525 (2018), doi:10.1109/TMI.2018.2837502.
