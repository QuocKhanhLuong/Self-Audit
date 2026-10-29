# ACDC L8 compact pre-KD decision — 2026-10-28

Decision: **NO_GO to start KD/guard with B52**. B8 teacher recovery is active in tmux `b8_ema_recovery_20261028`; no KD/guard arm has been launched.

## Teacher B52

The already local Drive backup ZIP hash is `61ff791ee2e6666722a68b06bb4f6fb7b1dc1baca7e9cf5e1b02816377aebc1c`; CRC passed. The selected `ema_full` payload is `edbce7e5da09e33ec46586b71fc5807102fe5b8849aee6eed7ad768257b2e89d`, manifest `9eec782e0d1aa20f99d47fb398197d4a5441812b3e9f3517c3e9fbba454c284a`, source closure `6e12885e44b725c132a01437dc6179cbf524b94c51dd5a5f5254701eb78cf442`. The historical loader selected raw `student` state at step 10000, not EMA weights. Labeled IDs are the same ACDC L8 union; B52 is a different batch/crop recipe from historical B8.

Original historical evaluator on TEST50 (50 patients, 100 labeled frames): Macro4 **0.861421**, foreground **0.816664**. BG 0.995690, RV 0.756322, MYO 0.793444, LV 0.900227. Versus reported B8, Macro4 delta -0.011532, foreground delta -0.015198. B52 misses the required Macro4 >= 0.87. This TEST50 cohort is already exposed development evidence; it is not a confirmatory holdout. The currently prepared manifest bytes differ from the earlier report, but all 50 raw cine hashes and 100 raw reference mask hashes match the historical data-preparation receipt, and prepared images/masks were checked voxelwise against those raw files.

## Compact supervised

The locked batch8/crop160/seed42 arm completed 10000 steps in 1195.4 training seconds. It has 603956 trainable parameters, Kaiming initialization, one AdamW optimizer owning encoder/decoder/head, no `tiny_test`, and no frozen decoder. Initial step weight-change norms: {'encoder': 0.21005690097808838, 'decoder': 0.10109606385231018, 'head': 0.0024739736691117287}. Final step norms: {'encoder': 0.03454923629760742, 'decoder': 0.01857505366206169, 'head': 0.0005798987112939358}. Peak CUDA allocated 533.2 MiB; reserved 676.0 MiB.

TRAIN L8 optimization-data fit (16 labeled frames): Macro4 0.976706, foreground 0.969180. VAL18 development: Macro4 0.730006, foreground 0.643522; BG 0.989456, RV 0.538352, MYO 0.657214, LV 0.735000. The model learns foreground; current generalization is below the deployment target. No TEST50 compact score has been used to tune it.

## Artifacts and state

- B52 quality: `/workspace/compact-acdc-l8/20261028/b52_test50_receipt/METRICS.json`.
- B52 physical/source/lineage verification: `/workspace/compact-acdc-l8/20261028/B52_VERIFICATION.json`.
- Compact training report/checkpoint/export: `/workspace/compact-acdc-l8/20261028/supervised_full/REPORT.json`, `/workspace/compact-acdc-l8/20261028/supervised_full/final`, `/workspace/compact-acdc-l8/20261028/supervised_final_export`.
- Compact VAL18 and TRAIN fit: `/workspace/compact-eval-val18-final/METRICS.json`, `/workspace/compact-acdc-l8/20261028/TRAIN_FIT_DIAGNOSTIC.json`.
- B8 recovery logs/checkpoints: `/workspace/compact-acdc-l8/20261028/b8_recovery/full` and `/workspace/compact-acdc-l8/20261028/b8_recovery/full.stderr.log`; still running.
- Source checks: `python scripts/check_repo.py` PASS; `python -m pytest -q` 790 passed, 2 warnings on the compact worktree. B8 historical source closure was separately verified before launch.
- MnM archive cleanup: three outer ZIPs and the nested Validation ZIP were CRC-verified against extracted files before removal. Verification receipt: `/workspace/compact-acdc-l8/20261028/MNM_ZIP_VERIFICATION.json`. No raw dataset or other scientific checkpoint was removed.
