# Q3D validation display terminology

This amendment changes presentation to train / val / test. It does not migrate persisted contracts or change the active server. No server connection, process change, run-directory access, manifest change, or deployment was performed. Server deployment and patient/GPU execution for this amendment are NOT_RUN. Verification below is SOFTWARE_ONLY.

The baseline is `3b8e3ab42d8cf5d729e7924ba2a91bb77a39013c`; the final source is `60124f4`. Historical files under `reports/`, all recipes/configs, manifest names and SHA values, and the frozen A/A-3D baselines remain unchanged. This directory is a new report.

| Presentation | Before | After |
| --- | --- | --- |
| Per-class viewer banner | DEV patient-macro Dice (%) | VAL patient-macro Dice (%) |
| Viewer foreground column | FG | val FG Dice |
| Teacher progress log | DEV patient_macro_fg_dice | VAL patient_macro_fg_dice |
| Campaign Markdown columns | teacher / C0 DEV Dice@K | teacher / C0 val FG Dice@K |
| Campaign Markdown description | selected development metrics | selected validation metrics |
| CLI help and readable errors | DEV, DEV18, DEVELOPMENT, DEV-only, DEV-selected | VAL, VAL18, VALIDATION, VAL-only, VAL-selected |
| Spec and README prose | TRAIN/development, DEV/DEV18 | train/val/test, VAL/VAL18 |

`contract_checks.json` records every one of the 42 changed Python string literals, with file, line, before and after values (including docstrings). Production AST structure is identical after masking string literals. The spec adds the requested compatibility sentence: `val = development set cũ (DEV18); key nội bộ giữ tên development.` It also records a display-only [AMEND]. Historical design entries keep their historical meaning.

Internal compatibility names remain unchanged: `development`, `role="development"`, `development_selection`, `development_epoch`, `teacher_dev_fg_dice`, `teacher_dev_per_class`, `c0_dev_fg_dice`, all JSON/CSV field names, existing `--dev-*` flags, filenames, receipt/checkpoint metadata, and persisted preflight reason/label-scope strings. No CLI aliases were added. Old selection records and current epoch records use the same internal contract; both print VAL in the viewer.

Tests exercise the viewer through its real `--once` CLI on old/new JSONL fixtures, check all RV/MYO/LV/FG values and epoch/update labels, and verify input files stay byte-identical. A separate regression pins all seven plan-v2 recipe file hashes captured before this change; `contract_checks.json` contains the identical before/after values. Existing campaign tests also verify friendly Markdown labels while JSON content and CSV field names stay unchanged.

Final acceptance on source `60124f4`: `python scripts/check_repo.py` PASS; `python -m pytest -q` reports `634 passed, 2 skipped in 196.63s (0:03:16)` with OMP_NUM_THREADS unset; `python scripts/smoke_q3d.py --all-arms --out /tmp/self-audit-q3d-val-display-final-smoke-20261112` PASS. Verbatim stdout and the full source SHA are in `software/`. `git status --porcelain` was empty after every command in the independent checkout. The original workspace had 905 pre-existing untracked files; all were preserved, and tracked files are clean after commit.

Independent read-only review of the main change found no contract or logic blocker and passed 44 focused tests. Its scope included old/new viewer compatibility and all seven recipe hashes. A second review confirmed no contract blocker in the final two-line follow-up, which changes only the remaining preflight help wording and a human-readable output-kind error.
