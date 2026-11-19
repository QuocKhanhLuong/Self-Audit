# CBKD-SA A4 Modal run: Phase 0 inspection (partial, 2026-11-19)

## Setup
- Modal client 1.6.4 installed in `~/.venvs/modal` (outside the repo); `~/.modal.toml` login works.
- Secret `rclone-gdrive` did not exist (only the old `selfaudit-rclone`); it was created from the Drive token the user supplied.
- Volume `self-audit-mount` created by the launcher on first use.
- Image builds (torch, MedNeXt, CineMA, nnunet deps); GPU import check not yet run.

## Data on the volume
- `raw/ACDC`: 7.3 GB, 1053 files; `training/` 100 patients (+ citation file), `testing/` 50 patients.
- `bundles/pg-k8-20261118/pg_k8_A1-A4_bundle.zip` (195 MB), unzipped to `.../x` (215 MB).

## Bundle inventory (answers to prompt item 4)
| Item | Present? |
|---|---|
| (a) S1 CineMA encoder artifact | **No.** Only `S1_RECEIPT.json` (names `student_init.pt`, sha256 `f596a68a…`). S1 must be retrained. |
| (b) A1–A4 `best_student` | Yes, inside `resume_A{1..4}.pt` (key `best_student`), plus `arms/*/{epochs.csv,diag.jsonl,run.log}`. |
| (c) S3 codebook | Yes, `S3/codebook.pt` (needed for O4). |
| (d) Teacher checkpoint | **No.** Only `S2_RECEIPT.json` (names `teacher_best.pt`, sha256 `5708cd36…`). S2 must be retrained. |

Consequence: required stages are S1, S2, S3-SA, A4-SA (the prompt's full list), so the budget in Phase 2 must cover all four.

## Not done yet
- Data re-preparation and manifest-hash comparison with `pg_k8_base.json`.
- O1, O2, O4, O6 oracles.
- GPU environment import check.
- Phases 1-3 (implementation, tests, timing probe, BUDGET.md, launch).
