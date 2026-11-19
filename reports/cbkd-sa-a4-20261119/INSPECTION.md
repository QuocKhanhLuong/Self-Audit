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

## Data preparation and hash comparison (done)
- Re-prepared with the repo preparer (`scripts/pg_prepare_acdc.py`); it fails closed on patient identity and passed:
  TRAIN80 / VAL18 / TEST50 / L8 lists equal the committed receipts (L8 = patients 007, 010, 011, 024, 026, 041, 077, 099).
- Manifest sha256: train `44274592...`, val `65839328...`, test `75cec96a...`. They differ from the `pg_k8_base.json`
  locks (train `ef7562a2...`) and from the real pg-k8 resolved config (train `7ccf6883...`) because manifests embed
  storage paths and a relocation lineage block. Patient/label identity is verified by the preparer, not by manifest hash.
  New locks are written into `A4_SA_CONFIG.json` by `scripts/cbkd_sa_make_config.py`.
- The download receipt was re-pointed (a copy in the volume) to the new inventory path; raw file hashes are still verified.

## Phase 0 oracles on VAL18 (no teacher; GT used for measurement only)
| | FG Dice | HD95 mm | LV | MYO | RV |
|---|---|---|---|---|---|
| O1 pooled-code oracle (1,8,8) | 0.884 | 2.93 | 0.923 | 0.808 | 0.919 |
| O2 shape-atom oracle (k-means K=128, 2x8x8) | 0.771 | 7.95 | 0.817 | 0.691 | 0.806 |

Mixed cells (no class > 0.7): 1.7 % of all cells and 40.6 % of cells containing MYO.

Reading: O2 (hard assignment of non-overlapping patches) is worse than O1 here. The P2 claim "pooled fractions cannot
carry MYO geometry" is not supported by this oracle: pooled fractions with trilinear upsampling reach MYO 0.81 against
0.69 for 128 hard atoms. The learned `D_A` is conditioned on teacher features and is not bounded by O2. Measurement
only, not a gate. O4 (old S3 codebook) and O6 (config diff) were not run.

## GPU environment and probe
- A100-SXM4-40GB, torch 2.15.0+cu132; `nnunet_mednext` and `cinema` import. The pinned MedNeXt checkout had to be added
  to the Modal image (`external/MedNeXt`).
- All four stage types (S1, S2, S3-SA, A4-SA) ran end to end for 100 updates; peak memory 18 GB (A4-SA).
- Early A4-SA student code usage at update 50 / 100: perplexity 1.7 / 1.5, 118-121 of 128 codes dead (random-init `P_S`,
  100 updates only; logged from the first CBKD update as required).

## Not done
- Launch: blocked by budget, see BUDGET.md.
