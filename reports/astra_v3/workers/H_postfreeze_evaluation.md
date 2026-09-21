# Astra Self-Audit v3: Independent Post-Freeze Evaluation Report (Worker H)

> **Root review status: investigation record, not accepted wholesale.** Read [root disagreement resolutions](../11_RED_TEAM.md) and the corresponding root report before using these claims. Worker completion indicates a delivered report, not scientific validation.

- **Audit Target Base SHA**: `d4f503b7107eca4e8ac85f37a5dd01ec50479666`
- **Orca Run ID**: `run_698eb3b460ef`
- **Task ID**: `task_eb2f88e0df1e`
- **Dispatch ID**: `ctx_92ca2e751ac9`
- **Role**: AGY Worker H, Independent Post-Freeze Evaluator
- **Evaluated Frozen Experiment**: `astra-v3-r0-dimensionless-seeds-20260920-v2`
- **Frozen Manifest**: `/private/tmp/astra_v3_r0_dimensionless_20260920_v2/FROZEN.json`
- **Expected Manifest SHA256**: `c0be95e4f3e6bf63277e576f80f0d89aacc949b48630bf211883e786d141196e` (**VERIFIED**)
- **Evidence Directory**: `reports/astra_v3/workers/H_evidence/`

---

## 1. Executive Summary

As the independent post-freeze evaluator (Worker H), an exhaustive audit and ground-truth (GT) evaluation of frozen experiment `astra-v3-r0-dimensionless-seeds-20260920-v2` was conducted. All integrity hashes for the manifest, config, generation script, split file, 16 input images, and 16 prediction archives were cryptographically verified prior to opening any manual GT. All native NIfTI array shapes and affine matrices matched exactly.

**Core Evaluation Verdict**:
The primary frozen candidate **`all_available`** and all six diagnostic arms **FAIL** the preregistered evaluation gate on both the `train` and `dev` splits.
- **Primary Dice** (mean patient macro volume Dice over RV, MYO, LV) for `all_available` is **0.1429** on `train` and **0.1391** on `dev`, substantially lower than the unconstrained `appearance_anchor` (**0.3681** train / **0.3524** dev).
- While `all_available` achieves high precision for LV (0.9918 train / 1.0000 dev) and RV (0.8826 train / 1.0000 dev), it fails the 0.95 precision gate on MYO (0.7304 train / 0.8726 dev) and collapses coverage below the 0.05 (5%) minimum (dev RV coverage is **1.09%**, train MYO coverage is **3.19%**).
- Harm analysis demonstrates that successive heuristic filtering cues (`plus_cross_slice`, `plus_augmentation`, `all_available`) induce large negative Dice deltas (-0.14 to -0.32 per patient) and destroy tens of thousands of correct accepted foreground voxels, collapsing entire cardiac phases to zero recall.

---

## 2. Integrity Verification & Audit Trail

### 2.1 Verification Timestamps & Execution Protocol
Evaluation was executed strictly under independent isolation rules:
- **Freeze Verification Start (UTC)**: `2026-09-20T16:21:07.555981+00:00`
- **Freeze Verification Complete (UTC)**: `2026-09-20T16:21:07.606241+00:00`
- **First Manual GT File Read (UTC)**: `2026-09-20T16:21:07.606312+00:00` (Strictly after verification complete)
- **First GT File Opened**: `/tmp/astra_event_acdc_training/training/patient093/patient093_frame01_gt.nii`
- **Post-Evaluation Re-verification Complete (UTC)**: `2026-09-20T16:21:08.858171+00:00`

### 2.2 Manifest & Root File Cryptographic Hashes
| File | Path | Verified SHA256 Hash |
| :--- | :--- | :--- |
| **Manifest** | `/private/tmp/astra_v3_r0_dimensionless_20260920_v2/FROZEN.json` | `c0be95e4f3e6bf63277e576f80f0d89aacc949b48630bf211883e786d141196e` |
| **Config** | `/private/tmp/astra_v3_r0_dimensionless_20260920_v2/config.json` | `c8ec87720cbea78ff3b6b287e888e79d5a379e3da8903e7b0ac32329295c9dfc` |
| **Script** | `reports/astra_v3/evidence/run_round0_seeds_dimensionless.py` | `acdea9e6127afa1212a6a2816ceb3585187d223d7eb265ef2fd66f6b9414d7f4` |
| **Split** | `/tmp/astra_event_acdc_training/acdc_patient_split_seed42.json` | `bc42574017276a6defc668bf533330fd41a64a5e90a75ea1b8958fc9e469265a` |

### 2.3 Manual Ground Truth Files Opened
Matched GT files were accessed by appending `_gt` to the source image basename within `/tmp/astra_event_acdc_training/training/`. All 16 GT files were read-only, verified, and re-verified:

| Split | Patient ID | Frame | GT File Path | GT SHA256 Hash |
| :--- | :--- | :--- | :--- | :--- |
| **train** | `patient093` | frame01 | `.../patient093/patient093_frame01_gt.nii` | `39b6ba0ba4aa6b287d6daa383ff8c2cd21b34f67750bf35df32e009c3ba84b16` |
| **train** | `patient093` | frame14 | `.../patient093/patient093_frame14_gt.nii` | `3af6df25b3acd416262b8c53edb1d9266de0396365a6b9f9c95487f31bd503bd` |
| **train** | `patient010` | frame01 | `.../patient010/patient010_frame01_gt.nii` | `bc8eefd60c8a2913afa650d024a3a67dec915eb606ea21a1f9278a826feec2b2` |
| **train** | `patient010` | frame13 | `.../patient010/patient010_frame13_gt.nii` | `427c9b73bb7dff19c577fd15379eff81b2707e9c845cfa779e3030435811d661` |
| **train** | `patient079` | frame01 | `.../patient079/patient079_frame01_gt.nii` | `0d89260df97fa620f4ef5824c0892c5750893f417f7390ea15f6e804f58c734e` |
| **train** | `patient079` | frame11 | `.../patient079/patient079_frame11_gt.nii` | `5e98dc85472855cf64ba7dc2a731ef17830b4ec7472ce7ca9a7bc72bc5ce5f9e` |
| **train** | `patient057` | frame01 | `.../patient057/patient057_frame01_gt.nii` | `9b36ea1f7b0a7ee1917f8a3ec95d8ebaaecfaec0745582c5a0833ea712a4ee6b` |
| **train** | `patient057` | frame09 | `.../patient057/patient057_frame09_gt.nii` | `aeeb616198f79fbc9571faebdfb38d30e52df6e0dd979a8e0f63a34fb081b7a2` |
| **dev** | `patient018` | frame01 | `.../patient018/patient018_frame01_gt.nii` | `a39031ef0e839e943c2c770cfa5ddf0fbdd626b9a81e3a6f44d8b6883ebcc074` |
| **dev** | `patient018` | frame10 | `.../patient018/patient018_frame10_gt.nii` | `1c890787d55d288d6c7574b6dbb60ea262c55b8813bc30f9a2ce23c723f338d8` |
| **dev** | `patient030` | frame01 | `.../patient030/patient030_frame01_gt.nii` | `ec8fc353a1ec24f5a35607db274483a936a2826ee11b65e90cb47e5b5e3fe94f` |
| **dev** | `patient030` | frame12 | `.../patient030/patient030_frame12_gt.nii` | `5cb8c5a2c4e201b15112dbf2d228f44d85605d3921764673be7fc2b978931eb6` |
| **dev** | `patient070` | frame01 | `.../patient070/patient070_frame01_gt.nii` | `fffe20904d9cf9e54d633c37b8b45f3dfd6fa33c7f39386d4e8eb17d6978536f` |
| **dev** | `patient070` | frame10 | `.../patient070/patient070_frame10_gt.nii` | `509a567fb4598d9ae4874b0cc6c483a213e414c77ea1b585324749f99db8bbdb` |
| **dev** | `patient076` | frame01 | `.../patient076/patient076_frame01_gt.nii` | `5a479aa55850ab89e023793f77341ebc96f272a85e8a6c8e31fc5495000969ad` |
| **dev** | `patient076` | frame12 | `.../patient076/patient076_frame12_gt.nii` | `43feef09b0b462ea84b553641b7117da8c1fbb46dbdbd09a2fa3ffb99c75bfcf` |

### 2.4 Native Geometry & Spatial Representation Policy
- **Native Shape & Affine Match**: Verified for all 16 cases. `GT shape == Image shape == Prediction shape`, and `GT affine == Image affine == Prediction affine`. No dimensional or spatial misalignments occurred.
- **Physical Representation Caveat**: Local header zooms disagree with NIfTI `sform` spacing, and spatial measurement units are marked unverified/unknown. In accordance with preregistered policy, this evaluation operates on the **STORED GRID** using dimensionless FOV fractions. No claims of physical metric millimeter distance, volume, or valid physical orientation are made.

### 2.5 Independent Evaluator Synthetic Validation
Before accessing any GT file, the evaluator executed 5 self-diagnostic synthetic unit tests (`reports/astra_v3/workers/H_evidence/evaluator.py:149-211`):
1. `test_unknown_as_fn`: Confirmed that UNKNOWN (255) pixels overlapping GT foreground correctly penalize Dice as False Negatives, while UNKNOWN pixels on background do not inflate False Positives.
2. `test_absent_predictions_null_precision`: Confirmed that classes with zero predictions return `null` / `None` precision rather than spurious numbers.
3. `test_both_empty_class_dice`: Confirmed that when both GT and prediction have zero voxels for a class, Dice returns `null` / `None` (preventing artificial 1.0 inflation).
4. `test_label_swaps`: Confirmed that simulated RV/LV anatomical swaps yield 1.0 swap rate.
5. `test_tampered_freeze_failure_before_gt`: Confirmed that hash verification detects simulated tampering and aborts execution before GT access.

---

## 3. Evaluation Results: All Seven Frozen Arms

Evaluation metrics are computed per 3D native phase volume, then macro-averaged across foreground classes $\{RV, MYO, LV\}$, then averaged across phases $\{ED, ES\}$, and finally averaged across patients.

### 3.1 Primary Split-Level Performance (Train Split)
*Train Split: 4 patients (`patient010`, `patient057`, `patient079`, `patient093`), 8 phase volumes, 6,400,000 total voxels.*

| Arm | Primary Dice | Acc. Prec. | UNK Frac. | Swap Rate | RV Prec / Cov | MYO Prec / Cov | LV Prec / Cov | Gate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`appearance_anchor`** | 0.3681 | 0.9727 | 0.7937 | 0.0702 | 0.7186 / 0.3202 | 0.4596 / 0.1314 | 0.7698 / 0.4541 | **FAIL** |
| **`plus_topology`** | 0.3681 | 0.9727 | 0.7937 | 0.0702 | 0.7186 / 0.3202 | 0.4596 / 0.1314 | 0.7698 / 0.4541 | **FAIL** |
| **`plus_boundary`** | 0.3656 | 0.9728 | 0.7938 | 0.0703 | 0.7186 / 0.3200 | 0.4588 / 0.1297 | 0.7704 / 0.4530 | **FAIL** |
| **`plus_cross_slice`** | 0.1692 | 0.9968 | 0.8047 | 0.0052 | 0.8941 / 0.1433 | 0.6590 / 0.0457 | 0.9846 / 0.2758 | **FAIL** |
| **`plus_augmentation`**| 0.2372 | 0.9907 | 0.8295 | 0.0074 | 0.7222 / 0.1403 | 0.5737 / 0.0614 | 0.9574 / 0.3477 | **FAIL** |
| **`all_available` (Primary)** | **0.1429** | **0.9977** | **0.8330** | **0.0056** | **0.8826 / 0.0998** | **0.7304 / 0.0319** | **0.9918 / 0.2527** | **FAIL** |
| **`unknown_control`** | 0.0000 | `null` | 1.0000 | `null` | `null` / 0.0000 | `null` / 0.0000 | `null` / 0.0000 | **FAIL** |

### 3.2 Primary Split-Level Performance (Dev Split)
*Dev Split: 4 patients (`patient018`, `patient030`, `patient070`, `patient076`), 8 phase volumes, 3,763,200 total voxels.*

| Arm | Primary Dice | Acc. Prec. | UNK Frac. | Swap Rate | RV Prec / Cov | MYO Prec / Cov | LV Prec / Cov | Gate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`appearance_anchor`** | 0.3524 | 0.9823 | 0.8481 | 0.0024 | 0.6359 / 0.2103 | 0.6656 / 0.2048 | 0.9410 / 0.5679 | **FAIL** |
| **`plus_topology`** | 0.3524 | 0.9823 | 0.8481 | 0.0024 | 0.6359 / 0.2103 | 0.6656 / 0.2048 | 0.9410 / 0.5679 | **FAIL** |
| **`plus_boundary`** | 0.3532 | 0.9827 | 0.8481 | 0.0024 | 0.6371 / 0.2103 | 0.6727 / 0.2048 | 0.9445 / 0.5679 | **FAIL** |
| **`plus_cross_slice`** | 0.1643 | 0.9986 | 0.8553 | 0.0000 | 1.0000 / 0.0120 | 0.8478 / 0.1006 | 0.9996 / 0.4122 | **FAIL** |
| **`plus_augmentation`**| 0.2478 | 0.9914 | 0.8927 | 0.0000 | 0.7581 / 0.0791 | 0.7380 / 0.1311 | 0.9654 / 0.4382 | **FAIL** |
| **`all_available` (Primary)** | **0.1391** | **0.9989** | **0.8960** | **0.0000** | **1.0000 / 0.0109** | **0.8726 / 0.0691** | **1.0000 / 0.3297** | **FAIL** |
| **`unknown_control`** | 0.0000 | `null` | 1.0000 | `null` | `null` / 0.0000 | `null` / 0.0000 | `null` / 0.0000 | **FAIL** |

---

## 4. Patient-Level Performance Breakdown

### 4.1 Primary Candidate (`all_available`) vs Baseline (`appearance_anchor`)

#### Train Split Patients:
- **`patient010`**:
  - `appearance_anchor`: Patient Macro Dice = **0.4190** (ED: 0.4497, ES: 0.3884)
  - `all_available`: Patient Macro Dice = **0.2814** (ED: 0.3388, ES: 0.2239) | $\Delta\text{Dice} = -0.1377$
- **`patient057`**:
  - `appearance_anchor`: Patient Macro Dice = **0.3936** (ED: 0.4751, ES: 0.3121)
  - `all_available`: Patient Macro Dice = **0.1736** (ED: 0.3472, **ES: 0.0000**) | $\Delta\text{Dice} = -0.2201$
- **`patient079`**:
  - `appearance_anchor`: Patient Macro Dice = **0.3114** (ED: 0.4106, ES: 0.2121)
  - `all_available`: Patient Macro Dice = **0.0890** (ED: 0.1363, ES: 0.0417) | $\Delta\text{Dice} = -0.2223$
- **`patient093`**:
  - `appearance_anchor`: Patient Macro Dice = **0.3483** (ED: 0.3786, ES: 0.3180)
  - `all_available`: Patient Macro Dice = **0.0278** (ED: 0.0556, **ES: 0.0000**) | $\Delta\text{Dice} = -0.3204$

#### Dev Split Patients:
- **`patient018`**:
  - `appearance_anchor`: Patient Macro Dice = **0.6326** (ED: 0.5968, ES: 0.6685)
  - `all_available`: Patient Macro Dice = **0.3626** (ED: 0.3690, ES: 0.3562) | $\Delta\text{Dice} = -0.2701$
- **`patient030`**:
  - `appearance_anchor`: Patient Macro Dice = **0.2186** (ED: 0.3285, ES: 0.1088)
  - `all_available`: Patient Macro Dice = **0.0138** (ED: 0.0277, **ES: 0.0000**) | $\Delta\text{Dice} = -0.2048$
- **`patient070`**:
  - `appearance_anchor`: Patient Macro Dice = **0.2432** (ED: 0.2834, ES: 0.2029)
  - `all_available`: Patient Macro Dice = **0.0986** (ED: 0.1058, ES: 0.0914) | $\Delta\text{Dice} = -0.1446$
- **`patient076`**:
  - `appearance_anchor`: Patient Macro Dice = **0.3152** (ED: 0.4582, ES: 0.1721)
  - `all_available`: Patient Macro Dice = **0.0814** (ED: 0.1628, **ES: 0.0000**) | $\Delta\text{Dice} = -0.2337$

*Critical Observation*: In 4 of the 8 evaluated patients (`patient057`, `patient093`, `patient030`, `patient076`), `all_available` completely failed to generate any foreground predictions on ES phases, collapsing ES Macro Dice to exactly **0.0000**.

---

## 5. Confusion Matrices

Confusion matrices are native 3D volume sums across all phases in the split. Format: Rows = GT $\{0: \text{BG}, 1: \text{RV}, 2: \text{MYO}, 3: \text{LV}\}$; Columns = Prediction $\{0: \text{BG}, 1: \text{RV}, 2: \text{MYO}, 3: \text{LV}, 4: \text{UNKNOWN (255)}\}$.

### 5.1 Train Split Confusion Matrices
#### `appearance_anchor` (Train):
```text
           Pred 0 (BG)  Pred 1 (RV)  Pred 2 (MYO)  Pred 3 (LV)  Pred 255 (UNK)
GT 0 (BG)    1,201,447       10,913         6,777        4,752       4,889,583
GT 1 (RV)            0       28,702           304        5,331          55,294
GT 2 (MYO)           0          173        11,556        1,028          75,189
GT 3 (LV)            0          154         6,506       37,151          38,004
```

#### `all_available` (Train):
```text
           Pred 0 (BG)  Pred 1 (RV)  Pred 2 (MYO)  Pred 3 (LV)  Pred 255 (UNK)
GT 0 (BG)    1,029,722        1,190            62            0       5,082,498
GT 1 (RV)            0        8,945             5          171          80,510
GT 2 (MYO)           0            0         2,804            0          85,142
GT 3 (LV)            0            0           968       20,676          60,171
```

### 5.2 Dev Split Confusion Matrices
#### `appearance_anchor` (Dev):
```text
           Pred 0 (BG)  Pred 1 (RV)  Pred 2 (MYO)  Pred 3 (LV)  Pred 255 (UNK)
GT 0 (BG)      511,883        4,464         1,999        1,205       3,044,478
GT 1 (RV)            0        7,796           105           76          29,095
GT 2 (MYO)           0            0         8,201           39          31,795
GT 3 (LV)            0            0         2,017       21,053          14,002
```

#### `all_available` (Dev):
```text
           Pred 0 (BG)  Pred 1 (RV)  Pred 2 (MYO)  Pred 3 (LV)  Pred 255 (UNK)
GT 0 (BG)      366,833            0            40            0       3,197,156
GT 1 (RV)            0          404             2            0          36,666
GT 2 (MYO)           0            0         2,768            0          37,267
GT 3 (LV)            0            0           362       12,222          24,488
```

---

## 6. Harm Analysis Against Baseline (`appearance_anchor`)

To determine whether the additional heuristic filters harmed or helped pseudo-label quality, we compute the per-patient Macro Dice delta ($\Delta\text{Dice} = \text{Dice}_{\text{arm}} - \text{Dice}_{\text{anchor}}$) and the net change in correctly accepted voxels ($\Delta\text{CorrectAccepted} = \text{Correct}_{\text{arm}} - \text{Correct}_{\text{anchor}}$):

### 6.1 Train Split Harm Ledger
| Patient | `plus_cross_slice` $\Delta\text{Dice}$ / $\Delta\text{Vox}$ | `plus_augmentation` $\Delta\text{Dice}$ / $\Delta\text{Vox}$ | `all_available` $\Delta\text{Dice}$ / $\Delta\text{Vox}$ |
| :--- | :---: | :---: | :---: |
| `patient010` | -0.1160 / -5,944 | -0.0598 / -19,049 | **-0.1377 / -23,240** |
| `patient057` | -0.1871 / -19,864 | -0.1399 / -101,161 | **-0.2201 / -107,500** |
| `patient079` | -0.1732 / -4,874 | -0.2010 / -45,866 | **-0.2223 / -46,267** |
| `patient093` | -0.3191 / -7,299 | -0.1230 / -36,638 | **-0.3204 / -39,702** |

### 6.2 Dev Split Harm Ledger
| Patient | `plus_cross_slice` $\Delta\text{Dice}$ / $\Delta\text{Vox}$ | `plus_augmentation` $\Delta\text{Dice}$ / $\Delta\text{Vox}$ | `all_available` $\Delta\text{Dice}$ / $\Delta\text{Vox}$ |
| :--- | :---: | :---: | :---: |
| `patient018` | -0.1845 / -7,267 | -0.1528 / -43,443 | **-0.2701 / -47,345** |
| `patient030` | -0.2047 / -2,979 | -0.0847 / -52,801 | **-0.2048 / -54,104** |
| `patient070` | -0.1362 / -1,591 | -0.0831 / -10,039 | **-0.1446 / -10,812** |
| `patient076` | -0.2268 / -5,459 | -0.0978 / -51,394 | **-0.2337 / -54,445** |

*Finding*: Every single heuristic cue added to `appearance_anchor` systematically **degraded** pseudo-label utility. `all_available` consistently induced the greatest destruction of anatomical recall, losing up to 107,500 correct voxels on a single patient.

---

## 7. Gate Evaluation & Diagnostic Findings

The preregistered gate configuration from `config.json` specifies three conditions per split:
1. `per_class_precision >= 0.95` for RV, MYO, LV (`null` fails).
2. `per_class_gt_coverage >= 0.05` for RV, MYO, LV.
3. `patient_fraction_with_all_three_classes >= 0.75` ($\ge 3$ of 4 patients predict all 3 classes).

### Gate Compliance Matrix
| Split | Candidate Arm | Precision $\ge 0.95$ | Coverage $\ge 0.05$ | Patient Diversity $\ge 0.75$ | Gate Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Train** | `all_available` | **FAIL** (MYO: 0.7304, RV: 0.8826) | **FAIL** (MYO: 0.0319) | **PASS** (1.0000) | **FAIL** |
| **Train** | `appearance_anchor` | **FAIL** (MYO: 0.4596, RV: 0.7186) | **PASS** (All $\ge 0.13$) | **PASS** (1.0000) | **FAIL** |
| **Dev** | `all_available` | **FAIL** (MYO: 0.8726) | **FAIL** (RV: 0.0109) | **PASS** (1.0000) | **FAIL** |
| **Dev** | `appearance_anchor` | **FAIL** (RV: 0.6359, MYO: 0.6656) | **PASS** (All $\ge 0.20$) | **PASS** (1.0000) | **FAIL** |

### Semantic RV/LV Swaps
- In `appearance_anchor` (Train): GT RV voxels predicted as LV totaled **5,331** voxels, yielding a swap rate of **0.0702** (7.02%).
- In `all_available` (Train): GT RV predicted as LV dropped to **171** voxels (swap rate **0.0056**).
- In `all_available` (Dev): Exactly **0** RV/LV swap errors occurred (swap rate **0.0000**), but this was achieved predominantly by abstaining on 98.9% of RV voxels.

---

## 8. Segregated Classifications

### Facts (Directly Measured and Logged)
- Manifest SHA256 matches preregistered expectation `c0be95e4f3e6...` exactly.
- All 16 GT NIfTI files match image dimensions and affine transformation matrices.
- First GT open occurred at `2026-09-20T16:21:07.606312+00:00`, strictly after verification completed.
- `all_available` achieves Primary Dice of **0.1429** (train) and **0.1391** (dev).
- `all_available` RV coverage on dev is **1.09%** (fails $\ge 5\%$ gate).
- `all_available` MYO coverage on train is **3.19%** (fails $\ge 5\%$ gate).
- `all_available` MYO precision is **0.7304** train / **0.8726** dev (fails $\ge 95\%$ gate).
- Overall gate verdict for `all_available` is **FAIL** on both train and dev.

### Inferences (Domain & Evaluative Deductions)
- Heuristic image-level filtering (cross-slice continuity, gamma augmentation, boundary contrast) successfully rejects false positive background pixels, but does so at an unacceptable cost to anatomical recall.
- By shrinking pseudo-label masks to conservative cores, the teacher leaves up to 89.6% of the volume as UNKNOWN.
- Because UNKNOWN voxels are penalized as False Negatives in 3D volume Dice, the aggressive abstention policy causes primary Dice to collapse by more than half compared to the unconstrained anchor.

### Proposals (Post-Freeze Recommendations)
- Do not deploy `all_available` seeds as supervision targets for student training without loosening the restrictive multi-cue intersection.
- Abandon dimensionless grid approximations and restore true physical voxel spacing and orientation vectors once data provenance is verified.
- The gate must require balanced precision-coverage trade-offs rather than permitting precision gains derived solely from severe coverage collapse.

### NOT RUN (Explicitly Prohibited Tasks)
- **Confidence Calibration**: NOT RUN because these heuristic seed arms produce hard discrete class assignments without calibrated posterior probabilities.
- **Motion & Temporal Consistency Cues**: NOT RUN because full cine series are unavailable in this frozen slice dataset.
- **Native Physical Orientation**: NOT RUN because sform and header spacing disagree, rendering physical patient orientation unverified.

---

## 9. Reproducibility & Commands

The evaluation is fully reproducible via the independent evaluator script:

```bash
# 1. Run unit tests, freeze verification, GT evaluation, and post-verification
PYTHONPATH=src /Users/alvinluong/miniforge3/bin/python reports/astra_v3/workers/H_evidence/evaluator.py
```

All raw metric outputs, patient logs, and cryptographic verification receipts are archived at:
`reports/astra_v3/workers/H_evidence/evaluation_results.json`.
