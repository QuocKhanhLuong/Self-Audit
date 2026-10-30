# Scalar Transfer-Utility Preflight Evidence Report

**Date:** 2026-10-30
**Role:** Evidence Worker
**Source Baseline:** `d93c4dea4f42895297ec5941bb938b47140aa9e4`
**Working Branch:** `QuocKhanhLuong/cptu-scalar-20261030`
**Implementation Contract:** [CPTU_IMPLEMENTATION_CONTRACT.md](../../../docs/CPTU_IMPLEMENTATION_CONTRACT.md)
**Governing Rules:** [AGENTS.md](../../../AGENTS.md)

---

## Executive Summary

1. **Independent Recomputation of the 761 / 7,028 Scalar Finding:**
   - Evaluated all 9,000 post-warmup transaction steps of `results/compact_acdc_l8_20261029/experiment_20261028/scalar_guard_full/TRAIN_LOG.jsonl.gz`.
   - Exactly **761 steps (10.828%)** out of 7,028 accepted scalar steps chose a scale below an equal-Dice feasible competitor.
   - The objective gap across all 1,276 tied candidate pairs decomposes into the algebraic ridge penalty $\frac{1}{2}\lambda m s^2$ with a maximum error of $1.0215 \times 10^{-17}$ (within machine precision).
   - Confirmed that 713 scalar steps accepted a nonzero update with objective worse than zero displacement ($1 - D_{base}^{FG}$), of which 693 had equal base foreground Dice, confirming that the historical objective $J_L$ does not measure transfer progress on unlabeled data.

2. **Model Architecture & Parameter Count Clarification:**
   - The actual model in `scalar_guard_full` is the **narrow** `CompactStudent` (widths `[16, 32, 64, 128]`), which has **603,956 parameters** and **2,415,824 raw float32 tensor bytes**.
   - This must not be confused with the wide model (widths `[32, 64, 128, 256]` with 2,410,596 parameters and 9,642,384 bytes) used in `wide_plain_kd`.

3. **Data Provenance & Augmentation:**
   - Context `[z-1, z, z+1]` represents adjacent **spatial Z** slices (through-plane context in 2D cine slices), **not temporal** frames.
   - Normalization provenance is pinned to `CANONICAL_PREPROCESSING` from `self_audit/shared_data.py`: **per-cine image-only 0.5/99.5 percentiles** clipping into $[0, 1]$. Validation of arrays in $[0, 1]$ reflects this percentile normalization, not simple min-max.

4. **Locked Pilot Budget Recommendation:**
   - The root configuration has **LOCKED 2,000 total steps**, comprising **1,000 warmup steps** and **1,000 active KD steps**, with evaluation checkpoints at **1,000 / 1,500 / 2,000** steps (final primary) across four matched arms: plain KD, scalar original, scalar larger-tie, and scalar cross-patient utility.
   - The flawed 500-step recommendation has been removed, as 500 steps would contain zero KD updates due to the 1,000-step warmup. No empirical claim is made that 500 steps is sufficient or that VRAM remains $< 800\text{ MB}$ under the new utility selector.
   - The user has explicitly authorized this locked 2,000-step pilot on compute; previous worker prohibition was task dispatch scope, not a global AGENTS block.

5. **Teacher Lineage & Historical Status:**
   - Recovered B8 student payload SHA-256: `0ae593ecb6b6f9dcb88d0a41d4a77d7b3b3a3edd87396624c3648afb1281f1b6` on L8 patient IDs (`007, 010, 011, 024, 026, 041, 077, 099`).
   - Historical teacher gate status `PASS` on TEST50 is a **RECORDED** receipt from `GO_DECISION_B8.json`, not a new certification. As documented, TEST50 was previously exposed during teacher selection and development, and is not a pristine confirmation set.
   - Historical b36 teacher mismatch: `b36bcafe...` (from commit `0f54455b`) was unavailable, blocking earlier runs until B8 recovery succeeded.
   - Physical weights status: Connection to `root@141.0.85.200:42465` refused; no newer SSH endpoint in receipts. Status: **`RECORDED_UNVERIFIED_PHYSICAL_ACCESS_BLOCKED`**.

---

## 1. Study Material Inventory & Provenance

Location: `/Users/alvinluong/.local/share/openresearch/files/self-audit/fewshot-design-20261028/latest-results-20261030/transfer-utility-study`

The inventory explicitly distinguishes **measured SHA-256** from **preexisting expected SHA-256** recorded in `STUDY_PROVENANCE.json` and `MPL_FIGURE_PROVENANCE.json`:

| Filename | Bytes | Measured SHA-256 | Preexisting Expected SHA-256 | Status | Document Provenance & Scope |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `GGR_FULLTEXT.txt` | 69,217 | `b55fb11bcd75b5583a4dc8212b2d76a03f7a418999d9b706f3a22200c225baac` | `b55fb11bcd75b5583a4dc8212b2d76a03f7a418999d9b706f3a22200c225baac` | MATCHED | Jiahe Chen, Qian Shao, Qiyuan Chen, Jiaying He, Jintai Chen, Jian Wu, Hongxia Xu (2026): "Geometric Gradient Rectification for Safe Open-Set Semi-Supervised Learning" |
| `METHOD_PROPOSAL_VI.md` | 20,475 | `eeb9aad4b47321b0bdf57ea7611be9b62d9a38240d64444dec65a4d58030cd6b` | `eeb9aad4b47321b0bdf57ea7611be9b62d9a38240d64444dec65a4d58030cd6b` | MATCHED | Method proposal: decoupling retention constraint from transfer utility |
| `MPL_FIGURE_PROVENANCE.json` | 493 | `5fb594ae02896d9d4d7c9d962ab2799910597201be51ad600314ca587b859520` | `5fb594ae02896d9d4d7c9d962ab2799910597201be51ad600314ca587b859520` | MATCHED | Clip coordinates and provenance for MPL Figure 1 |
| `MPL_FULLTEXT.txt` | 88,066 | `d04247f322d60715da06a66bb84d8c936a6872c63a473b5ac03869aa1c51a4b7` | `d04247f322d60715da06a66bb84d8c936a6872c63a473b5ac03869aa1c51a4b7` | MATCHED | Hieu Pham, Zihang Dai, Qizhe Xie, Minh-Thang Luong, Quoc V. Le (CVPR 2021): "Meta Pseudo Labels" |
| `STUDY_PROVENANCE.json` | 2,696 | `597696823c9e2c9e1a4897b51402353d8fd537943412caff343f0fefa34b3800` | `597696823c9e2c9e1a4897b51402353d8fd537943412caff343f0fefa34b3800` | MATCHED | Study provenance record with literature references and baseline hashes |
| `TRIAL_SELECTION_AUDIT.json` | 10,706 | `91e01f0d33a7c26ac354ce37c8133d30af1520d57fbe552baad70eba6aad946b` | `91e01f0d33a7c26ac354ce37c8133d30af1520d57fbe552baad70eba6aad946b` | MATCHED | Post-hoc trial selection receipts for scalar and joint arms |
| `audit_trial_selection.py` | 4,699 | `0a3a308351e5017ed7b325b3a569947f97ba2505646237ae21ebfebd4a8ad5d3` | `0a3a308351e5017ed7b325b3a569947f97ba2505646237ae21ebfebd4a8ad5d3` | MATCHED | Original trial selection analysis script |
| `extract_mpl_figure.py` | 893 | `9b82de4f93e3926e317b9cc017f9b654476637d51488f9c0f6ce5cd093cef372` | `9b82de4f93e3926e317b9cc017f9b654476637d51488f9c0f6ce5cd093cef372` | MATCHED | PDF clipping script for MPL Figure 1 |
| `meta-pseudo-labels.pdf` | 1,245,388 | `43cd1f89528d54bc5652f4ab5688cc4c09246827b93d29dde1d71ea2bf2cd4e7` | `43cd1f89528d54bc5652f4ab5688cc4c09246827b93d29dde1d71ea2bf2cd4e7` | MATCHED | Original CVPR 2021 PDF of Meta Pseudo Labels |
| `mpl-figure1.png` | 34,374 | `c2680e03f36ef9fb0d68d1edf827c31f5a8ef88ae7ef908cec9e0083f95e359f` | `c2680e03f36ef9fb0d68d1edf827c31f5a8ef88ae7ef908cec9e0083f95e359f` | MATCHED | Raster extract of MPL Figure 1 |

> [!IMPORTANT]
> **Copyright & CI Portability:**
> - In compliance with guidelines, full-text copyrighted papers and PDFs are inspected locally and are **not** redistributed or committed into git.
> - Automated unit tests in `tests/test_transfer_utility_evidence.py` run via temporary fixtures (`tmp_path`) so that tests pass cleanly on Linux/CI without requiring `/Users/alvinluong/` external directories.

---

## 2. Baseline Source Hashes (Archive `d93c4de`)

The worktree status column is a snapshot at evidence capture, before the final
integration. The delivery verification report binds the final source bytes.

Hashes are derived directly from `git show d93c4dea4f42895297ec5941bb938b47140aa9e4:<path>`:

| Source File | Archived Hash (`d93c4de`) | Worktree Status | Working File Role |
| :--- | :--- | :--- | :--- |
| `ACTIVE_PIPELINE.json` | `618de5c3ab6bbb4b2e88a9ff4bc7ca0eb90594e281c7728e041adf3e675abccf` | Modified by peers | Preserved; read-only for evidence |
| `docs/METHOD.md` | `d7a62f699eae6035496333178406c3d2c243f4364aed9d84ba1923e9ededb7b6` | Modified by peers | Preserved; read-only for evidence |
| `docs/COMPACT_STUDENT.md` | `dffb3952e9569add30093f81908a50b88eec8c8dd18730f993114b364bd98ea3` | Modified by peers | Preserved; read-only for evidence |
| `self_audit/compact_control.py` | `17878be6becada6f6fefa2ed7d56f52357c7d049f6ff07f853ea319e91d082b8` | Modified by peers | Preserved; read-only for evidence |
| `self_audit/compact_training.py`| `b7a5123ebc93e9271e111430f4818b8654d0c63d1b31e7a5b34dc05749c06a70` | Unmodified | Read-only |
| `self_audit/compact_teacher.py` | `973a93bc202beaacb4b6ab24c5468e16c5621d54d9cb4ffeb580e32a222b061d` | Unmodified | Read-only |
| `self_audit/compact_artifacts.py`| `fbba8da509400967b9a3823be48dabe654102a757eb58627a9f04850a08165dc` | Modified by peers | Preserved; read-only for evidence |
| `self_audit/recovery_gates.py` | `f9008c2f1742616b0f91add632fb5c4fb51bc9df5a65302f79f2aef4bfb96a51` | Unmodified | Read-only |
| `train.py` | `ec671ef385b7ef6aea0cbe4975cdb69a743c937998170529f9bdbb9226ad68fe` | Unmodified | Read-only |
| `scripts/check_repo.py` | `515383134071c224db443822e33032940e532accdde334434c1051cbbcdb2f9e` | Unmodified | Read-only |

---

## 3. Independent Selection Oracle & Ridge Verification

Recomputed by [`scripts/audit_transfer_utility_evidence.py`](../../../scripts/audit_transfer_utility_evidence.py) on `results/compact_acdc_l8_20261029/experiment_20261028/scalar_guard_full/TRAIN_LOG.jsonl.gz`:

### Recomputation Statistics

- **Total Post-Warmup Steps Evaluated:** 9,000 (steps 1,000 to 9,999)
- **Accepted Steps:** 7,028
- **Rejected Steps (no feasible trial):** 1,972
- **Independent Oracle Agreement:** 7,028 / 7,028 (100.0%)
- **Steps Selecting Scale Below Largest Feasible:** 1,971
- **Total Feasible Pairs with Scale > Chosen Scale:** 3,458
- **Steps with Equal Foreground Dice on Larger Scale:** **761** (10.828% of accepted steps)
- **Total Larger Equal-Dice Feasible Candidate Pairs:** 1,276
- **Larger Trials with Strictly Better Foreground Dice:** 12 pairs
- **Steps Choosing Candidate Worse Than Zero Displacement:** 713 steps (693 with identical base foreground Dice)

### Ridge Penalty Decomposition

In the original controller objective:
$$J_L(\alpha) = (1 - D_L^{FG}(\theta_s + \Delta(\alpha))) + \frac{\lambda}{2} \|\alpha\|_2^2$$
For a scalar trial $\alpha = s \cdot \mathbf{1}_M$, the penalty term is $\frac{\lambda}{2} M s^2$.
When two feasible trials $t_1, t_2$ have identical foreground macro Dice $D_L^{FG}(t_1) = D_L^{FG}(t_2)$, the objective gap is:
$$\Delta J = J_L(t_2) - J_L(t_1) = \frac{\lambda}{2} M \left(s_2^2 - s_1^2\right)$$

- **Observed Maximum Decomposition Error:** $1.0214939411201044 \times 10^{-17}$
- **Conclusion:** On foreground-Dice ties, the historical controller strictly discriminates against larger steps solely due to the quadratic ridge penalty.

---

## 4. Current Pinned Baseline & Historical Invariants

1. **Eight Candidate Scales:**
   - In `self_audit/compact_control.py`:
     `base_scales = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125][:max_trials]`
2. **Absence of Zero Candidate in Historical Baseline:**
   - Zero displacement ($\alpha = \mathbf{0}$) is not included in `base_scales`.
   - `best_objective` initializes to $\infty$, not $1 - D_{base}^{FG}$.
   - Consequently, 713 scalar steps accepted a update with objective worse than zero displacement.
3. **Guard Tolerances:**
   - Strict non-regression: `risk_budget = 0.0`, `absent_fp_budget = 0.0`, numerical tolerance $= 10^{-6}$.

---

## 5. Teacher Identity, Lineage & Recorded Status

- **Active Teacher Artifact:** Recovered B8 student checkpoint
  - Payload SHA-256: `0ae593ecb6b6f9dcb88d0a41d4a77d7b3b3a3edd87396624c3648afb1281f1b6`
  - Manifest SHA-256: `ae36d026ce26ca188627e6ea4699695e9eb99501312ce05d5324b81e44d3ddaa`
  - Recorded Historical Gate Status: **`PASS`** (`b8_test50_macro4 = 0.880049 >= 0.88`, `b8_test50_foreground = 0.841210 >= 0.84`)
  - Scope: PASS is a **RECORDED** historical outcome from `GO_DECISION_B8.json`. The TEST50 cohort was previously exposed for development and teacher selection, and is **not** a pristine confirmation cohort.
- **Declared L8 Labeled Patient Budget:**
  - `['patient007', 'patient010', 'patient011', 'patient024', 'patient026', 'patient041', 'patient077', 'patient099']`
- **Manifest Identities:**
  - `train_manifest_sha256`: `e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4`
  - `val18_manifest_sha256`: `cac4426393ca11959f56df2876869e825443916cdec02c6d95ec8da3fac0ca50`
- **Historical b36 Teacher Mismatch:**
  - Checkpoint `b36bcafe9613325d85ee70dabd6a36fef7c17429882967831a151c0cce6d8437` (commit `0f54455b`) was unavailable, blocking earlier runs (`TEACHER_BLOCKER.json`) until B8 recovery succeeded.
- **Physical Weights Verification Status:**
  - Connection to `root@141.0.85.200:42465` refused; no newer SSH endpoint in receipts. Status: **`RECORDED_UNVERIFIED_PHYSICAL_ACCESS_BLOCKED`**.

---

## 6. Architecture, Training Specifications & Locked Pilot Budget

### Architecture & Parameter Counts

| Model Variant | Widths | Parameters | Raw Float32 Tensor Bytes | Role in Archive |
| :--- | :--- | :--- | :--- | :--- |
| **Narrow CompactStudent** | `[16, 32, 64, 128]` | **603,956** | **2,415,824** | **Used in `scalar_guard_full` and `joint_guard_full`** |
| Wide CompactStudent | `[32, 64, 128, 256]` | 2,410,596 | 9,642,384 | Used in `wide_plain_kd` and capacity profile |

### Training & Data Specifications

| Component | Specification | Details |
| :--- | :--- | :--- |
| **Supervised Optimizer** | AdamW | `lr = 0.0003`, `weight_decay = 0.0001`, `betas = (0.9, 0.999)` |
| **Distillation Loss** | Soft Cross-Entropy | Temperature: 1.0, Coefficient: 1.0, Pseudo threshold: 0.7 |
| **Context** | 3 Slices | **Spatial Z** `[z-1, z, z+1]` (through-plane context in 2D cine, **NOT temporal**) |
| **Data Augmentation** | Native spatial crop | `crop_size = 160`, 50% foreground slice sampling on labeled pool |
| **Normalization** | Percentile clipping | `CANONICAL_PREPROCESSING`: **per-cine image-only 0.5/99.5 percentiles** into $[0, 1]$ |
| **Batch Sizes** | $B_L = 8$, $B_U = 8$ | Per transaction step |

### Archived Run Profiles

| Experiment Arm | Steps | Elapsed Time | Steps / Sec | Peak VRAM |
| :--- | :--- | :--- | :--- | :--- |
| `supervised_full` | 10,000 | 1,200.35 s (20.0 min) | 8.37 | 533.2 MB |
| `wide_plain_kd_full` | 10,000 | 2,325.10 s (38.8 min) | 4.34 | 1,299.9 MB |
| `scalar_guard_full` | 10,000 | 12,662.36 s (3.52 hours) | 0.79 | 624.8 MB |
| `joint_guard_full` | 10,000 | 11,355.92 s (3.15 hours) | 0.88 | 624.8 MB |

### Locked Pilot Budget (Root Authorized)

- **Total Steps:** **2,000 steps** (including **1,000 warmup steps**, leaving 1,000 active KD transaction steps).
- **Evaluation Checkpoints:** Steps **1,000 / 1,500 / 2,000** (final only).
- **Four Arms:** `plain_kd`, `scalar_guard_original`, `scalar_guard_larger_tie`, `scalar_guard_cross_patient_utility`.
- **Correction Note:** Removed the flawed 500-step proposal, which would have contained zero KD updates due to the 1,000 warmup steps. No empirical claim is made that 500 steps is sufficient or that VRAM remains $< 800\text{ MB}$ under the new utility selector.
- **Authorization:** The user has explicitly authorized this locked 2,000-step pilot on compute; previous worker prohibition was task dispatch scope, not a global AGENTS block.

---

## 7. True NOT_RUN Statuses

| Item | Status | Reason |
| :--- | :--- | :--- |
| **Authorized Four-arm Patient Pilot** | `NOT_RUN` | Server SSH refused; no physical asset/GPU preflight possible |
| **TEST Mask Inspection / Tuning in This Task** | `NOT_RUN` | No TEST masks opened; historical TEST50 was already observed and is not a new holdout |
| **Physical Weight Hash on Remote GPU** | `NOT_RUN_SSH_REFUSED` | SSH to `root@141.0.85.200:42465` refused; no newer endpoint in receipts |
| **Patient Generalization Certification** | `NOT_CERTIFIED` | Audit Brier and transfer utility proxy do not certify unseen patient Dice |

---

## 8. Automated Test Verification

All tests executed via `/Users/alvinluong/miniforge3/bin/python`:

1. `python scripts/check_repo.py`: **PASS** (`{"status": "PASS", "architecture": "dinov3_convnext_tiny_native_decoder_v2", "supervision": "semi_supervised; dense TRAIN subset incl audit; not no-GT"}`)
2. `python -m pytest tests/test_transfer_utility_evidence.py -v`: **9 passed in 3.85s**
   - `test_study_material_hashes_with_fixture`: PASSED (CI-portable; passes without `/Users/alvinluong/`)
   - `test_study_material_nonexistent_directory`: PASSED (graceful handling of missing study dir)
   - `test_scalar_larger_feasible_equal_dice_finding`: PASSED (independent replay confirms 761/7028)
   - `test_ridge_gap_exact_decomposition`: PASSED (error $< 10^{-15}$)
   - `test_zero_competitor_behavior`: PASSED (713 worse-than-zero steps confirmed)
   - `test_teacher_identity_and_b36_mismatch`: PASSED (payload `0ae593ec...` and b36 mismatch confirmed)
   - `test_narrow_student_parameters_and_pilot_specs`: PASSED (**603,956 params, 2,415,824 raw bytes**, spatial Z, percentile norm, 2k budget)
   - `test_source_hashes_derivation_from_git`: PASSED (graceful derivation from `git show d93c4de`)
   - `test_full_evidence_audit_entrypoint`: PASSED (full audit pipeline execution)
