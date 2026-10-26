# Native-Volume Correction Owner Report (G9)

**Lane**: `agy_worker_correction` (Gemini coding lane)
**Role**: Correction Owner (G9 Executable native-3D point/correction workflow, prompt-conditioned refinement, provenance/manual preservation, replay/CLI, iterative persistence)
**Date**: 2026-10-26
**Status**: COMPLETE / VERIFIED (All 42/42 unit, regression, CLI I/O spy, and adversarial tests passing on CPU; full volume test suite 244 passed)

---

## 1. Scope & Exclusive Ownership

In strict compliance with `docs/native_3d/IMPLEMENTATION_CONTRACT.md`, `AGENTS.md`, and task instructions, the owned implementation files are exclusively:
- `self_audit/volume/correction.py`: Core correction engine, point prompt representation, geodesic/bilateral propagation, hard manual preservation, iterative persistence, strict allowlist simulation authorization, authoritative manifest binding, and callable interactive protocol.
- `scripts/volume_correct.py`: Standalone executable CLI supporting native volume NIfTI/NPY/NPZ inputs, geometry sidecars, prompt fixtures, provenance exports, pre-access mask authorization, authoritative `--train-manifest` binding, independent export paths, and TRAIN-L / synthetic fixture simulation.
- `tests/test_volume_correction.py`: Comprehensive test suite verifying coordinate mapping, boundary respect, hard manual preservation, CLI I/O spy gates preventing mask access before authorization, sparse unknown preservation, iterative manual persistence, NIfTI meter/micron conversion, unknown units rejection, and interactive protocol execution.
- `reports/native_volume_pilot/agy_worker_correction.md`: Durable documentation and audit trail.

No modifications were made to baseline modules (`self_audit/*.py`), old configs, or other workers' files.

---

## 2. Review Findings & Implemented Fixes (Witnesses 1–4)

### 2.1 Witness 1: Negative / Background Prompt Semantics & Hard Manual Preservation
- **Defect Identified**:
  When `prior_labels` was all 3, `manual_only`, and `PointPrompt((2,0,0), target_class=0, prompt_type="negative")` was supplied, the output voxel remained 3 with `PROVENANCE_ORIGINAL` and `manual_voxels = 0`. Because `simulate_clicks_from_ground_truth` emits negative prompts for background (`target_c == 0`), negative prompts were skipped by a `continue` in `VolumeCorrector.correct`, leaving `manual_mask` False. Untouched prior restoration subsequently overwrote the negative correction with the old prior label.
- **Implemented Fix**:
  1. Updated `VolumeCorrector.correct` to apply declared prompt semantics uniformly across positive, negative, and edit prompts:
     - Negative prompts targeting class 0 (or foreground classes) explicitly mark the prompt voxel (and spherical radius if `radius_voxels > 0.5`) in `manual_mask = True` with `manual_labels = 0`.
     - Seed class 0 is registered with `weight = prompt.weight` in `resolved_seeds_by_class` for propagation.
     - `affected_slices` records the prompt slice index `iz`.
  2. Hard manual preservation enforces exact label 0 and probability 1.0 at manual voxels:
     - Voxels are assigned `PROVENANCE_MANUAL` (2).
     - Untouched prior restoration explicitly excludes manual voxels.
     - `summary["manual_voxels"]` accurately reflects background corrections.
  3. Iterative persistence via `prior_provenance`: earlier background manual edits are strictly preserved in subsequent correction rounds unless explicitly superseded by a new conflicting prompt.

### 2.2 Witness 2: Unknown (255) Preservation & Prior Provenance Category Lineage
- **Defects Identified**:
  1. Without prior labels/probabilities (`prior_labels is None and prior_probabilities is None`) and without prompts, `correct()` returned background 0 with `PROVENANCE_ORIGINAL` (1) instead of `UNKNOWN_LABEL` (255) and `PROVENANCE_UNKNOWN` (255). No reference means no original supervision.
  2. No-op replay with `prior_labels = [0, 1, 2, 255, 3]` and `prior_provenance = [ORIGINAL 1, PSEUDO 4, PROPAGATED 3, UNKNOWN 255, MANUAL 2]` returned provenance `[1, 1, 1, 255, 2]`, improperly flattening `PSEUDO` and `PROPAGATED` lineage into `ORIGINAL`.
- **Implemented Fix**:
  1. No prior / no prompts: when neither prior labels, prior probabilities, nor prior provenance is provided, all untouched voxels strictly evaluate to `UNKNOWN_LABEL` (255) and `PROVENANCE_UNKNOWN` (255).
  2. Full prior provenance preservation: when `prior_provenance` is provided, all untouched voxels retain their exact prior provenance category (`ORIGINAL`, `PSEUDO`, `PROPAGATED`, `UNKNOWN`, or prior `MANUAL`). No lineage is erased or overwritten during no-op replay or partial iterative updates.
  3. Geometric invalidity: voxels with `geometric_valid == False` strictly enforce `UNKNOWN_LABEL` (255) and `PROVENANCE_UNKNOWN` (255).

### 2.3 Witness 3: Physical Geometry Fail-Closed, Unit Conversion & Synthetic Fixture Gate
- **Defects Identified**:
  - `_load_volume` silently accepted NIfTI headers with `"unknown"` or `0` spatial units as millimeters.
  - `VolumeCorrector.correct` without `affine` or `spacing_xyz_mm` silently invented `(1.0, 1.0, 1.0)` spacing.
  - `CorrectionResult.export_nifti` without `affine` silently invented identity affine `np.eye(4)`.
- **Implemented Fix**:
  1. Implemented centralized `validate_geometry(affine, spacing_xyz_mm, synthetic_fixture=False)`:
     - Real data: fails closed (`ValueError`) if geometry is missing, if affine is non-finite (NaN/Inf), if affine is singular (`abs(det) < 1e-8`), if spacing is non-positive, or if affine spacing disagrees with provided `spacing_xyz_mm`.
     - Synthetic test fixtures: can opt in via `synthetic_fixture=True` to enable deterministic identity geometry.
  2. NIfTI unit validation in `_load_volume`:
     - Declared meters (`meter`, `m`, 1) are converted to millimeters (`* 1000.0`).
     - Declared microns (`micron`, `um`, 3) are converted to millimeters (`* 0.001`).
     - Declared millimeters (`mm`, 2) are preserved.
     - Unknown units (`unknown`, 0, None) fail closed with `ValueError` for real data operations.
  3. `export_nifti` validates that affine is finite and nonsingular; missing affine fails closed unless `synthetic_fixture=True`.

### 2.4 Witness 4: Authoritative TRAIN-L Manifest Binding & Non-Human Accounting
- **Defects Identified**:
  Click simulation previously accepted a user-declared `--dataset-role TRAIN-L` without verifying authoritative membership against an actual training manifest.
- **Implemented Fix**:
  1. Enforced strict authoritative manifest binding in `authorize_simulation` and CLI:
     - Real TRAIN-L simulation requires `--train-manifest` (with optional `--train-manifest-sha256`).
     - Checks manifest records before touching or hashing the GT mask path:
       - Patient ID must be in `manifest["labeled"]` (or train split).
       - Patients in `manifest["unlabeled"]` or `evaluation_ids` fail closed immediately.
       - Mask filename/path must match the authorized frame mask in manifest.
     - Roles such as `VAL`, `TEST`, `TRAIN-U`, `TYPO`, and unproven `TRAIN` fail closed before any mask access, preserving CLI I/O spy guarantees.
  2. Output reporting and non-human annotation accounting:
     - All simulation and interactive outputs mark `"protocol": "simulated_additional_supervision"`.
     - Explicitly records `"human_annotation_claimed": False` and `"human_annotation_time_sec": None`.
     - Human annotation time is never claimed or fabricated.

---

## 3. Verified Source Hashes (SHA-256)

| File | SHA-256 |
|---|---|
| `self_audit/volume/correction.py` | `c3b8d26f5fa7b78754d55e0ec4d43e063d89faaace9bcfb3a278bfc5bd23ab97` |
| `scripts/volume_correct.py` | `1e3cd85f78346b74f363848ae80761692e9da19e5a230c6d886b84600d86688c` |
| `tests/test_volume_correction.py` | `8078db45e0854e6e8decae4fbb5cd5f2d1872c86f244c36f244b7c4a51b6f60b` |

---

## 4. Verification & Test Evidence

### 4.1 Owned Test Suite (`tests/test_volume_correction.py`)
Command:
```bash
python3 -m pytest tests/test_volume_correction.py -q
```
Output:
```text
..........................................                               [100%]
42 passed in 7.38s
```

All 42 tests pass, including:
- 35 baseline and first-repair tests
- `test_negative_prompt_background_correction_applied_and_persisted` (Witness 1)
- `test_no_prior_no_prompts_returns_unknown_255` (Witness 2)
- `test_noop_replay_preserves_all_prior_provenance_categories` (Witness 2)
- `test_physical_geometry_fail_closed_and_validation` (Witness 3)
- `test_correctly_aligned_nifti_positive_case` (Witness 3)
- `test_authoritative_train_manifest_binding` (Witness 4)
- `test_cli_real_train_manifest_simulation` (Witness 4)

### 4.2 Full Volume Test Suite
Command:
```bash
python3 -m pytest tests/test_volume_*.py -q
```
Output:
```text
244 passed in 14.38s
```

### 4.3 Repository Health Check
Command:
```bash
python3 scripts/check_repo.py
```
Output:
```json
{"status": "PASS", "architecture": "dinov3_convnext_tiny_native_decoder_v2", "supervision": "semi_supervised; dense TRAIN subset incl audit; not no-GT"}
```

### 4.4 Astra Replayable Adversarial Harness Verification
All 3 correction adversarial checks in `reports/native_volume_pilot/review_geometry_aux.md` (`forbidden_mask_io`, `forbidden_role_allowlist`, `unknown_retention`) verify 100% `PASS`.

---

## 5. Tooling Coordination Note (I00 vs I01 & New CLI Flags)

To tooling and root coordinators:
- **I00 vs I01 Distinction**:
  - `I00` (`automatic_only`): zero prompts, zero manual voxels, zero propagated voxels, preserving model-only prior.
  - `I01` (`interactive_extra_supervision`): prompt-conditioned refinement consuming explicit click budget.
- **New CLI Flags in `scripts/volume_correct.py`**:
  - `--train-manifest <path>`: Required for real `TRAIN-L` simulation to verify authoritative patient membership before opening GT masks.
  - `--train-manifest-sha256 <hash>`: Optional pinned SHA-256 for manifest integrity verification.
  - `--synthetic-fixture`: Explicit flag for test fixtures to opt into known identity geometry without real manifest files.

---

## 6. Execution & Evidence Disclaimer

- **CPU Verification**: 100% PASS on Apple Silicon (macOS, Python 3.11.16).
- **GPU Runs**: PRO4000 GPU is not currently rented per user confirmation; all GPU / patient training runs remain `NOT RUN`.
- **Annotation Time**: All simulated clicks and interactive receipts explicitly declare `human_annotation_claimed=False` and `human_annotation_time_sec=None`.
