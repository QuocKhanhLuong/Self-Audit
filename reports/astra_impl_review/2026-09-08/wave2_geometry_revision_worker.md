# Wave 2.3 Geometry Safety — Revision Worker Report

**Date:** 2026-09-08
**Task ID:** `task_ef15692f46cb`
**Dispatch ID:** `ctx_bf85f8a0daf5`
**Worker Terminal:** `term_4991158f-02ba-4966-80ad-dfb1b2b608f5`
**Coordinator Terminal:** `term_bb8fdfa8-8643-4694-8cda-d05efb24504b`
**Status:** SUCCEEDED (All five review notes resolved with regression verification)

---

## 1. Executive Summary

- **What was done:** Resolved all five coordinator review notes from `wave2_geometry_review_notes.md`: removed the unneeded `verified` argument and enforced fail-closed deferral on `strict_physical` while rejecting malformed, single-sided, sheared, and spacing-inconsistent affines and suppressing unverified native-mm metrics; added finite, positive, and dimensional boundary validation for physical spacing in `surface_metrics` and `annotation_metrics` before empty-mask returns; confirmed multi-case collation parity across diverse shapes and mixed spacing in the PyTorch `DataLoader`; guarded preprocessing against partial-pair silent overwrites, checked both volume and mask shapes on disk, enforced global target-size consistency and malformed JSON rejection before any writes, and preserved historical metadata across retained arrays during subset processing; and added a real two-resize production integration test spanning NIfTI preprocessing, saved metadata, and `ACDCDataset` while eliminating all trailing whitespace.
- **What was found:** All 11 focused regression tests in `tests/test_geometry_safety.py` pass cleanly along with the full 192-test repository suite and `py_compile`, proving that unverified discrete inverse-resizing can never emit native millimeter metrics, empty-mask shortcuts cannot bypass spacing validation, partial preprocessing runs retain historical metadata without corruption, and `git diff --check` is completely clean.
- **What is left:** Wave 2.3 revision is complete and ready for final coordinator signoff, with Wave 3 (checkpoint and calibration binding) remaining as the scheduled subsequent wave.

---

## 2. Corrections by Review Note

### Note 1: Native Physical Deferral & Attempted Bypass Rejections (`volume_inference.py`)
- **Removed `verified` Boolean API:** Stripped caller `verified` flag as requested.
- **Fail-Closed on `strict_physical=True`:** Raises `ValueError` declaring native physical reconstruction as `DEFERRED` pending raw NIfTI continuous transform chain provenance.
- **Malformed & Single Affine Rejection:** Rejects non-4x4 matrices (e.g. `[[1]]`), non-numeric/infinite matrices, and single affines lacking a paired target affine.
- **Shear & Axis-Alignment Guard:** Verifies spatial block is orthogonal/diagonal; non-axis-aligned or sheared matrices raise `ValueError`.
- **Spacing-Affine Consistency Guard:** Compares supplied physical spacing against affine column norms; inconsistencies raise `ValueError`.
- **Suppression of Native-mm Metrics:** Unverified path evaluates native Dice on `METRIC_SPACE_VOLUME_NATIVE` while suppressing `hd95_mm`, `assd_mm`, `per_class_hd95_mm`, and `per_class_assd_mm`, explicitly reporting `native_mm_available=False` and `native_reconstruction_verification="DEFERRED"`.

### Note 2: Physical Spacing Boundary Validation (`metrics.py`)
- **Boundary Validation Before Empty Masks:** In `surface_metrics`, spacing values are validated for length, positivity (`> 0.0`), and finiteness (`np.isfinite`) *before* empty-mask early returns (`(empty_score, empty_score)` or `(inf, inf)`).
- **Upfront Spacing Validation in `annotation_metrics`:** When `spacing_known=True`, non-positive, NaN, infinite, or dimensionally mismatched spacing raises `ValueError` immediately.
- **Unit Pixel Grid:** When `spacing_known=False`, evaluation strictly uses `(1.0,) * pred.ndim`.

### Note 3: Actual Default DataLoader Collation Parity (`common.py` / `tests`)
- **Collation Across Cases:** Tested an actual PyTorch `DataLoader` with 4 distinct cases having heterogeneous source shapes (`(200, 300)`, `(216, 256)`, `(180, 180)`, `(240, 320)`) and mixed known/unknown spacing.
- **Parity & Types:** Batch tensors collate to `[4, 3, 128, 128]` and `[4, 128, 128]`, with boolean `spacing_known` collating to `Tensor([True, False, True, False])` and preserving per-sample source dimensions and effective spacings.

### Note 4: Preprocessing Preflight, Partial Pairs, and Subset Metadata Preservation (`preprocess_acdc.py`)
- **Partial-Pair Guard:** Checks whether volume and mask exist on disk. If one exists but not the other, raises `ValueError` refusing silent overwrite without `--no-skip`.
- **Both Array Shapes Checked on Disk:** Validates both existing volume and mask shapes on disk against `target_size` and confirms `vol.shape == mask.shape`.
- **Global Preflight Before Writes:** Validates existing metadata, rejects malformed JSON (rather than resetting to `{}`), and rejects target-size changes globally before writing any new arrays (protecting retained arrays even when raw input contains only new cases).
- **Subset Processing Preserves Retained Metadata:** Merges existing metadata with newly processed cases; retained arrays not touched by a subset run are preserved in `metadata.json` and verified to still exist on disk before writing.

### Note 5: Real Two-Resize Production Integration & Clean Whitespace (`preprocess_acdc.py`, `tests/test_geometry_safety.py`)
- **Production Chain Integration:** Added `test_two_resize_production_chain_integration` testing an end-to-end synthetic NIfTI volume ($H=216, W=256, Z=5$, zooms $1.4 \times 1.6 \times 10.0$ mm) through `preprocess_acdc` (resize 1: $224 \times 224$), verified saved `metadata.json`, and subsequent `ACDCDataset` load and resize (resize 2: $256 \times 128$), verifying exact compound effective spacing ($10.0, 1.4 \cdot 216 / 256, 1.6 \cdot 256 / 128$).
- **Zero Trailing Whitespace:** Cleaned all trailing whitespace across modified lines; `rtk git diff --check` passes with zero warnings or errors.

---

## 3. Explicitly Deferred Native Paths

In alignment with the architecture plan:
1. **Continuous 3D Coordinate Resampling:** DEFERRED (discrete nearest-neighbor inverse resize cannot claim millimeter precision).
2. **Oblique / Sheared Grid Distance Transforms:** DEFERRED / REJECTED (spacing-only distance fields require orthogonal axes).
3. **Raw Clinical ACDC NIfTI Native Verification:** DEFERRED (raw clinical NIfTI archives are not present in this workspace).

---

## 4. Verification Evidence

### Source Compile
```bash
rtk python3 -m py_compile $(find src tests scripts -name "*.py")
# Output: Exit code 0, 0 errors
```

### Git Diff Whitespace Check
```bash
rtk git diff --check
# Output: Exit code 0, clean
```

### Focused Regression Suite
```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 rtk python -m pytest -v tests/test_geometry_safety.py
```
```text
tests/test_geometry_safety.py::test_two_resize_production_chain_integration PASSED [  9%]
tests/test_geometry_safety.py::test_dataset_pixel_mask_parity PASSED     [ 18%]
tests/test_geometry_safety.py::test_unknown_spacing_stays_unknown PASSED [ 27%]
tests/test_geometry_safety.py::test_annotation_metrics_uses_unit_grid_when_spacing_not_known PASSED [ 36%]
tests/test_geometry_safety.py::test_physical_spacing_boundary_validation_and_empty_masks PASSED [ 45%]
tests/test_geometry_safety.py::test_actual_dataloader_collates_multiple_cases_with_mixed_geometry PASSED [ 54%]
tests/test_geometry_safety.py::test_validate_image_mask_geometry_mismatches PASSED [ 63%]
tests/test_geometry_safety.py::test_preprocess_partial_pair_fails PASSED [ 72%]
tests/test_geometry_safety.py::test_preprocess_skip_validates_both_shapes_and_preserves_subset_metadata PASSED [ 81%]
tests/test_geometry_safety.py::test_evaluate_volume_native_strict_physical_deferred_and_bypasses_rejected PASSED [ 90%]
tests/test_geometry_safety.py::test_evaluate_volume_native_unverified_suppresses_native_mm PASSED [100%]

============================== 11 passed in 0.88s ==============================
```

### Full Repository Test Suite
```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 rtk python -m pytest -q -p no:cacheprovider tests
```
```text
........................................................................ [ 37%]
........................................................................ [ 75%]
................................................                         [100%]
=============================== warnings summary ===============================
tests/test_self_audit_core.py::test_dynamic_window_coordinates_and_backward_are_valid
  /Users/alvinluong/Self-Audit/tests/test_self_audit_core.py:32: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
  Consider using tensor.detach() first.
    assert float(coordinates.min()) >= -1.0

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
192 passed, 1 warning in 5.61s
```
