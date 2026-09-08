# Wave 2.3 Geometry Safety — Worker Completion Report

**Date:** 2026-09-08
**Task ID:** `task_14bdb20de9d2`
**Worker Terminal:** `term_4991158f-02ba-4966-80ad-dfb1b2b608f5`
**Coordinator Terminal:** `term_bb8fdfa8-8643-4694-8cda-d05efb24504b`
**Status:** SUCCEEDED (Ready for Astra evaluation)

---

## 1. Executive Summary

- **What was done:** Implemented minimum geometry safety, provenance preservation, and fail-closed guards across `src/self_audit/data/common.py`, `scripts/preprocess_acdc.py`, `src/self_audit/evaluation/metrics.py`, and `src/self_audit/evaluation/volume_inference.py`, accompanied by 7 comprehensive regression tests in `tests/test_geometry_safety.py`.
- **What was found:** In-plane resize now correctly scales physical spacing to $(s_z, s_h \cdot H / H_2, s_w \cdot W / W_2)$ while keeping pixel/mask outputs bitwise identical; preprocessing validates image-mask shape and affine compatibility while preserving historical metadata in the skip path; unverified spacing strictly scores on a unit grid in true pixels; and native volume evaluation fails closed on missing affine or sheared grids while marking full native reconstruction as `DEFERRED`.
- **What is left:** Wave 2.3 requirements are 100% complete. Full repository test suite passes with 188 tests (181 baseline + 7 new). Ready to advance to Wave 3.

---

## 2. Implemented Changes

### A. `src/self_audit/data/common.py` (`VolumeSliceDataset.__getitem__`)
- **Effective Spacing Scaling:** When in-plane resize $(H, W) \to (H_2, W_2)$ occurs, effective spacing is calculated as $(s_z, s_h \cdot H / H_2, s_w \cdot W / W_2)$.
- **Grid Descriptors:** Emits explicit metadata keys:
  - `"source_shape"`: `(volume.shape[0], orig_h, orig_w)`
  - `"network_shape"`: `(volume.shape[0], net_h, net_w)`
  - `"source_spacing"`: `(sz, sh, sw)` or `(1.0, 1.0, 1.0)`
  - `"effective_spacing"`: scaled physical tuple or `(1.0, 1.0, 1.0)`
  - `"spacing"`: effective spacing
  - `"spacing_known"`: boolean
  - `"spacing_units"`: `"mm"` if known, else `"pixel"`
  - `"source_axis_order"`: `"ZHW"`
  - `"network_axis_order"`: `"ZHW"`
- **Preserved Pixel Recipe:** Pixel values and mask array outputs remain bitwise identical to previous implementations; DataLoader collation compatibility is fully preserved.

### B. `scripts/preprocess_acdc.py`
- **Geometry Compatibility Validation:** Added `validate_image_mask_geometry(img_nii, mask_nii, case_name)` verifying `img_nii.shape == mask_nii.shape` and `np.allclose(img_nii.affine, mask_nii.affine, atol=1e-3)`; raises `ValueError` on any mismatch.
- **Rich Schema Version 2 Metadata:** Emits `schema_version: 2`, preserving `orig_affine`, `orig_orientation`, `orig_shape`, `orig_spacing`, and the explicit `resize_chain`.
- **Existing-Output Skip Path Safety:** Checks existing array shapes on disk against requested `target_size`. If target size has changed, fails loudly with `ValueError`. When skipped, historical metadata is preserved verbatim without stamping fresh provenance or rewriting historical datasets.

### C. `src/self_audit/evaluation/metrics.py` (`annotation_metrics`)
- **Unit Grid Pixel Metrics:** When `spacing_known=False`, surface metrics (HD95, ASSD) strictly use the unit grid `(1.0,) * pred.ndim`.
- **Independent Distance Space:** Pixel-space distances are truly measured in pixels and cannot be altered or mislabeled by any supplied non-unit physical spacing values.

### D. `src/self_audit/evaluation/volume_inference.py` (`evaluate_volume_native`)
- **Fail-Closed Guard for `strict_physical=True`:** Requires verified physical spacing and affine transform metadata. Missing spacing or missing affine raises `ValueError`.
- **Shear / Non-Axis-Aligned Grid Guard:** Detects non-diagonal spatial affine blocks (shear / oblique angles) and rejects them (`ValueError`), as spacing-only distance transforms cannot measure Euclidean distance on non-orthogonal grids.
- **Affine Compatibility:** Verifies `np.allclose(affine, target_affine, atol=1e-3)` when both are supplied.
- **Deferred Reconstruction Marking:** Caller boolean `verified=True` is not accepted as evidence without underlying affine metadata. The output block explicitly marks `"native_reconstruction_verification": "DEFERRED"` and documents concrete checks and limitations.

---

## 3. Explicitly Deferred Native Paths

In accordance with the task specification and architecture plan, the following paths are deliberately left **DEFERRED**:

1. **Continuous 3D Spatial Coordinate Resampling:**
   - *Status:* DEFERRED.
   - *Rationale:* True continuous inverse resampling to raw acquisition grids requires 3D spline/BSpline interpolation, continuous coordinate transforms, and raw NIfTI headers. Adding this would introduce a large geometry subsystem beyond minimum safety guards.
2. **Oblique / Sheared Non-Axis-Aligned Distance Fields:**
   - *Status:* DEFERRED (Guarded & Rejected).
   - *Rationale:* SciPy `ndimage.distance_transform_edt` operates exclusively on orthogonal grids parameterized by axis spacing. Sheared or oblique geometries cannot be measured accurately without full 3D reorientation and resampling.
3. **End-to-End Verification Against Raw ACDC NIfTI Files:**
   - *Status:* DEFERRED.
   - *Rationale:* Raw clinical ACDC NIfTI archives are not stored in this repository checkout. Synthetic fixtures and preprocessed arrays verify bookkeeping and fail-closed safety without fabricating native ground-truth claims.

---

## 4. Verification Evidence

### PyCompile
```bash
rtk python3 -m py_compile $(find src tests scripts -name "*.py")
# Output: Exit code 0, 0 errors
```

### Focused Regression Tests
```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 rtk python -m pytest -v tests/test_geometry_safety.py
```
```text
tests/test_geometry_safety.py::test_anisotropic_two_resize_spacing_fixture PASSED [ 14%]
tests/test_geometry_safety.py::test_dataset_pixel_mask_parity PASSED     [ 28%]
tests/test_geometry_safety.py::test_unknown_spacing_stays_unknown PASSED [ 42%]
tests/test_geometry_safety.py::test_annotation_metrics_uses_unit_grid_when_spacing_not_known PASSED [ 57%]
tests/test_geometry_safety.py::test_validate_image_mask_geometry_mismatches PASSED [ 71%]
tests/test_geometry_safety.py::test_preprocess_skip_path_preserves_metadata_and_rejects_changed_size PASSED [ 85%]
tests/test_geometry_safety.py::test_evaluate_volume_native_guards PASSED [100%]

============================== 7 passed in 0.87s ===============================
```

### Full Repository Test Suite
```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 rtk python -m pytest -q -p no:cacheprovider tests
```
```text
........................................................................ [ 38%]
........................................................................ [ 76%]
............................................                             [100%]
=============================== warnings summary ===============================
tests/test_self_audit_core.py::test_dynamic_window_coordinates_and_backward_are_valid
  /Users/alvinluong/Self-Audit/tests/test_self_audit_core.py:32: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
  Consider using tensor.detach() first.
    assert float(coordinates.min()) >= -1.0

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
188 passed, 1 warning in 5.74s
```
