# Region & Temporal Module Implementation & Defect Resolution Report

**Lane**: `agy_worker_regions_temporal` (Gemini 3.8 Flash worker)
**Role**: Region and Temporal Owner (G7 deterministic physical region learning + motion-aligned temporal consistency)
**Task ID**: `task_0821e33f987e`
**Dispatch ID**: `ctx_9b6916d1478b`
**Date**: 2026-10-26
**Status**: COMPLETE / VERIFIED (Per-Sample Identity Cache & Per-Sample Displacement Metric Delivered)

---

## 1. Executive Summary

In authoritative followup `task_0821e33f987e` / `ctx_9b6916d1478b`, we resolved the remaining independent Astra issue from `reports/native_volume_pilot/review_models_trainer.md` and the coordinator temporal metric correction (`msg_2e0063b8ab4e`):

1. **Per-Sample Identity Cache & Companion Invariance (`self_audit/volume/regions.py`)**:
   - **Defect**: `RegionManager.targets()` hashed the entire batch tensor as a single key. Under `refresh_every=5`, step 0 with logical batch `[A, B]` (teacher class 1) followed by step 1 with `[A, C]` (teacher class 2) caused a cache miss for the full batch `[A, C]`, recomputing A's targets with class 2. Conversely, microbatch chunking `[A]` followed by `[C]` hit the cache for A, returning class 1 (max diff 1.0). Thus gradient accumulation changed the training objective.
   - **Resolution**:
     - Refactored `RegionManager` to maintain a per-sample identity cache. Each sample `i` in batch `B` is keyed independently by its own image bytes, validity mask bytes, and normalized loader metadata, combined with per-sample cadence `interval = step // refresh_every`.
     - In batch calls, per-sample targets are individually retrieved from cache or computed fresh, and combined into batch tensors (`region_labels`, `region_weights`, `target_probabilities`).
     - Sample A's cache key is completely invariant to companion identity or batch slot position (`[A, B]` vs `[A, C]` vs `[C, A]`).
     - Executable witness verified: full logical batch B=2 and chunked B=1 both retain cached A (class 1) with **max difference strictly 0.0**.
     - Bounded LRU eviction capacity and exact state serialization/resume are fully preserved.

2. **Per-Sample Physical Displacement Metric (`self_audit/volume/temporal.py`)**:
   - **Defect**: `temporal_consistency` previously evaluated `metadata[0]` spacing for the entire batch. For duplicate dx=2 registration pairs with batch size B=2 and anisotropic spacings `[1, 1, 4]` and `[3, 1, 4]`, sample 0 (2mm) and sample 1 (6mm) reported mean displacement 2.0mm instead of 4.0mm.
   - **Resolution**:
     - Evaluates physical displacement in mm per sample using each sample's distinct `spacing_xyz_mm`.
     - Duplicate dx=2 pair now correctly reports mean physical displacement **4.0mm**.
     - Validates that metadata length matches batch size `B`, and all spacing values are 3 finite positive numbers (failing closed otherwise).
     - Verifies retention of finite directional autograd gradients into student logits.

All 36 owned tests (22 region, 14 temporal) pass. All 13 Astra adversarial checks in `review_geometry_aux.md` continue to pass. Repository integrity check (`scripts/check_repo.py`) reports `PASS`.

---

## 2. Technical Implementation Details

### 2.1 Per-Sample Caching Architecture (`self_audit/volume/regions.py`)

- `compute_sample_identity_hash`:
  Accepts either single volume items `(1, C, Z, H, W)` / `(C, Z, H, W)` or full batches `(B, C, Z, H, W)`. Hashes contiguous bytes and normalized metadata item dictionaries.
- `RegionManager.targets`:
  Iterates over each batch item `bi in range(b)`:
  - Generates independent sample hash `compute_cache_hash(sub_img, sub_val, sub_meta, step, mode, ...)`.
  - Queries `self._cache`: on hit, updates LRU order and retrieves cached target; on miss, computes deterministic supervoxels, boundary weights, and boundary splitting/rejection for the sample, adding to `self._cache` and trimming to `max_cache_size`.
  - Combines individual sample targets into unified `RegionTargets` preserving batch dimension order.
  - Returns `self._cached_targets` directly on single-sample or all-hit batches to preserve exact object identity.

### 2.2 Per-Sample Spacing Metric (`self_audit/volume/temporal.py`)

- Replaced single `metadata[0]` lookup with per-batch element iteration:
  ```python
  disp_mm_list = []
  for bi in range(student_logits.shape[0]):
      sp = metadata[bi].get("spacing_xyz_mm")
      sx, sy, sz = float(sp[0]), float(sp[1]), float(sp[2])
      u_x, u_y, u_z = flow_xyz[bi, 0], flow_xyz[bi, 1], flow_xyz[bi, 2]
      disp_mm_i = torch.sqrt((u_x * sx) ** 2 + (u_y * sy) ** 2 + (u_z * sz) ** 2).mean().item()
      disp_mm_list.append(disp_mm_i)
  metrics["mean_displacement_mm"] = float(np.mean(disp_mm_list))
  ```

---

## 3. Test Evidence

### 3.1 Owned Test Suite
```sh
rtk proxy python -m pytest tests/test_volume_regions.py tests/test_volume_temporal.py
============================== 36 passed in 1.93s ==============================
```

- `tests/test_volume_regions.py` (**22 passed**):
  - `test_per_sample_cache_accumulation_and_companion_invariance`: verifies full B=2 vs split B=1 repeated-A/changing-companion invariance with max diff 0.0, and `[C, A]` permutation invariance.
  - `test_r2_boundary_rejection_no_cross_class_interior_pooling`: Astra R2 witness (loss < 1e-4, correct student gradients, zero cross-class pooling, accurate valid voxel counts).
  - `test_reject_missing_detached_ema_source`: missing detached probability rejection before cache/build, features-only rejection.
  - `test_lru_resume_capacity_restoration`: `max_cache_size` persistence and eviction across resume.
  - Prior 18 region tests: determinism, physical spacing awareness, fail-closed metadata, stop-gradient, etc.

- `tests/test_volume_temporal.py` (**14 passed**):
  - `test_temporal_per_sample_spacing_displacement_metric`: duplicate dx=2 pair B=2 with spacings `[1, 1, 4]` and `[3, 1, 4]` reports mean displacement 4.0mm; metadata length and positive spacing validation; finite student gradients.
  - `test_unobservable_constant_image_rejection`: constant all-ones, all-zeros, and flat images reject pre-apodization, yielding 0 accepted voxels and 0 student gradient.
  - Prior 12 temporal tests: identity diagnostic, known motion phantom, support-aware invalid texture isolation, fractional boundary contamination rejection, out-of-bounds rejection, subpixel motion, etc.

### 3.2 Astra Review Independent Adversarial Checks (`review_geometry_aux.md`)
```json
[
  {"check": "sparse_frames", "status": "PASS"},
  {"check": "z_padding", "status": "PASS"},
  {"check": "source_geometry_binding", "status": "PASS"},
  {"check": "synthetic_identity_gate", "status": "PASS"},
  {"check": "cross_patient_cache", "status": "PASS"},
  {"check": "immutable_targets", "status": "PASS"},
  {"check": "region_boundary", "status": "PASS"},
  {"check": "registration_invalid_input", "status": "PASS"},
  {"check": "registration_out_of_bound", "status": "PASS"},
  {"check": "through_z_motion", "status": "PASS"},
  {"check": "forbidden_mask_io", "status": "PASS"},
  {"check": "forbidden_role_allowlist", "status": "PASS"},
  {"check": "unknown_retention", "status": "PASS"}
]
{"pass": 13, "fail": 0}
```

### 3.3 Repository Integrity
```sh
rtk proxy python scripts/check_repo.py
{"status": "PASS", "architecture": "dinov3_convnext_tiny_native_decoder_v2", "supervision": "semi_supervised; dense TRAIN subset incl audit; not no-GT"}
```

---

## 4. Source SHA-256 Hashes

| File | SHA-256 |
|---|---|
| `self_audit/volume/regions.py` | `1474c539eb8ca64e92b6eaa367107a45206b1eefad55394034e14c54f200bb9c` |
| `self_audit/volume/temporal.py` | `854bf74222402446d9d984dfe1eccabec44dd152a7ef52cdd43932e48f26881d` |
| `tests/test_volume_regions.py` | `42e78808f96858a15d0ea794ac3f171accd26e1fe5b878b6e598e013583d9f59` |
| `tests/test_volume_temporal.py` | `32dc8df4c4939115ae3036d27b2c889b8cbd423a5181e233b2495d2c32e15afc` |
| `reports/native_volume_pilot/agy_worker_regions_temporal.md` | this report |
