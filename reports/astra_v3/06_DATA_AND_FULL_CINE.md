# Data and full-cine feasibility

**Local available data: 100 ACDC patients, 200 static phase volumes, 1,902 slices, zero 4D images in the audited root.** M&Ms and full-cycle cine are unavailable in the inspected locations. This is an inventory statement, not proof about every disk or remote server.

Root independently opened all 200 image headers, never GT arrays in the generator. All have unknown spatial units and affine column norms inconsistent with header zooms. See [root header inventory](evidence/root_header_geometry.json). An invertible numeric affine and `sform_code=2` do not certify scanner coordinates. Array-grid restoration/Dice can be evaluated when matching shapes/affines agree; physical distances, volumes and patient orientation remain unverified.

[The official ACDC dataset description](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html) describes cine acquisitions, phase metadata and variable cycle coverage. The local copy is not that complete acquisition. Worker E's Info.cfg frame counts are sidecar declarations, not observed intermediate frames; they cannot supply missing image data. ED and ES endpoints must never be substituted for adjacent temporal frames.

## Loader findings from executable code

- `scripts/preprocess_acdc.py:60` selects only ED/ES, requires the paired mask, opens mask arrays and saves them. It is unsuitable for mask-free training.
- That script **does preserve** `orig_affine`, orientation, shape, spacing and a resize chain in schema-v2 `metadata.json` (lines 85–134). Worker E's blanket “all geometry destroyed” claim is incorrect. Resized masks still cannot reconstruct original native GT boundaries.
- `VolumeSliceDataset` builds z-neighbor triplets, not full cine temporal triplets.
- Maskfree `discovery._frame_metadata` recognizes 4D axes and checks time units; `_resolve_protocol` explicitly rejects `cine_predictive` because temporal transport is not implemented. Auto routing remains spatial.
- Maskfree `ImageOnlyDataset`, `_native_stack` and `_build_unit_tensors` select a frame and z neighbors. Existing producer losses implement image objectives, not a complete v3 temporal teacher.
- Supervised `src/self_audit/data/mnms.py` pairs masks; its default raw→unified map is `{0:0,1:3,2:2,3:1}`. The new loader must enumerate images independently. This numeric mapping belongs in the evaluator for GT, not in image discovery.

## Image-only cine contract to implement

```text
study_id, patient_id, dataset, acquisition_id, vendor/site if actually supplied
source_image_hash, native_shape_XYZ[T], raw_dtype
native_affine, qform, sform, codes, declared_units
header_spacing, affine_spacing, geometry_consistency_status
axis_order, orientation_codes, trusted_patient_axes OR null
frame_index_t, total_frames_T, time_stamps OR null, timing_units/status
phase_label ED/ES/cine/unknown, phase_metadata_source OR null
slice_index_z, z_total, spatial_neighbors, temporal_neighbors
neighbor_validity, wrap_policy, acquisition_continuity_status
image_triplets: prev/cur/next each [3,H,W], absent neighbor explicitly null
normalization_parameters, resize/pad/crop chain, inverse-grid transform
split_id, metadata_allowlist, all source/transformation hashes
```

For a genuinely complete periodic cine, wrap time only if acquisition metadata confirms cycle coverage. Otherwise boundary neighbors are absent/masked, not fabricated. For z boundaries use declared replication with a neighbor-validity flag. Short-axis slices may have separate breath holds; a rectangular `[X,Y,Z,T]` array is not proof of perfect cross-slice temporal registration.

Use full FOV with aspect-preserving resize and pad to 224; no mask-based heart crop. Store original shape and exact resampling semantics. With `align_corners=False`, inverse pixel-center mapping is `(u+0.5)*input_size/output_size-0.5`, after undoing padding. Do not swap row/column conventions implicitly. Nibabel voxel-to-world convention is RAS; axis code `LPS` describes voxel axis directions, not a claim that the affine outputs LPS millimeters. Worker E's sample mapping omits the half-pixel convention and assumes both units and coordinate convention; it is not an accepted physical roundtrip test.

Validate reconstruction with non-square synthetic landmarks and image-only native grids, plus trusted acquisition geometry before physical exports. Never infer missing metadata. Preserve the original affine bytes in an array-grid export while flagging geometry as unverified; clinical physical export remains blocked until repaired from an authoritative source.

## Splits and cross-dataset contract

Historical split SHA256: `bc42574017276a6defc668bf533330fd41a64a5e90a75ea1b8958fc9e469265a`, 80 train/20 development, disjoint patients. The 20 have prior development use and are not an untouched final test set. Round-0 chose four patients per side by a fixed image-independent hash. Training must never fit on development images unless a separately declared transductive setting is used.

Two separate M&Ms experiments are mandatory: (1) native application of the same recipe, with independent patient splits and provenance; (2) frozen ACDC-trained student evaluated on M&Ms without adaptation. Contrary to Worker E, the second experiment **requires** retaining the ACDC checkpoint. Joint ACDC+M&Ms training is not cross-domain generalization. Add vendor/site strata where genuine metadata exists; do not invent site labels or require random initialization for the frozen-transfer arm.

M&Ms dataset-specific timing/geometry and actual label convention must be verified on acquisition staging. The official [M&Ms site](https://www.ub.edu/mnms/) was temporarily unavailable during this audit. No M&Ms accuracy, timing or full-cine feasibility result is claimed.
