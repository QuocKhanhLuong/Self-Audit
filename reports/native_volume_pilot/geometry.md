# G0 native physical geometry/data delivery

Status: IMPLEMENTED / G4 REPAIRED (CPU/manufactured software checks only; no
patient assets or GPU run).

Owned paths:

- `self_audit/volume/geometry.py`: versioned `self_audit.volume.geometry.v1`
  sidecar validation, affine/spacing checks, world/voxel conversion and
  conservative `invert_probabilities` native-grid export.  Native export now
  carries geometric support through resampling, requires a full linear
  interpolation stencil, and normalizes probabilities only where support is
  complete.  Physical NIfTI validation checks source hashes, mm units,
  orientation, frame mapping, optional mask/cine-frame voxel correspondence,
  and the canonical raw-XYZT -> percentile-normalized TZYX prepared-image and
  prepared-mask voxel bridge. U-sidecar mask references are recursively
  rejected before raw validation/path/hash access.
- `self_audit/volume/data.py`: hash-pinned `VolumeTrainingData` bridge from
  `self_audit.data.v2`, contiguous native slabs, global z-neighbour triplets,
  coupled XY crops/flips, optional explicit physical resampling, separate
  `geometric_valid`/`label_valid` masks, UNKNOWN=255 handling and adjacent-
  frame temporal pairs.  Labeled sampling draws only declared mask frame IDs;
  out-of-grid Z planes retain their physical indices but are zero-filled and
  invalid, while valid edge centers retain the historical clipped triplet.
- `self_audit/volume/fixtures.py`: deterministic manufactured fixture creator;
  `make_volume_fixture(folder, ...)` returns
  `(manifest_path, manifest_sha256, geometry_path, geometry_sha256)`.
- `scripts/volume_prepare.py`: executable image-only sidecar preparation CLI;
  rejects missing physical affine/source metadata, caller/manifest identity
  mismatches, duplicate TRAIN IDs and incomplete labeled source coverage, and
  accepts explicit `--physical-metadata` only after hash-pinned NIfTI
  image/mask/frame and prepared-payload correspondence. It prints
  `mask_paths_read: 0` for the image-only path and the actual labeled-source
  count when physical metadata is supplied (U/VAL/TEST masks are never
  discovered).
- `tests/test_volume_geometry.py`, `tests/test_volume_data.py`: affine/spacing,
  asymmetric affine/resampling support, full-stencil native inversion,
  NIfTI/frame correspondence, raw-to-prepared binding, sparse labeled frames,
  unknown/padding support, forbidden nested U mask fields, real
  identity/provenance gates and temporal grid coupling.

Verification receipt:

```text
rtk pytest -q tests/test_volume_geometry.py tests/test_volume_data.py
19 passed
rtk pytest -q
335 passed

Independent replay harness from review_geometry_aux.md: 12/13 checks passed;
the sole failure is the unrelated pre-existing region-boundary witness. The
G0/G4 data/geometry witnesses (sparse frames, Z padding, source-hash binding,
synthetic identity/provenance and raw-to-prepared physical path) pass.
```

The current test evidence is manufactured CPU/software evidence. No real TRAIN,
VAL or TEST masks were opened, no server/GPU was contacted, and no patient Dice
or throughput claim is made. The loader deliberately fails closed for missing
or inconsistent affine/physical metadata, synthetic identity mismatches,
forbidden nested U mask references, incomplete real raw provenance, and
raw-to-prepared image/mask correspondence failures. Real preparation still
requires the hash-pinned source geometry and raw NIfTI image/mask/frame
receipts. Physical resampling is explicit and records its reversible transform;
it does not create through-plane information.

Final repair source hashes (2026-10-26):

| File | SHA-256 |
|---|---|
| `self_audit/volume/geometry.py` | `3fdb0986e4ff572c6855743414fa1f443e631d57450fe3b7803e5e52f3e1c9bd` |
| `self_audit/volume/data.py` | `34d1a2727762ae1111865a71cde63cef000c56be4adc8d052116a3a52b321493` |
| `self_audit/volume/fixtures.py` | `f1b88049b6748f6ce50247a575500abad885b52a072ae2f759a9b0ad88810a90` |
| `scripts/volume_prepare.py` | `32929bb2e1fec70dbd6ff9ca2381c6312b2e0a3433a700e9854e486eeefed881` |
| `tests/test_volume_geometry.py` | `08c3dda96c2d8abb0896b64774476afc8d737dbae82d7ae408f73e6e95bfcc3f` |
| `tests/test_volume_data.py` | `36fc3f2e107dc8d653a447cde0064e64de3a869ef3528f5c423d29dfe21a7a76` |

Integration dependency: the integration owner should import
`VolumeTrainingData` from `self_audit.volume.data` and pass the four constructor
arguments from the prepared manifest/sidecar receipt. Export calls should pass
one emitted sample's `metadata` and C,Z,Y,X probabilities to
`invert_probabilities` before any argmax. New model code consumes the emitted
B,3,Z,H,W tensor and may use only the center channel where declared.

Real preparation uses the explicit CLI contract below (metadata JSON is keyed
by patient; U entries contain `source_image` only and never `mask_sources`):

```text
python scripts/volume_prepare.py --manifest TRAIN/train.json \
  --manifest-sha256 <manifest_sha256> --physical-metadata physical.json \
  --out TRAIN/geometry.json
```

Each physical entry supplies `affine_xyz_to_world_mm`, `orientation`,
`slice_order`, `temporal_frame_mapping`, `source_image: {path, sha256}` and,
for labeled TRAIN patients, `mask_sources` keyed by frame with raw mask and
same-frame cine source paths/hashes. Missing or inconsistent fields fail
closed; the prepared NPY payload hashes remain separately pinned to the v2
manifest.
