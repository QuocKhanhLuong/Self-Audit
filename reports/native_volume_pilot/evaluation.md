# Frozen native evaluation / deployment report

## Implemented scope

Owned files are `self_audit/volume/evaluation.py`,
`scripts/volume_evaluate.py`, `scripts/volume_export.py`,
`tests/test_volume_evaluation.py`. Existing training, model, loss, engine and
configuration files were not edited.

`predict_native_volume` performs complete TZYX image-only inference with
contiguous Z slabs and XY tiles, explicit overlap, geometric support, and
probability averaging after `geometry.invert_probabilities`. Argmax is applied
only on the merged native grid. `evaluate_manifest` accepts a hash-pinned
development/external-frozen evaluation manifest and a separate hash-pinned
geometry sidecar, chooses student or EMA explicitly, freezes all predictions
before any optional mask read, and reports frame confusion plus exact existing
patient-frame mean Dice aggregation. Labels are inaccessible by default and
require `allow_labels=True`/CLI `--score`.

`load_checkpoint_student_only` and `export_student` provide a compact student
payload and strict config/source/payload hashes. Export loading never creates a
teacher, optimizer, region cache or external bank and does not require the
original pretrained weight file. It calls the model owner's explicit
`build_volume_model(..., architecture_only=True)` seam; for a pinned DINO
architecture that seam loads only the clean local source entrypoint with
`pretrained=False`. No process-global monkeypatch, URL/download, or fallback
architecture is allowed. Synthetic provenance is explicit and must match the
manifest/checkpoint rather than defaulting to true.

## Verification

```text
rtk python -m pytest -q tests/test_volume_evaluation.py
```

Result: **17 passed** (CPU, Python 3.11.16, torch 2.14.0; 2026-10-26). The
fixtures cover odd native shapes and nonidentity affine, partial support and
overlapping slab/tile merge, image-only versus explicit label reads, student
versus EMA choice, frozen model-state preservation, independent patient/frame
aggregation, strict manifest/geometry/split rejection, student-only export
payload tamper rejection, translated-mask and frame/source metadata rejection,
nonidentity/permuted affine acceptance, masked-target support, and CLI
execution from an arbitrary working directory. The follow-up adds a true
neighboring-receptive-field invalid-value invariance oracle and manufactured
real-format NIfTI image-only mismatch/positive controls. Image-only evaluation
was verified with nonexistent mask paths and read zero labels.

```text
rtk python -m py_compile self_audit/volume/evaluation.py scripts/volume_evaluate.py scripts/volume_export.py tests/test_volume_evaluation.py
rtk git diff --check
```

Both pass. Existing compatibility checks also passed (2 selected tests) for
`tests/test_volume_integration_followup.py::test_frozen_evaluation_does_not_mutate_source_model`
and `tests/test_volume_resume.py`; concurrent region/temporal integration
failures are outside this ownership and were not changed.

The geometry repair adds an explicit frozen-mask metadata contract. Unknown
mask fields, translated or otherwise contradictory native affines, units/frame
claims, invalid source hashes, and mismatched cache frame identities fail
closed. For non-synthetic scoring, the hash-pinned sidecar must provide a raw
image plus one raw mask source per cached frame; after all image-only
predictions are frozen, `validate_nifti_correspondence` verifies raw NIfTI
units/affines/frame mapping and exact raw-to-cache image/mask voxel
correspondence before metrics are computed. Image-only mode never opens mask
paths or raw mask sources. `evaluate_frozen` now applies the same geometric
and label-valid support to targets before confusion, preventing false
negatives from invalid target voxels.

The P1 configured-support repair now zeroes every native-invalid image voxel
before resampling/context construction, carries declared support through the
same full linear interpolation stencil as the image, and zero-pads unsupported
FOV tiles. After inverse probability merge, native support is intersected again
with the caller's `native_valid` mask and unsupported probabilities are zeroed.
Configured tiles/slabs are first averaged on the complete resampled current
frame and inverted once, so linear native stencils cannot be split at tile or
shifted-slab boundaries; constant-class controls are tile/probability invariant.
The independent convolutional oracle confirms that changing an invalid voxel
cannot alter supported predictions, while the raw model remains intentionally
neighbor-sensitive. Real image-only manifests now pass the already-loaded
prepared TZYX image to `validate_nifti_correspondence` before inference (with
`masks=None`); a shape/affine/unit-valid but voxel-mismatched raw XYZT NIfTI is
rejected before model calls, and a canonical normalized bridge is accepted
without reading labels.

Evaluation now consumes the checkpoint's declared `geometry` recipe. For
`configured`/`configured_physical_fov` modes it requires positive
`spacing_xyz_mm` (and the declared FOV where required), reuses the native
resampling bridge, applies configured flip, fixed physical XY-FOV windows and
FOV-derived slab depth (with invalid edge padding), and passes each tile's
reversible resample transform through
`invert_probabilities`. Predictions are therefore merged and argmaxed on the
native grid; configured spacing cannot silently fall back to native-only
inference. A resampling probe confirms configured `(Z,Y,X)` tiles differ from
native geometry while returned probabilities/support retain native shape.

Scoped geometry/evaluation checks: **25 passed** (`tests/test_volume_evaluation.py`
and `tests/test_volume_geometry.py`); `python scripts/check_repo.py`: **PASS**.
The broad `tests/test_volume_cli.py` run was **NOT ACCEPTED** at this moving
snapshot because shared config changes reject V04/R01/I01 on a
`native_z/native geometry cannot declare spacing_xyz_mm` validation error; no
evaluation source caused that failure. No patient arrays, real masks, GPU,
network or long training were used.

This is CPU/manufactured software evidence only. No patient arrays, U/VAL/TEST
labels, GPU run, network, download, or long training was used; patient Dice is
**NOT RUN**. The coordinator still owns the repository full-suite gate and
integration exports from `self_audit.volume.__init__`.

## API receipt and hashes

```text
predict_native_volume(model, image_tzyx, affine_xyz_to_world_mm, slab_depth=3,
                      tile_size=256, tile_overlap=0, slab_overlap=0,
                      native_valid=None, device="cpu")
evaluate_manifest(model, manifest, manifest_sha256, geometry, geometry_sha256,
                  prediction_model="student", allow_labels=False, synthetic=False,
                  config=None, predictions_dir=None)
load_checkpoint_student_only(checkpoint, checkpoint_sha256, device="cpu")
export_student(model, folder, config=..., source_checkpoint_sha256=...,
               prediction_model="student", synthetic=...)
```

Exact working-tree SHA-256 receipt:

```text
rtk shasum -a 256 self_audit/volume/evaluation.py scripts/volume_evaluate.py scripts/volume_export.py tests/test_volume_evaluation.py
```

```text
90e5b7207af274a841a5944a33775e57e9d656af5730f8a9289e6c4455ae82c6  self_audit/volume/evaluation.py
9a73e678ae9f8b8b0829ce7cb087d2c6f6121a4af6713f92fbc29f5fc99e5acf  scripts/volume_evaluate.py
0ed301b677864d50ff949b4cc4a180b8929d1dd6145bd93aab91eb763c84a247  scripts/volume_export.py
1bb1eb0295b4a0044468c2748322a37664601669668470450c13c1f5beeaadd8  tests/test_volume_evaluation.py
```

The coordinator should regenerate this receipt after any integration edit;
source digests intentionally include the complete `self_audit/volume/*.py`
package and therefore change when another owner lands code.
