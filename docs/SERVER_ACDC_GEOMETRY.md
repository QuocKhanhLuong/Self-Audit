# Server ACDC prepared geometry revalidation

This is a bounded compatibility rule for revalidating recovered, already
prepared ACDC artifacts. It does not rerun normalization, modify source NIfTI
files, resample images, or launch training.

## Recovered-source evidence

The server investigation receipt is
`/workspace/selfaudit-20261101/receipts/ACDC_GEOMETRY_INVESTIGATION.json`.
The recovered preparation contains 202 source files, and its `train.json` has
the exact historical SHA256
`e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4`. All 18
VAL prepared image and mask hashes, raw cine hashes, and cine affine hashes
match `results/compact_acdc_l8_20261029/experiment_20261028/val18_native.json`
(SHA256 `cac4426393ca11959f56df2876869e825443916cdec02c6d95ec8da3fac0ca50`).

The frame-sidecar discrepancy is narrowly identified: 10 patients (20 frame
files) have the original placeholder affine `diag(-1,-1,1,1)` with sform code
2 and qform code 0. The corresponding cine headers have sform and qform codes
0, so nibabel derives the cine affine from its voxel spacings. The decoded
voxel array of each affected frame is exactly equal to its selected raw cine
frame. This establishes array correspondence; it does not establish an
independent physical-space transform for the frame sidecar.

## Validation contract

`self_audit.modal_data_prep.validate_prepared` keeps strict frame/cine affine
validation by default. The recovered artifact may be revalidated only by
explicitly setting:

```python
validate_prepared(
    raw,
    prepared,
    split,
    historical_val,
    allow_historical_placeholder_geometry=True,
)
```

The opt-in admits a mismatch only when all of these conditions hold:

1. The frame affine is exactly `diag(-1,-1,1,1)`, its sform matrix has that
   same value, its sform/qform codes are exactly 2/0, and the cine header's
   sform/qform codes are exactly 0/0.
2. `numpy.array_equal(frame.dataobj, cine.dataobj[..., frame_index])` succeeds.
3. The TRAIN manifest has an exact historical semantic proof: exact pinned
   bytes, or complete parsed-manifest equality after stripping only parent
   paths with every payload hash unchanged.
4. Every one of the 18 pinned VAL rows matches on prepared image hash, both
   prepared and manifest mask hashes, raw cine hash, and canonical cine affine
   hash (a 4x4 little-endian float32 matrix at NIfTI storage precision, plus
   the existing 1e-5 matrix tolerance).

An arbitrary affine difference, a changed frame array, an absent or changed
TRAIN proof, or any VAL hash/affine mismatch fails closed. Unlabeled TRAIN
records remain image-only; only their cine images are read. Reference frame
images and prepared masks are read only for labeled L8 and the pinned VAL
frames.

## Receipt fields

The returned validation receipt records, per selected frame, both the frame
and cine affines and their canonical SHA256 values; each header's coded sform
and qform matrices/codes; the exact array-bridge result; and whether the exact
historical placeholder signature was observed. The top-level geometry
compatibility receipt records the opt-in state, historical TRAIN semantic
proof, pinned VAL manifest hash, all-18 VAL match result, affected frame IDs,
and the physical-space limitation. The generated `val18_fresh_native.json`
continues to store the raw cine affine as authoritative and includes the
frame-sidecar bridge evidence for each VAL patient.

The compatibility result is not a claim that the placeholder affine is
correct in world coordinates. Exact voxel equality is the bridge, no resampling
is performed, and the original raw MRI files remain untouched.

## Verification

The synthetic geometry tests cover opt-in versus default strict behavior,
receipt contents, rejection of arbitrary affines, rejection of mismatched
frame voxels, missing or changed historical TRAIN proof, and VAL affine-hash
failure. These tests validate software behavior only; they do not establish
new patient or training evidence.
