# Server evaluation reference preparation

`scripts/prepare_server_evaluation.py` builds a separate, hash-pinned
`self_audit.data.v2` evaluation manifest for an explicitly selected ACDC or
M&Ms split role. It prepares image arrays and reference masks only. It does not
run training, inference, or metrics.

The caller must provide the raw root, exact split JSON and SHA-256, role
(`development` selects the explicit `val` IDs; `test` selects the explicit
`test` IDs), and a new output directory. The adapter requires explicit
TRAIN/VAL/TEST lists in the split and rejects overlap or conflicting
representations. It opens raw payloads only for the selected role IDs; TRAIN
and unlabeled image or mask payloads are neither opened nor searched by ID.

## ACDC

The ACDC adapter uses native `<patient>_4d.nii[.gz]` cines and only that
patient's `<patient>_frameNN.nii[.gz]` and matching
`<patient>_frameNN_gt.nii[.gz]` reference pairs. It requires one cine per
selected patient, matching frame/reference shapes and affines, and an exact
voxel equality bridge from each frame file to the declared zero-based cine
frame. The cine affine is retained in the evaluation manifest. If frame files
have placeholder affine headers, the receipt and record preserve the actual
frame affine delta from the cine affine; the adapter does not treat those
headers as equal physical geometry.

The TEST role resolves only the pinned split's `test` list. It is intended for
the already authorized final ACDC TEST50 reference preparation after VAL best
selection. This command prepares references only; a separate frozen inference
and evaluation action is still required.

```bash
python scripts/prepare_server_evaluation.py \
  --dataset acdc \
  --raw-root /workspace/selfaudit-20261101/raw/acdc_test50 \
  --split-json splits/acdc_reviewed_locked_train80_val18.json \
  --split-sha256 b826eaf494760079aa559b0c0c12f7444951d779ebbb5ed291c0398f20bf835f \
  --role test \
  --out /workspace/selfaudit-20261101/prepared/acdc_test50_reference_eval
```

Use `--role development` to prepare the same pinned split's `val` IDs. The
adapter has no default TEST or development split and does not modify the locked
TRAIN/VAL/TEST membership.

## M&Ms

M&Ms requires an exact SHA-pinned CSV with the `External code`, `ED`, and `ES`
columns. ED and ES values are interpreted as zero-based frame indices. The
selected VAL or TEST IDs must have metadata rows, and the eight known
quarantined duplicate IDs are removed and listed in the receipt.

Raw files are resolved by exact external-code names: `<External code>_sa.nii`
or `.nii.gz` and `<External code>_sa_gt.nii` or `.nii.gz`, at any depth under
the caller's raw root. Ambiguous or missing matches fail. The mask must be a
native XYZT volume with the cine's spatial shape and affine. For ED and ES,
scaling roundoff after integer rounding must be strictly below `1e-4`; the
canonical shared mapping utility is then applied once:

| Raw mask | Canonical mask |
| --- | --- |
| 0 | 0 |
| 1 | 3 |
| 2 | 2 |
| 3 | 1 |
| 255 | 255 |

No split, M&Ms metadata CSV, or metadata hash is guessed. Supply the actual
server paths and SHA-256 values explicitly:

```bash
python scripts/prepare_server_evaluation.py \
  --dataset mnm \
  --raw-root "$MNM_RAW_ROOT" \
  --split-json "$MNM_SPLIT_JSON" \
  --split-sha256 "$MNM_SPLIT_SHA256" \
  --metadata-csv "$MNM_METADATA_CSV" \
  --metadata-csv-sha256 "$MNM_METADATA_CSV_SHA256" \
  --role development \
  --out "$MNM_EVALUATION_OUTPUT"
```

The environment values above must be set to the exact approved inputs; the
command has no M&Ms TEST default.

## Outputs and provenance

The output directory contains `<role>.json`, one normalized TZYX float32 image
per prepared patient, one canonical uint8 ZYX mask per referenced frame, and
`PREPARATION.json`. Manifest image and mask paths are relative to the manifest;
native geometry source paths are absolute and accompanied by their source file
hashes. The manifest stores `axes: ["T", "Z", "Y", "X"]`, the native cine
affine, and `geometry_source_image: {path, sha256}` for every patient.

The receipt binds the configuration, source code, split, metadata, cohort,
prepared arrays, raw source files, frame geometry checks, and output manifest
hashes. It records zero TRAIN payload reads, zero TRAIN mask discoveries, zero
unlabeled mask discoveries, and zero inference, metric, or training runs. A
failed validation removes the staging directory and does not publish a partial
output directory. The manifest's `normalization` is `prepared`, matching
compact inference's already-normalized input mode.
