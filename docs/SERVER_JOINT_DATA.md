# Server joint ACDC + M&Ms data contract

This is the prepared-data handoff for the supervised compact server protocol:
ACDC L4 + M&Ms L4 = total L8, seed 42. It uses the existing
`self_audit.data.v2` shared manifest and quarantine rules. The compact model is
supervised from the eight declared TRAIN masks; the protocol is not no-GT.

## Source audit and current missing inputs

The repository has one hash-pinned prior M&Ms preparation report:

- `reports/baseline_recovery/MNM_PREPARATION_SOURCE.json`
- SHA-256: `82f6a05b38fa8e69607b0fd6fcecf0062809353dfb47cc8107cbbfb865bf030b`

It records the older prepared manifest
`/workspace/Self-Audit/data/prepared/MnM_L8_S42_verified/train.json` at SHA
`a7fc92227c555d0de3895b5b37154e49bd3de5b44d26958bde34863d86dfaae3` and a
prior loading artifact at
`/workspace/Self-Audit/reports/automatic_pilot_pro4000/mnm_loading_smoke.json`
with declared SHA `f00c9a6e7b6d8c67d8bcafba14108aa77283ab1513250c2ac2890ec419e0cec1`.
The previous manifest is L8/U162, so it cannot establish the new L4 selection
from the full source pool. Its server artifact and prepared manifest are not in
this checkout. The checkout also has no `data/` directory.

That report pins the eight M&Ms quarantine IDs already used by
`shared_data.py`: `A7G0P5`, `A8C9H8`, `C5Q2Y5`, `C8J7L5`, `C8O0P2`, `E9L4N2`,
`K3R0Y7`, and `Q0Q1Y4`. Its duplicate audit states that original splits and
source files were left unchanged, and that all members of those duplicate
groups are excluded from derived training and evaluation manifests.

The coordinator located these Drive archive entries as metadata only:

- `Training-001.zip`, Drive ID `1Zy9mcXndDXzjsOcuthVOmdf7U725zU1K`
- `M&M-20251118T142205Z-1-002.zip`, Drive ID `1Fhtu7fIrT3LTBF_vyJoBY__SBKgDJraD`
- `Testing-003.zip`, Drive ID `19MxZEMIgOWyf-GEIkZkzCkJzx-Y21PZ9`

The prepared-data folder `prepared_data/mnm` (Drive ID
`1y3rUZ0RxleyrMNm4qonzaVBdTRt9_3st`) is an unverified candidate source; its
preprocessing lineage is not established by the prior receipt. The exact
authoritative full split metadata path and SHA are still missing from this
checkout. The worker did not access Drive, SSH, or download these files.

Do not run real preparation until the coordinator supplies the extracted
official M&Ms root, the exact locked split metadata and its SHA-256, a hash-pinned
raw source manifest built from that split, prepared ACDC source manifest, and
complete prepared evaluation maps. `prepare_server_mnm.py` does not guess the
cohort from directory contents, use MnM2, or treat the old L8 manifest as the
new source split.

## Inputs to `prepare_server_mnm.py`

The command takes a prepared single-domain ACDC TRAIN manifest, an extracted
official M&Ms root, a hash-pinned M&Ms source manifest, and a hash-pinned index
of generic evaluation manifests:

```bash
python scripts/prepare_server_mnm.py \
  --acdc-manifest /path/to/acdc_train.json \
  --acdc-manifest-sha256 <acdc_train_sha256> \
  --mnm-root /path/to/extracted/official/MnM \
  --mnm-source-manifest /path/to/mnm_raw_source.json \
  --mnm-source-manifest-sha256 <mnm_source_sha256> \
  --evaluation-maps /path/to/evaluation_maps.json \
  --evaluation-maps-sha256 <evaluation_maps_sha256> \
  --out /path/to/new/joint_prepared
```

The prior receipt path and SHA above are pinned defaults. The M&Ms source
manifest uses `self_audit.mnm.raw_source.v1`; it points at the exact split JSON
under `--mnm-root` and lists every official TRAIN ID exactly once, divided into
annotated candidates and image-only U records. The split JSON must contain
explicit patient-ID arrays under `train`, `val`, and optional `test` keys (or
the equivalent `train_patients` / `splits` / `locked_splits` representation
used by `prepare_acdc.py`). The union of raw-source TRAIN records must match the
pinned TRAIN IDs exactly. The split file itself is SHA checked before any image
or mask payload is read.

Example source-manifest structure:

```json
{
  "schema": "self_audit.mnm.raw_source.v1",
  "dataset": "M&Ms",
  "synthetic": false,
  "source_split": {
    "path": "Training-001/locked_train_val_test.json",
    "sha256": "<exact_split_sha256>"
  },
  "train_labeled_candidates": [
    {
      "id": "A5C2D2",
      "image": "Training-001/Training/Labeled/A5C2D2/A5C2D2_sa.nii.gz",
      "image_sha256": "<raw_cine_sha256>",
      "masks": {
        "9": {
          "path": "Training-001/Training/Labeled/A5C2D2/A5C2D2_sa_gt.nii.gz",
          "sha256": "<raw_mask_sha256>",
          "source_frame": 9
        }
      }
    }
  ],
  "train_unlabeled": [
    {
      "id": "B1C2D3",
      "image": "Training-001/Training/Unlabeled/B1C2D3/B1C2D3_sa.nii.gz",
      "image_sha256": "<raw_cine_sha256>"
    }
  ]
}
```

Every record path is relative to `--mnm-root` and every raw file hash is
mandatory. A frame key is an explicit zero-based prepared time index. A
3D-per-frame mask omits `source_frame`; a 4D mask must provide its explicit
source frame. The script does not infer ED/ES indices or search for mask files.
Unlabeled records must contain only `id`, `image`, and `image_sha256`; a mask
path on U is rejected before payload I/O.

The evaluation map index uses `self_audit.server_evaluation_maps.v1`:

```json
{
  "schema": "self_audit.server_evaluation_maps.v1",
  "maps": {
    "acdc_val": {
      "domain": "acdc",
      "manifest": "/path/to/acdc_val.json",
      "sha256": "<acdc_val_manifest_sha256>"
    },
    "mnm_val": {
      "domain": "mnm",
      "manifest": "/path/to/mnm_val.json",
      "sha256": "<mnm_val_manifest_sha256>"
    }
  }
}
```

Map names are caller-defined. Each manifest is independently hash-checked and
must use one of compact inference's supported evaluation schemas and roles.
The index may include validation, test, or external-frozen maps. Their patient
IDs must cover every declared ACDC source `evaluation_id` and every M&Ms VAL/TEST
ID from the pinned split. The preparation reads only evaluation image arrays
for overlap checks; evaluation masks remain unopened for freeze-before-score
evaluation.

## Preparation order and receipts

The script performs these checks in order:

1. Verify the ACDC manifest, M&Ms source manifest, split metadata, evaluation map
   index, each evaluation manifest, and the pinned prior M&Ms receipt.
2. Check TRAIN IDs against the official pinned split and apply the existing
   ACDC/M&Ms quarantine before any raw patient image or mask read.
3. Select ACDC L4 and M&Ms L4 with the existing deterministic `label_subset`
   hash ordering and seed 42. The candidate pool is disclosed as
   `restricted_manifest_labeled`; ACDC selection is limited to the labeled IDs
   in its input manifest.
4. Normalize each clean TRAIN M&Ms cine independently using its image-only 0.5
   and 99.5 percentiles, clip to `[0,1]`, transpose native XYZT to TZYX, and do
   not resize. ACDC inputs are checked as already prepared normalized TZYX.
5. Validate train/evaluation IDs and image byte/voxel overlap before opening
   any reference mask.
6. Open only selected M&Ms TRAIN4 masks, validate native geometry and integer
   representation roundoff, then apply raw labels `0->0`, `1->3`, `2->2`,
   `3->1`, and `255->255` exactly once into new canonical arrays. Validate only
   selected ACDC TRAIN4 masks. No U mask is discovered, hashed, or opened.

The new output directory must not exist and must be outside every source root.
The script writes `train.json`, `PREPARATION.json`, and `EVALUATION_MAPS.json`.
The shared TRAIN manifest has namespaced IDs, exact 4+4 label counts, and
image-only U records. The preparation receipt binds all input paths and hashes,
selected IDs, quarantines, preprocessing, source/output mask hashes, frame
roundoff, mapping, and mask-access counters. Real image-quality, GPU, and
training claims are outside this preparation receipt.

## Compact domain sampling and resume

For shared manifests, `CompactTrainingData.sample_indices` alternates the
domain by global outer step: ACDC on even steps and M&Ms on odd steps. It then
draws the supervised patient uniformly from that domain's L pool, a different
audit patient from the same L pool, and the U patient from the same domain's U
pool. Every even-length prefix therefore has exact 50/50 domain exposure. All
patient draws use the compact trainer's checkpointed NumPy data RNG, so an exact
resume continues the same patient schedule. ACDC-only manifests keep the
pre-existing compact sampler behavior.

The coordinator-owned runner supplies the generated TRAIN manifest and its
SHA-256 to the supervised compact trainer and reads arbitrary map names from
`EVALUATION_MAPS.json` for validation/evaluation. There is one student and one
fixed initialization; no ACDC segmentation checkpoint is reused for the joint
protocol.

## Verification boundary

`tests/test_server_joint_data.py` uses manufactured arrays and a fake NIfTI
reader to check split/hash enforcement, selected-only mask reads, label
remapping, image-only U records, generic map handling, exact 50/50 domain
sampling, sampler-state replay, and unchanged ACDC-only compact sampling.
These tests are software checks only. No patient preparation, dataset download,
GPU run, or patient training was performed by this worker.
