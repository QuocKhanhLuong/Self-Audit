# Cardiac Dataset Adapters and Four-Source Training Design

## Status

Approved design for implementation after repository and local-data audit.

## Goal

Add safe, auditable adapters for CMR-MULTI and CMRxMotion and allow an
explicitly configured mixed-training cohort containing ACDC, M&Ms, CMR-MULTI,
and CMRxMotion, while preserving the current Self-Audit model, preprocessing,
losses, optimizer, schedule, inference, and checkpoint flow.

The change is data-pipeline work. It must not overwrite or modify
`data/CMR-MULTI/` or `data/CMRxMotion/`.

## Evidence-locked current contract

The active repository uses a 2.5-D, center-slice contract, despite the task
brief using the broader term “3.5D”:

```text
model input:       [B, 3, H, W]
channels:          z-1, z, z+1 in one spatial volume/unit
target:            [B, H, W] mask for the center z slice
boundary:          replicate (z=0 -> [z0,z0,z1], last -> [z-1,z,z])
resize:            in-plane H/W only, default 256x256
normalization:     per-volume 0.5/99.5 percentile clipping then z-score
through-plane:     no interpolation/resampling
labels:            0=Background, 1=RV, 2=MYO, 3=LV
augmentation:      identical geometric operation on all three channels and mask
```

This contract is evidenced by `self_audit.data.common`,
`self_audit.models.encoder`, the active YAML schema, and `docs.md`. Existing
ACDC and M&Ms loaders remain behaviorally unchanged.

## Scope and non-goals

In scope:

1. Audit scripts and CSV reports for both new datasets.
2. Deterministic CMR-MULTI flattened-Z/T reconstruction with explicit
   confidence and manual-review flags.
3. Read-only CMRxMotion archive discovery, image/mask pairing, missing-GT
   reporting, and subject/acquisition/phase parsing.
4. Dataset-specific raw-label remapping into the locked common schema.
5. A shared logical-unit/cardio 2.5-D dataset implementation that reuses the
   current normalization, resize, context, boundary, and tensor code paths.
6. Unified configuration support for four-source mixed training.
7. Subject-level split manifests and leakage tests.
8. Dataset-balanced, subject-aware sampling for mixed training.
9. Visual QC overlays, unit tests, loader smoke tests, and model forward tests.
10. README and compatibility documentation.

Out of scope:

- changing `SelfAuditNet`, the encoder channel count, loss, optimizer,
  scheduler, curriculum, inference modes, or checkpoint format;
- isotropic or through-plane resampling;
- fake masks for unlabeled CMRxMotion volumes;
- PNG-slice export as the primary preprocessing representation;
- silently accepting an ambiguous CMR-MULTI Z/T factorization;
- changing the source archives or source NIfTI files.

## Dataset findings and restrictions

### CMR-MULTI

The local `CINE_MULTI/SAX_TR` tree contains 105 image files and 105 matching
annotation files. The local workbook provides SAX patient IDs and direct image
and annotation paths. The observed arrays have shape `[X,Y,N]`; raw labels are
`{0,1,2,3}` and image/mask shapes and affine matrices match in the sampled
audit.

Only SAX cine is eligible for this project. The public CMR-MULTI metadata
defines SAX cine labels as raw `1=LV myocardium`, `2=LV cavity`, and `3=RV
cavity`; the adapter therefore uses:

```python
CMR_MULTI_RAW_TO_COMMON = {0: 0, 1: 2, 2: 3, 3: 1}
```

The flattened third axis is not treated as spatial depth until a per-case
candidate is verified. Candidate factors use configurable ranges with the
audited defaults `T in [18,35]` and `Z in [6,24]`. For each candidate, the
audit records:

- adjacent temporal foreground Dice;
- cyclic temporal wrap Dice;
- adjacent spatial foreground Dice;
- temporal-minus-spatial continuity margin;
- a deterministic composite continuity score;
- the top-1/top-2 score margin;
- derived spacing only when a trustworthy source spacing is available.

The composite score is an evidence summary, not a learned or arbitrary class
decision: `temporal_dice + wrap_dice + max(temporal_dice - spatial_dice, 0)`.
If the source does not provide reliable physical Z spacing after flattening,
`derived_z_spacing` is recorded as `NA` and is not invented or used as a
hidden tie-breaker.

The default audit acceptance gate is explicit and reportable:

```text
temporal_dice >= 0.90
wrap_dice >= 0.80
temporal_dice - spatial_dice >= 0.03
top1_minus_top2_score >= 0.02 when a second candidate exists
```

Cases failing a gate are written to the report with `manual_review` status and
are excluded from supervised training unless a user-approved Z/T override is
provided. Accepted reconstruction is `[X,Y,Z,T]`; each temporal unit is
`image[:,:, :,t]` and context is always selected along Z within that same t.

### CMRxMotion

The local source currently consists of two ZIP archives. Discovery must scan
both without extracting or rewriting them:

- training archive: 20 subjects, 80 acquisition cases, 160 image volumes,
  139 labeled volumes;
- challenge validation archive: 5 subjects, 40 image volumes, no labels;
- combined observed cohort: 25 subjects, 200 images, 139 labeled volumes.

The training archive has 11 acquisition cases missing at least one ED/ES
mask. Missing GT is represented as `has_mask=false` and is excluded from
supervised train/validation datasets. No missing mask is replaced by a
background array. Validation-only unlabeled images remain auditable but are
not eligible for supervised training.

CMRxMotion images are 4-D with a singleton final dimension and labels are
3-D; the adapter squeezes only a validated singleton dimension and checks
that image and mask arrays then have identical `[X,Y,Z]` shape. Raw labels are
officially `0=background, 1=LV, 2=MYO, 3=RV`, so the adapter uses:

```python
CMRXMOTION_RAW_TO_COMMON = {0: 0, 1: 3, 2: 2, 3: 1}
```

There are four observed image/mask affine mismatches in P015-1 and P015-2
(both ED and ES). The arrays are shape-compatible and retained only with an
explicit `affine_equal=false` metadata flag and QC coverage. The adapter does
not independently canonicalize image and mask arrays when their affines
disagree, because doing so could destroy array alignment. Any future policy to
exclude these units must be controlled by configuration, not silently inferred.

## Architecture

### Common logical-unit interface

Add a focused data module rather than a second training framework. A logical
unit is one paired 3-D image/mask volume presented to the existing 2.5-D
sample contract. The adapter interface is:

```python
class Cardiac35DAdapter(Protocol):
    def list_units(self, *, split: str | None = None,
                   labeled_only: bool = True) -> list[CardiacUnit]: ...
    def load_unit(self, unit: CardiacUnit) -> LoadedCardiacUnit: ...
    def remap_labels(self, mask: np.ndarray) -> np.ndarray: ...
```

`CardiacUnit` carries `dataset`, `case_id`, `subject_id`, image/mask source
references, phase/time metadata, shape, spacing, affine status, and a stable
split identity. `LoadedCardiacUnit` carries native `[Z,H,W]` image and mask
arrays plus metadata. CMR-MULTI units are per `(case,t)` after reconstruction;
CMRxMotion units are per `(acquisition,phase)`.

`Cardiac35DSliceDataset` owns only the shared sample mechanics:

```text
adapter.load_unit
-> finite/shape/integer-label validation
-> adapter.remap_labels
-> common percentile_clip_and_zscore
-> common build_25d_triplet
-> common resize_sample
-> optional existing geometric transform
-> image [3,H,W], mask [H,W], common metadata
```

This keeps file discovery, archive interpretation, subject identity, phase,
and source label semantics out of the training loop. Existing ACDC/M&Ms
datasets continue to use `VolumeSliceDataset`; a mixed wrapper namespaces
their metadata and the new adapter metadata to make default collation safe.

### CMR-MULTI adapter

`CMRMultiAdapter` discovers only `CINE_MULTI/SAX_TR/image` and `anno`,
validates one-to-one filename pairing, reads the workbook patient ID when
available, and loads NIfTI arrays without modifying source files. It consumes
the generated Z/T audit report and refuses unaccepted cases by default.

### CMRxMotion adapter

`CMRxMotionAdapter` scans ZIP members and parses:

```text
P<subject>-<acquisition>-ED.nii.gz
P<subject>-<acquisition>-ES.nii.gz
P<subject>-<acquisition>-ED-label.nii.gz
P<subject>-<acquisition>-ES-label.nii.gz
```

The parser is independently unit-tested. Archive members are loaded lazily
into memory using the existing NIfTI dependency; the source ZIP remains
read-only. A future offline extraction step can be added without changing the
logical-unit interface.

## Configuration and mixed training

Existing single-dataset YAML remains valid. The unified schema gains an
explicit mixed form:

```yaml
dataset:
  name: mixed
  sources:
    - name: acdc
      data_root: preprocessed_data/ACDC
      split_manifest: splits/acdc_patient_split_seed42.json
      class_mapping: {0: 0, 1: 1, 2: 2, 3: 3}
    - name: mnms
      data_root: preprocessed_data/mnm
      class_mapping: {0: 0, 1: 3, 2: 2, 3: 1}
    - name: cmr_multi
      data_root: data/CMR-MULTI
      split_manifest: splits/cmr_multi_subject_split_seed42.json
      class_mapping: {0: 0, 1: 2, 2: 3, 3: 1}
    - name: cmr_motion
      data_root: data/CMRxMotion
      split_manifest: splits/cmrxmotion_subject_split_seed42.json
      class_mapping: {0: 0, 1: 3, 2: 2, 3: 1}
  sampling_strategy: dataset_subject_balanced
```

The exact YAML fields follow existing strict-schema style. Each source has
its own root, optional manifest, split aliases, mapping, and preprocessing
settings inherited from the common dataset section. All sources must resolve
to the same four-class contract and a common image size. Existing ACDC/M&Ms
single-source configs must produce byte-equivalent legacy config fields after
parsing.

The mixed sampler uses replacement for a fixed epoch length equal to the
combined logical sample count. It assigns equal expected mass to each source,
then equal expected mass to each subject inside a source, then distributes a
subject's mass across its center-slice samples. This is explicitly recorded in
the cohort descriptor and checkpoint lineage. Validation loaders remain
deterministic and unweighted.

Mixed sample metadata is namespaced as `<dataset>:<id>` for `dataset`,
`case_id`, and `patient_id`; this prevents collisions such as `P001` across
independent source datasets and keeps existing trainer membership checks valid.

## Splits and leakage

Split unit is subject, never slice, frame, phase, or acquisition:

- CMR-MULTI uses workbook patient IDs and groups every cine time unit of a
  patient together.
- CMRxMotion uses the `P###` prefix and keeps all acquisitions and ED/ES
  phases of a subject together.
- ACDC and M&Ms preserve their current split resolvers.

Generated manifests are written under `splits/`, never into dataset roots.
Validation checks both within-source split disjointness and the effective
loader memberships. New tests assert pairwise disjoint subject sets for every
source and assert that all units of one source subject have one split.

## Audit, reports, and QC

The implementation must provide:

```text
reports/cmr_multi_audit.csv
reports/cmr_multi_zt_inference.csv
reports/cmrxmotion_audit.csv
reports/dataset_compatibility.md
reports/qc/cmr_multi/*.png
reports/qc/cmrxmotion/*.png
```

Audit CSVs include the requested paths, shapes, dtypes, spacings, affine
status, labels, intensity statistics, mask voxel counts, subject/case/phase
identity, and inclusion/restriction reason. The compatibility report answers
the full task matrix and ends each dataset with one of
`USE_FOR_TRAINING`, `USE_WITH_RESTRICTIONS`, `USE_FOR_VALIDATION_ONLY`, or
`EXCLUDE`.

QC images show context channels, center image, color-coded target overlay,
dataset/case/subject, Z and T/phase. CMR-MULTI QC covers at least ten cases,
multiple inferred T values, and apex/mid/base slices. CMRxMotion QC covers
multiple subjects, acquisitions, ED/ES, missing-GT exclusions, and the P015
affine-mismatch units.

## Tests and verification

Add tests for:

- CMR-MULTI pairing, workbook subject parsing, Z/T reconstruction, candidate
  ambiguity refusal, raw-label mapping, shape, and boundary behavior;
- CMRxMotion filename parsing, ZIP discovery, singleton-dimension squeeze,
  missing-GT exclusion, affine flagging, raw-label mapping, and shape;
- common sample shape `[3,H,W]`/`[H,W]` and label set `{0,1,2,3}`;
- subject-level split disjointness and mixed metadata namespacing;
- dataset-subject-balanced sampler weights;
- one-batch loader smoke for each new dataset and a mixed loader;
- CPU model forward with a mixed batch using the current model constructor.

Integration tests use tiny temporary NIfTI/ZIP fixtures and do not depend on
the large local archives. A separate command runs the real-data audit and
smoke/QC against configured paths.

## Training-flow compatibility

No optimizer, model, loss, schedule, inference, or checkpoint code changes
are planned. The only training-facing changes are:

1. factory dispatch for new source names and mixed source lists;
2. strict schema fields for source configuration and sampling;
3. a sampler-aware extension of the existing `build_data_loader` helper;
4. metadata namespacing needed by the existing cohort/leakage checks.

The first experiment after adapter verification is an explicit ablation:

```text
E0 current data only
E1 current + CMR-MULTI
E2 current + CMRxMotion
E3 current + both new datasets
```

All four runs hold model, seed, optimizer, scheduler, epochs, augmentation,
validation policy, and metrics fixed. The adapter task itself does not claim
that any new dataset improves generalization.

## Known limitations

- CMR-MULTI's flattened NIfTI third-axis header does not expose a trustworthy
  physical Z spacing; physical HD95/ASSD claims for those units remain
  unavailable unless external metadata is supplied.
- CMR-MULTI reconstruction is a continuity-based interpretation and must keep
  manual-review cases visible.
- CMRxMotion validation images are unlabeled and cannot contribute to
  supervised training or validation metrics.
- CMRxMotion affine mismatches are retained only as explicitly flagged,
  array-aligned units; the compatibility report must not call them affine-safe.
- M&Ms remains dependent on the release-specific raw label contract and the
  existing M&Ms data root; the mixed config cannot fabricate missing M&Ms
  data.

## External metadata references

- CMR-MULTI dataset README and SAX label table:
  <https://huggingface.co/datasets/Jackloveaa/CMR-MULTI/blob/main/README.md>
- CMRxMotion label definitions:
  <https://cmrx.chihucloud.com/Task2-T1-T2-mapping.html>
- CMRxMotion challenge description:
  <https://arxiv.org/abs/2210.06385>
