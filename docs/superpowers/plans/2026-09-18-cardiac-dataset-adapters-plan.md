# Cardiac Dataset Adapters and Four-Source Training Implementation Plan

## Goal

Add audited, subject-safe adapters for CMR-MULTI and CMRxMotion and expose a
four-dataset training configuration (ACDC, M&Ms, CMR-MULTI, CMRxMotion) while
preserving the existing model and training loop contract.

The current contract is fixed by code evidence:

- model input is a three-channel 2.5-D tensor with shape [B, 3, H, W];
- channels are adjacent spatial slices [z-1, z, z+1];
- boundaries use replicated slices;
- target is the center slice;
- in-plane resize is applied without Z resampling;
- intensity preprocessing is volume-wise 0.5/99.5 percentile clipping followed
  by z-score;
- unified labels are 0 Background, 1 RV, 2 MYO, 3 LV.

## Guardrails

- Source directories data/CMR-MULTI and data/CMRxMotion remain read-only.
- Existing ACDC/M&Ms loaders and baseline configuration remain compatible.
- Splits are made by subject, never by slice, frame, phase, or acquisition.
- Missing masks are excluded from supervised samples, never replaced with
  fabricated background masks.
- CMR-MULTI flattened Z/T reconstruction is accepted only when confidence
  gates pass; ambiguous cases are reported and excluded.
- Image/mask affine mismatches are retained only as explicitly flagged cases
  after visual QC.
- Dataset-specific parsing and label mapping stay inside adapters. Shared
  sample preprocessing is reused from self_audit.data.common.
- No core training-loop change is allowed unless a failing contract test proves
  it is necessary.

## Task 1: Centralize label and subject-split contracts

Files:

- Create src/self_audit/data/label_schema.py.
- Update src/self_audit/data/common.py, acdc.py, mnms.py, and data/__init__.py.
- Create tests/test_label_schema.py and tests/test_subject_split.py.

Implement:

- one source of truth for class names, IDs, and validated remapping;
- dataset-independent remap_labels(mask, mapping);
- subject_level_split(case_to_subject, ratios, seed) and a validator that
  proves disjoint subject sets;
- compatibility shims so existing ACDC and M&Ms callers keep their API.

TDD order:

1. Write failing tests for the schema, remapping, deterministic split, and
   disjointness.
2. Implement the smallest compatible primitives.
3. Run the focused tests and the existing split tests.

## Task 2: Add a reusable lazy cardiac 2.5-D dataset

Files:

- Create src/self_audit/data/cardiac35d.py.
- Create tests/test_cardiac35d_dataset.py.

Implement:

- CardiacUnit and LoadedCardiacUnit records containing dataset, case, subject,
  source paths, phase/time metadata, and spatial shape;
- Cardiac35DAdapter protocol/base interface for discovery, loading, metadata,
  and source-label remapping;
- Cardiac35DSliceDataset that lazily loads logical 3-D units, reuses
  percentile_clip_and_zscore, build_25d_triplet, and resize_sample, and emits
  image [3,H,W], center target [H,W], dataset/case/subject/z metadata;
- bounded per-worker caching without pre-exporting PNG slices.

Tests must verify channel order, replicated boundaries, target center semantics,
shape/dtype/label validation, and metadata namespacing.

## Task 3: Implement the CMR-MULTI audit and adapter

Files:

- Create src/self_audit/data/cmr_multi.py.
- Create tests/test_cmr_multi_adapter.py.

Implement:

- strict image/annotation pairing and SAX-only workbook discovery;
- workbook parsing without adding a dependency just for XLSX metadata;
- flattened third-axis reconstruction from [X,Y,N] to [X,Y,Z,T] with time
  fastest order;
- candidate search with T in 18..35 and Z in 6..24;
- continuity score:
  temporal Dice + wrap Dice + max(temporal Dice - spatial Dice, 0);
- acceptance gates temporal >= .90, wrap >= .80,
  temporal minus spatial >= .03, and top-1/top-2 score gap >= .02;
- explicit manual-review status for failed or ambiguous candidates;
- raw label mapping 0->0, 1->2, 2->3, 3->1;
- no invented flattened-axis spacing; retain source metadata and mark
  spacing uncertainty.

Tests must prove N equals Z*T, time is the fastest flattened dimension, labels
are remapped into the unified schema, invalid pairing fails, and ambiguous
inference is not silently accepted.

## Task 4: Implement the CMRxMotion audit and adapter

Files:

- Create src/self_audit/data/cmrxmotion.py.
- Create tests/test_cmrxmotion_adapter.py.

Implement:

- discovery from extracted roots and ZIP archives;
- parser for subject, acquisition, and ED/ES phase;
- in-memory NIfTI member loading;
- squeeze only the singleton image axis, retaining true spatial dimensions;
- grouping all acquisitions and phases by subject;
- raw label mapping 0->0, 1->3, 2->2, 3->1;
- audit records for missing masks and affine mismatches;
- supervised discovery that excludes missing-mask units and does not fabricate
  labels;
- native aligned-array handling for the known affine-mismatch cases, with a
  report flag and QC requirement.

Tests must cover filename parsing, grouping, image/mask pairing, missing-mask
exclusion, singleton-axis handling, label remapping, and affine mismatch
flags.

## Task 5: Extend configuration and dataset factory without breaking legacy
configurations

Files:

- Update src/self_audit/training/unified_config.py.
- Update src/self_audit/training/_utils.py.
- Create tests/test_new_dataset_config_contract.py and
  tests/test_dataset_factory_adapters.py.

Implement:

- accepted dataset names acdc, mnms, cmr_multi, cmr_motion, and mixed;
- source entries with dataset-specific roots, manifests, mappings, and audit
  reports;
- optional sampling_strategy and strict audit settings;
- factories and split validation for both new adapters;
- backward-compatible flat legacy conversion for existing configs;
- mixed configuration with all four datasets and explicit subject-level split
  manifests.

Do not modify model construction, loss, optimizer, scheduler, or the existing
training loop.

## Task 6: Add namespaced mixed-dataset sampling

Files:

- Create src/self_audit/data/mixed.py.
- Create tests/test_mixed_sampling.py.

Implement:

- MixedCardiacDataset using namespaced IDs dataset:subject/case;
- subject-balanced, dataset-balanced sampling with a deterministic seed;
- no sample-count dominance by a dataset with more slices or frames;
- loader integration only for mixed training, while single-dataset loaders
  continue to use their current path.

Add tests that prove source and subject balance, deterministic index streams,
and no cross-split subject leakage.

## Task 7: Add audit, split, and compatibility commands

Files:

- Create scripts/audit_cmr_multi.py.
- Create scripts/audit_cmrxmotion.py.
- Create scripts/generate_subject_splits.py.
- Create scripts/build_dataset_compatibility_report.py.
- Create tests/test_audit_scripts.py.

Commands must accept paths through CLI/config and write only new output
locations. Generate:

- reports/cmr_multi_audit.csv
- reports/cmr_multi_zt_inference.csv
- reports/cmrxmotion_audit.csv
- reports/dataset_compatibility.md
- subject-level split manifests for each dataset

The compatibility report must use evidence-backed statuses:
USE_FOR_TRAINING, USE_WITH_RESTRICTIONS, USE_FOR_VALIDATION_ONLY, or EXCLUDE.

## Task 8: Add visual QC and adapter/model smoke verification

Files:

- Create scripts/generate_dataset_qc.py.
- Create scripts/verify_dataset_adapters.py.
- Create tests/test_adapter_smoke.py.

Generate overlays under reports/qc with class colors and captions for dataset,
case, subject, spatial index, and phase/time. Cover at least ten CMR-MULTI
cases across multiple inferred T values and apex/mid/base positions, plus
multiple CMRxMotion subjects, acquisitions, ED/ES phases, and flagged affine
cases.

The smoke command must load one batch per new dataset and one mixed batch,
then run the existing model forward pass on CPU with no shape, dtype, or class
count errors.

## Task 9: Add configs, run real-data audits, and document usage

Files:

- Create configs/cmr_multi.yaml.
- Create configs/cmrxmotion.yaml.
- Create configs/self_audit_mixed_four_datasets.yaml.
- Update README.md and docs.md.
- Generate reports, manifests, and QC outputs from the actual local datasets.

Document:

- dataset paths and expected archive/tree layouts;
- audit and split commands;
- adapter smoke-test command;
- four-dataset training command;
- restrictions and known limitations;
- exact current 2.5-D contract and unchanged core training flow.

## Verification checklist

Run focused tests after each task, then:

    python -m pytest -q
    python -m compileall -q src scripts tests
    git diff --check
    python scripts/verify_dataset_adapters.py --config configs/self_audit_mixed_four_datasets.yaml --device cpu --num-workers 0

Before claiming completion, inspect generated CSVs, compatibility decisions,
split manifests, and QC images. Confirm source directories have no modified
timestamps or content, and confirm git status contains no unintended changes.

## Commit checkpoints

Use small commits after each coherent task:

1. shared contracts;
2. reusable dataset;
3. CMR-MULTI adapter;
4. CMRxMotion adapter;
5. config/factory;
6. mixed sampling;
7. audit and QC tooling;
8. configs, reports, docs, and final verification.

