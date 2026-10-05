# ADNet Execution Audit 2026-10-03

## Scope

ADNet is audited here as a native few-shot labeled-support baseline. The audit
uses ACDC converted to ADNet-compatible assets and excludes STEGO/PICIE or
image-only baseline fairness targets from acceptance.

Canonical labels:

```text
0 BG
1 RV
2 MYO
3 LV
```

## Source Pin

- ADNet upstream: `https://github.com/sha168/ADNet`
- Required ADNet commit: `c6bba85040c12ad1d2f351cdd8f72850daaaf3fb`
- Source index: `reports/ADNET_SOURCE_INDEX_2026-10-01.json`
- Local source root: `baseline/ADNet`

## Implemented Local Artifacts

- `scripts/convert_acdc_to_adnet_assets.py`
  - Converts paired ACDC raw NIfTI or preprocessed NPY volumes to `[Z,H,W]`
    ADNet assets.
  - Writes `query/`, `support/`, `gt/`, `adnet_assets.json`, and
    `conversion_receipt.json`.
  - Keeps query records image-only; GT is emitted only for evaluator manifest
    preparation.
- `scripts/capture_adnet_environment.py`
  - Captures Python, PyTorch, CUDA, GPU/driver, repo identity, ADNet identity,
    and optional artifact hashes.
- `scripts/setup_adnet_source.py`
  - Clones/pins upstream ADNet when missing.
  - Verifies the checkout against `reports/ADNET_SOURCE_INDEX_2026-10-01.json`
    by commit and per-file SHA-256 before GPU execution.
- `docs/adnet_acdc_converted_runbook.md`
  - Contains the GPU command sequence and acceptance gates.

## Required GPU Artifact Paths

```text
reports/adnet_assets/acdc_converted/fold0/adnet_assets.json
reports/adnet_assets/acdc_converted/fold0/conversion_receipt.json
reports/adnet_manifests/acdc_converted/fold0/query_manifest.json
reports/adnet_manifests/acdc_converted/fold0/support_manifest.json
reports/adnet_manifests/acdc_converted/fold0/gt_manifest.json
reports/adnet_training/acdc_converted/fold0/adnet_source_setup_receipt.json
reports/adnet_training/acdc_converted/fold0/model.pth
reports/adnet_training/acdc_converted/fold0/environment_receipt.json
reports/adnet_run/acdc_converted/fold0/evaluation.json
```

## Execution Commands

Setup ADNet source:

```bash
python scripts/setup_adnet_source.py \
  --adnet-root baseline/ADNet \
  --source-index reports/ADNET_SOURCE_INDEX_2026-10-01.json \
  --receipt reports/adnet_training/acdc_converted/fold0/adnet_source_setup_receipt.json
```

Convert ACDC:

```bash
python scripts/convert_acdc_to_adnet_assets.py \
  --data-root /path/to/acdc_or_preprocessed_acdc \
  --split-manifest splits/acdc_patient_split_seed42.json \
  --output-root reports/adnet_assets/acdc_converted/fold0 \
  --query-split val \
  --support-split train \
  --target-size 224
```

Prepare manifests:

```bash
python scripts/prepare_adnet_fewshot_manifests.py \
  --spec reports/adnet_assets/acdc_converted/fold0/adnet_assets.json \
  --asset-root reports/adnet_assets/acdc_converted/fold0 \
  --path-style relative-to-asset-root \
  --output-dir reports/adnet_manifests/acdc_converted/fold0
```

Capture environment:

```bash
python scripts/capture_adnet_environment.py \
  --output reports/adnet_training/acdc_converted/fold0/environment_receipt.json \
  --adnet-root baseline/ADNet \
  --checkpoint reports/adnet_training/acdc_converted/fold0/model.pth \
  --asset-spec reports/adnet_assets/acdc_converted/fold0/adnet_assets.json \
  --query-manifest reports/adnet_manifests/acdc_converted/fold0/query_manifest.json \
  --support-manifest reports/adnet_manifests/acdc_converted/fold0/support_manifest.json \
  --gt-manifest reports/adnet_manifests/acdc_converted/fold0/gt_manifest.json
```

Preflight:

```bash
python scripts/preflight_adnet_fewshot.py \
  --query-manifest reports/adnet_manifests/acdc_converted/fold0/query_manifest.json \
  --support-manifest reports/adnet_manifests/acdc_converted/fold0/support_manifest.json \
  --checkpoint reports/adnet_training/acdc_converted/fold0/model.pth \
  --output-root reports/adnet_run/acdc_converted/fold0 \
  --query-image-root reports/adnet_assets/acdc_converted/fold0 \
  --support-asset-root reports/adnet_assets/acdc_converted/fold0 \
  --required-classes 1,2,3
```

Run:

```bash
python scripts/run_adnet_fewshot.py \
  --query-manifest reports/adnet_manifests/acdc_converted/fold0/query_manifest.json \
  --support-manifest reports/adnet_manifests/acdc_converted/fold0/support_manifest.json \
  --checkpoint reports/adnet_training/acdc_converted/fold0/model.pth \
  --output-root reports/adnet_run/acdc_converted/fold0 \
  --query-image-root reports/adnet_assets/acdc_converted/fold0 \
  --support-asset-root reports/adnet_assets/acdc_converted/fold0 \
  --device cuda
```

Evaluate:

```bash
python scripts/evaluate_adnet_fewshot.py \
  --outputs-root reports/adnet_run/acdc_converted/fold0 \
  --gt-manifest reports/adnet_manifests/acdc_converted/fold0/gt_manifest.json \
  --gt-root reports/adnet_assets/acdc_converted/fold0 \
  --output reports/adnet_run/acdc_converted/fold0/evaluation.json \
  --classes 1,2,3
```

## Acceptance Gates

- Converted images and masks load as numeric `[Z,H,W]`.
- ADNet source setup receipt reports `status=READY`.
- ADNet source audit verifies commit
  `c6bba85040c12ad1d2f351cdd8f72850daaaf3fb` and indexed file hashes.
- Query images are non-constant and finite.
- GT masks contain only labels `0,1,2,3`.
- Support masks are binary and non-empty on declared support slices.
- Query manifest contains no GT, mask, label, oracle, or metric fields.
- Preflight returns `status=READY`.
- `model.pth` loads through `scripts/run_adnet_fewshot.py`.
- Sealed outputs bind checkpoint SHA-256, query manifest SHA-256, and support
  manifest SHA-256 in each `metadata.json`.
- Evaluator consumes sealed outputs plus GT manifest only.

## Current Status

Local implementation and contract tests are ready. The full GPU training,
preflight, smoke, full run, and evaluation remain pending until the target ACDC
data root and GPU runtime are available.

## Known Limitations

- The current runner supports ADNet 2D inference.
- Upstream ADNet training is supervoxel-oriented; the training receipt must
  document the exact ACDC converted training adapter or recipe used to produce
  `model.pth`.
- Support selection is fixed before inference by the asset spec and manifest;
  no dynamic query-aware support selection is allowed.
