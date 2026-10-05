# ADNet ACDC Converted GPU Runbook

This lane audits ADNet as a native few-shot labeled-support baseline. It does
not compare against image-only baselines and does not pass query GT to the
producer.

## 1. Freeze Environment

Fetch and audit the upstream ADNet source:

```bash
python scripts/setup_adnet_source.py \
  --adnet-root baseline/ADNet \
  --source-index reports/ADNET_SOURCE_INDEX_2026-10-01.json \
  --receipt reports/adnet_training/acdc_converted/fold0/adnet_source_setup_receipt.json
```

The setup receipt must report `status=READY` and an audit status of `READY`.
It clones ADNet when missing, pins the source to commit
`c6bba85040c12ad1d2f351cdd8f72850daaaf3fb`, and verifies indexed source file
hashes before the producer can run.

```bash
python scripts/capture_adnet_environment.py \
  --output reports/adnet_training/acdc_converted/fold0/environment_receipt.json \
  --adnet-root baseline/ADNet
```

`baseline/ADNet` should be pinned to upstream commit
`c6bba85040c12ad1d2f351cdd8f72850daaaf3fb`, matching
`reports/ADNET_SOURCE_INDEX_2026-10-01.json`.

## 2. Convert ACDC To ADNet Assets

```bash
python scripts/convert_acdc_to_adnet_assets.py \
  --data-root /path/to/acdc_or_preprocessed_acdc \
  --split-manifest splits/acdc_patient_split_seed42.json \
  --output-root reports/adnet_assets/acdc_converted/fold0 \
  --query-split val \
  --support-split train \
  --target-size 224
```

The converter writes:

- `query/*.npy`: image-only query volumes, `[Z,H,W]`, `float32`
- `support/*.npy`: support images and class-binary support masks
- `gt/*.npy`: evaluator-only canonical GT, labels `0 BG`, `1 RV`, `2 MYO`, `3 LV`
- `adnet_assets.json`: `adnet.asset_spec.v1`
- `conversion_receipt.json`: conversion provenance and support policy

Support policy is deterministic: for each foreground class, choose the first
support-split volume containing that class and the median non-empty slice.

## 3. Prepare Manifests

```bash
python scripts/prepare_adnet_fewshot_manifests.py \
  --spec reports/adnet_assets/acdc_converted/fold0/adnet_assets.json \
  --asset-root reports/adnet_assets/acdc_converted/fold0 \
  --path-style relative-to-asset-root \
  --output-dir reports/adnet_manifests/acdc_converted/fold0
```

Producer inputs are `query_manifest.json` and `support_manifest.json`.
`gt_manifest.json` is evaluator-only.

## 4. Train Checkpoint

Train ADNet 2D on the converted ACDC lane and save:

```text
reports/adnet_training/acdc_converted/fold0/model.pth
```

The training receipt must record:

- command line and working directory
- ADNet source commit
- Self-Audit repo commit and dirty status
- ACDC source root and split signature
- checkpoint SHA-256
- training log path
- Python, PyTorch, CUDA, GPU, and driver versions

After training, refresh the environment receipt with artifact hashes:

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

## 5. Preflight

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

Acceptance: JSON output has `"status": "READY"`.

## 6. Run And Evaluate

Smoke on one or two query cases first by creating a temporary query manifest
subset. Then run the full split:

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

```bash
python scripts/evaluate_adnet_fewshot.py \
  --outputs-root reports/adnet_run/acdc_converted/fold0 \
  --gt-manifest reports/adnet_manifests/acdc_converted/fold0/gt_manifest.json \
  --gt-root reports/adnet_assets/acdc_converted/fold0 \
  --output reports/adnet_run/acdc_converted/fold0/evaluation.json \
  --classes 1,2,3
```

Each sealed output must contain `metadata.json`, `semantic_map.npy`,
`validity_map.npy`, and `class_binary/class_*.npy`.
