# ADNet Few-Shot Manifest Examples

ADNet is evaluated as a labeled-support baseline. The producer may read frozen
support images and support masks, but query GT must live only in the evaluator
GT manifest.

## Query Manifest

Producer input. Do not include `mask`, `label`, `gt`, metric, or oracle fields.

```json
{
  "schema": "adnet.query_manifest.v1",
  "records": [
    {
      "sample_id": "patient001_frame0001",
      "image_path": "query/patient001_frame0001.npy",
      "image_sha256": "<sha256>",
      "split": "dev",
      "metadata": {
        "patient_id": "patient001",
        "frame": "ED"
      }
    }
  ]
}
```

## Support Manifest

Producer input. This is the only labeled evidence ADNet may consume.

```json
{
  "schema": "adnet.support_manifest.v1",
  "class_mapping": {
    "RV": 1,
    "LV-MYO": 2,
    "LV-BP": 3
  },
  "supports": [
    {
      "support_id": "support_patient004_rv",
      "class_name": "RV",
      "canonical_class_id": 1,
      "image_path": "support/patient004.npy",
      "image_sha256": "<sha256>",
      "mask_path": "support/patient004_rv_mask.npy",
      "mask_sha256": "<sha256>",
      "slice_indices": [8]
    }
  ]
}
```

Canonical class ids follow the Self-Audit cardiac schema:

```text
0 BG
1 RV
2 MYO
3 LV
4 VOID
```

## GT Manifest

Evaluator-only input. This file must not be passed to
`scripts/run_adnet_fewshot.py`.

```json
{
  "schema": "adnet.gt_manifest.v1",
  "records": [
    {
      "sample_id": "patient001_frame0001",
      "gt_path": "gt/patient001_frame0001.npy",
      "gt_sha256": "<sha256>"
    }
  ]
}
```

## Commands

```bash
python scripts/preflight_adnet_fewshot.py \
  --query-manifest query_manifest.json \
  --support-manifest support_manifest.json \
  --checkpoint model.pth \
  --output-root reports/adnet_run \
  --required-classes 1,2,3

python scripts/run_adnet_fewshot.py \
  --query-manifest query_manifest.json \
  --support-manifest support_manifest.json \
  --checkpoint model.pth \
  --output-root reports/adnet_run \
  --device cuda

python scripts/evaluate_adnet_fewshot.py \
  --outputs-root reports/adnet_run \
  --gt-manifest gt_manifest.json \
  --output reports/adnet_run/evaluation.json \
  --classes 1,2,3
```
