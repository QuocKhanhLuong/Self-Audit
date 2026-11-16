# PG ACDC data preparation

## Locked identities and input provenance

The preparer creates native arrays and role manifests under
`/root/dataset/prepared/ACDC/pg-k8-20261116` from original files under
`/root/dataset/ACDC/raw/{training,testing}`. It requires the server download
receipt to say `DOWNLOADED_VERIFIED` and verifies that it pins both
`/root/dataset/receipts/ACDC_FILES.json` and the matching
`/root/dataset/receipts/ACDC_SOURCE.json`. It checks the complete NIfTI path
inventory and sizes, all CINE SHA256 values, and the SHA256 values of L8 and
VAL/TEST masks when those masks are prepared. Original raw files remain
read-only.

VAL18 comes from the committed Track-A patient metrics receipt, TEST50 from
the one-time test transaction, and L8 from the K8 phase receipt. L8 is
`patient007`, `patient010`, `patient011`, `patient024`, `patient026`,
`patient041`, `patient077`, and `patient099`. TRAIN80 is the 100 Drive
training folders minus VAL18 and the two reviewed duplicate-image exclusions,
`patient055` and `patient076`, documented in
`reports/automatic_pilot_pro4000/FINAL_REPORT.md`. The raw folder identities
must match these lists exactly. The historical K8 source-lock SHA256 is
`60dd11747570f3ac183e8636050a3ac3d73651cb43b37deba9ff7422b442a433`.

## Exact server command

From the repository root on the prepared server, after the ACDC downloader
reports `DOWNLOADED_VERIFIED`:

```sh
python scripts/pg_prepare_acdc.py \
  --raw-root /root/dataset/ACDC/raw \
  --raw-inventory /root/dataset/receipts/ACDC_FILES.json \
  --source-inventory /root/dataset/receipts/ACDC_SOURCE.json \
  --download-receipt /root/dataset/receipts/ACDC_DOWNLOAD.json \
  --val-receipt reports/fuse-semi-20261113/track-A/PATIENT_METRICS.json \
  --test-receipt reports/fuse-semi-20261113/track-A/TEST_ONCE/TEST_TRANSACTION.json \
  --l8-receipt reports/fuse-acdc-k8-aug-20261114/PHASE0_L8.json \
  --exclusion-receipt reports/automatic_pilot_pro4000/FINAL_REPORT.md \
  --mask-inventory reports/fuse-rebuild-20261113/BUILD_RECEIPT.json \
  --out /root/dataset/prepared/ACDC/pg-k8-20261116
```

The default `--mask-inventory` pins original uncompressed `.nii` masks. When
raw sources are `.nii.gz`, the script uses the exact extension-specific Drive
inventory SHA instead of comparing compressed bytes with the historical
uncompressed container SHA. CINE and mask discovery accepts only exact `.nii`
or `.nii.gz` suffixes, so `.nii.interrupted-*` files cannot be mistaken for
valid input. The optional three `--source-{train,val,test}` manifests must be
provided together and match all three historical SHA pins; the fresh raw-only
path needs no old normalized arrays.

## Materialized contract

The script uses the existing FUSE `normalize_cine` routine to create float32
TZYX arrays and the existing Q3D ACDC geometry repair with the authorized
`acdc_pixdim_native_v1` policy. It records original/effective affine and
spacing evidence. Invalid headers, split identity mismatches, mask shape
errors, and fractional, nonfinite, or out-of-range labels fail closed. Only
TRAIN L8 masks are read in the training path. VAL/TEST masks are read only in
their evaluation preparation path. No TRAIN non-L mask is opened.

The `PREPARATION_RECEIPT.json` uses schema
`self_audit.candidate_pg.materialization.v1` and status `READY`. It pins the
download/source inventories, receipt evidence, exact split and labeled IDs,
and generated TRAIN/VAL/TEST manifest hashes. `source_manifest_hashes` always
has the three role keys; raw-only receipt derivation records null historical
role-manifest values there. `manifests` and `artifacts` pin generated files.
`source_lock` refers to the new derived lock inside the output tree;
`historical_source_lock` separately records the original source-lock hash and
path. If `--locked-k8` is supplied, that file is validated and never
overwritten. Every manifest lineage records the historical K8 source hash and
the derived-lock hash. The generated role manifests are TRAIN v2 and VAL/TEST
v1, each accepted by `FuseData`; new local hashes remain explicit and are not
substituted silently for historical pins.

## PG loader pinning and evidence boundary

`open_pg_data` and the server queue require explicit role manifest SHA pins.
The PG runtime config must pin the emitted `train`, `val`, and `test` hashes,
and loader validation must preserve the original K8 lineage while checking
the derived lock. The preparation receipt does not disable or bypass those
checks. Config and loader integration is owned by the coordinator.

The scoped synthetic fixture opens all three generated roles through the
actual `open_pg_data`/`FuseData` loader, checks exact identities and L8/U
records, and verifies that the historical lock stays byte-identical. Four
scoped tests pass. No real data were materialized by this worker because the
server raw tree is not mounted in this workstation; its synthetic checks do
not establish server READY state, training, GPU behavior, or efficacy. M&Ms
remains preparation-only; ACDC training order is A4, A3, A2, A1.
