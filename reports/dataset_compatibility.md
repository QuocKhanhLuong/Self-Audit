# Dataset compatibility

This report is generated from the local audit CSVs. It preserves the
existing [B,3,H,W] spatial 2.5-D contract and does not alter source data.

| Criterion | ACDC | M&Ms | CMR-MULTI | CMRxMotion |
|---|---|---|---|---|
| SAX anatomy compatible | existing contract | existing contract | evidence required: 105 cases | evidence from audit |
| LV/RV/MYO compatible | verified current schema | verified current mapping | raw 0/1/2/3 remapped | raw 0/1/2/3 remapped |
| GT quality sufficient | current training source | labeled source | 101/105 Z/T accepted | 139/200 labeled |
| Input convertible to current 2.5-D | yes | yes | yes, Z/T then spatial Z | yes, native 3-D |
| Spatial context meaningful | yes | yes | only within accepted Z/T cases | adjacent native Z slices |
| Z/affine risk | current pipeline | current pipeline | flattened-axis spacing not used | affine mismatches flagged |
| Missing/uncertain data | current policy | current policy | manual-review cases: 4 | missing masks: 61; affine flags: 4 |
| Subject leakage controllable | subject split | subject split | case/subject manifests | subject groups all acquisitions/phases |
| Decision | USE_FOR_TRAINING | USE_WITH_RESTRICTIONS | USE_WITH_RESTRICTIONS | USE_WITH_RESTRICTIONS |

## Dataset-specific restrictions

- CMR-MULTI: exclude cases whose Z/T inference is not accepted; do not
  train on temporal context or invent flattened-axis spacing.
- CMRxMotion: supervised training uses labeled cases only. Affine
  mismatches require native aligned-array handling plus visual QC; strict
  mode excludes them.
- Mixed training must use subject/dataset-balanced sampling and the same
  model, loss, augmentation, and validation contract as the baseline.
