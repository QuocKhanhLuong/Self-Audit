# Maskfree150 performance-artifact retention manifest

Status: bounded publication inventory accepted by the coordinator on 2026-09-15.  This note is a retention/staging contract only; it makes no GPU, real-data, clinical, or production-performance claim.

## Retention boundary

The six CPU profiling folders below are the accepted before/after pairs, including the density-refinement pair.  Keep their direct receipts together so a profile can be interpreted with its configuration, hotspot summary, batch journal, and (for an after run) the reference pointer and matched report.  The batch journals are retained locally for provenance but are not publication paths: each full `profile_report.json` already contains the 25 batch records.

| run | profile_report.json | matched_report.json | batch_metrics.jsonl | resolved_config.json | hotspot_summary.md | pointer | direct bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `before_ordinary` | 2,255,651 | — | 1,287,485 | 834 | 922 | — | 3,544,892 |
| `after_ordinary` | 2,937,570 | 633,009 | 1,287,442 | 832 | 926 | 96 | 4,859,875 |
| `before_instrumented` | 2,407,510 | — | 1,373,150 | 842 | 1,574 | — | 3,783,076 |
| `after_instrumented` | 3,108,878 | 633,348 | 1,385,598 | 840 | 1,644 | 96 | 5,130,404 |
| `after_density_ordinary` | 2,938,076 | 633,009 | 1,287,464 | 848 | 921 | 96 | 4,860,414 |
| `after_density_instrumented` | 3,109,485 | 633,348 | 1,385,669 | 856 | 1,646 | 96 | 5,131,100 |
| **total** | **16,757,170** | **2,532,714** | **8,006,808** | **5,052** | **7,633** | **384** | **27,309,761** |

All paths in that table are under `reports/maskfree150/performance_cpu/<run>/`.  A before run intentionally has no `matched_report.json` or `reference_snapshot_pointer.json`; this is not an execution-gap claim.  The after-run pointer files resolve to the immutable original snapshot below.

## Snapshot variants to retain

Retain exactly these three existing snapshot locations and only the files listed by each `snapshot_manifest.json` (plus that manifest):

| variant | location | listed payload | clean bytes |
| --- | --- | ---: | ---: |
| original | `reports/maskfree150/local_baseline/source_snapshot` | 32 source/config files + manifest | 841,642 |
| v1 | `reports/maskfree150/performance_cpu/after_ordinary/source_snapshot` | 33 source/config files + manifest | 937,245 |
| final | `reports/maskfree150/performance_cpu/after_density_ordinary/source_snapshot` | 33 source/config files + manifest | 938,113 |

The on-disk original directory also has ignored `__pycache__` files; those are not part of the immutable 32-file snapshot and must not be staged.  Do not copy the other corrected snapshot directories.  The after-run pointers to `local_baseline/source_snapshot` are the equivalence reference for those runs, not missing execution artifacts.

## Exact bounded publication whitelist

Stage only the following paths after the coordinator's final tree review (the six `batch_metrics.jsonl` journals remain local-only):

```text
reports/maskfree150/performance_artifact_manifest.md
reports/maskfree150/local_baseline/.gitignore
reports/maskfree150/performance_cpu/.gitignore
reports/maskfree150/local_baseline/corrected_fixture/fixture_manifest.json
reports/maskfree150/local_baseline/source_snapshot/snapshot_manifest.json
reports/maskfree150/local_baseline/source_snapshot/files/{configs/maskfree_acdc_150.yaml,configs/maskfree_mnms_150.yaml,src/self_audit_maskfree/__init__.py,src/self_audit_maskfree/auditor.py,src/self_audit_maskfree/config.py,src/self_audit_maskfree/contracts.py,src/self_audit_maskfree/data/__init__.py,src/self_audit_maskfree/data/dataset.py,src/self_audit_maskfree/data/discovery.py,src/self_audit_maskfree/data/firewall.py,src/self_audit_maskfree/data/geometry.py,src/self_audit_maskfree/data/partition.py,src/self_audit_maskfree/epoch_validation.py,src/self_audit_maskfree/evaluation/__init__.py,src/self_audit_maskfree/evaluation/epoch_reference.py,src/self_audit_maskfree/evaluation/freeze.py,src/self_audit_maskfree/evaluation/metrics.py,src/self_audit_maskfree/evaluation/native_reference_geometry.py,src/self_audit_maskfree/evaluation/partitions.py,src/self_audit_maskfree/evaluation/reference.py,src/self_audit_maskfree/evaluation/verification.py,src/self_audit_maskfree/experiments.py,src/self_audit_maskfree/export.py,src/self_audit_maskfree/hypotheses.py,src/self_audit_maskfree/losses.py,src/self_audit_maskfree/models.py,src/self_audit_maskfree/observation.py,src/self_audit_maskfree/ontology.py,src/self_audit_maskfree/progress.py,src/self_audit_maskfree/reporting.py,src/self_audit_maskfree/runtime.py,src/self_audit_maskfree/trainer.py}
reports/maskfree150/performance_cpu/after_ordinary/source_snapshot/snapshot_manifest.json
reports/maskfree150/performance_cpu/after_ordinary/source_snapshot/files/{configs/maskfree_acdc_150.yaml,configs/maskfree_mnms_150.yaml,src/self_audit_maskfree/__init__.py,src/self_audit_maskfree/auditor.py,src/self_audit_maskfree/config.py,src/self_audit_maskfree/contracts.py,src/self_audit_maskfree/data/__init__.py,src/self_audit_maskfree/data/cache.py,src/self_audit_maskfree/data/dataset.py,src/self_audit_maskfree/data/discovery.py,src/self_audit_maskfree/data/firewall.py,src/self_audit_maskfree/data/geometry.py,src/self_audit_maskfree/data/partition.py,src/self_audit_maskfree/epoch_validation.py,src/self_audit_maskfree/evaluation/__init__.py,src/self_audit_maskfree/evaluation/epoch_reference.py,src/self_audit_maskfree/evaluation/freeze.py,src/self_audit_maskfree/evaluation/metrics.py,src/self_audit_maskfree/evaluation/native_reference_geometry.py,src/self_audit_maskfree/evaluation/partitions.py,src/self_audit_maskfree/evaluation/reference.py,src/self_audit_maskfree/evaluation/verification.py,src/self_audit_maskfree/experiments.py,src/self_audit_maskfree/export.py,src/self_audit_maskfree/hypotheses.py,src/self_audit_maskfree/losses.py,src/self_audit_maskfree/models.py,src/self_audit_maskfree/observation.py,src/self_audit_maskfree/ontology.py,src/self_audit_maskfree/progress.py,src/self_audit_maskfree/reporting.py,src/self_audit_maskfree/runtime.py,src/self_audit_maskfree/trainer.py}
reports/maskfree150/performance_cpu/after_density_ordinary/source_snapshot/snapshot_manifest.json
reports/maskfree150/performance_cpu/after_density_ordinary/source_snapshot/files/{configs/maskfree_acdc_150.yaml,configs/maskfree_mnms_150.yaml,src/self_audit_maskfree/__init__.py,src/self_audit_maskfree/auditor.py,src/self_audit_maskfree/config.py,src/self_audit_maskfree/contracts.py,src/self_audit_maskfree/data/__init__.py,src/self_audit_maskfree/data/cache.py,src/self_audit_maskfree/data/dataset.py,src/self_audit_maskfree/data/discovery.py,src/self_audit_maskfree/data/firewall.py,src/self_audit_maskfree/data/geometry.py,src/self_audit_maskfree/data/partition.py,src/self_audit_maskfree/epoch_validation.py,src/self_audit_maskfree/evaluation/__init__.py,src/self_audit_maskfree/evaluation/epoch_reference.py,src/self_audit_maskfree/evaluation/freeze.py,src/self_audit_maskfree/evaluation/metrics.py,src/self_audit_maskfree/evaluation/native_reference_geometry.py,src/self_audit_maskfree/evaluation/partitions.py,src/self_audit_maskfree/evaluation/reference.py,src/self_audit_maskfree/evaluation/verification.py,src/self_audit_maskfree/experiments.py,src/self_audit_maskfree/export.py,src/self_audit_maskfree/hypotheses.py,src/self_audit_maskfree/losses.py,src/self_audit_maskfree/models.py,src/self_audit_maskfree/observation.py,src/self_audit_maskfree/ontology.py,src/self_audit_maskfree/progress.py,src/self_audit_maskfree/reporting.py,src/self_audit_maskfree/runtime.py,src/self_audit_maskfree/trainer.py}
reports/maskfree150/performance_cpu/{before_ordinary,after_ordinary,before_instrumented,after_instrumented,after_density_ordinary,after_density_instrumented}/{profile_report.json,resolved_config.json,hotspot_summary.md}
reports/maskfree150/performance_cpu/{after_ordinary,after_instrumented,after_density_ordinary,after_density_instrumented}/{matched_report.json,reference_snapshot_pointer.json}
reports/maskfree150/local_baseline/{profile_report.json,hotspot_summary.md,resolved_config.json}
```

The six profile/matched/config/hotspot/pointer receipts total 19,302,953 bytes before the snapshot and diagnostic additions; the fixture manifest is 647 bytes, and the three clean snapshot trees total 2,717,000 bytes (2.591133 MiB).  Including the original diagnostic profile/hotspot/config (208,127 bytes), the bounded artifact payload is 22,228,727 bytes (21.198966 MiB), excluding all batch journals.  The exact staged total must be taken from `git diff --cached --stat` after staging, because this note and the two local ignore files are also publication paths.

## Deliberately local-only material

Do not stage or delete generated data/checkpoints: `corrected_fixture/synthetic_patient_42.npy`, every `trainer_workspace/` tree and checkpoint, any `__pycache__/`, the six accepted `batch_metrics.jsonl` journals, and the redundant `corrected_*`/`early_2plus5` diagnostic directories.  The original local-baseline direct `profile_report.json`, `batch_metrics.jsonl`, `hotspot_summary.md`, and `resolved_config.json` remain available as diagnostic provenance; only the profile/hotspot/config subset is staged, and none substitutes for the six accepted CPU receipt sets.  The fixture can be reproduced by the existing harness using its recorded path identity; no fixture-content or outcome claim is made here.
