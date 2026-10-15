# Documentation index

Runbooks describe commands for a specific pipeline. Dated reports, source
snapshots and verification receipts describe the revision that produced them;
they do not certify a later checkout. Keep each experiment's supervision,
split, configuration and artifact schema together when reproducing results.

## Runnable research pipelines

| Pipeline | Start here | Additional contracts |
| --- | --- | --- |
| Frozen SSL guidance | [Native-data full pipeline](stable_guidance_full.md) | [Method, stage commands and limitations](stable_guidance.md) |
| Image-only pseudo-label v3 | [Review and execution runbook](pseudolabel_v3_review_runbook.md) | [System contract](pseudolabel_system_v3.md), [progress and tracking](pseudolabel_v3_live_progress.md) |
| Experimental v3 bootstrap | [Bootstrap protocol](pseudolabel_v3_bootstrap_experimental.md) | [Counterfactual quotient audit](pseudolabel_v3_cqa.md), [foreground diagnostics](pseudolabel_v3_foreground_diagnostics.md) |
| MaskFree150 | [Method and execution](maskfree150.md) | [Large-batch settings](maskfree_large_batch.md) |
| Supervised Self-Audit reference | [Training contract](../docs.md) | [Repository overview](../README.md) |
| Supervised REW | [16 GB runbook](rew_16gb.md) | Separate supervision contract from the no-GT tracks |

## Baselines and reproduction

- [Baseline implementations](../baseline/) and [benchmark freezes](../benchmark_freezes/)
  retain comparison and ablation identities.
- [Dataset contracts](../README_AGENT_CMR_DATASETS.md) cover shared data adapters.
- [ADNet converted ACDC](adnet_acdc_converted_runbook.md) and
  [few-shot manifests](adnet_fewshot_manifests.md) describe a mask-supported comparison.
- [STEGO/PiCIE port provenance](stego-picie-port-provenance.md) records that port's lineage.
- [Legacy reproduction](legacy_reproduction.md) pins retired runner revisions.

## Evidence and historical work

- [Reports](../reports/) contain dated audits, measurements, failures and source snapshots.
- [v3 checkup](pseudolabel_v3_checkup_20261005.md) records the October 5 repairs.
- [Design specifications](superpowers/specs/) and [implementation plans](superpowers/plans/)
  are historical design records; use the relevant runbook for current commands.

Documentation cleanup preserves these records and their paths. A test pass is
software evidence; anatomy quality and GPU performance require their own
experiment receipts.
