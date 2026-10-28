# Real-patient PRO4000 ACDC execution

Base commit: `93f7c7a1f9606a68955f849dd580b90462ad329e`; executable patch: `SOURCE_PATCH_B128_FINAL.patch`.

ACDC: TRAIN80 (L8/U72), VAL18, TEST50; locked split unchanged; excluded055/076. M&Ms results are outside this ACDC-only report.

Batch 128 across all arms; full_ema accumulates one effective batch across microbatches of 40 and takes one optimizer/EMA step. All metrics below are measured raw-student native predictions, frozen before reference scoring.

| Comparison | Arm | Steps | RV | MYO | LV | FG mean | Phase s | Export s | Peak reserved MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| equal_update_B128_M40 | scalar_brier | 128 | 0.0 | 0.0 | 0.0 | 0.0 | 587.595373456832 | 60.35798978898674 | 20616.0 |
| equal_update_B128_M40 | joint_brier | 128 | 0.0 | 0.0 | 0.0 | 0.0 | 574.9324312349781 | 55.47955488716252 | 20616.0 |
| equal_update_B128_M40 | scalar_dice | 128 | 0.0 | 0.0 | 0.0 | 0.0 | 582.9438339318149 | 55.04571333201602 | 20616.0 |
| equal_update_B128_M40 | joint_dice | 128 | 0.0 | 0.0 | 0.0 | 0.0 | 587.9398568880279 | 55.09046270814724 | 20616.0 |
| equal_update_B128_M40 | frozen | 128 | 0.0 | 0.0 | 0.0 | 0.0 | 175.2432176428847 | 55.23393214819953 | 20616.0 |
| equal_update_B128_M40 | support_only | 128 | 0.0 | 0.0 | 0.0 | 0.0 | 176.70932618295774 | 54.84480545902625 | 20616.0 |
| equal_update_B128_M40 | plain_head_u | 128 | 0.0 | 0.0 | 0.0 | 0.0 | 245.44327570102178 | 55.16219560010359 | 20616.0 |
| equal_update_B128_M40 | extra_supervised | 128 | 0.0 | 0.0 | 0.0 | 0.0 | 173.99631453002803 | 55.718169185798615 | 20616.0 |
| equal_update_B128_M40 | smaller_lr | 128 | 0.0 | 0.0 | 0.0 | 0.0 | 245.92411302402616 | 55.25552252517082 | 20616.0 |
| equal_update_B128_M40 | full_ema | 128 | 0.41711249728342187 | 0.4083722376846927 | 0.6189768026778182 | 0.48148717921531103 | 524.0544183528982 | 55.56024826900102 | 19770.0 |
| equal_time_B128_M40 | scalar_brier | 131 | 0.0 | 0.0 | 0.0 | 0.0 | 602.1802955789026 | 59.49232500302605 | 20616.0 |
| equal_time_B128_M40 | joint_brier | 130 | 0.0 | 0.0 | 0.0 | 0.0 | 603.4313673391007 | 55.35493357293308 | 20616.0 |
| equal_time_B128_M40 | scalar_dice | 131 | 0.0 | 0.0 | 0.0 | 0.0 | 604.2454279079102 | 55.253288165898994 | 20616.0 |
| equal_time_B128_M40 | joint_dice | 132 | 0.0 | 0.0 | 0.0 | 0.0 | 604.4033178579994 | 55.091514858882874 | 20616.0 |
| equal_time_B128_M40 | frozen | 461 | 0.0 | 0.0 | 0.0 | 0.0 | 601.2105827741325 | 55.07628329494037 | 20616.0 |
| equal_time_B128_M40 | support_only | 474 | 0.0 | 0.0 | 0.0 | 0.0 | 600.8188348440453 | 55.066507427953184 | 20616.0 |
| equal_time_B128_M40 | plain_head_u | 326 | 0.0 | 0.0 | 0.0 | 0.0 | 601.9323966300581 | 55.212433443870395 | 20616.0 |
| equal_time_B128_M40 | extra_supervised | 470 | 0.0 | 0.0 | 0.0 | 0.0 | 600.5703474588227 | 55.308977456064895 | 20616.0 |
| equal_time_B128_M40 | smaller_lr | 324 | 0.0 | 0.0 | 0.0 | 0.0 | 600.4330938260537 | 55.07490901998244 | 20616.0 |
| equal_time_B128_M40 | full_ema | 151 | 0.5038281720618222 | 0.5230152058336998 | 0.6985252166723928 | 0.5751228648559715 | 607.7955633969977 | 55.40151931508444 | 19770.0 |

## Decision: equal_update_B128_M40

{
  "decision": "INCONCLUSIVE_NOT_RUN",
  "quality_status": "NOT_RUN",
  "reason": "Matched compute verification failed or missing in verified_context.",
  "gates": {
    "joint_beats_scalar": null,
    "dice_guard_beats_smaller_lr": null,
    "u_beats_extra_supervised": null
  }
}

## Decision: equal_time_B128_M40

{
  "decision": "NO_GO",
  "quality_status": "MEASURED",
  "gates": {
    "joint_beats_scalar": false,
    "dice_guard_beats_smaller_lr": false,
    "u_beats_extra_supervised": false
  },
  "scores": {
    "joint_dice": 0.0,
    "scalar_dice": 0.0,
    "smaller_lr": 0.0,
    "extra_supervised": 0.0
  },
  "per_class_primary_results": {
    "joint_dice": {
      "RV": 0.0,
      "MYO": 0.0,
      "LV": 0.0
    },
    "scalar_dice": {
      "RV": 0.0,
      "MYO": 0.0,
      "LV": 0.0
    },
    "smaller_lr": {
      "RV": 0.0,
      "MYO": 0.0,
      "LV": 0.0
    },
    "extra_supervised": {
      "RV": 0.0,
      "MYO": 0.0,
      "LV": 0.0
    }
  },
  "matched_compute": {
    "status": "VERIFIED_MATCHED",
    "comparison_mode": "equal_time",
    "steps": null,
    "time_budget_seconds": 600.0,
    "device": "cuda",
    "hardware": "PRO4000 24GB"
  },
  "reason": "One or more independent query gates failed."
}

## Limitations

Single patient-label seed and single run per arm; no statistical or clinical certification. Drive TEST50 was previously exposed in this repository and is not a new confirmatory cohort. Drive label provenance is not independently challenge certified. Equal-update is not equal-compute evidence. Fresh equal-time phase spread must be within 5%; export and common warm costs are reported separately. M&Ms results are excluded from this report under the current ACDC-only scope. DINO weights match the authorized backup and recorded full SHA; independent upstream full-hash certification unavailable. Full-EMA follows the executable base commit and trains encoder/body, despite runbook frozen-encoder wording. AMP is disabled to preserve finite argmax guard semantics. No W&B credentials; structured file logging is active.

## Locations

Checkpoints/native compressed predictions: `/workspace/Self-Audit/runs/pro4000/`. Prepared data: `/workspace/Self-Audit/data/prepared/`. Raw data: `data/ACDC/`, `data/MnM/`. All original archives retained in `data/archives/`. Reproduction: `execute_B128.sh`, `run_stage.py`, `evaluate_stage.py`, `PLAN.json`, `SOURCE_PATCH_B128_FINAL.patch`, pinned data/model/checkpoint manifests. No raw patient data or checkpoints are included in Git.
