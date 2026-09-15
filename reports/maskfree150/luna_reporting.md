# Bounded image-only reporting

`src/self_audit_maskfree/reporting.py` owns the final, image-only summary for a
Maskfree150 run.  The coordinator calls it after all compared predictions and
checkpoint identities have been frozen, `validate_freeze` has passed, and the
image-only verification rows have been produced.

## Callable interface

```python
write_image_only_reports(
    run_root,
    manifest,
    epoch_history,
    verification_rows,
    experiment_rows,
    coverage_rows,
    epoch,
    checkpoint,
) -> {"json": Path, "csv": Path, "md": Path}
```

`run_root` is the run directory.  The writer creates `run_root/reports/` and
emits `image_only_summary.json`, `image_only_summary.csv`, and
`image_only_summary.md`.  When rows carry more than one split, it also writes
the same three files below `run_root/splits/<split>/reports/` and adds a
`split_reports` path index to the all-split JSON.  `manifest` and each row collection may be an
in-memory mapping/sequence or an existing JSON/JSONL path.  The function does
not accept models or tensors and does not read image, annotation, or reference
files.

## JSON schema

The top-level payload is `maskfree150.reporting.v1` and carries the required
provenance in both `provenance` and the compact fields `dataset`, `split`,
`protocol`, `epoch`, `checkpoint`, `population`, `count`, `available`,
`contract_version`, and `unit`.

```text
{
  "schema_version": "maskfree150.reporting.v1",
  "provenance": {...},
  "method_to_candidate": method -> candidate id
                         # with >1 unit, method -> {unit -> candidate id}
  "verification": {
    "candidates": [{"unit_id": ..., "patient_id": ..., "candidate_id": ...,
                    "verify_score": {"nll_sum": ..., "count": ...,
                                     "normalized_nll": ..., "available": ...},
                    "selection_to_verification_gap": ...}],
    "method_summaries": {
      method: {
        "candidate_ids_by_unit": {...},
        "patient_macro_normalized_nll": float | null,
        "raw_pooled_normalized_nll": float | null,
        "patient_macro_selection_gap": float | null,
        "unit_count": int,
        "patient_count": int,
        "raw_count": int,
        "available": bool,
        "reason": str | null
      }
    },
    "patient_macro": method -> value,
    "raw_pooled": method -> value,
    "paired_patient_bootstrap": pair -> result,
    "diagnostics": {
      "degenerate_flags": {...},
      "ambiguity": {...},
      "ranking_agreement": {
        "available": bool,
        "metrics": {...},
        "note": "top-2 verification-ranking agreement is not repeated-annotation stability"
      },
      "stability": {
        "available": bool,
        "protocol": "fit_noise_sigma_0.05" | null,
        "scope": "pre_freeze_image_only" | null,
        "methods": {
          method: {"agreement": float | null,
                   "valid_agreement": float | null,
                   "common_valid_count": int | null,
                   "count": int, "patient_count": int}
        }
      },
      "sensitivity": {...},
      "temporal": {"available": bool, "value": float | null,
                   "reason": str | null}
    }
  },
  "coverage": {
    "natural_coverage": {"student_no_audit": {...}, "student_audited": {...}},
    "matched_coverage": {
      "support_pixels": int | null,
      "fraction": float | null,
      "quality": {
        "value": null,
        "available": bool,
        "by_arm": {
          "student_no_audit": {"normalized_nll": float | null, ...},
          "student_audited": {"normalized_nll": float | null, ...}
        },
        "reason": str | null
      }
    },
    "valid_foreground": {...},
    "valid_fg_collapse": method -> bool | null
  },
  "epoch_audit": {"metric_rows": [...]},
  "rows": [...],
  "metric_rows": [...]
}
```

Every item in `metric_rows` is rebuilt through the strict W6 `metric_row`
helper.  It therefore includes `dataset`, `split`, `protocol`, `epoch`,
`checkpoint`, `population`, `count`, `available`, `unit`, and
`contract_version`.  Nonfinite or missing values are emitted as `value: null`
with a reason.

## Aggregation and scope rules

- Verification candidate rows retain raw NLL sum, observed-pixel count,
  normalized NLL, and the selection-to-verification gap.  Method summaries
  first average unit values within each patient and then average patients.
  `raw_pooled_normalized_nll` separately divides summed raw NLL by summed
  observed-pixel count.
- Paired bootstrap comparisons are E5 against E1, E2, E3, E4, plus
  `student_audited` against `student_no_audit`.  A pair enters only with a
  finite normalized NLL and positive observed-pixel count.  The resampling
  unit is a patient; fewer than two paired patients remain unavailable.
- Natural support counts and matched support counts are separate.  A matched
  support count does not create a matched-quality result.  Quality remains
  unavailable with an explicit reason unless a caller supplies a metric
  actually scored on the matched masks.  W6's nested `student_coverage`
  payload is consumed when present, with raw NLL/count/normalized NLL retained
  per student arm and an audited-minus-no-audit comparison.
- `valid_fg_collapse` is a coverage diagnostic.  It is true only when every
  supplied coverage row reports zero valid foreground support; incomplete rows
  leave it unavailable.
- Degenerate outcomes, semantic ambiguity, perturbation sensitivity, and
  temporal status are copied/summarized only when their fields are present.
  An explicit `repeated_annotation_stability` payload from the fixed-noise
  pre-freeze diagnostic is aggregated per method, averaging units within each
  patient before averaging patients. Without that payload, stability remains
  unavailable with a reason.
  W6 top-2 verification-ranking agreement is reported under
  `ranking_agreement`; true repeated-annotation stability stays unavailable
  until an independent repeat is supplied.  The spatial profile explicitly
  reports temporal predictive verification unavailable because fit-time
  temporal transport is not available.
- Appearance evidence is predictive NLL in nats per observed pixel.  It is not
  segmentation accuracy and is never described as a correctness probability.

The report writer has no import path to any reference evaluator.  Reference
metrics remain a separate post-freeze process and cannot enter training,
selection, checkpointing, or this image-only summary.

## Focused software check

```bash
PYTHONPATH=src rtk pytest -q tests/test_maskfree_reporting.py
```

The synthetic checks pass (`2 passed`).  They verify patient-first
aggregation, raw pooled denominators, all requested paired comparisons,
temporal unavailability, natural/matched support separation, foreground
collapse reporting, split-specific aggregation, and nested matched student
quality.  It is CPU software evidence only; no real dataset, GPU, or
segmentation-quality result is claimed.
