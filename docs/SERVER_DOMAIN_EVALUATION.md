# Shared compact domain evaluation

`self_audit.server_evaluation.evaluate_shared_domains` evaluates the same
exported raw compact student separately on the pinned ACDC and M&Ms development
cohorts. It builds one namespaced combined manifest and calls
`run_two_phase_evaluation`, which freezes and hashes predictions for both
domains before it opens reference masks. The historical evaluator returns
independent per-domain BG, RV, MYO, and LV patient macro metrics, Macro4,
foreground Macro, frame counts, and patient coverage.

Checkpoint selection uses this fixed rule:

```text
selection_score = (ACDC class_first_macro4 + M&Ms class_first_macro4) / 2
```

The rule gives each dataset equal weight regardless of cohort size; pooled
patient metrics are never used for selection. A sidecar
`<out_dir>_SELECTION_LOCK.json` records the rule and exact training, evaluation
index, domain manifest, and export SHA-256 pins before evaluation begins. The
domain report is written to `<out_dir>/DOMAIN_EVALUATION.json`, alongside the
canonical `METRICS.json` and `PREDICTIONS_FROZEN.json` receipts.

Example call from the server runner:

```python
from self_audit.server_evaluation import evaluate_shared_domains

result = evaluate_shared_domains(
    export_path=export_dir,
    export_sha256=export_sha,
    train_manifest_path=train_path,
    train_manifest_sha256=train_sha,
    evaluation_maps_path=evaluation_maps_path,
    evaluation_maps_sha256=evaluation_maps_sha,
    out_dir=validation_dir / "domain_eval",
    device=device,
    batch_size=eval_batch_size,
    normalization="prepared",
)
macro4_for_best_step = result["selection_score"]
```

The pinned map index uses schema `self_audit.server_evaluation_maps.v1`. It must
contain exactly one manifest with role `development` for each domain. The
helper accepts either source entries `{domain, manifest, sha256}` or prepared
entries `{domain, manifest_path, manifest_sha256, role, patient_ids}`. It
rejects mismatched map roles, dataset metadata, duplicate IDs, and conflicting
namespace prefixes. Existing `acdc:<id>` and `mnm:<id>` IDs remain unchanged;
unprefixed IDs receive their declared domain prefix. The combined manifest is
passed with the exact shared TRAIN manifest to the canonical evaluator, which
checks TRAIN ID overlap, image byte/voxel overlap, export lineage, native
geometry, and complete prediction/scoring coverage.

The helper accepts only the shared semi-supervised data schema with
`label_budget=8` and `domain_budgets={"acdc": 4, "mnm": 4}`. This is a joint
ACDC L4 + M&Ms L4 training identity, distinct from an ACDC-only L8 teacher.
The evaluator binds the selected export to that exact shared TRAIN manifest;
the runner protocol remains responsible for pinning random seed 42
initialization and for rejecting teacher initialization in the joint run. An
ACDC TEST50 manifest is not scored here; the main runner handles it only after
the best development checkpoint is frozen.
