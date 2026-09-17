# W6 evaluation implementation - Luna handoff

Status: **PASS with bounded integration limitations**.

This review was completed in /Users/alvinluong/Self-Audit on the shared
checkout. No worktree, commit, push, GPU, SSH, remote command, or training run
was used. The owned implementation paths are:

- src/self_audit_maskfree/experiments.py
- src/self_audit_maskfree/evaluation/
- scripts/evaluate_maskfree_reference.py (existing isolated entry point)

## Implemented contract

run_bank_experiments(...) fits every exported selector/control on FittingView
before finalization and returns:

- fitted_predictions: method name to the exact pre-freeze FittedHypothesis,
  including E0-E5 and available controls;
- control_fitted: exact pre-freeze fits for all available control methods;
- fitted_controls: the historical degenerate-only alias;
- bank.candidate_slots: the preregistered positional correspondence used by
  the shuffled-evidence control.

E2 records confidence_constant, selection_basis, and a note that one-hot draft
confidence is constant, so the criterion is disclosed as smoothness-only.
Ontology refusal marks the E0 validity map invalid and records the failure
rather than treating unresolved groups as valid anatomy.

ForeignEvidenceCache stores dataset/run scope, study and unit identity, and
candidate-slot evidence. Cross-study shuffling uses an actual cached entry and
the shared slot map. Missing or incompatible evidence is unavailable with a
reason; no placeholder is reported as a successful method.

verify_frozen_bank(...) accepts the following integration arguments:

    verify_frozen_bank(
        fitted,
        verify_view,
        model,
        freeze_manifest,
        *,
        selection_scores=None,
        degenerate_fitted=None,
        control_fitted=None,
        method_to_candidate=None,
        selected_candidate_id=None,
        student_fitted_by_arm=None,
        seed=42,
        allow_repeat=False,
        resume_cached=False,
        ledger_path=None,
        freeze_root=None,
        required_methods=None,
    )

It calls only the authoritative export.validate_freeze, deduplicates shared
fitted candidates, scores each one once, and keeps verification ranking
diagnostic. Perturbation and degenerate comparisons use the preselected
candidate supplied by selected_candidate_id; an invalid id fails closed.
Verification reuse is keyed by freeze_id + study_id + unit_id + partition_id.
An optional JSON ledger atomically stores the completed result so
resume_cached=True can return it without rescoring.

verify_student_coverage(...) computes natural and equal-count predictive scores
from the frozen student fits. Its result is:

    {
      available, reason,
      natural: {arm: {score, supported_pixels, available, reason}},
      matched: {arm: {score, supported_pixels, available, reason}},
      supported_pixels_by_arm, matched_supported_pixels, note
    }

The matched result is image-only predictive NLL from frozen validity support; it
is not a reference quality claim.

The isolated reference evaluator now keeps
method -> patient -> volume/frame keys, resolves one explicit reference per
volume, computes volume metrics before patient macro aggregation, and preserves
volume identifiers in surfaces, edit attribution, coverage, and diagnostics.
Ambiguous patient-only mappings are rejected. Reference raw integer labels require
a fixed dataset-level reference_label_map with a bijection covering background,
RV, MYO, and LV; native and synthetic configured cases must declare it before
evaluation. No implicit or per-case label inference is used, and the global
oracle permutation is diagnostic only.

The requested split is matched against each frozen prediction entry's actual
split; ``all`` and ``mixed`` explicitly evaluate all frozen splits. Native
surface distances require valid frozen prediction geometry, a native reference
NIfTI, converted spacing agreement, and matching prediction/reference affines.
Overlap metrics reject affine mismatches and do not treat same-shaped arrays as
registered grids.

load_freeze_manifest preserves the decoded producer payload without injecting
metadata that would invalidate its manifest digest. This allows root
finalization to wrap per-unit verification in W7's verified_freeze_session(...)
while validate_freeze_manifest passes the original mapping object to
export.validate_freeze.

## Focused evidence

Commands and results:

    PYTHONPATH=src /Users/alvinluong/miniforge3/bin/python -m pytest tests/test_maskfree_evaluation.py -q
    3 passed

    PYTHONPATH=src /Users/alvinluong/miniforge3/bin/python -m pytest tests/test_maskfree_export.py -q
    9 passed

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/alvinluong/miniforge3/bin/python -m py_compile src/self_audit_maskfree/experiments.py src/self_audit_maskfree/evaluation/freeze.py src/self_audit_maskfree/evaluation/reference.py src/self_audit_maskfree/evaluation/verification.py
    passed

The evaluation suite covers paired selectors, all six controls, cache
unavailability, equal-cost random edits, metric edge cases, per-patient
aggregation, freeze tamper rejection, no-refit verification, degenerate
challengers, and the one-time reuse guard. The export suite covers native
round trips, missing geometry, manifest hashes, and immutability.

## Root integration requirements and limits

Root finalization should:

1. preseed foreign cache entries with the same dataset/run scope used by each
   FittingView, using one real representative per distinct study;
2. pass method_to_candidate, control_fitted, selected_candidate_id, and both
   student fits to verification;
3. pass a durable ledger_path and use resume_cached=True only when a matching
   completed result is present;
4. use the same original freeze mapping object inside
   verified_freeze_session(...).

The broader firewall suite was not used as a W6 gate because four failures are
in the shared data/contracts layer: FittingView.validate() rejects the existing
source_hash and spacing metadata. That issue is outside this ownership. No real
ACDC/M&Ms masks, native reference labels, GPU execution, cine transport,
anatomical accuracy, clinical benefit, novelty, or calibration claim is
established here.
