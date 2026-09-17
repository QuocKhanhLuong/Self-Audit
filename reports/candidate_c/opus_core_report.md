# Candidate C core delivery — models, replay, solver

Worker: Claude Opus 5 (core). Base HEAD verified at dispatch and unchanged for my files:
`9a612485b7ead1273082d7dd52879f8e9e7a3e83`. Implements the user-selected Candidate C
per `reports/candidate_c/architecture_contract.md`. **No effectiveness, Dice or novelty
conclusion follows from anything in this report.** Everything below is software behaviour
on tiny synthetic CPU tensors.

## 1. Changed files

| File | Change |
|---|---|
| `src/self_audit/models/dynamic_window.py` | Coordinate override, preclamp/identity metadata, `free` offset mode, normalized-pixel helper, override validation |
| `src/self_audit/models/annotation_expert.py` | Per-internal-iteration geometry capture, `ExpertReplayRecord`, functional frozen `replay`, `state_identity`, `feature_only` conditioning |
| `src/self_audit/models/self_audit_net.py` | `window_mode`/`CandidateCConfig`, bounded accepted-transition record, C1/C2/C3 solver, direct-rollback control, per-sample diagnostics, `capture_geometry` payload |
| `src/self_audit/models/__init__.py` | Exports (ownership approved by coordinator) |
| `tests/test_candidate_c.py` | New, 60 tests |
| `reports/candidate_c/opus_core_report.md` | This report |

No file owned by another worker was touched. No commit, no push.

## 2. C1 / C2 / C3 mapping

### C1 — exact frozen factual-support replay

`AnnotationExpert.replay(record, coordinates=None)` re-executes a recorded transition from
`ExpertReplayRecord`: detached clones of shared features, pre-transition logits, the prior
audit *input*, the original turn index, internal-iteration index and depth, the realized
post-clamp supports of **every** internal iteration, and the factual candidate/delta/gate.

The replay is **functional** (`torch.func.functional_call` with detached parameters and
buffers), so no `requires_grad` flag is mutated, no `.grad` buffer is touched and there is
no flag-restoration window to leak through. When an override is supplied the window
generator is not consulted at all — the realized support fully determines the downstream
computation — which is why the factual replay reproduces the factual output with
`max_abs_err == 0.0` exactly, not approximately (`test_c1_factual_replay_is_exact_and_multi_iteration`).

C1 runs **before** any optimization. It is checked with strict `replay_atol`/`replay_rtol`
(default `1e-5`), and the stored factual output is separately checked against the *current
retained state*. Either failure — and a `StaleReplayRecordError` from the record no longer
matching the live model state — **disables C for that transition and routes the row to the
ordinary annotator for that turn**, logging `replay_failure` / `stale_record` /
`stale_record_state`. The row's fallback candidate is bitwise equal to the ordinary
annotator's candidate, it still goes through the official Auditor, and no solver backward or
candidate check is spent on it. Only `StaleReplayRecordError` is caught; any other exception
from the replay surfaces. If that fallback transition is accepted it becomes a new
**ordinary** record at the current turn (never labelled C), built from the same single
ordinary forward as the rest of the ordinary partition so its factual batch composition is
preserved even in a group where other rows' C1 succeeded.

> **Correction (1 of 2).** An earlier revision of this report stated that a C1 failure "falls
> through to the ordinary annotator". That was false in the code: the row received a
> zero-innovation identity candidate, and a `StaleReplayRecordError` propagated out of
> `infer`.
>
> **Correction (2 of 2).** The first attempt at that fix was *still* wrong, and my 52-test
> suite did not catch it. The routing test was written as `replay_ok is False`, but
> `c1_passed[offset]` is a 0-dim `torch.bool` tensor and `tensor(False) is False` is always
> false, so a purely numeric replay failure stayed on the identity path. My own corruption
> fixture masked it: it perturbed the *stored factual logits*, which also set the
> retained-state `stale` flag, and `stale` was compared with `bool(...)`, so the fallback
> fired through the staleness branch instead. The check is now tri-state
> (`replay_ok is not None and not bool(replay_ok)`, where `None` means the mode performs no
> replay), and there is an independent **numeric-only** fixture that perturbs nothing but the
> replay output. That fixture fails against the previous code with exactly the reported
> symptom (`accepted_path: factual_support` instead of `ordinary`) while the other 55 tests
> pass, which is the evidence that the old suite could not close this.

### C2 — bounded restitution solver

Objective, on the thresholded predicted-REGRESS set only:

```
weighted CE( replay(Q), argmax A_pre_transition ) / max(eps, sum w)  +  lam * mean|| (Q - P) / pixel_step ||^2
```

* Decision variable is the realized sampling coordinates `Q` only, at the same recorded
  state, indices and depth. Nothing else can differ — the replay API accepts no other input.
* Trust region: axis-wise L-infinity box of `rho_feature_pixels` (default 1.0) around the
  realized factual `P`, converted with `2/(size-1)` in `align_corners=True` units, `x` =
  width, `y` = height. A **degenerate size-one axis gets a zero step**, so it is frozen
  rather than silently divided by zero. Domain is the normalized `[-1,1]` square; because
  `P` is already in-domain, the post-box clamp can only move toward `P`, so both
  constraints hold.
* Displacement regularizer is in **feature-pixel units**, so `lam` is dimensionless.
* **FIX protection**: **every** pixel with `E_FIX >= fix_threshold` is protected — there is
  no margin-based exclusion. The predicted class is `factual_logits.argmax(dim=1)`, which is
  a deterministic function of the stored factual output and is defined for a tied pixel
  exactly as for any other (lowest winning index), so a tied pixel has a predicted class to
  preserve like every other thresholded FIX pixel. Protection is two conditions, both over
  the full protected set:
  1. the candidate winning margin against the factual winner is at least
     `margin_fraction * factual_margin` (for a tie the factual margin is `0`, so the
     required margin is `0` and this condition is vacuous there); and
  2. `candidate_logits.argmax(dim=1)` equals the factual winner.

  Condition 2 is stated explicitly rather than inferred from condition 1 because condition 1
  cannot decide an exact tie: a candidate that flips the class at exactly equal logits has a
  winning margin of exactly `0.0`, which clears a required margin of `0.0`. The **final C3
  check applies the same two-condition test to the actual transferred candidate over the
  same full protected set**, and a row that fails reverts to the factual support with reason
  `c3_preservation_failed`. The factual point is feasible by construction under both
  conditions. `num_protected` therefore counts **all** thresholded predicted FIX pixels;
  `num_ties_excluded` is retained at a constant `0` for backward diagnostic compatibility
  (nothing is excluded any more) and the new `num_fix_ties` reports, for information only,
  how many protected pixels sit at or below the `1e-6` tie epsilon. `candidate_c_no_fix`
  still removes the constraint and nothing else: it reports the same `num_protected`.
* Budget: exactly one differentiable factual replay (which *is* the C1 check and the base
  objective — there is no hidden second base forward), exactly one backward, and at most
  `max_backtracks` **total** candidate checks. `max_backtracks` is hard-bounded to the
  integers `0..2` by `CandidateCConfig.__post_init__` itself, so no entry point — YAML,
  `SelfAuditNet(...)`, or a direct `CandidateCConfig(...)` — can construct an unbounded
  loop. `rho_feature_pixels` must be strictly positive and every probability threshold
  (`fix_threshold`, `regress_threshold`, `margin_fraction`, `min_regress_mass`) must lie in
  `[0, 1]`. No optimizer, no double backward, no optimizer on model parameters.
* Proposal rule, in **feature-pixel coordinates** `z = (Q - P) / pixel_step` with
  `grad_z = grad_Q * pixel_step`: `step_z = -rho * grad_z / max|grad_z|` per sample across
  every internal depth, then `step_Q = step_z * pixel_step`. Across one sample the largest
  pixel-gradient component therefore moves exactly `rho` feature pixels on **both** axes of a
  non-square grid. An explicit `lr` replaces the normalization with a step in feature pixels
  per unit pixel-gradient.
* The two candidate points are the single projected proposal `Q1` and the midpoint
  `P + 0.5*(Q1 - P)`, so the second check genuinely **halves the realized displacement** and
  both points satisfy the box and the domain.

> **Correction.** An earlier revision normalized the step in raw normalized-coordinate units
> (`scale = 1/max|grad_Q|`, giving `max|direction| = 1.0`) and then clamped it against a box
> of about `0.03`. Nearly every component saturated, so the "half step" re-evaluated almost
> the same point and the claim that the largest component moves exactly `rho` pixels was
> false. Both are fixed; a non-square-grid test now verifies the pixel bound and that the
> half check really halves the displacement, including at the domain boundary.
* Eligibility is an **explicit intersection with `record.accepted`**. A row whose ordinary
  transition was rejected stays in the replay only to preserve the factual batch composition
  and is reported as `record_row_not_accepted`; it is never inferred to be ineligible from
  `retained != factual`, which would be a coincidence rather than a rule.
* Returns the factual support (**exactly zero** innovation) on: empty/negligible REGRESS
  mass, non-finite objective, zero or non-finite gradient, non-finite replay, infeasible
  candidates, or no strict finite improvement.

### C3 — innovation transfer

```
candidate = current_retained_state + sigmoid(recorded_gate) * (counterfactual_logits - factual_logits)
```

The gate is the recorded per-pixel sigmoid gate, spatial and **class-shared**. The entire
innovation and gate are detached; gradients reach the annotator only through the retained
state. When the solver returns `Q = P` the innovation is an explicitly constructed zero
tensor, not a replay residue, so the identity fallback is exactly zero.

Because an accepted transition means `current_state == factual_logits`, C3 is a convex
mixture at every pixel. That argument is *verified rather than assumed*: a final
preservation check confirms the actual transferred candidate keeps the factual winning
class at every protected pixel, and any row that fails reverts to the factual support.
The check is not redundant even for a feasible counterfactual: the mixture is convex between
the *factual output* and the candidate, whereas `current_state` is only required to match the
factual output to within the C1 tolerance (`replay_atol`/`replay_rtol`), so a tied protected
pixel can still flip at the transfer. `test_final_c3_check_catches_a_tie_that_survived_the_counterfactual`
exercises exactly that path.

The official detached Auditor still judges the final candidate; acceptance remains strictly
`delta_q > tau_accept` and rejection still HALTs. No ground truth enters the proposal,
the replay or the evidence — `oracle_target` only affects the acceptance rule, never the solver.

## 3. Temporal contract (the part that is easy to get wrong)

A C-produced accepted transition is **not** an ordinary `AnnotationExpert` update of its
immediate pre-state, so it can never be replayed by C1. The implementation therefore:

* records only accepted **ordinary** transitions (`RECORD_KIND_ORDINARY`), one per row, for
  one `infer` call;
* on an accepted restitution or rollback action, **clears** that row's ordinary record and
  leaves only an ineligible `previous_kind` marker, which forces the next active turn back
  onto the ordinary annotator;
* applies this **even when the restitution fell back to the factual support** — an identity
  action taken on the restitution path is still not an ordinary update, and relabelling it
  would be exactly the dishonest shortcut the contract forbids;
* never retains an older ordinary record as "the most recent accepted action";
* produces no recursive restitution-on-restitution records.

**Exact factual batch composition is preserved.** The record keeps *every* row of the
factual ordinary forward, not only the accepted ones, and marks accepted rows with an
`accepted` mask; the replay then re-executes at that original batch shape and discards rows
that have since halted. Replaying only the survivors would change kernel tiling and
reduction order and could break the strict C1 tolerance under AMP — which must never be
worked around by loosening the tolerance. The C1 tolerance stays at `1e-5`.

Records are runtime-only: never a buffer, never in `state_dict`, never serialized, never
spanning batches. `candidate_c` adds **zero parameters and zero buffers**: the state dict is
key-identical and parameter-count-identical to the `current` baseline in every mode. `state_identity` is a blake2s digest over per-parameter and per-buffer
**object identity and `_version`** (an optimizer step bumps `_version` in place), dtype,
device, module mode, conditioning/offset mode and an explicit invalidation generation —
never a sum of versions. Replay against a changed state raises `StaleReplayRecordError`
(fail closed). `SelfAuditNet.invalidate_candidate_c_records()` lets a trainer make the
guarantee explicit around `optimizer.step()`.

## 4. Gradient policy

The solver runs inside `torch.inference_mode(False)` plus `enable_grad()`, on ordinary
detached clones materialized from possibly-inference tensors. Only the coordinate tensors
carry `requires_grad`, and the single gradient is taken with `torch.autograd.grad` rather
than `backward()`, so no `.grad` buffer anywhere in the process is written or cleared. Verified by test: parameter `requires_grad` flags are unchanged, a
pre-seeded `.grad` is bit-identical afterwards, no other parameter acquires a gradient, the
input image gets no gradient, and the encoder and Auditor get no gradient — under both a
caller `no_grad()` and a caller `inference_mode()`, with results identical to each other.
The ordinary joint-training path keeps its gradients (`infer(...).backward()` still
populates annotation-expert gradients).

Agreed gradient policy for joint training: the restitution innovation and its gate are
**always** detached; `detach_innovation=False` is deliberately not exposed. Differentiating
through the solver is out of scope.

## 5. Switches

`SelfAuditNet(window_mode=...)`, no duplicated model, zero new learned parameters:

| Mode | Behaviour |
|---|---|
| `current` | Shipped baseline. Verified **bitwise identical** to HEAD. |
| `feature_only` | Both audit entry paths removed (input-projection channels zeroed *and* window conditioning dropped); parameter shapes unchanged. |
| `free_offsets` | Only the pre-existing free channels; no ellipse/ring term; identical parameter shapes and count. The unit `tanh` outputs are rescaled to the **same maximum per-axis reach as `structured`** (`max_center_displacement + max_radius + max_residual_offset` = 0.92 by default), so the control is not handicapped by the tighter 0.12 residual bound. Rescaling the unit `tanh` rather than dividing the bounded residual keeps this defined when `max_residual_offset` is zero. |
| `candidate_c` | C1+C2+C3 with FIX protection. |
| `candidate_c_no_fix` | Removes the FIX constraint only; keeps C1, bounds, budget and the Auditor gate. |
| `direct_rollback` | No solver, no replay; mixes retained logits toward pre-transition logits on the **thresholded** predicted-REGRESS set with the same recorded gate and the same audit gate. |

`CandidateCConfig` is strict (unknown keys, negatives, non-finite and out-of-range values
rejected) and accepts a validated mapping. There is no redundant `enabled` / `enforce_fix` /
`detach_innovation` field — `window_mode` alone selects activation and the constraint.
I touched no config file.

## 6. Diagnostics

`infer` returns `candidate_c_diagnostics`: JSON-compatible **per-sample** attempt rows with
`turn`, `sample_index`, `eligible`, `record_kind`, `c1_passed`, `c1_max_abs_err`,
`c1_max_rel_err`, `regress_mass`(+`_raw`), `fix_mass`, `num_protected`,
`num_ties_excluded` (constant `0`; kept for backward compatibility), `num_fix_ties`
(additive), `feasible`, `improved`, `objective_before`, `objective_after`,
`constraint_violation`, `coordinate_displacement`, `innovation_magnitude`,
`fallback_reason`, `accepted_path`, `evals`, `solver_group_size`, `delta_q`, `accepted`,
`proposal_checks_run`, `proposal_displacement`, and
`record_turn` — the turn index of the accepted ordinary transition being replayed, so an
offline triplet aligns as `transition_previous[record_turn]`,
`transition_candidates[record_turn]`, `transition_candidates[turn]` with no extra evaluation
and no redundant factual-logit list. `record_turn` is `None` when there is no record.

`num_fix_ties` is **additive**: it is a new key on the row, and both the summariser
(`self_audit.evaluation.candidate_c`, which reads rows with `row.get(...)`) and the
diagnostics tests use subset checks, so no existing consumer breaks. I deliberately did not
bump `CANDIDATE_C_DIAGNOSTIC_SCHEMA_VERSION` or extend `ROW_COUNT_FIELDS`: both live in
`src/self_audit/evaluation/candidate_c.py`, which is not mine in this dispatch. A follow-up
owner of that file may want `num_fix_ties` added to `ROW_COUNT_FIELDS` so it appears in the
summary; until then it is present on every row but not summarised. `num_ties_excluded` is
still emitted and is now **always `0` by construction** — it is kept only so existing
readers of the schema keep working, and it no longer carries information.
`None` always means *unmeasured*; it is never a stand-in zero.

**Field semantics for the diagnostics consumer (changed in this revision).**
`feasible`, `improved`, `objective_after` and `constraint_violation` now describe the
proposal that was actually **checked** — the accepted one, else the last one evaluated — and
are no longer synonyms for acceptance. A proposal can be `feasible: true, improved: false`
(reported as `no_improvement`) or `feasible: false, improved: true` (reported as
`infeasible`); the earlier revision set both fields to the acceptance flag and so labelled
every rejection infeasible. A failed feasibility check now exposes its measured
`constraint_violation` instead of a null. `constraint_violation` and `objective_after` are
null only when no proposal was evaluated at all. Two fields were added:
`proposal_checks_run` (0, 1 or 2) and `proposal_displacement` (the checked proposal's
displacement). `c1_failure_kind` names which half of C1 failed: `"numeric"` when the replay
ran but did not reproduce the factual output, `"record_state"` when the record no longer
matched the live model state. **C1 covers a valid recorded state, not only the numeric
comparison**, so a preflight `StaleReplayRecordError` now reports `c1_passed: false` — a
trainer counting replay failures observes invalid-state attempts instead of skipping them —
while `c1_max_abs_err` / `c1_max_rel_err` stay `null`, because no comparison happened and a
fabricated `0.0` would be a lie. A third field, `solver_invocation_id`, is stable within one `infer` call and
is shared by every row of one batched solver invocation: those rows all report that
invocation's `evals`, so **deduplicate on `solver_invocation_id` before summing eval
counts**. It is `null` for rows that ran no solver. `record_turn` is deliberately preserved
on an ordinary C1-fallback row, so a consumer can still align the original triplet. `coordinate_displacement` now describes the **selected** support — exactly
`0.0` on the identity fallback, and null only where no support selection took place (an
ordinary-path row).

`infer(..., capture_geometry=True)` additionally returns `candidate_c_geometry`, one element
per diagnostics row in the same order, carrying depth-ordered `factual`/`chosen` supports,
preclamp coordinates and attention — matching the schema the diagnostics worker's
`evaluation/candidate_c.py` documents.

Each depth entry's `turn_index` and `iteration_index` are the **effective** conditioning
identities the generator actually used, read back from the capture and stored on the record
— never re-derived. The raw request can exceed the embedding table and be clamped: at
`turn_index = 2` the three internal depths request `(2, 3, 4)` but the table stops at 3, so
the effective identities are `(2, 3, 3)`. An earlier revision reported the raw `4`, which
misdescribed the computation. The raw value is kept alongside as
`requested_iteration_index`. Runtime replay indices are unchanged by this; it is a reporting
fix only. I verified integration directly:
`geometry_row_metrics` reports `available: True` and `evaluate_candidate_c` runs on my
output. Raw tensors appear only behind this flag; the ordinary rollout output stays serializable.

Eval counts are honest about scope: they are per solver invocation and shared by the rows of
one batched group, with `solver_group_size` naming how many rows shared them.

## 7. Tests actually run

All with `/private/tmp/self-audit-torch241/bin/python` (verified Python 3.10.21, torch 2.4.1), CPU.

* `tests/test_candidate_c.py` — **60 passed in ~18s** (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`). Demonstrates: exact replay identity
  (`max_abs_err == 0.0`); multi-iteration override across all 3 internal iterations;
  override reproducing the generated support bit-for-bit with Q/K/V untouched; override
  validation rejecting out-of-domain/wrong-shape/non-finite/non-floating input without
  silent clamping; a **successful feasible improving solve** with a real non-zero coordinate
  displacement (not the identity fallback); FIX margin preservation checked on the final C3
  candidate; bounded evals (1 replay + 1 backward + <= 2 candidate checks); backtracking
  rescuing a step that violates the margin; every fallback (`no_regress_evidence`,
  `stale_record`, `replay_failure`, `infeasible`, `no_improvement`) returning exactly zero
  innovation; exact innovation subtraction; detached gradients with no leakage under
  `no_grad` and `inference_mode`; Auditor authority halting a rejected candidate;
  non-recursive record kinds; `feature_only` removing both audit paths; config validation;
  the geometry payload; a **B>1 mixed accept/halt batch** in which the solver replays the
  full four-row factual composition (`solver_group_size == 4`) while only the survivors are
  transferred, with C1 still exactly `0.0`; `record_turn` presence and correctness; and that
  no mode adds a parameter or buffer; a C1 replay failure inside real `infer` taking the
  ordinary annotation path with a candidate **bitwise equal** to the `current`-mode ordinary
  run and provably not the identity; an injected `StaleReplayRecordError` falling back
  without crashing while an unrelated `RuntimeError` still propagates; a **mixed group**
  where one row's C1 fails and another's succeeds, after which the fallback row's new
  ordinary record replays at `c1_max_abs_err == 0.0`; rejected record rows never eligible;
  feasible-versus-improved separation with exposed violations; hard config bounds at the
  direct runtime constructor; and a **non-square 8x12 grid** verifying the per-axis
  feature-pixel bound and that the half check halves the realized displacement; the free
  control's reach matching `structured` exactly, including the zero-residual-bound case; and
  solver invocations being identifiable for eval-count deduplication; an **independent
  numeric-only replay failure** (the replay output alone is perturbed, record and retained
  state untouched) taking the ordinary path with no staleness, zero backward and zero
  candidate checks; a preflight state failure counted as `c1_passed: false` with
  `c1_failure_kind: "record_state"` and null numeric errors; and effective conditioning
  identities `(2, 3, 3)` at turn 2 depth 3 matching the underlying capture.

  **Tied and near-tied protected FIX pixels** (this revision). A real forward pass cannot be
  asked to place a pixel at an exact logit tie, so four tests script the replay through
  `_ScriptedReplayExpert`: `candidate_logits = base + shift * sensitivity` where `shift` is
  the signed total coordinate displacement from the factual supports. `shift` is identically
  zero at the factual supports, so C1 holds by construction (`c1_abs_error == 0.0` is
  asserted, not assumed) and the solver takes its ordinary path; only the logits it observes
  are controlled. Each fixture is one row on a 2x2 grid with one REGRESS driver pixel
  (`E_REGRESS = 0.95`) and three FIX pixels (`E_FIX = 0.95`), so `num_protected == 3` and
  `num_ties_excluded == 0`, with `num_fix_ties == 2` (`1` in the exact-flip fixture). The
  near-tied pixel uses `1.0 - 5e-7`, whose float32 winning margin is `4.77e-7` — strictly
  positive and at or below the `1e-6` tie epsilon, i.e. exactly the case the old rule dropped.

  - `test_tied_and_near_tied_fix_pixels_are_protected_not_excluded` — the exact tie
    (margin `0.0`, deterministic winner class `0`) and the near tie are both protected. The
    proposal is a genuine improvement (`objective_before = 1.0986`, `objective_after =
    0.0058`, `checked_improved is True`) and would flip both predicted classes, so it is
    refused as `infeasible` with `violation_after = 0.80`, `settled` all false, exactly zero
    innovation and `selected_displacement == 0.0`. Both candidate checks run
    (`candidate_checks == 2`), so the half step is refused too.
  - `test_the_refused_step_is_feasible_for_strict_margin_pixels_alone` — the control that
    pins the change to the protected *set*. On the same fixture the accepted `no_fix` step
    satisfies the margin constraint on the single strictly-positive-margin pixel
    (`max(required - candidate_margin) <= 0.0`) and keeps its class, while the tied and
    near-tied predicted classes are destroyed both in the counterfactual and after the gated
    transfer. Under the old strict-margin-only set that same step would have been accepted.
  - `test_counterfactual_class_check_refuses_an_exactly_tied_flip` — isolates the
    class-identity condition. The protected pixel is `[0, 1, 1]` (winner class `1`,
    margin `0`); the candidate raises class `0` to exactly `1.0`, so the candidate winning
    margin is exactly `0.0` and `violation_after == 0.0` — the margin test measures **no**
    violation — yet the step is still refused (`checked_feasible is False`, reason
    `infeasible`, zero innovation) because the predicted class changed. Reverting only the
    `class_kept_row` term in the source makes this test fail while the rest of the file
    passes, so the condition is independently necessary.
  - `test_final_c3_check_catches_a_tie_that_survived_the_counterfactual` — the final class
    condition. The tied pixel does not move under the proposal, so the counterfactual passes
    both halves (`checked_feasible is True`, `violation_after == 0.0`), but `current_state`
    carries the tie the other way by `2e-6` (inside the `1e-5` replay tolerance, so `stale`
    is false). The transfer is refused with reason `c3_preservation_failed`, zero innovation
    and `selected_displacement == 0.0`.

  Each of these tests fails against the previous `factual_margin > 1e-6` protected set and
  passes against the corrected one. In every one the `candidate_c_no_fix` control
  (`enforce_fix=False`) accepts the same step and the assertions show the predicted FIX class
  actually changing — the violation the constraint prevents is exposed, not merely asserted
  away. The pre-existing success, gradient-isolation, C1-exactness and mixed-batch tests are
  unchanged and still pass; `test_solver_preserves_every_protected_fix_pixel` was updated to
  assert the full protected set (on the real untrained net: `num_protected == 512` per row,
  `num_fix_ties == 0`, `num_ties_excluded == 0`, covering all `1024` thresholded FIX pixels
  in the batch).
* `tests/test_self_audit_core.py`, `tests/test_self_audit_audit.py`, `tests/test_self_audit_regressions.py`
  — **30 passed** in the previous dispatch. **NOT re-run in this revision**: the coordinator
  scoped this pass to the focused file and compile only, and the root integration pass will
  retest the quiescent tree.
* `python -m compileall src/self_audit/models/` — clean.
* **Baseline equivalence**: built the model from `git archive HEAD` and from the working tree
  with `window_mode="current"` under identical seeds; `self_audit`, `always_accept_refinement`
  and `forward_annotation` outputs are **bitwise identical** (`max_abs_err = 0.0`).
  A0 / ConvNeXt construction, checkpoint and calibration provenance are untouched by me.

## 8. Limitations — replay, fallback, compute

1. **The FIX constraint is only as loose as the weakest protected factual margin, and the
   protected set now includes zero-margin pixels.** On an untrained net the smallest
   protected margin observed was `7.3e-4`, so `rho = 1` feature pixel is
   routinely infeasible and the solver returns the factual support. Protecting ties can only
   tighten this: a tied pixel imposes no margin requirement but does impose the
   class-identity requirement, so on a checkpoint that produces ties the solver returns the
   factual support at least as often as the earlier revision did. That is the requested
   behaviour — a predicted FIX class is never traded away — but solver acceptance rates are
   not comparable across the two revisions. This is the mechanism
   behaving correctly, not a bug, but it means `rho` and `margin_fraction` interact strongly
   with model calibration and must be tuned on real checkpoints. My feasible-solve tests use
   `rho = 0.01` for this reason and the report should not be read as evidence that the
   default `rho = 1.0` is usable.
2. **An untrained auditor produces no usable evidence.** With random weights the local
   softmax sits near 1/3 per channel, so `E_REGRESS >= 0.5` selects nothing and every
   attempt falls back with `no_regress_evidence`. End-to-end C behaviour therefore cannot be
   judged from these tests; it needs a trained auditor.
3. **Compute is several annotator evaluations per attempted correction**, not free.
   Preserving the factual batch composition means halted rows are replayed and discarded, so
   the replay cost is set by the recorded group size rather than by the surviving active
   count. That is a deliberate trade of compute for C1 exactness under AMP. Per
   solver invocation: 1 differentiable replay forward, 1 backward, up to 2 candidate
   forwards, plus the unchanged final audit — and activations for all replay iterations, not
   merely coordinates. In record-consuming modes the ordinary path additionally runs with
   geometry capture, and a turn whose active rows split across both paths runs **two**
   expert forwards. This is not equivalent to ordinary inference cost.
4. **C is structurally weak by construction.** An all-UNCHANGED transition returns identity,
   and C cannot repair errors the previous transition never touched. After an accepted
   restitution the next turn is always ordinary, so C fires at most every other turn.
5. **Numerics.** C1 was exact (`0.0`) on CPU float32 and under CPU bf16 autocast at B=1 and
   B=3, including a mixed accept/halt batch, because the replay re-executes inside the same
   ambient autocast context *and* at the recorded batch composition. I did not loosen
   tolerances to make it pass. GPU/TF32/channels-last determinism is **NOT TESTED** — no authorized GPU
   was available in this dispatch.
6. **Not run by me**: any GPU, ACDC or M&Ms gate; training integration; full suite.
   Those are other workers' and the coordinator's gates, and remain unmet from my side.
7. Batched solving makes feasibility and improvement decisions per row, but the two candidate
   forwards are shared by the whole group, so one row's settled step does not reduce another
   row's remaining budget.
8. A C1 or stale failure costs one extra ordinary forward for that row relative to a clean
   C turn, because the failing rows join the ordinary partition only after the replay has
   been attempted. The replay itself is the C1 check and cannot be skipped.
9. The trust region interacts strongly with calibration now that the step is correctly
   normalized: on the untrained fixture `rho = 0.2` is already infeasible under FIX
   protection, `rho = 0.5` needs the half step, and `rho = 1.0` stops improving at all. The
   default `rho = 1.0` still should not be assumed usable without tuning on a real checkpoint.

## 9. Coordination notes

* The diagnostics worker's `evaluation/candidate_c.py` already documented
  `infer(..., capture_geometry=True) -> candidate_c_geometry`, which my dispatch did not
  name. I implemented it to their published schema and verified their functions consume my
  payload rather than asking them to change.
* **Revision dispatch (REVISE) — acted on all six review items:** (1) the blocking C1/stale
  fallback now routes to the ordinary annotator with narrow `StaleReplayRecordError`
  handling and a correctly-labelled new ordinary record; (2) hard runtime bounds on
  `max_backtracks` (0..2), `rho_feature_pixels` (> 0) and every probability threshold
  ([0, 1]) at every entry point; (3) the proposal is rebuilt in feature-pixel coordinates so
  the half step genuinely halves; (4) solver eligibility explicitly intersects
  `record.accepted`; (5) `feasible` and `improved` are separated and violations exposed, with
  `proposal_checks_run` / `proposal_displacement` added and `coordinate_displacement`
  redefined as the selected support; (6) `validate_coordinate_override` compares full device
  equality including index, and three unused imports were removed. Two further coordination
  items landed in the same pass: the `free_offsets` control was rescaled to the structured
  operator's full per-axis reach (it previously had only the 0.12 residual bound against
  structured's 0.92, which made it a strictly weaker and therefore unfair control), and a
  stable `solver_invocation_id` was added so the diagnostics consumer can deduplicate the
  per-group eval counts.
* **Second review dispatch:** root independently reproduced a C1 blocker my 52-test suite did
  not close — the `replay_ok is False` tensor-identity check — plus two recorded-metadata
  defects. All three are fixed: tri-state C1 routing with an independent numeric-only
  fixture, preflight state failures reported as explicit C1 failures with a
  `c1_failure_kind` and no invented numeric error, and effective post-clamp conditioning
  identities in the geometry payload. `current`-mode output remains bitwise identical to
  HEAD.
* **For root to apply:** the `free_offsets` reach change (residual-only 0.12 -> structured
  0.92 per-axis) is not yet cross-referenced in `reports/candidate_c/baseline_interfaces.md`.
  That file belongs to the released config worker, so I did not edit it; a small final-report
  cross-reference is needed. `current`-mode output is
  still **bitwise identical** to HEAD after all of this; Q/K/V semantics, A0, the parameter
  and buffer sets, the official Auditor gate and the 130-epoch recipe are untouched.
* Acted on all four coordinator follow-ups in the previous dispatch: `record_turn` added as a JSON
  scalar row field; `autograd.grad` replacing `backward()`; exact factual batch composition
  preserved end to end (record creation no longer slices, replay no longer slices); and B>1
  plus `inference_mode` plus bf16 autocast tested. GPU remains authorization-blocked, so
  GPU/TF32/channels-last determinism is **NOT RUN**, not passed.
* Config/schema/builder wiring (`training/unified_config.py`, `training/_utils.py`, YAML) is
  the config worker's; I only added the `window_mode` / `candidate_c` kwargs to
  `SelfAuditNet.__init__` and `build_self_audit_net`.
