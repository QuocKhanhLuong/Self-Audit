# W4 — Frozen transition bank readiness (worker report)

Worker: Claude (Orca dispatch `ctx_8bb7caf52c29`, task `task_cd3854a284b2`).
Date: 2026-09-08. No commit, no push, no training, no architecture/objective/
config/split/checkpoint change, Proposal 2/3 unopened.

## Scope actually delivered

A tested export path for a frozen transition bank: real proposal generation
from a real model, a separate evaluator that attaches actual quality
afterwards, and a strict-JSON artifact that is re-validated at both the
serialization and the loading boundary.

**No real bank was created in this environment**, and nothing in this task
claims one. Every test runs a randomly initialised `SelfAuditNet` over tiny
fixture volumes. That is enough to test the contract and is explicitly not a
diagnostic result about a trained model.

### Files owned and added (three new files, plus this report)

| File | Role |
|---|---|
| `src/self_audit/evaluation/transition_bank.py` | Generation, evaluator, schema, provenance blocks, validation, strict JSON I/O |
| `scripts/export_transition_bank.py` | CLI: bound checkpoint + real cohort → bank artifact |
| `tests/test_transition_bank.py` | 54 regressions, including two subprocess CLI integration tests |
| `reports/astra_impl_review/2026-09-08/wave4_bank_worker.md` | This report |

No existing production file was edited. `src/self_audit/evaluation/__init__.py`
was deliberately **not** touched (parallel-ownership rule); every consumer uses
the direct import `from self_audit.evaluation.transition_bank import ...`.

## Design: generation and evaluation are separate by construction

`generate_on_policy_proposals(model, images, *, patient_ids, case_ids,
slice_indices, tau_accept, t_max, rollout_policy)` is the deployable entry
point. It has no ground-truth parameter at all, and it *refuses a batch
mapping* (`TypeError`) because a dataset batch carries `mask` — passing the
batch dict would put query GT one lookup away from the rollout. The CLI holds
`batch["mask"]` back and hands it only to the evaluator.

The rollout is not a reimplementation. It calls `SelfAuditNet.infer` and reads
the trace the model already exports (`transition_previous`,
`transition_candidates`, `audits[t]["local_logits"/"delta_q"/"accepted"/
"active_mask"]`), so exporter behaviour cannot drift from deployed behaviour —
there is no second rollout loop to diverge. Inactive rows are filtered by the
model's own `active_mask`.

`generate_synthetic_proposals(model, images, ground_truth, ...)` is the
GT-consuming path. It reuses `CounterfactualGenerator` and stamps every row
`gt_used_in_generation=True`, `source="synthetic"`. `validate_bank` refuses a
synthetic row that claims `gt_used_in_generation=False` **and** refuses an
on-policy row that claims GT was used.

`TransitionEvaluator.evaluate(proposals, targets=...)` is the only component
that reads reference labels. It reuses the W1 helpers verbatim
(`score_transition`, `compute_dice_from_stats`, `SufficientStatistics`,
`resolve_metric_contract`/`validate_contract`) and rejects a volume metric
space, because a bank row is a per-slice proxy and must never be labelled a
volume score.

## Schema (per row)

`schema_version`, `patient_id`, `case_id`, `slice_index`, `stage`,
`trajectory_id`, `state_representation`, `previous_state_id`,
`candidate_state_id`, `delta_q` (predicted **signed** score, not a
probability), `accepted` (what the rollout actually did),
`hypothetical_gate {tau_accept, accept}` (what the threshold would decide —
recorded separately), `source`, `rollout_policy`, `gt_used_in_generation`,
`generation_operation`, `generation_valid`, `halted_after`, `local_evidence`
(declared FIX/UNCHANGED/REGRESS channel semantics, mean probability and
argmax pixel fraction), plus the evaluator block: `q_previous`, `q_candidate`,
`delta_dice`, the three `*_defined` flags, `delta_class`, `neutral_margin`,
`per_class_dice_previous/candidate`, `metric_contract`,
`metric_contract_version`, `metric_space`, `empty_policy` and the per-side
`sufficient_statistics`.

Header: `bank_schema_version`, `generation`, `evaluation`, `protocol`, `rows`,
`integrity`.

### State identifiers

`state_tensor_id(state, representation=...)` digests the **actual state tensor
content** (via the W3.1 `provenance.state_digest` primitive, which frames key,
shape, dtype and raw bytes) together with the declared representation. It is
not derived from a stage number, and it refuses a batched tensor so an id can
never be batch-position-dependent. Ids are full `repr:sha256:<64 hex>`; a
truncated digest is rejected.

### Source tags and rollout policy

`on_policy` / `always_accept_prefix` / `synthetic`, each pinned to exactly one
rollout policy (`self_audit` / `always_accept_refinement` /
`synthetic_perturbation`). `always_accept_prefix` is recorded as an analysis
trajectory: its rows carry `accepted=True` while their `hypothetical_gate`
records the rejection the deployable gate would have made. `validate_bank`
enforces the source/policy pairing and enforces that a `self_audit` row's
`accepted` equals its own threshold decision.

### HALT

A rejected on-policy sample halts. The generator raises if the model ever
emits a later stage for a halted row, and `validate_bank` rejects any bank
whose trajectory continues past a rejected on-policy transition. State
continuity is enforced in both directions: an accepted candidate must be the
next previous state; a rejected candidate must leave the state unchanged.

### Undefined quality

Undefined `Q` (every foreground class empty in both prediction and reference
under `empty_policy="exclude"`) keeps its row, its per-class nulls and its
sufficient-statistics coverage counters, and serializes as JSON `null`.
`dump_bank` writes with `allow_nan=False`; `load_bank` parses with a
`parse_constant` hook that rejects `NaN`/`Infinity` tokens outright.

### Provenance, kept separate

`generation_provenance_record(...)` and `evaluation_provenance_record(...)` are
two independent blocks, so re-scoring a frozen bank under a different contract
can never read as a change to the model that produced the rows. An unknown
producer stays `"unknown"`. The split signature keeps its narrower label
`membership_only_not_content_hash` (from `provenance.MEMBERSHIP_COVERAGE`) and
`validate_bank` rejects an attempt to upgrade that string to a content-coverage
claim. The bank's own `integrity.content_signature` *is* a content hash of the
header and rows, and is labelled as such.

Export identity has exactly two admissible shapes, named in
`generation.export_identity_class`:

* `bound_checkpoint_export` — requires a complete `CheckpointBinding`
  (`checkpoint_path`, `checkpoint_sha256`, `state_digest`,
  `file_state_digest`, `model_identity`, `producer`; all three digests full
  SHA-256) **and** a recorded cohort identity. `checkpoint_bound=True` with a
  null checkpoint or a null cohort is rejected.
* `unverified_demo_export` — only admissible with an explicit
  `unverified_demo_reason`. Constructing an unbound record without one raises.

## Review corrections applied (coordinator msg_d87826e42700, msg_a0b67cf6b3fe)

1. **CLI identity fallback removed.** `_identity` now raises when a batch does
   not carry `patient_id` / `case_id` / `slice_idx`; the positional
   `patient_0`-style placeholders and the `range(batch)` slice fallback are
   gone. A real export cannot invent an identifier. Two subprocess CLI
   integration tests were added against a real tiny ACDC-layout dataset, a real
   split manifest and a real `save_checkpoint` artifact.
2. **Semantic validation, not just a content hash.** `validate_bank` now
   recomputes `q_previous`, `q_candidate` and every per-class Dice from each
   row's own `sufficient_statistics` with `compute_dice_from_stats` under the
   bank's contract, and rejects any disagreement, any `*_defined` flag that
   contradicts the recomputed definedness, a `delta_class` that disagrees with
   `classify_delta` at the row margin, a `neutral_margin` or `metric_space`
   that disagrees with the contract, and a `delta_dice` that is not
   `q_candidate - q_previous`. The tamper tests re-seal the integrity
   signature first, so the semantic check — not the hash — is what fires.
3. **Complete bound identity is now mandatory** (see above), with the explicit
   unverified-demo format as the only alternative.
4. **Protocol and type strictness.** Protocol `tau_accept` must match every
   row's gate tau; `t_max` must cover every non-synthetic stage; `sources` must
   be a non-empty list of known tags that covers every row's source. Local
   evidence must declare the exact channels, carry finite values in `[0,1]`
   and sum to 1 on both blocks. State ids must be full SHA-256. Integers and
   booleans are read with strict readers (`True` is not an integer; `1` is not
   a boolean; floats are not truncated). One trajectory must belong to exactly
   one patient/case/slice/source/policy/representation.
5. **Non-finite values are rejected before `json_safe` can null them.**
   `build_bank` runs `_reject_nonfinite_scores` on the *raw* rows first:
   `delta_q` and `tau_accept` must always be finite, and a quality score may be
   non-finite only where its own `*_defined` flag says it is undefined. A
   non-finite value carried under a `defined=True` flag is an error, not a
   silent `null`.
   The CLI also re-runs `verify_bound_state(..., boundary=
   "transition_bank_export_complete")` after the whole export, so a mutation
   midway cannot be recorded under the identity bound at the start, and it
   refuses to write an empty bank.

On msg_8d0e1745dbe8 (deployable path parity): parity is structural — the
exporter has no rollout loop of its own, it reads `SelfAuditNet.infer`'s trace,
so proposals, decisions, local evidence and residual state semantics are the
deployed ones by construction. No model source was edited.

## The four bypasses in `wave4_bank_review_notes.md`

Each was reproduced the way the reviewer reproduced it — mutate a real
generated bank, **recompute `bank_content_signature`**, then call the public
`validate_bank` — so the content hash cannot be what rejects it. All four now
fail, each with a dedicated regression:

| Mutation | Draft | Now | Test |
|---|---|---|---|
| `sufficient_statistics.previous.tp["1"] = 999999`, `Q` unchanged | ACCEPTED | rejected: `q_previous ... does not match the score recomputed from its sufficient statistics` | `test_review_bypass_1_inflated_previous_tp_with_unchanged_quality` |
| `checkpoint_bound=True` with the `checkpoint` block removed | ACCEPTED | rejected: `A bound export must carry its complete checkpoint identity` | `test_review_bypass_2_checkpoint_bound_with_the_block_removed` |
| Row `metric_space="volume_native"` under a slice-proxy header | ACCEPTED | rejected: `metric_space ... disagrees with the bank contract metric space` | `test_review_bypass_3_row_metric_space_volume_under_a_slice_header` |
| Row `schema_version=1.5` | ACCEPTED | rejected: `schema_version must be an integer` | `test_review_bypass_4_fractional_row_schema_version` |

## Producer source-content identity (msg_75f20d5cb437)

The bank does not compare revisions — it is not a lineage consumer — and it
never compares `git_sha`. It embeds whatever `provenance.git_source_provenance()`
returns, so W3.1's new fields already appear verbatim at
`generation.generation_code.source_content_signature` and
`evaluation.evaluation_code.source_content_signature` (confirmed present in the
current tree, `source_content_signature_known=True`).

What the bank does enforce is honesty about them: `source_content_signature_known=True`
with a value that is not a full SHA-256 digest is rejected, and
`source_content_signature_known=False` with a signature stored anyway is
rejected — an unverified signature must stay `"unknown"`. A record written
before the field existed is left alone rather than back-filled
(`test_unknown_source_signature_may_not_be_recorded_as_known`).

## GT firewall and its positive control

`test_gt_firewall_generation_is_invariant_to_permuted_reference`: images and
weights fixed, reference masks permuted between the two arms. Every
generation-side field — `previous_state_id`, `candidate_state_id`, `delta_q`,
`accepted`, `hypothetical_gate`, `local_evidence`, `trajectory_id`, source and
policy tags — is asserted bit-identical, while the evaluator's
`(q_previous, q_candidate, delta_dice)` triples are asserted to move.

`test_positive_control_gt_dependent_generation_does_change`: the same
permutation applied to a *deliberately* GT-dependent generator (positive
counterfactual repair towards the reference, same RNG seed in both arms). The
GT-free previous state is asserted identical; the candidate state ids are
asserted to differ. Without this control the invariance test above would pass
even if the probe were blind.

No GT-dependent slice selection is used anywhere in this task, and no dataset
configuration was changed.

## Verification — exact commands and output

Interpreter (differs from the one recorded in `docs.md`; disclosed rather than
assumed):

```text
python=3.11.16 pytorch=2.13.0 pytest=9.1.1 numpy=2.4.6
exe=/Users/alvinluong/miniforge3/bin/python
```

Focused suite:

```text
$ python -m pytest -q tests/test_transition_bank.py
......................................................                   [100%]
54 passed in 15.20s
```

Source compilation:

```text
$ python -m py_compile $(find src scripts tests -name "*.py")
COMPILE_OK
```

Full suite, run at the moment this report was written. The tree was **not**
quiescent: `tests/test_calibration_lineage.py` currently fails to import a
symbol W3.2 has not finished adding, and a collection error aborts the whole
run, so the second command excludes that one module to obtain a number at all:

```text
$ python -m pytest -q
E   ImportError: cannot import name 'MEASUREMENT_SEMANTICS_VERSION' from
    'self_audit.evaluation.calibration_lineage'
ERROR tests/test_calibration_lineage.py
1 error in 0.98s

$ python -m pytest -q --ignore=tests/test_calibration_lineage.py
FAILED tests/test_checkpoint_binding.py::test_runner_post_training_branch_binds_best_at_every_consumer
FAILED tests/test_checkpoint_binding.py::test_calibrated_tau_is_verified_against_starting_state
FAILED tests/test_checkpoint_binding.py::test_explicit_cli_tau_cannot_skip_artifact_verification
FAILED tests/test_checkpoint_binding.py::test_uncalibrated_run_does_not_consult_or_verify_anything
FAILED tests/test_metric_contract_and_replay.py::test_cli_calibrate_threshold_roundtrip
FAILED tests/test_proposal1_cli_integration.py::test_cli_pipeline_positive_matching_end_to_end
FAILED tests/test_proposal1_cli_integration.py::test_cli_pipeline_replaced_weights_fails_before_calibrated_report
FAILED tests/test_proposal1_cli_integration.py::test_cli_tau_override_cannot_bypass_invalid_calibration
FAILED tests/test_proposal1_cli_integration.py::test_cli_disjoint_independent_evaluation_cohort_positive
9 failed, 291 passed, 1 warning in 47.28s
```

Every one of those nine failures is in `tests/test_checkpoint_binding.py`
(W3.1), `tests/test_metric_contract_and_replay.py` /
`tests/test_calibration_lineage.py` (W3.2 calibration) or
`tests/test_proposal1_cli_integration.py` (AGY, `ctx_084f9722dc62`) — all files
under active concurrent edit and none owned by W4. None of them imports
`transition_bank`. Per msg_18ee3d92d382 the authoritative full-suite run is the
coordinator's, once all writers are quiescent; no check was weakened here to
make a concurrent-edit failure disappear.

Whitespace check on the three owned files (no trailing whitespace, no tabs):

```text
$ grep -nP '[ \t]+$' <each owned file>   # no output
$ grep -nP '\t' <each owned file>        # no output
```

`git diff --check` covers tracked files only; all three owned files are still
untracked, and staging them to satisfy the check would have disturbed the
shared index other workers are using, so the equivalent whitespace check above
was run directly instead.

## Concurrent-snapshot disclosure

This ran in parallel with W3.1 (producer) and W3.2 (calibration) on the same
working tree, and the tree moved between runs. Observed on three consecutive
full-suite runs within a few minutes, with no change of mine in between:

* `2 failed, 359 passed` — `test_calibration_lineage.py::test_overlapping_patients_are_rejected_even_when_authorized` plus the calibrate CLI test
* `6 failed, 355 passed` — five `test_calibration_lineage.py` identity-mutation cases plus the calibrate CLI test
* `1 failed, 360 passed` — the calibrate CLI test only
* `1 collection error` / `9 failed, 291 passed` — the final state above, after
  W3.2 and AGY landed further in-flight edits

`tests/test_transition_bank.py` passed on every one of those runs. The
`test_cli_calibrate_threshold_roundtrip` failure that is present throughout
reports `"...metric contract or cohort it was measured on. Re-cache with
scripts/cache_validation_transitions.py, which records one."` — a
producer/consumer strictness issue in W3.2's ownership, not a bank issue, and
the coordinator has already flagged (msg from the read-only QA worker) that this
test file belongs to no ownership row. No compatibility bypass was added on my
side to make it or any other cross-owner failure pass.

## Dependencies and open items

* Imports `json_safe` from `self_audit.evaluation.threshold`, which W3.2 owns
  and is actively editing. It is a read-only dependency (strict JSON coercion);
  if its semantics change, the bank's serialization must be re-checked.
* Imports `state_digest`, `git_source_provenance`, `resolve_model_identity`,
  `cohort_identity`, `preprocessing_descriptor`, `MEMBERSHIP_COVERAGE` and
  `UNKNOWN` from `self_audit.provenance` (W3.1), and
  `bind_evaluation_checkpoint` / `verify_bound_state` / loader builders from
  `training/_utils.py` (W3.1). No producer API change was requested; the
  current interfaces were sufficient.
* `evaluation/__init__.py` deliberately does not re-export this module. If the
  coordinator wants it in the package surface, that edit belongs to whoever
  owns that file after the parallel phase.
* Interface/integration PASS is not claimed here: it waits on W3.1/W3.2
  review, per the parallel execution update. What is claimed is that the
  export path itself is implemented, exercised end-to-end through the CLI
  against a real tiny dataset and a real saved checkpoint, and covered by the
  regressions listed above.
* Remote GitHub Actions remains waived by the user as a blocking gate; it is
  not claimed as PASS, and no CI file was touched.
