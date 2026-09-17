# Joint self-audit from epoch 1 — core verification

Date: 2026-09-12
Scope: training-core verification only. The config profile, the runner, and the schedule
YAML are owned by a separate worker.

## Decision

The user chose to train the Annotator and the Auditor jointly, with the Auditor's
accept/reject feedback live from the first step, explicitly accepting that the Auditor
begins randomly initialised. **No new objective, loss, or acceptance policy was added.**
The schedule places the already-approved `retained_final_annotation` / `threshold_gate`
interval at epoch 0 and runs it for all 130 epochs.

An earlier warm-up design (a `joint_bootstrap` objective, helper, and `[0,10)` interval)
was implemented and then **fully reverted** on the user's steering. Nothing from it remains:
`src/self_audit/training/joint_bootstrap.py` is deleted and `schedule.py` is byte-identical
to its pre-task state — no new objective, no new transition population, no relaxed
cross-field rule.

## Files

| File | Status |
|---|---|
| `src/self_audit/training/unified_trainer.py` | edited earlier — telemetry only, already reviewed, not expanded in this revision |
| `tests/test_joint_from_start.py` | 15 tests |
| `reports/candidate_c/joint_from_start_core.md` | this report |

No model core file, `finetune_joint.py`, existing config, script, test, or `docs.md` was
touched by this worker.

## The production change (unchanged in this revision)

Purely additive telemetry, derived from the interval contract, with no objective, loss, or
policy effect:

```
train_stats["rollout/feedback_gates"]   = "active" | "inactive"   # rollout == threshold_gate
train_stats["rollout/joint_trainable"]  = 1.0 | 0.0               # trainable == "all"
train_stats["rollout/interval_rollout"] = interval.rollout
log_payload["schedule/feedback_gates"]  = "active" | "inactive"
log_payload["schedule/joint_trainable"] = 1.0 | 0.0
```

Why: the existing `rollout/predicted_history_*` flags describe the **optional auxiliary
second rollout**, which this profile leaves off. Without these keys, a reader sees
`predicted_history_exposure = 0` for the whole run and can wrongly conclude that Auditor
feedback was inactive. It is live from step 1 inside the gated objective's own single
rollout. The keys apply uniformly to every interval of every profile — no objective branch.

## How the accept and reject cases are reached

Not by a seed, and not by overriding any decision. Each test pins the Auditor's **final
scoring layer** as a test-only initialisation: the last `Linear` of `auditor.global_head`
gets a zero weight matrix and a bias of exactly `±0.05`, so `delta_q` is that constant for
every row. The real, unmodified gate in `SelfAuditNet.infer` — `delta_q > tau_accept`,
strict — then does the deciding.

* Pinned: two Auditor scoring parameters.
* Untouched: the gate expression, `tau`, the halt rule, the annotation path, the audit
  loss, and every other Auditor parameter — the local head still produces real logits and
  still receives real gradient.

`test_pinned_score_drives_the_real_unmodified_gate` asserts that every scored transition's
`delta_q` equals the pinned value and that the resulting accept/reject and halt behaviour is
what the unmodified rule produces. This removes the earlier seed-dependent fixtures, which
relied on random sign and were not portable across PyTorch/timm versions or backends.

## What was verified

**The zero-based `[0, 130)` schedule works.** `total_epochs = 130`, one interval,
`get_interval(0)` and `get_interval(129)` both resolve to it, `is_interval_start(0)` and
`is_reset_boundary(0)` are true, and the interval round-trips through `to_dict`/`from_dict`
still satisfying the unchanged cross-field contract (including `reset_optimizer = True`).
The trainer builds all three parameter groups — `encoder`, `annotation_heads`, `auditor` —
at epoch 0, with every parameter trainable.

**Accepted case: a unit gradient step trains both families.** Gradients are asserted
**before** any optimizer step and **per term**: the annotation loss gives the
`annotation_expert` non-zero gradient and the Auditor exactly zero; the audit loss gives the
Auditor non-zero gradient and the annotator exactly zero. The step then runs with
`weight_decay = 0` (asserted on every parameter group), so observed movement cannot come
from decay acting on a gradientless parameter. The test is explicit that this is a **unit
gradient step, not the profile's first scheduled step**: the warmup-cosine schedule returns
`max(step / warmup_steps, 1e-8)`, so the real first step applies `base_lr * 1e-8`.

**All-reject case: the explicit limitation, asserted rather than hidden.** With the gate
pinned below `tau`, the row halts after exactly one attempted turn and exactly one
transition is scored. The Auditor **does** receive gradient from that rejected transition;
the annotation term reaches `initial_head` and the encoder but gives the `annotation_expert`
**exactly zero** — the rejected candidate is discarded from the graph. Random Auditor
feedback from step 1 therefore permits Auditor and A0/encoder gradients even on a
rejected transition, as observed in this fixture; it does not guarantee useful learning
or nonzero gradients on every real batch. Nothing was added to force acceptance, soften the halt, add a
candidate-side GT loss, or change A0 strength.

**Auditor inputs and ground truth.** Every tensor handed to the Auditor has
`requires_grad = False`, and the rollout is called with the image only — never with the
mask.

**One rollout, under either auxiliary-exposure setting.** Parametrised over
`predicted_history_exposure` ∈ {False, True}: exactly one `infer` and one `encode` per
batch, `auxiliary_batches == 0`, and both auxiliary losses zero, in both cases. Positive
evidence that feedback was live in that single rollout: the joint rollout's own counters
report `rollout_batches == 1`, `accepted_history_count > 0`, `real_evidence_attempts > 0`.

**Candidate-C record precondition, on a real `candidate_c` model.** The earlier version of
this check ran `window_mode: current`, where no replay record is ever opened, so it could
not prove the precondition. It is replaced by a `window_mode = "candidate_c"`,
`max_turns = 2`, batch-1 fixture that reads the published diagnostics:

* **Turn 0** — `solver_invocation_id is None`, `record_turn is None`, `eligible is False`,
  `fallback_reason == "no_accepted_history"`, `c1_passed is None`: no accepted ordinary
  transition exists yet, so the solver is not attempted at all.
* **Turn 1** — `solver_invocation_id == "turn1"` and `record_turn == 0`: the record built by
  the accepted ordinary transition at turn 0 is consumed here, and only here.
* `invalidate_candidate_c_records()` returns the next replay generation. This
  assertion verifies invalidation, not physical memory reclamation of tensor references.

Feasibility and improvement of the solve are deliberately **not** asserted — that would make
the test depend on the numerical outcome of the solve rather than on the precondition under
test.

**Invalidation is a real generation bump.** On the same `candidate_c` model, a two-batch
epoch reports `rollout/candidate_c_invalidations >= 5` (epoch start, before each batch,
after each optimizer step) and the replay generation counter advances by exactly that
number — so every reported invalidation actually happened rather than being a no-op hook
call.

**Best selection.** `best_selection_min_epoch = 0`, `best_metric =
final_foreground_macro_dice`. Selection is gated in the trainer on
`interval.rollout == "threshold_gate"`, so it is open from epoch 0 precisely because this
single interval is the gated one, and a real epoch-0 validation does produce that metric.

## Tests

`tests/test_joint_from_start.py`, 15 tests, run in isolation on CPU:

1–2. `test_joint_from_start_schedule_is_valid_and_gated_at_epoch_zero`,
   `test_trainer_builds_all_three_parameter_groups_at_epoch_zero`
3–4. `test_pinned_score_drives_the_real_unmodified_gate[True]` / `[False]`
5. `test_accepted_unit_gradient_step_trains_both_families`
6. `test_all_reject_trains_the_auditor_but_only_a0_and_the_encoder`
7. `test_auditor_inputs_are_detached_and_ground_truth_never_enters_the_model`
8–9. `test_joint_runs_one_rollout_whatever_the_auxiliary_exposure_flag_says[False]` / `[True]`
10. `test_candidate_c_record_is_unavailable_at_turn_zero_and_consumed_only_next_turn`
11. `test_candidate_c_records_are_invalidated_around_every_optimizer_step`
12. `test_train_epoch_logs_feedback_as_active_and_joint_from_the_first_epoch`
13. `test_best_selection_opens_at_epoch_zero_only_through_the_gated_metric`
14–15. `test_every_epoch_resolves_to_the_single_gated_interval[0]` / `[129]`

Observed environment for this run: torch 2.14.0, numpy 2.4.6, CPU. Only this file was run
in this revision; the broader regression suites were run against the same production state
in the previous dispatch and the production code has not changed since.

No GPU run, no full-suite run, no real training run, no download, no commit, no push.

## Risks

1. **Reject collapse is a real possibility, not a solved problem.** If the random Auditor
   rejects most rows early, the refinement expert receives little or no gradient while the
   Auditor keeps learning on a population of rejected transitions. The tests document the
   mechanism; they do not prevent it. Watch `accepted_history_count` and
   `attempted_turn_samples` against `rollout_samples` over the first epochs.
2. **The pinned-gate fixtures are constant-score fixtures.** They prove the gate's behaviour
   under a constant score; they say nothing about the score distribution a real random
   Auditor produces, and therefore nothing about how often acceptance actually occurs at the
   start of a real run.
3. **No end-to-end run.** Nothing here executes a real epoch on real data; the 130-epoch
   behaviour of joint-from-epoch-0 is unverified by construction and can only come from an
   actual run.
4. **Test ownership overlap.** `tests/test_joint_from_start_configs.py` belongs to the config
   worker and covers the profile itself; this file covers the training core. The small
   overlap on schedule shape is deliberate redundancy, not a conflict.
