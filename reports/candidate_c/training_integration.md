# Candidate C — training integration (Worker 3)

Base HEAD `9a612485b7ead1273082d7dd52879f8e9e7a3e83`. Files edited, all within this
stream's exclusive ownership:

* `src/self_audit/training/finetune_joint.py`
* `src/self_audit/training/unified_trainer.py`
* `tests/test_candidate_c_training.py` (new)
* `reports/candidate_c/training_integration.md` (this file)

No config, core-model, data or script file was touched. No commit, push or
worker spawn was made.

## What this delivers, and what it does not

It delivers the **opt-in curriculum plumbing** described in the contract: a
bounded auxiliary accepted-predicted-history rollout that is **off by default**,
plus sample-level rollout telemetry emitted for every schedule interval, plus
replay-record invalidation around optimizer updates.

It delivers **no evidence** that predicted history is informative. During the
annotation bootstrap the Auditor is untrained, so its local evidence is cold and
uncalibrated; during the auditor-only interval the same evidence is
non-stationary because the Auditor is being optimised underneath it. Both facts
are logged explicitly (`rollout/evidence_calibration`, values
`cold_untrained_predicted` and `nonstationary_training_predicted`) rather than
being implied away. No test in this stream claims trained-evidence use.

## Exact schedule behaviour

The 130-epoch schedule, the interval boundaries (0–100 / 100–120 / 120–130), the
learning rates, the ConvNeXt encoder settings, the primary bootstrap objective
and the A0 stage weighting are all unchanged. Nothing in this change alters which
objective an interval runs or how long it runs.

Per interval, with `training.rollout.predicted_history_exposure = false`
(the default):

| Interval | Objective | Behaviour |
|---|---|---|
| 0 `annotation_bootstrap` | `weighted_a0_a3` | Bit-identical to base: `phase_a_loss` on the zero-history `forward_annotation` output, scaled by `annotation_weight`. No rollout runs. |
| 1 `auditor_training` | `counterfactual_audit` | Bit-identical to base: `_auditor_batch` only. No rollout runs. |
| 2 `joint_self_audit` | `retained_final_annotation` | Bit-identical to base: `compute_joint_losses` only. The joint rollout runs as it always did; only its counters are now read out of it. |

With the option **on** (`predicted_history_exposure: true`,
`predicted_history_weight` default `0.1`):

* **Interval 0** — the primary zero-history loss is untouched. An additional term
  `annotation_weight × predicted_history_weight × annotation_loss(retained_final_logits, GT)`
  is added, where the retained final logits come from the official deployable
  path `model.infer(mode="self_audit", tau_accept=training.rollout.tau,
  t_max=training.rollout.max_turns)`. Same official Auditor gate, same strict
  `delta_q > tau` acceptance, a rejection still HALTs the row. No GT is passed
  into the proposal, no acceptance is forced, no oracle or synthetic audit map is
  injected. Skipped entirely when `annotation_weight == 0`.
* **Interval 1** — the annotator stays frozen (`set_module_trainability("auditor")`
  already sets `requires_grad=False` and `eval()` on every non-Auditor module).
  The rollout itself runs under `torch.no_grad()`, and every Auditor input —
  shared features, previous probabilities, candidate probabilities — is a
  detached leaf. The collected on-policy pairs are re-scored by `model.auditor`
  with grad enabled, and an audit loss with GT-derived transition targets is
  added as `audit_weight × predicted_history_weight × aux`. The base
  `_auditor_batch` term is unchanged; when it yields no transition the auxiliary
  term alone can still carry the batch, and when neither produces a term the
  batch is skipped exactly as before. Skipped entirely when `audit_weight == 0`.

  **Added cost of this path, stated exactly.** `infer` already publishes the
  shared features the official Auditor was conditioned on for this trajectory,
  so they are reused and there is **exactly one encoder forward** — no second
  encode, and no fallback substituting raw images for shared features (a missing
  `shared_features` key raises rather than inventing one). Rows are selected
  without copying the full feature map when every row is active.

  Under the **baseline** `window_mode: current` the path costs 1 encoder
  forward, `T` annotator forwards, and `2 × T` Auditor forwards for `T` active
  turns: `T` **official Auditor collection passes** inside `infer` under
  `no_grad` that decide acceptance, plus `T` **gradient-bearing Auditor loss
  passes** here. One backward runs, through the Auditor only.

  **That `T` figure is baseline-only and must not be quoted for a restitution
  run.** Under `candidate_c` / `candidate_c_no_fix` / `direct_rollback` each
  record-consuming attempt adds one differentiable historical replay of the
  stored support and up to two candidate checks per solver group, plus a
  coordinate backward inside the solver; because supports are grouped, the
  factual replay batch can exceed the number of active rows. Those evaluations
  belong to `infer`, not to the auxiliary loss — the auxiliary path neither
  performs nor double-counts them — but it inherits them, so the real annotator
  cost under a restitution mode is strictly greater than `T`.
* **Interval 2** — **no auxiliary rollout**. The joint objective already rolls out
  with accepted predicted history, so adding one would be a duplicate rollout at
  double cost. `auxiliary_batches` is therefore 0 here by design, and the
  counters are measured from the joint rollout itself.

### Restricted-schedule disclosure

Per the contract's "report the exact restricted schedule" clause: exposure is
implemented for intervals 0 and 1 as an *auxiliary* term only. The primary
bootstrap objective never sees predicted history, and interval 2 is unchanged.
So it is not the case that "all stages learned from evidence"; what is true is
that, when the option is enabled, intervals 0 and 1 additionally see the
accepted-predicted-history distribution at a configurable, bounded weight.

## Gradient paths

* **Interval 0 auxiliary.** Gradient flows from the retained final state back
  through the ordinary annotation path (encoder, initial head, annotation
  expert). The audit evidence the expert consumes is detached inside `infer`, and
  `delta_q` is detached before the gate, so the discrete acceptance decision
  carries no gradient. Auditor parameters have `requires_grad=False` in this
  interval and are absent from the optimizer's parameter groups. Candidate-C
  restitution innovations are detached by the core module; only the ordinary
  retained path trains, which is exactly the contract's "C solve innovation
  detached, primary ordinary retained graph still trains".
* **Interval 1 auxiliary.** The only differentiable module in the graph is
  `model.auditor`. Verified by test, not by inspection: every non-Auditor
  parameter has `grad is None` or a zero gradient after `aux.backward()`, and at
  least one Auditor parameter has a non-zero gradient.
* **Interval 2.** Unchanged from base — annotation term on the retained final
  state, audit term on detached transition inputs.
* **Ground truth.** GT is used only as a supervised target
  (`annotation_loss(..., batch["mask"])`, `build_transition_targets(..., GT)`) and
  for evaluation. It never enters a runtime proposal, a replay, or an acceptance
  decision.

## Counters

Six sample-level counters are emitted for **every** interval, under the
`rollout/` prefix, in `train_stats` → the JSON report row → the W&B payload (the
payload copies the `rollout/` keys verbatim, so the three views cannot disagree).
Counts are in sample-turn attempts, never in batches.

| Counter | Definition |
|---|---|
| `real_evidence_attempts` | Attempts where the annotator consumed the **latest** real predicted accepted evidence: rows accepted at turn `t-1` **and** attempted again at turn `t` (`state_masks[t-1] & active_masks[t]`). A rejected attempt HALTs and can never contribute. Whatever produced that acceptance counts — under a restitution window mode it may be a Candidate-C or direct-rollback action. |
| `accepted_history_count` | **All** accepted transitions **created** by the rollout (`sum(state_masks)`), ordinary and restitution alike. It is a creation count, not a reuse count — it includes a final-turn acceptance that no later turn could consume. Attempted-but-rejected history is never counted. |
| `candidate_c_attempts` | Rows that actually reached the solver holding a replayable ordinary record: a valid `record_turn` with `record_kind == "ordinary"` (an explicit `attempted` boolean from the core overrides this inference). Counted **whether or not the solve was eligible or succeeded**. A fresh ordinary turn holding no record is not an attempt, and neither is a turn whose held record is itself a restitution (`record_kind == "candidate_c"`, `record_turn` null). |
| `candidate_c_feasible` | Attempted rows with `feasible is True`. |
| `candidate_c_fallback` | Attempted rows with a non-null `fallback_reason`. |
| `candidate_c_replay_failure` | Attempted rows with `c1_passed is False`. `None` is *unmeasured* and is not counted. |

The attempt predicate is deliberately **not** `eligible is True`. Gating on
eligibility excluded exactly the rows that represent real solver failures — a
C1 replay failure or a stale record makes the core mark the turn ineligible and
route it back to normal annotation — so `candidate_c_replay_failure` could never
increment for a genuine core failure, and a zero-REGRESS turn never reached the
denominator either. Feasible / fallback / replay-failure are all counted inside
this actual-attempt denominator. Core's revision that routes a numeric or state
C1 failure to normal annotation while still logging the origin `record_turn` is
handled correctly: such a turn counts as an attempted C **and** as ordinary
annotation exposure.

One further counter is published **alongside** the six, clearly named so it
cannot be mistaken for one of them:

| Counter | Definition |
|---|---|
| `ordinary_annotation_real_evidence_attempts` | Post-acceptance attempts whose **own** forward is the trainable ordinary annotation generator: accepted at `t-1`, attempted at `t`, and the row **for turn `t`** reports `accepted_path == "ordinary"` — counted as forward exposure regardless of whether turn `t` was itself accepted. Under a baseline ordinary rollout no other path exists, so every post-accept attempt qualifies and the count equals `real_evidence_attempts`. `ordinary_accepted_path_from_diagnostics` records which derivation was used (`1.0` = per-sample `accepted_path`). |

**The evidence's source does not determine which module receives it.** The
counter keys on the *current* turn's path, not the previous turn's:

* previous **ordinary** → current **C**: the evidence is consumed inside the
  solver. Does **not** count as generator exposure.
* previous **C** → current **ordinary**: the evidence is consumed by the
  trainable ordinary annotation generator. **Does** count.
* a C1 replay failure routed back to normal annotation: counts here (it really
  did feed the previous predicted evidence into the ordinary generator) *and*
  counts as a `candidate_c_attempts` with `candidate_c_replay_failure`.

An earlier revision had this reversed, keying on the previous turn's path. The
totals of a symmetric ordinary → C → ordinary walk are identical under both
conventions, which is why the fixtures below assert the **per-turn** breakdown
rather than the total. Measured against the real solver
(`window_mode="candidate_c"`, `t_max=3`, 2 samples, all turns accepted):
`real_evidence_attempts = 4`, `ordinary_annotation_real_evidence_attempts = 2`,
with the whole contribution on turn 2 (the ordinary turn) and none on turn 1
(the counterfactual one) — `candidate_c_attempts = 2`, `feasible = 2`.

**This counts forward exposure, not a guaranteed parameter gradient.** An
accepted conditional update, or a gradient probe confirming the evidence
actually carried into a weight update, would be separate measurements. Neither
is claimed here.

Denominators published alongside, so no ratio has to be guessed:
`rollout/attempted_turn_samples` (total sample-turn attempts),
`rollout/rollout_samples`, `rollout/rollout_batches`,
`rollout/predicted_history_exposure`, `rollout/predicted_history_weight`.

**Zero vs. unmeasured.** A zero here is a measured zero *only* when
`rollout/rollout_batches > 0`. The Candidate-C family carries its own
availability flag, `rollout/candidate_c_diagnostics_available`: it is `1.0` only
when `model.window_mode` is `candidate_c` or `candidate_c_no_fix` **and** the
`infer` output actually carried `candidate_c_diagnostics`. Under `current`,
`feature_only`, `free_offsets` or `direct_rollback` there is no C solver, so the
four C counters are zero with the flag at `0.0` — per the coordinator's
amendment, those zeros are never presented as valid C telemetry. On stage
aggregation the flag is a conjunction over rollout batches: one batch without
diagnostics makes the whole epoch's aggregate unavailable.

**Auxiliary cost is reported separately** from the primary loss:
`rollout/auxiliary_batches`, `rollout/auxiliary_seconds`,
`rollout/auxiliary_annotation_loss`, `rollout/auxiliary_audit_loss` (the two loss
figures are means over `auxiliary_batches`; both are `0.0` when that count is 0).

## Record invalidation

`UnifiedTrainer.invalidate_candidate_c_records()` calls the model hook of the
same name when the core module exposes it, and reports `False` otherwise so a
missing hook is never logged as an invalidation that happened. It is called at
the start of every training epoch, before every batch's forward, and immediately
after every optimizer update (both the in-loop step and the trailing partial
step). A replay record therefore cannot span a batch boundary or a weight
update. The per-epoch count is logged as `rollout/candidate_c_invalidations`.

## Tests

`tests/test_candidate_c_training.py` — 23 tests, **23 passed in 6.4 s** on CPU
with `/private/tmp/self-audit-torch241/bin/python` (PyTorch 2.4.1):

* rejected history is never credited as accepted history; `real_evidence_attempts`
  counts only accepted→attempted continuations;
* Candidate-C counters are gated on the actual `window_mode` across all six
  modes, `c1_passed is None` does not inflate the failure count, and a missing
  diagnostics list reports unavailable rather than measured;
* the C attempt denominator is record-based, not eligibility-based: an
  ineligible C1-failure row and a zero-REGRESS row both count as attempts (so
  `candidate_c_replay_failure` can increment for a real core failure), an
  eligible row holding no record does not, a row whose held record is itself a
  restitution does not, and an explicit `attempted` flag overrides the
  inference;
* availability is conjoined correctly across rollout batches;
* **option off**: interval-0 loss is `allclose` to the `phase_a_loss` reference,
  `auxiliary_batches == 0`, all six counters zero with `rollout_batches == 0`;
* settings validation rejects a negative weight, a NaN weight and a non-bool
  flag, and defaults to `(False, 0.1)` when the schema does not carry the fields;
* **option on, interval 0**: the total equals baseline + `weight × aux` to 1e-5,
  differs from the baseline, the rollout is measured
  (`attempted_turn_samples > 0`), and the term really reaches annotation-expert
  parameters;
* **option on, interval 1**: annotator parameters are frozen, and after
  `aux.backward()` no non-Auditor parameter has a non-zero gradient while the
  Auditor does;
* **option on, interval 2**: loss is `allclose` to `compute_joint_losses`,
  `auxiliary_batches == 0`, counters still measured;
* invalidation is called exactly `1 + batches + optimizer_steps` times per epoch
  and the count matches the logged counter;
* every stage emits all six counters with coherent denominators;
* the W&B payload carries every `rollout/` field the report row carries;
* **against the real solver**: a `SelfAuditNet(window_mode="candidate_c")`
  rollout is run and its actual `candidate_c_diagnostics` rows are classified by
  the documented rules — availability `1.0`, attempts equal to the rows holding
  a replayable ordinary record, the first-turn row holding no record correctly
  excluded, and the per-turn ordinary exposure independently recomputed from the
  published `accepted_path` and required to match the counter;
* **temporal fixtures, asserted per turn** (totals alone cannot distinguish the
  two conventions):
  * 2 turns × 2 samples, ordinary → C: `real_evidence_attempts = 2` but
    `ordinary_annotation_real_evidence_attempts = 0`, per-turn `[0, 0]` — the
    solver consumed that evidence;
  * 3 turns × 2 samples, ordinary → C → ordinary: per-turn `[0, 0, 2]`,
    `real = 4`, `ordinary = 2`. A reversed convention would give `[0, 2, 0]`,
    the same total on the wrong turn;
  * 1 sample, ordinary → C1-failure-routed-to-ordinary: counted as a C attempt
    **and** a replay failure **and** ordinary exposure, per-turn `[0, 1]`;
  * the original 3-turn single-sample walk now also asserts per-turn `[0, 0, 1]`;
  * a baseline rollout with no restitution path yields `2 / 2` with the flag at
    `0.0`; a HALTed row contributes to neither counter;
* **auxiliary auditor cost**: the encoder is called exactly once for the whole
  auxiliary path, the gradient-bearing Auditor passes receive the rollout's own
  shared features (same values *and* same storage pointer, so no per-turn clone),
  and the official no-grad gate passes are equal in number to the loss passes;
* a rollout output without `shared_features` raises `KeyError` rather than
  falling back to raw images.

Regression, same interpreter: `tests/test_unified_trainer.py` +
`tests/test_unified_logging.py` — **45 passed in 105 s**. Compile/import of both
edited modules verified.

## Limitations

1. The core and config streams landed during this work and their names match
   what is integrated here: `SelfAuditNet.window_mode`,
   `SelfAuditNet.invalidate_candidate_c_records()` and
   `RolloutConfig.predicted_history_exposure` / `.predicted_history_weight`
   (defaults `false` / `0.1`). Both integrations remain written defensively
   (`getattr` on `window_mode`, `callable` check on the hook) so this stream does
   not hard-fail on a core revision that drops them. The Candidate-C counters are
   now exercised against the **real** solver's diagnostics rows as well as
   against synthetic rows; what is still unobserved is a `candidate_c_feasible`
   or `candidate_c_replay_failure` hit — the smoke rollout produced only
   fallbacks — so those two branches are covered by synthetic rows only.
2. Tests inject the two curriculum fields through a `RolloutConfig` subclass
   rather than editing a config file, because the config worker owns the schema
   and the canonical YAMLs. The trainer's fallback default `(False, 0.1)` is
   asserted so the option stays off even on a config that omits the fields.
3. No long training run and no GPU run was performed. Nothing here is evidence of
   effectiveness, Dice, or novelty; the required RTX4070 ACDC/M&Ms gates are
   outside this dispatch and remain NOT RUN from this stream's side.
4. `rollout/auxiliary_annotation_loss` and `rollout/auxiliary_audit_loss` are
   `0.0` when `auxiliary_batches == 0`; read them together with that count rather
   than as a measured loss of zero.
5. Provenance, threshold selection, calibration lineage, `best.pt`/`last.pt`
   commit protocol and the one-config execution path are untouched — this change
   adds keys to the epoch row and the W&B payload and adds no new artifact.
6. `ordinary_annotation_real_evidence_attempts` measures forward exposure only.
   It does not prove the evidence influenced a weight update; an accepted
   conditional update count or a gradient probe would be separate work.
7. When a record-consuming window mode emits no diagnostics rows at all, the
   ordinary split degrades to "every acceptance is ordinary" and
   `ordinary_accepted_path_from_diagnostics` is `0.0` — read the flag before
   quoting that counter under a restitution mode.
