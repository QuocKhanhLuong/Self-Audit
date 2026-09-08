# Parallel invariant QA — independent read-only gate

Reviewer: independent QA worker (task `task_eae7eff0f81e`, dispatch `ctx_33e22c2eaa7e`).
Scope: Gate A architectural invariants plus cross-worker integration risk, while three
implementation workers are actively writing (producer `ctx_1729fcee7642`,
calibration `ctx_b0cee3308413`, bank `ctx_8bb7caf52c29`).

This reviewer edited no production, test, config, split or checkpoint file, made no
commit, and started no training or long job. The only file authored here is this report.

## Verdict

| Gate | Verdict |
|---|---|
| Gate A — reject→HALT / per-sample masks / local feedback state machine | PASS |
| Gate A — audit stop-gradient | PASS |
| Gate A — Phase A/B/C objective, backbone, pretraining recipe | PASS (source recipe unchanged; the approved typed entropy correction is the only behavioural change and it does change model behaviour — see the caveat below) |
| Gate A — diagnostic paths use real model trace, no query GT in decisions | PASS, scoped to the deployable decision path inspected here |
| Integration risk | **REVISE** — one existing test fails against the current tree (I-1, already assigned to W3.2); the remaining items are recorded caveats, not change requests |

Overall for this bounded scope: **REVISE**. Nothing found here is a BLOCK on the
architecture, and no invariant is breached. The REVISE stands on the one failing test
(I-1), whose ownership the coordinator has since settled. The two contract-semantics items —
the entropy-correction recalibration boundary and the deliberate training/evaluation
contract split (I-2) — are **recorded caveats about score semantics and threshold
portability**. Neither asks for a change to the objective, to `audit/targets.py`, or to the
contract split.

## Snapshot this evidence is scoped to

```text
HEAD                = 1305692914c8f50f11135d00d9035b4059f297a5
tracked diff sha256 = c497cc727257ff30... (first 16 hex of `git diff HEAD | shasum -a 256`)
snapshot taken      = 2026-09-08T19:01:36+0700
interpreter         = /Users/alvinluong/miniforge3/bin/python (torch 2.13.0, pytest 9.1.1)
```

**The digest above covers tracked changes only.** `git diff HEAD` does not see untracked
files, so a large part of what this review actually inspected is *outside* the fingerprint:
`src/self_audit/provenance.py`, `src/self_audit/evaluation/calibration_lineage.py`,
`src/self_audit/evaluation/contracts.py`, `src/self_audit/evaluation/transition_bank.py`,
`scripts/export_transition_bank.py`, and the new `tests/test_calibration_lineage.py`,
`test_checkpoint_binding.py`, `test_dataset_split_safety.py`, `test_entropy_contract.py`,
`test_geometry_safety.py`, `test_metric_contract_and_replay.py`, `test_transition_bank.py`.
Those modules can change without moving the digest, so the digest cannot be used on its own
to decide whether this report is still current. Re-verify against the integration gate's
combined diff, not against the digest alone.

The Anaconda interpreter named in `docs.md` (`/Users/alvinluong/opt/anaconda3/bin/python`)
no longer exists on this machine; the miniforge interpreter above is the one that has both
torch and pytest, and is what every command below ran under. `docs.md` should be corrected
by whoever owns it — this reviewer did not edit it.

Three workers are still writing. File mtimes observed during the review
(`calibration_lineage.py` 18:57:13, `transition_bank.py` 18:57:49, versus review start
~18:58) confirm the tree was moving under this snapshot. Every finding below is scoped to
the snapshot fingerprint above and may be stale against a later tree.

## Gate A.1 — reject→HALT, per-sample masks, local feedback state machine

**PASS. The state machine source is byte-identical to HEAD.**

```bash
git diff --stat HEAD -- src/self_audit/audit/gate.py src/self_audit/models/self_audit_net.py \
  src/self_audit/models/auditor.py src/self_audit/models/dynamic_window.py \
  src/self_audit/audit/counterfactual.py src/self_audit/models/encoder.py \
  src/self_audit/training/train_annotation.py src/self_audit/training/train_auditor.py
# empty output — zero changed lines in all eight files
```

The invariant sites are therefore untouched:

- `src/self_audit/models/self_audit_net.py:247` — `if mode == "self_audit" or mode == "oracle_accept": active = accepted`. A rejected row leaves the active set; `always_accept_refinement` deliberately walks to the cap.
- `src/self_audit/models/self_audit_net.py:210` — `state = torch.where(accepted[:, None, None, None], candidate, state)`. Per-sample state update, not a batch-wide branch.
- `src/self_audit/models/self_audit_net.py:215-219` — `previous_audit = torch.where(accepted[...], local_evidence, torch.zeros_like(local_evidence))`. The local feedback fed into the next turn is zeroed for a rejected row, so a rejected row's evidence cannot leak forward.
- `src/self_audit/models/self_audit_net.py:227-233` — every turn publishes `active_mask=attempted` and `state_mask=accepted` per sample.
- `src/self_audit/audit/gate.py:19-21` — `threshold_accept` still detaches `delta_q` before the comparison, so the gate stays non-differentiable.

Runtime probe, forced rejection (toy software probe on a randomly initialised model, not a
scientific result — it proves control flow, not model quality):

```python
m = SelfAuditNet(pretrained_encoder=False).eval()
o = m.infer(torch.randn(4, 3, 64, 64), mode="self_audit", tau_accept=1e9, t_max=3)
```

```text
turns emitted: 1
0 active [True, True, True, True] accepted [False, False, False, False]
halt_turn [0, 0, 0, 0]  accepted_count [0, 0, 0, 0]  num_attempted [1, 1, 1, 1]
final state == initial: True
```

Exactly one transition is emitted, all four rows halt at turn 0, and the returned state is
the initial state. With `tau_accept=1e9` replaced by the baseline `0.0` the same model
accepts every row and runs to the cap, so the branch is genuinely threshold-driven.

The replay side agrees. `src/self_audit/evaluation/threshold.py:421-422` reads
`# A rejected transition halts only that sample` / `active = accepted`, and eligibility is
`eligible = active & valid[:, turn]` (`threshold.py:399`), so a row inactive in the recorded
model trace cannot be revived by replay.

## Gate A.2 — audit stop-gradient

**PASS. `compute_joint_losses` is untouched.**

The changed *old-file* line numbers in `src/self_audit/training/finetune_joint.py` are
`20, 24, 35, 36` (imports) and then nothing below `568`:

```bash
git diff HEAD --no-color -U0 -- src/self_audit/training/finetune_joint.py \
  | grep -E '^@@' | awk -F'[ ,]' '{print $2}' | tr -d '-' | sort -n | head -8
# 20 24 35 36 568 576 588 589
```

`compute_joint_losses` spans `finetune_joint.py:215-313`, entirely inside the untouched
band. Its stop-gradient block is intact:

- `finetune_joint.py:278-279` — `previous_audit = _select_batch_rows(previous.detach(), ...)`, `candidate_audit = _select_batch_rows(candidate.detach(), ...)`.
- `finetune_joint.py:294` — `total = annotation_term + float(lambda_audit) * audit_term`, with the audit branch fed only detached tensors, so audit gradients reach Auditor parameters only.
- `src/self_audit/models/auditor.py` is unmodified, so the Auditor's own defensive internal detach also stands.

All of the Phase C diff lands in `collect_validation_transition_cache`
(`finetune_joint.py:578-836`), which is decorated `@torch.no_grad()` at
`finetune_joint.py:577` and is validation-only.

## Gate A.3 — Phase A/B/C objective, backbone, pretraining recipe

**PASS.**

- `git status --porcelain configs/` is empty. No YAML recipe changed.
- `training/train_annotation.py`, `training/train_auditor.py`, `models/encoder.py`, `models/self_audit_net.py`, `audit/counterfactual.py` all have zero changed lines (command above).
- `src/self_audit/training/_utils.py` changed inside `build_patient_dataset`, `validate_dataset_splits` and `save_checkpoint` (which now stamps a `provenance` block carrying `state_digest` and `model_identity`). **`load_checkpoint` itself is unchanged**: its hunk is `@@ -921,4 +942,185 @@`, four context lines and zero deletions, so the 181 added lines are five *new* helpers appended after it — `_select_checkpoint_candidate`, `_checkpoint_binding_from_payload`, `bind_evaluation_checkpoint`, `bind_existing_evaluation_state`, `verify_bound_state`. `git diff HEAD -- src/self_audit/training/_utils.py | grep -c build_model_from_config` returns `0`, so the model constructor, its optimizer parameter grouping, and the `pretrained_encoder` path are untouched.
- `src/self_audit/audit/targets.py` gained only a docstring, `CONTRACT_NAME`, and `CONTRACT_VERSION` (`+16 -1`). No numeric change to `build_transition_targets` or `multiclass_dice`; the training target remains `audit_target_legacy_one_v1`.
- `scripts/train_self_audit.py`'s large hunk (`@@ -704,67 +1067,3 @@`) is a pure extraction of the inline post-training calibration block into `run_post_training_calibration` (`train_self_audit.py:473`), which preserves `select_threshold` (`:548`), `save_calibration` (`:553`) and the hex round-trip check (`:578-581`), and additionally stamps `build_lineage` (`:530`).

### The typed entropy correction: synthetic-only probe, real effect unmeasured

`src/self_audit/models/annotation_expert.py` replaced range-heuristic dispatch with
explicit `entropy_from_logits` / `entropy_from_probabilities`, and
`annotation_expert.py:276` now calls `entropy_from_logits` directly. Old-versus-new probe
(toy software probe, deterministic seed, not a scientific result):

```text
SYNTHETIC INPUTS ONLY — torch.manual_seed(0), no checkpoint loaded
logits N(0,1)                          maxabsdiff=2.384e-07  old_mean=0.801404  new_mean=0.801404
logits N(0,8) peaked                   maxabsdiff=3.986e-07  old_mean=0.160577  new_mean=0.160577
logits all in [0,1] (heuristic trap)   maxabsdiff=5.955e-01  old_mean=0.875419  new_mean=0.978173
```

**What this probe does and does not establish.** The three inputs are hand-constructed
Gaussian and uniform tensors. They are *not* samples of any real checkpoint's logit
distribution, and this reviewer has no trained checkpoint to draw one from. The probe
therefore establishes only that the two implementations agree on the arithmetic they share
for those particular synthetic inputs. It establishes **nothing** about how often real
`A_t` logits land in the range where old and new diverge, and it is not evidence that the
correction is numerically inert in a real forward pass. An earlier draft of this report
claimed exactly that; the claim was unsupported and has been withdrawn.

The correct statement is narrower: the *source* recipe — configs, optimizer groups, encoder,
objective — is unchanged, and the entropy correction is the single approved behavioural
change. Because entropy conditions the `AnnotationExpert` at `annotation_expert.py:276`, a
changed entropy value changes the expert's update, and therefore can change real
predictions, `DeltaQ`, and everything calibrated downstream of them. **Any checkpoint or
calibration artifact produced before this correction must be treated as measured under the
old entropy definition and recalibrated, not carried across.** `MODEL_ENTROPY_VERSION`
(`annotation_expert.py:16`, `"2.0.0"`) exists to make that boundary explicit and should be
recorded in checkpoint and calibration provenance so a stale pairing fails loudly rather
than silently.

The divergent synthetic case — an all-non-negative logit tensor, where the old heuristic
sum-normalised instead of applying softmax — is the defect the correction removes. It is now
unreachable by accident because `input_type` defaults to `"logits"` and any other value
raises (`annotation_expert.py`, `annotation_entropy`).

No production caller passes probabilities: `grep -rn "annotation_entropy|entropy_from_logits|entropy_from_probabilities" src/ scripts/` hits only `models/__init__.py` re-exports and `annotation_expert.py` itself.

`entropy_version: str = MODEL_ENTROPY_VERSION` at `annotation_expert.py:215` is a plain
class attribute on an `nn.Module` (the `@dataclass` at `annotation_expert.py:190` decorates
`AnnotationExpertOutput`, not `AnnotationExpert`), so it introduces no dataclass-field or
`state_dict` side effect.

Non-blocking observation, pre-existing at HEAD and unchanged: `models/auditor.py:102` and
`:107` compute their fallback entropy as an unnormalised `-(p·log p)` sum, not divided by
`log(C)`, so the Auditor's fallback is on a different scale from the expert's normalised
entropy. In practice `self_audit_net.py:175-179` always supplies both entropies, so the
fallback is dead in the current graph. Flagged for the record only; it is not a regression
introduced by this wave and no edit is requested.

## Gate A.4 — diagnostic paths use a real model trace, and no query GT reaches a decision

**PASS, scoped to the deployable decision path inspected here.**

Scope of this claim: it covers the inference/rollout decision path in
`volume_inference.infer_patient_volume`, `SelfAuditNet.infer`, and
`transition_bank.generate_on_policy_proposals` — the places where an accept/reject decision
is made. It does **not** extend to data preprocessing, split construction, or dataset
provenance, which are a separate concern owned by other gates and were not audited here. No
powered GT-leakage positive control was run by this reviewer.

- `src/self_audit/evaluation/volume_inference.py:517-518` still refuses `oracle_accept` in the deployable path. Probe: `infer_patient_volume(m, x, mode="oracle_accept")` raises `ValueError: oracle_accept requires an explicit analysis wrapper; GT is not an inference input`.
- `src/self_audit/models/self_audit_net.py:116-117` keeps `oracle_accept` analysis-only and requires an explicit `oracle_target`.
- `src/self_audit/evaluation/transition_bank.py:409-413` refuses a dataset batch **mapping** outright, on the stated grounds that the mapping carries `mask`; the caller must pass `batch["image"]` and identity metadata separately. This is **API isolation** — it removes one route by which GT could arrive. It is not a proof that the whole GT firewall holds, and it is weaker evidence than a powered positive-control test, which this reviewer did not run.
- `transition_bank.py:429-437` obtains rows from a real `model.infer(...)` rollout under `torch.no_grad()`, with `rollout_policy` constrained to `self_audit` / `always_accept_refinement` (`transition_bank.py:96-97`) — `oracle_accept` is not reachable.
- `transition_bank.py:449` reads `audit["active_mask"]` from that trace rather than reconstructing it, and `transition_bank.py:456-459` raises `BankValidationError` if the rollout ever emits a stage for a sample after it halted. `transition_bank.py:1200-1215` re-checks the same property at validation time on the serialised rows.
- On-policy rows carry `gt_used_in_generation=False` (`transition_bank.py:481`); `generate_synthetic_proposals` is explicitly documented as training-side and never GT-free deployment evidence (`transition_bank.py:513-515`), and `transition_bank.py:1067` fails a row whose `source` and `gt_used_in_generation` disagree.
- `evaluation/audit_decomposition.py` changed only its import style plus one new reporting key `"audit/metric_contract": AUDIT_TARGET_LEGACY_ONE_V1` (`audit_decomposition.py:414`) — no decision logic.

## Snapshot test and build evidence

```bash
python -m pytest -q tests/test_entropy_contract.py tests/test_self_audit_hardening.py \
  tests/test_self_audit_regressions.py tests/test_self_audit_masks.py \
  tests/test_self_audit_core.py tests/test_self_audit_audit.py tests/test_self_audit_volume.py
# 67 passed, 1 warning in 2.30s

python -m py_compile $(find src scripts tests -name "*.py")
# COMPILE_OK

git diff --check HEAD
# rc=0, no whitespace errors

grep -rn -e '^<<<<<<< ' -e '^>>>>>>> ' src scripts tests
# no output — no conflict markers, no evidence of one worker clobbering another

python -m pytest -q
# 1 failed, 337 passed, 1 warning in 55.70s
```

Remote CI is a waived blocker for this wave. **No CI PASS is claimed here**; the evidence
above is local only.

## Integration findings

### I-1 (RESOLVED BY COORDINATOR — assignment already made) — fail-closed lineage breaks a previously unowned test

**Status update:** the coordinator has assigned `tests/test_metric_contract_and_replay.py`
migration to the calibration worker W3.2 (`msg_866246f55fff`). The ownership question this
finding raised is settled; what follows is retained as the evidence record only, and no
further coordinator decision is requested.

`tests/test_metric_contract_and_replay.py::test_cli_calibrate_threshold_roundtrip` fails:

```text
CLI failed with stderr: Error: Cached transitions carry no 'lineage' block, so the
calibrated threshold could not be tied to the weights, preprocessing recipe, metric
contract or cohort it was measured on. Re-cache with
scripts/cache_validation_transitions.py, which records one.
tests/test_metric_contract_and_replay.py:600: AssertionError
```

Reproduce: `python -m pytest -q tests/test_metric_contract_and_replay.py`
(`1 failed, 31 passed in 2.47s`).

The refusal itself is **correct and is the calibration gate's stated requirement**
(`scripts/calibrate_threshold.py:97-103`, `_artifact_lineage`). The problem is ownership:
the failing test builds a synthetic cache payload at
`tests/test_metric_contract_and_replay.py:566-579` with no `lineage` key, and that test file
appears in **no** ownership row of `parallel_execution_update.md`. mtimes make the sequence
explicit — `tests/test_metric_contract_and_replay.py` 15:53:05 versus
`scripts/calibrate_threshold.py` 18:51:21 — so a later, correct tightening broke an earlier,
unowned test.

Resolution in flight under W3.2: the migrated test should either supply a minimal valid
`lineage` block or assert the fail-closed refusal as the expected behaviour. This finding is
newly observed in this snapshot; it is not a restatement of the initial calibration or bank
malformed-schema probes already on record at root.

### I-2 (NOT A DEFECT — intended design; recorded as a score-semantics caveat) — training contract and evaluation contract are deliberately distinct

**Correction to an earlier draft of this report.** The split between the training target
contract and the evaluation contract is an **explicit user requirement**, adopted precisely
so that the Phase B/C objective is preserved rather than retuned to match a new metric. It
is not a newly discovered invariant conflict, and this report does **not** request any
change to the objective, to `audit/targets.py`, or to the contract split. The material
below is retained only to state the caveat that follows from the design.

`collect_validation_transition_cache` now records `q_previous`, `q_candidate` and
`actual_delta_dice` under `foreground_dice_exclude_v1`
(`finetune_joint.py:606-608, 736-745`), while the auditor's `delta_q` is trained against
`build_transition_targets` / `multiclass_dice` under `audit_target_legacy_one_v1`
(`audit/targets.py`, contract name declared at `targets.py:24`). The legacy values are
preserved as `legacy_actual_delta_dice` (`finetune_joint.py:725-726, 819`), so the mismatch
is fully auditable rather than hidden — that part is good practice.

Threshold selection consumes the evaluation contract: `_evaluate_threshold_core` classifies
harmful/beneficial from `actual` (`threshold.py:400-402`) and gates on
`quality[:, turn] > tau_accept` (`threshold.py:402`). Background-class inclusion and the
both-empty convention differ between the two contracts, so the two scales are genuinely
different quantities rather than a relabelling.

**The caveat to record — not a change request.** Because `tau_accept` is selected against
`foreground_dice_exclude_v1` while `delta_q` is produced under `audit_target_legacy_one_v1`,
a selected `tau_accept` carries the *evaluation* contract's score semantics and is only
meaningful when quoted together with the contract name it was selected under. It follows
that a threshold is not portable across contracts, and that any change to either contract —
or to the entropy definition discussed above — invalidates a previously selected threshold
and requires recalibration. The cache already preserves both scales
(`legacy_actual_delta_dice`, `finetune_joint.py:725-726, 819`), which is what makes this
caveat checkable rather than an assumption. No number is claimed here: this reviewer has no
trained checkpoint and measured nothing about the size of the gap.

### I-3 (accepted, no action) — two cache producers, one lineage stamp

`collect_validation_transition_cache` deliberately emits no `lineage` key
(`grep -n lineage src/self_audit/training/finetune_joint.py` returns nothing). Both real
producers stamp it afterwards: `scripts/cache_validation_transitions.py:92` and
`scripts/train_self_audit.py:530`. A third-party caller that used the library function and
saved the result directly would produce a cache the calibration CLI refuses. That refusal is
fail-closed and therefore safe; recorded so the seam is a known design choice rather than an
accident. This is the same mechanism that surfaces as I-1.

### I-4 (low, informational) — two cache provenance fields are weaker than they read

Both in `collect_validation_transition_cache`, both new in this diff:

1. `"excluded_empty_slice_count": 0` (`finetune_joint.py:790`) is now a hardcoded constant, because blank slices are intentionally no longer dropped. A consumer reading it as a live count sees a reassuring `0` that was never measured. An explicit `null`, or a renamed key, would carry the intent better.
2. `case_ids` (`finetune_joint.py:832-834`) is populated only when the batch happens to expose `case_id` / `subject_id` / `volume_id`, and is silently omitted otherwise. It is keyed on case, not patient. This is **not** a leakage risk today: cohort disjointness is enforced from `provenance.cohort_identity`, which records `patient_ids` separately (`src/self_audit/provenance.py:460-467`) and is checked patient-wise in `calibration_lineage._verify_cohort` (`calibration_lineage.py:435-436, 464`). Flagged only so nobody later mistakes the cache's `case_ids` for a disjointness basis.

### I-5 (verified sound, no action) — blank-slice retention does reach the replay metric

The Phase C cache stopped dropping NaN-Dice blank slices, and the old docstring warned that
doing so would make `final_macro_dice` NaN at every grid point. That warning no longer
applies: `threshold.py:426` uses `np.nanmean`, and `select_threshold` raises
`NoFeasibleThresholdError` when every row is NaN (`threshold.py:1159-1163`) rather than
returning an arbitrary row.

The stated purpose — detecting blank-to-hallucination regressions — also holds, but only
through the `q_candidate` path. When A0 and GT are both blank, `q_previous` is NaN and the
delta is undefined; a hallucinating candidate produces a finite `q_candidate` of 0.0. At
`threshold.py:414-415` the accepted state score is taken directly
(`final = np.where(accepted, cand_scores[:, turn], final)`), so that 0.0 does enter
`final_macro_dice`. The legacy fallback at `threshold.py:417`
(`np.nan_to_num(delta_turn, nan=0.0)`) would *not* catch it, but it only runs for a cache
with no `q_candidate`, which the current producer always writes. Undefined transitions are
separately counted as `undefined_accepted_total` / `undefined_rejected_total`
(`threshold.py:409-412`), so they are reported rather than absorbed. Verified sound at this
snapshot; recorded because the guarantee depends on `q_candidate` continuing to be written.

## Not covered by this gate

- Remote CI (waived blocker; no CI result is claimed).
- The producer worker's exact best/state checkpoint binding and the bank's schema and positive-control internals beyond the invariant and firewall checks above — those belong to the producer, calibration and bank gates.
- The initial calibration and bank malformed-schema probes already recorded at root, which were deliberately not repeated.
- Any scientific claim about model quality. Every runtime probe in this report ran on a randomly initialised `SelfAuditNet` or on hand-constructed synthetic tensors, and establishes control flow and implementation arithmetic only — never the distribution or parity of real-checkpoint values.
- Data preprocessing, split construction, and dataset provenance. The Gate A.4 GT claim is scoped to the deployable decision path only.
- Any powered positive-control test for GT leakage. None was run here.

## Standing caveat

Three workers were writing while this ran. Everything above is scoped to the snapshot at the
top of this report — and, as noted there, the tracked-diff digest does not cover the
untracked modules this review inspected, so the digest alone cannot certify that this report
is current. Re-verify against the integration gate's combined diff before any whole-phase
PASS is recorded.
