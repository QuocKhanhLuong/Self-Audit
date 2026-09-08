# W4 — Frozen transition bank readiness (TASK SPEC, not implemented)

Execute only after explicit Astra dispatch and W1–W3 review. AGY owns a narrow diagnostic
export module/CLI, necessary existing diagnostic integration, and regression tests.
Read docs.md and current provenance/contracts first. Preserve other edits. No model,
training objective, checkpoint, split, historical report edits, training, commit/push.

## Objective

Make a tested export path ready, not create a real bank in this environment. Reuse exact
metric/provenance helpers, with proposal generation separated from evaluator GT attachment.
Do not label a schema-only dataclass or hand-written rows as an integrated diagnostic path.

## Minimum schema and behavior

- Explicit schema version, patient ID, case ID, slice index, stage, deterministic previous
  and candidate state identifiers, delta_q (predicted signed score, not probability),
  accepted/rejected, Q_previous/Q_candidate and explicit undefined-score validity.
- Transition source enum: on_policy, synthetic, always_accept_prefix. Record rollout policy
  separately from hypothetical gate decision; always-accept is not the deployable trajectory.
- Local evidence summary with declared channel semantics, metric contract/version,
  checkpoint file hash and actual state digest, split/protocol/cohort identity.
- Evaluator computes actual Q only after proposal generation. Deployable generation API
  does not accept query GT. Synthetic generation may use training GT but must be explicitly
  marked; never claim those edits are GT-free deployment.
- Reuse source/model validation from W3. Stable state IDs refer to actual state tensor
  content and representation, not merely stage numbers. Validate state continuity and
  required identity/contract fields at serialization/loading boundaries.
- On-policy rejected samples HALT: emit no fabricated later transitions for them.
  Empty foreground/undefined scores must retain row/coverage information, strict JSON null
  handling instead of nonstandard NaN tokens or dropped examples.
- Persist generation/evaluation provenance separately. Unknown producer remains unknown.
  Do not assert content-hash coverage if only membership is hashed.

## Mandatory tests / PASS

Tiny deterministic actual generation -> evaluator -> JSON roundtrip; per-sample reject/HALT;
state ID reproducibility and change sensitivity; contract/provenance/schema tamper rejection;
GT-firewall with image/weights fixed and reference GT permuted: proposals, state IDs and
decisions identical while evaluator quality changes. Positive control deliberately using GT
must change, proving probe sensitivity. Cover source tags and undefined Q preservation.
No real checkpoint/dataset integration claim from these tests. Run focused then full tests/
and source compile; exact output in new wave4_bank_worker.md.

Remote GitHub Actions is waived as a blocking gate by the user, not claimed PASS. CI changes
are outside this bounded task unless separately dispatched. Proposal 2/3 remain unopened.
