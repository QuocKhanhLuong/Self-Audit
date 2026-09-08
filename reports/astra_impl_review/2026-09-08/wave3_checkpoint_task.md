# W3.1 — producer checkpoint binding (TASK SPEC, not implemented)

## Worker authority update

User explicitly requested "gọi claude code nốt đi" after AGY quota exhaustion.
Claude Code via Orca now replaces AGY as implementer for remaining Proposal 1 tasks;
Astra remains coordinator/independent reviewer. References to AGY below describe the
original role, not a restriction against the newly authorized Claude worker. Start here,
read implementation_review.md current handoff and wave3_checkpoint_review_notes.md.
Existing W1/W2 edits belong to prior work: preserve them, do not restart or overwrite.

Execute only after Astra dispatch and W2 dependencies. Read docs.md, execution_plan.md
W3.1/W3.2 and current code. AGY implements; Astra independently reviews. No retraining,
commits/push, old-checkpoint edits, architecture/objective changes or weakened pretrained.

## Evidence / objective

Unified runner collects validation cache from live last model, then chooses best.pt path
for artifact identity. Need exact state binding before collector, calibration and associated
diagnostics. Hashing a different checkpoint file beside a live model is insufficient.

## Ownership and change

scripts/train_self_audit.py, training/_utils.py checkpoint helpers, new narrow provenance
helper if needed, scripts/cache_validation_transitions.py, focused tests.

1. Select checkpoint -> load exact state strictly -> verify actual model state digest ->
   collect cache -> calibrate -> diagnostics. Best selection criterion/training recipe
   remains unchanged. Declare explicit last fallback if best absent; missing both fails.
2. Capture file SHA256 and deterministic state digest over parameter/buffer keys, shapes,
   dtypes and bytes. Verify digest at relevant consumer boundaries; don't merely attach
   arbitrary caller metadata. Do not restore optimizer/scheduler/RNG for evaluation loading.
3. Persist checkpoint-producing git SHA from checkpoint provenance, separately from current
   evaluation/calibration code SHA. New save_checkpoint payloads record producing metadata;
   legacy unknown producer stays unknown, never replaced with current HEAD as if historical.
   Record dirty/source semantics identity where needed; report-only commits must not pretend
   model code changed. No scanning provider/account/environment secrets for provenance.
4. Resolve actual model config/backend and entropy semantic version after construction/load,
   not from checkpoint filename or directory. Preserve ImageNet initialization for training;
   no new teacher/backbone. Strict state/config mismatch fails.
5. Cache lineage foundation: checkpoint SHA, state digest, resolved model configuration,
   model semantics, producer SHA, metric contract/schema, preprocessing descriptor/signature,
   effective split membership signature and exact calibration cohort identity, t_max and
   fixed class mapping. Use W2 resolver/geometry descriptors rather than new parallel logic.
   Membership hashes are not content hashes: label their coverage honestly.
6. Keep legacy general checkpoint loading backward compatible for inspection/resume as
   appropriate; strict calibration/export validity is a distinct gate. W3.2 will finalize
   artifact consumers. Do not claim W3 complete from producer binding alone.

## Tests / PASS

- Tiny best != last fixture: exact best parameter/buffer/state digest observed at collector,
  calibration callback and both associated diagnostics. Last fallback declared/tested.
- Exercise the actual unified runner post-training branch (extract a small called helper
  if needed). Testing an unused helper is not enough. No long training required; stub the
  training work or use synthetic callbacks/state dicts.
- Missing checkpoint, state/config/backend mismatch fail; modifying live state after binding
  is detected; no unintentional optimizer/RNG restore at evaluation-only load.
- Newly saved provenance weights-only roundtrip; legacy producer never fabricated.
- Focused -> full tests/ -> source compile, exact outputs and concise wave3_checkpoint_worker.md.

No real bank, real-checkpoint validation or Dice improvement claimed from these fixtures.
