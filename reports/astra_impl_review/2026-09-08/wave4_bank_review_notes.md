# W4 draft independent review — REVISE, 2026-09-08

Root read transition_bank.py and export_transition_bank.py. Actual tiny SelfAuditNet
generation and evaluator fixture from test_transition_bank.py produced a baseline bank.
Each independent mutation below was followed by recomputing bank_content_signature, then
calling public validate_bank. This tests semantic validity, not only transport integrity.

| Mutation | Draft actual | Required |
|---|---|---|
| Previous TP for class 1 becomes 999999; Q unchanged | ACCEPTED | Reject stats/Q inconsistency |
| checkpoint_bound=True, checkpoint block removed | ACCEPTED | Reject missing bound checkpoint provenance |
| Row metric_space becomes volume_native under slice header | ACCEPTED | Reject metric-space mixing |
| Row schema_version=1.5 | ACCEPTED | Reject fractional schema version |

Sent review requirements in `msg_d87826e42700`: strict bound identity/cohort/protocol,
per-row semantic consistency and quality recomputation from stats, strict numeric/schema
types, finite values, full state hash syntax, evidence simplex, no fabricated CLI identities,
actual tiny CLI integration and post-generation state verification. Content hashing does
not establish semantic consistency; tests must not rely exclusively on stale-hash rejection.

The source generation correctly reuses model.infer traces. Keep this path and filter
inactive rows; do not add a duplicate recurrent rollout. This is a synthetic diagnostic
probe, not real-bank validation, training, Dice improvement or novelty evidence.
# Final validator closure — 12:18 UTC

Root reran 54 tests PASS in 9.36s, then independently generated six actual tiny-model
proposal rows using test helpers (`make_proposals`, tau=-1), evaluated structured reference
masks, and validated the unmodified bank successfully. After deepcopy and recomputing the
integrity signature, each individual row-0 mutation remained accepted:
empty_policy=legacy_one, metric_contract_version=1.5, halted_after=True.
These prove semantic validation gaps, not hash failures. Assigned final bounded closure
task `task_4017b559c3ee`; AGY recovery dispatch `ctx_0ab9fbe2228d` owns only bank module,
tests and new final worker report. Other exact schema/identity checks are in the task spec.
No real trained checkpoint or bank was used; these remain software fixtures.
