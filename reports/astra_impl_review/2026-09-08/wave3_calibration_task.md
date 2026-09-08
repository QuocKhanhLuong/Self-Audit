# W3.2 — strict calibration consumer (TASK SPEC, not implemented)

Only execute after Astra dispatch and W3.1 producer review. Read docs.md, current producer
helpers/contracts and plan W3.2. No training, checkpoint/split edits, commit/push, architecture
or objective changes. AGY implements; Astra reviews actual diff and tests.

## Objective / ownership

Strict artifact lineage through threshold.py, calibrate_threshold.py, audit_checkpoint.py,
train_self_audit.py and necessary focused tests. Reuse W3.1 provenance, W1 contracts and
W2 effective split/geometry semantics; do not duplicate independent signature logic.

## Requirements

- Versioned artifact schema. Required verified lineage includes exact checkpoint SHA256,
  actual state digest, resolved model config/backend/entropy semantics, checkpoint-producing
  git SHA, metric contract/schema, preprocessing signature, effective split/protocol and
  calibration cohort identity, t_max and class mapping. Current calibration code provenance
  separate from checkpoint-producing commit. Unknown historical provenance never fabricated.
- Every calibration consumer used for evaluation validates expected runtime lineage before
  using tau or emitting calibrated results. Missing fields or any mismatch hard-fail.
  CLI tau override does NOT skip artifact lineage checks if artifact is supplied/consulted.
- Audit checkpoint CLI already builds model and dataset before resolve_tau_accept: derive
  expected lineage from those actual objects/checkpoint, not from artifact itself.
- Unified runner roundtrip uses W3.1 loaded state. Its earlier Phase-C tau artifact branch
  must also reject incompatible lineage before training begins; preserve manual/default tau
  recipe. Do not silently apply an old calibration to different starting weights.
- Historical schema may be inspected under explicit unverified/legacy API if needed, but
  cannot enter deployable calibrated evaluation by a compatibility fallback or warning.
- Calibration cohort and independent evaluation cohort are different roles: protocol declares
  exact permitted membership for each role. Do not require eval cohort equal calibration,
  and do not accept arbitrary replacement cohort. Positive disjoint eval fixture required.
- Avoid matching against artifact's own values to manufacture success. Expected contract,
  preprocessing, weights, t_max/class map and evaluation role must come from runtime/protocol.
- Strict JSON output (NaN handled explicitly), no serialization-only "verification" claim.

## Mandatory tests / PASS

Matched producer -> artifact -> actual consumer positive roundtrip; best != last full
synthetic runner calibration branch; mutate each lineage field separately -> hard-fail;
missing schema/fields -> fail; weight changes under same filename -> fail;
config/backend/entropy/metric/preprocessing/split/protocol/t_max/class mismatch -> fail;
CLI override cannot bypass; allowed disjoint evaluation cohort accepted, arbitrary cohort
rejected. Legacy inspect remains explicitly unverified, never a valid calibrated result.

Focused tests -> full tests/ -> compile. Exact outputs in concise wave3_calibration_worker.md.
No real checkpoint or dataset metrics; no novelty or Dice gain claims.
