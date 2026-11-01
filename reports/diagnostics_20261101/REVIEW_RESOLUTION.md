# Integrated review resolution

`INDEPENDENT_REVIEW.md` is an immutable review of its listed intermediate source
hashes. Its padding finding was fixed before final delivery: native validity is
false-padded to the compact encoder canvas before area reduction. The 33x35 →
40x40 fixture now admits 16 complete cells, not 25.

The coordinator also found and resolved these integration issues:

- Historical module aliases and the old teacher without a `teacher_memory`
  attribute are accepted through a checked structural interface; physical model
  loaders retain strict hash/source/state checks. D0 remains independently
  compared with the native student forward.
- D1/D2 explicitly compute output-cotangent VJPs and compare them with direct
  autograd. Independent tests compare diagnostic KD with the actual canonical
  class-displacement builder at temperature 1 and 2.3, and compare the teacher
  fit tap with a hook on its actual native forward.
- D3 permutes valid scalar feature coordinates, including channels, and shares
  D2's norm/fallback. A singleton coordinate is logged as identity, not an
  additional rejection. Both-active cosine uses the same valid subspace.
- The shipped smoke no longer substitutes fake models or text weights. It
  serializes real synthetic NativeStudent and CompactStudent models, strictly
  reloads them, and executes both CLIs over distinct three-slice inputs.
- GT guards now watch the actual fixture mask paths. The integrated test denies
  all TRAIN/development mask opens during preflight, prediction and fit-P, and
  checks both freezes plus the exposure ledger before scoring opens GT.
- B scoring accepts the existing authorized A exposure ledger while late
  fitting/locking/prediction remains forbidden. Changed frozen logits fail
  before GT access. Historical A configs without the optional fixture flag
  remain compatible with a real B config.
- Batch KD, common denominators and forward/backward counts are recorded once
  per batch; per-sample norms/statuses link to those records. Real patient row
  arrays serialize correctly. A-only fixture reports cannot claim patient
  results, and nonfinite CLI failures have an explicit invalid status.
- Streaming scorer arrays are copied only for the current chunk before tensor
  conversion, avoiding a writable-tensor view of a read-only memmap. Frame Dice
  validation rejects out-of-range values before aggregation.

The existing canonical trainer, model architecture, KD recipe and historical
results were not changed. Final test and subprocess-smoke evidence, including
source/config/fixture weight/manifest hashes, is in `VERIFICATION.json` and
`VERIFICATION.md`. This closes software findings only; patient/GPU evidence and
the pre-VAL scientific choices remain separate prerequisites.

Orchestration used Luna max after AGY reported quota exhaustion. Context,
kernel/fit and metrics workers delivered bounded implementations/tests. The
runtime worker stopped after an Orca `consumer_fenced` error; the coordinator
completed its integration, actual CLI smoke, regression tests and final checks.
No server jobs or patient data were accessed during this work.
