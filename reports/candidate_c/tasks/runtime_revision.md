# Runtime report finalization — code PASS pending integration
Root reviewed the actual provenance diff and focused tests. Keep production code unchanged. You are not alone; preserve others edits. No commit/push.
Update only reports/candidate_c/runtime_review.md to remove stale current claims while keeping historical investigation evidence explicitly historical:
- Coordinator accepts fail-closed old last.pt resume and old source-bound calibration behavior; no silent migration. Old runs require original checkout or restart, new calibration only ACDC for external flow.
- Mode signature hole fixed, _bind_execution_mode temporary helper now removed by config stream. Do not state it remains.
- No dedicated checked-in Candidate C config required; copy canonical YAML to a bounded-run config and set model.window_mode; report actual config signature. Both canonical configs default current baseline.
- Coordinator attempted BatchMode read-only SSH nvidia-smi to alias 4070; exit 255 Permission denied (publickey,password). No GPU/data/checkpoint verified. Do not include host IP or credentials.
- Replay records are local infer structures, not persistent AnnotationExpert record attributes. Latest core preserves original factual batch even when rows halt; memory estimate must name assumptions and distinguish one batched record references from per-sample duplication.
- Keep tests on moving tree provisional; root will run quiescent gates. Mark no more tests needed for this report-only revision.
- Final verdict code PASS / CUDA blocked / controlled experiment not yet authorized, with remaining numerical and resume limits explicit.
Settle with report path and changed section list; no broad tests.
