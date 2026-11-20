# W2S-Match on the 4070: wiring and deviations (not yet trained)

Branch claude/server4070-20261119. Runner: scripts/run_pipeline_w2s_match.sh (resumable markers under runs/w2s-match-20261119/markers).

## What was added
- scripts/w2s_prepare_val20.py: VAL20 = VAL18 + patient055 + patient076 (seed-42 split keeps them in VAL; the PG preparer excludes them). Hash-pinned, raw files checked against the verified inventory. Receipt: VAL20_RECEIPT.json.
- self_audit/candidate_pg/w2s_data.py: VAL20 view (wrapper over the receipt-validated VAL18 loader). Active only if data.w2s_val20 is configured; then selection, teacher selection and S3 reconstruction all use 20 patients.
- engine.py (small edits): VAL20 hook, cbkd_sa_gated treated as an SA variant, S0 needs no S2 receipt, codebook only required if an arm uses it, W2S diagnostics in diag.jsonl, watchdogs wired.
- config.py: optional data.w2s_val20 {path, sha256}.
- scripts/w2s_make_configs.py, w2s_stage_link.py (S1 reuse verification), w2s_saturation.py, w2s_eval_teacher.py, w2s_cb_check.py, w2s_finalize.py; tests/test_w2s_engine_wiring.py.
- Arm mapping onto the engine: S0 -> A2 (target_mode self), S2arb -> A2 (w2s_arbiter, cbkd off), S2 -> A4 (w2s_arbiter + cbkd_sa_gated); one run directory per arm.

## Deviations from the W2S prompt (to report)
1. Patients 055 and 076 are in VAL (VAL20), not TRAIN, in splits/acdc_patient_split_seed42.json. TEST patient118 duplicates patient055, so TEST49 and VAL20 overlap in one image pair.
2. VAL18 and TEST49 are derived from per-patient results (patient macro), not separate evaluation passes.
3. W2S arm configs do not pin cinema_init.hf_revision (S1 was trained with b1251ee5...); irrelevant for arms, which load the S1 artifact.
4. TEST50 is excluded from eval_splits; the finalizers evaluate it once on the VAL20-selected checkpoint.
5. w2s_checks.Watchdogs: the veto-saturation window starts at the first low reading after alpha onset.
6. S1 was trained on the 4070 (not an A100) under commit 1d8242a; arms run on merged code.
