Q-3D integration of the corrected spec — SOFTWARE_ONLY

Base commit: `016706f08202f39262426bd542935130a8a9c7d8`. This delivery fixes patient-weight cancellation in the actual codeKD and feature-KD engine paths, phase alignment, and sharing of stage artifacts. It does not establish patient efficacy, clinical validity, or a real-data gate pass.

| Phase | Delivered behavior | Evidence |
| --- | --- | --- |
| B1 | Public KD losses return per-cell maps; the engine applies the common class reducer and cohort omega separately. L is always 1. V1-u uses the same U eligibility gate with unit omega. Full U-cycle metrics record mean/max/zero fraction/ESS. | Pure loss and engine tests include one real synthetic J optimizer update, cohort rescaling, patient-relative changes, C1, and zero-support gradients. |
| B2/B3 | Main captures `forward_with_features[0][3]` without a module hook. Grid step/offset come from actual modules. Feature sampling and target fractions/confidence share native centers; target pooling excludes padding. | Actual pinned MedNeXt source and tiny/student ramp probes; native interior tolerance <=0.5 px and linear alignment <=1e-5. Decoder child hook fires once. |
| B4 | New plans share one T per identity and main Q/P, with separate source/O1 Q/P. J imports sealed stage artifacts with checkpoint/content/source/cohort/phase SHA checks. CLI accepts expected manifest SHA. Completed-stage publication can recover interrupted export without overwriting published artifacts. | Shared V1/V1-u/C1 stage-import tests, tamper/recovery tests and a mocked DAG runner. No campaign process was executed. |
| H1-H3/M | Cold-load mirrors parent dtype, deterministic flags and torch thread count. P0 inventories the exact source and computes L-only coverage when data are available. g and interface norm ratio are logged at evaluation. Baseline tests/docs are restored byte-for-byte. | Cold-load tested at default threads and 1; 33 baseline tests and 16 archived docs match a03aadb; baseline source/config diff is empty. |

The new immutable plan is `../q3d-shared-plan-20261112/PLAN.json`: datasets ACDC/M&Ms, protocols A/B, K=2/5/10, seed42; 180 experiment settings, **T=12, Q=60, P=60, J=168**, all NOT_RUN. Existing plans are preserved. The shared T region-head reference is explicitly not an independently trained MedNeXt K-label baseline; the independent baseline launcher is preserved and its result remains NOT_RUN in the shared plan, especially under Protocol B.

Validation commands use `/Users/alvinluong/miniforge3/bin/python`. `OMP_NUM_THREADS` and `MKL_NUM_THREADS` are unset for the final full suite and smoke. The optional source probe is enabled with `Q3D_OFFICIAL_MEDNEXT_SOURCE=/tmp/q3d-mednext-phase-pin`; that clean checkout is pinned to `0b78ed869fbd1cc2fd38754d2f8519f1b72d43ba`. No weights or patient data were downloaded.

Final acceptance: **448 passed, 6 warnings in 125.88s (0:02:05)**; `check_repo.py` PASS; **15/15** synthetic smoke recipes completed. Final command output and machine receipts are stored beside this report. `pytest.txt` is verbatim stdout/stderr, including upstream checkpoint warnings. `smoke.json` records measured tokenizer gates and the synthetic-only override; it must not be read as a real gate pass. Physiology gates for the shared V1/V1-u comparison pass without an override, and their omega maps differ while teacher/C0/cache hashes match.

NOT_RUN / OPEN:

- Real-data acceptance P0-P5, patient training/evaluation, GPU, clinical metrics and campaign execution.
- Real L-GT MYO coverage (no declared TRAIN manifests); the 2 percent OPEN threshold is unchanged.
- Real decoder_stride8 and stride4_decoder_block alignment: the pinned upstream upsample branch adds low-side padding before a skip sum and its native center differs from the encoder skip. No single phase satisfies the current alignment contract. These ablations fail closed pending a separate reviewed contract. Tiny proxies passing smoke do not close this OPEN item.
- Real tokenizer gate, official pretrained-weight quality/overlap and independent MedNeXt baseline efficacy.

Claude Code's initial read-only review found artifact/runner integration defects; the dependency shape, cache hash key, missing DAG execution, C1 artifact handling, resume publication and live cache identity checks were corrected and covered by tests. Its initial findings are preserved in `claude-pre-review.md` as historical review evidence, not a statement of the final code state. A fresh independent review is requested after pushing the accepted code SHA.

Git whitespace validation excludes only the byte-identical restored `tests/test_candidate_a3d_semi_plan.py`, whose historical EOF blank line is preserved. All new edits pass the whitespace check.
