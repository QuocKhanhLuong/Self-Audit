# Final bounded validation plan and evidence rules

Coordinator executes these only after all production workers settle. Worker totals from moving trees are provisional. No full scientific training or Dice comparison is authorized.

Environment: `/private/tmp/self-audit-torch241/bin/python`, torch2.4.1 CPU, with `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`. All shell calls prefixed `rtk`. Capture exact commands/results and source signature before/after in final_integration_review.md.

1. Compile changed/new Python modules, import core/config/trainer/evaluation entrypoints. No import downloads.
2. Run new Candidate C focused suites (core, config, training, diagnostics, runtime, M&Ms and native pipeline if supplied). Require a successful coordinate solve, not only fallback tests.
3. Selected existing regression suites: unified trainer/logging/config; Self-Audit core/audit/regressions; runtime checkpoint and resume; checkpoint binding and calibration lineage; M&Ms data/external evaluator and bounded runtime pipeline.
4. Independently compare current-mode outputs to a temporary git-archive of base HEAD under identical seed/weights/CPU/threads; self_audit, always_accept_refinement, forward_annotation, mixed B>1.
5. Real RTX4070 ACDC native, native M&Ms and frozen ACDC->M&Ms bounded smokes require working remote authentication, datasets and a source-compatible frozen checkpoint/calibration. Use CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1. None of the CPU fixture tests substitutes for these gates.

Root read-only SSH nvidia-smi to alias4070 failed exit255 Permission denied(publickey,password); local preprocessed ACDC/M&Ms folders absent and CUDA unavailable. Required real GPU gates remain BLOCKED/NOT RUN until access is provided. No GREEN/commit/main push under the current gate contract while required gates remain unmet.

Large real-data validation, full130epochs, allablation arms, clinical effectiveness and novelty claims are intentionally outside this phase.

## Final coordinator evidence (2026-09-11)

- Gate 1: py_compile of 21 changed/new Python files into a temporary cache, then 7 critical imports. PASS. `final_test_receipt.json` records exact files.
- Selected Gate 2/6: one combined run of 20 focused/new/existing files with `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /private/tmp/self-audit-torch241/bin/python -m pytest -q -ra ...`; exact argv in `final_test_receipt.json`, output in `final_pytest.log`.
- Archived original HEAD vs current baseline, identical seed, CPU B2 32x32: 325 tensors bitwise equal. `baseline_parity.json`.
- `.../python reports/candidate_c/run_synthetic_flow_probe.py`: Candidate C + predicted history through synthetic ACDC 3 intervals (6 steps), best/last, calibration, bank export and frozen external M&Ms; independent native M&Ms 3 intervals (6 steps), max_val_batches1. Native CLI correctly reports incomplete/partial validation and refuses best.pt. Initial harness incorrectly asserted completion; corrected assertions were applied to the existing artifacts without another training run. `synthetic_flow_receipt.json` states this precisely.
- `rtk proxy git diff --check`: PASS.
- Required real GPU gates: NOT RUN. Read-only SSH to alias4070 failed authentication, local torch CUDA false, no local real ACDC/M&Ms data or trained checkpoint. Requested CUDA_DEVICE_ORDER/ CUDA_VISIBLE_DEVICES setting is unchanged.

## Final post-cleanup verification

Root removed an unconsumed runtime record tensor after all workers settled. `post_cleanup_test_receipt.json` records the exact six-file pytest command:184 passed, no skips,59.75s; all changed/new Python compiled and4 imports passed. `run_synthetic_flow_probe.py` then ran successfully end to end on the final source (the earlier bounded-validation harness expectation was corrected). Final receipts supersede older fixture artifacts; pre-cleanup735-test evidence remains explicitly labelled. Replay storage probe confirms1MiB/row saved. No additional production change followed these gates.
