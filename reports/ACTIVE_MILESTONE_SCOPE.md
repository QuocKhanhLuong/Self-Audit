# Active milestone scope (2026-10-01)

## In scope

| Component | Environment | Official entrypoints and enforcement |
|---|---|---|
| Self-Audit core | `self-audit-canonical` v1 | `scripts/train_self_audit.py`, `train_maskfree.py`, `src/self_audit/training/{train_annotation,train_auditor,finetune_joint}.py`, `scripts/evaluate_external_mnms.py`, `evaluate_maskfree_{epoch,reference}.py`: refuse a non-canonical environment before `main()` |
| shared_benchmark | `self-audit-canonical` v1 | `scripts/evaluate_cardiac_baseline_reference.py`, `scripts/evaluate_visualize_shared_benchmark.py`: refuse before `main()` |
| CUTS | `self-audit-canonical` v1 | Training: `cardiac_benchmark.train_stage1` refuses and records the contract via `provenance.environment_identity()`. Generation: `scripts/run_cuts_scientific.py` is bound by the active historical-224 freeze v12, so it runs through `scripts/run_in_official_environment.py` |
| DSS-US | `self-audit-canonical` v1 | `baseline/DSS_US/scripts/run_native.py`: every profile is BLOCKED_PROTOCOL; a profile that ever becomes executable must pass the environment check before any input, model or output access |
| SGSCN | `self-audit-canonical` v1 | `baseline/SGSCN/scripts/run_native.py`: refuses before heavy imports; identity in `provenance.environment_contract` |

The environment is defined in `environments/self-audit-canonical/environment.json`
(CPython 3.10 series, concrete lock 3.10.20; torch 2.4.1; torchvision 0.19.1;
NumPy 1.26.4; CUDA 12.1 for GPU). `SELF_AUDIT_ALLOW_UNOFFICIAL_ENVIRONMENT=1`
only enables non-official portability runs and records `official: false`.

## Out of scope (legacy)

**DFC, STEGO and PiCIE** are not part of this milestone. Their code, tests,
freezes and artifacts are kept unchanged for history (`baseline/{DFC,STEGO,PICIE}/LEGACY_OUT_OF_SCOPE.md`).
They are not canonical-environment consumers, have not been made compatible,
do not enforce the environment, and no DFC/STEGO/PiCIE result or readiness status
counts toward the milestone. Shared-benchmark tests that still exercise their
historical contracts remain part of the shared regression suite.

## Must pass before an official run

1. `python scripts/check_environment.py --variant <cpu|cu121>` exits 0.
2. Tests in the canonical environment: Self-Audit core/environment-contract tests,
   `tests/shared_benchmark`, `baseline/CUTS/tests`, `tests/native_baselines`,
   `baseline/SGSCN/tests`, `baseline/DSS_US/tests`.
3. `scripts/native_protocol_readiness.py` matches the committed readiness receipt
   `dss_us_sgscn_native_readiness_paper_evidence_20261001.json` (no native gate changed).
4. CUTS: the active historical-224 freeze v12 validates the checkout.

## Native readiness (unchanged)

- DSS-US: all 11 CAMUS producer profiles BLOCKED_PROTOCOL (row recipes, CAMUS
  cohort, CRF parameters, Step II clusters); Step I Track B evaluator
  EVALUATOR_READY; Step II Track B BLOCKED_PROTOCOL (evaluated stage, label
  consistency); DINO checkpoint not supplied. Evidence:
  `dss_us_sgscn_paper_protocol_evidence_20261001.md`.
- SGSCN: paper profiles BLOCKED_PROTOCOL (architecture, loss reduction/weight,
  stopping, cohort/input); Track B BLOCKED_PROTOCOL (overlap measure, ties, HM/XOR);
  official-code reference profiles REFERENCE_READY.
- CAMUS, PH2, SYSU-US: BLOCKED_DATA. Native Track A: BLOCKED_ADAPTER.
