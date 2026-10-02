# Active milestone scope (2026-10-01)

## In scope

| Component | Environment | Official entrypoints and enforcement |
|---|---|---|
| Self-Audit core | `self-audit-canonical` v1 | `scripts/train_self_audit.py`, `train_maskfree.py`, `src/self_audit/training/{train_annotation,train_auditor,finetune_joint}.py`, `scripts/evaluate_external_mnms.py`, `evaluate_maskfree_{epoch,reference}.py`: refuse a non-canonical environment before `main()` |
| shared_benchmark | `self-audit-canonical` v1 | `scripts/evaluate_cardiac_baseline_reference.py`, `scripts/evaluate_visualize_shared_benchmark.py`: refuse before `main()` |
| CUTS | `self-audit-canonical` v1 | Training: `cardiac_benchmark.train_stage1` refuses and records the contract via `provenance.environment_identity()`. Generation: `scripts/run_cuts_scientific.py` is bound by the active historical-224 freeze v12, so it runs through `scripts/run_in_official_environment.py` |
| DSS-US | `self-audit-canonical` v1 | `baseline/DSS_US/scripts/run_native.py`: refuses a non-canonical environment before any input, model or output access; only the declared-conventions Step II DSS-baseline profile is runnable |
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
   `dss_us_sgscn_native_readiness_paper_faithful_runnable_20261002.json`.
4. CUTS: the active historical-224 freeze v12 validates the checkout.
5. ACDC native adaptation: `scripts/acdc_native_readiness.py` reports the per-method statuses below.

## Native readiness by profile class (2026-10-02)

Every profile reports its own class, and the classes are never merged. A paper-faithful profile can be RUNNABLE
while exact PAPER_REPRODUCTION stays BLOCKED_PROTOCOL.

| Class | DSS-US | SGSCN |
|---|---|---|
| PAPER_REPRODUCTION | 11 profiles BLOCKED_PROTOCOL | 2 profiles BLOCKED_PROTOCOL |
| PAPER_FAITHFUL_REIMPLEMENTATION (templates) | 11 BLOCKED_PROTOCOL (required values) | 2 BLOCKED_PROTOCOL (required values) |
| PAPER_FAITHFUL_WITH_DECLARED_CONVENTIONS | RUNNABLE: `step2_dss_baseline_dss_paper_faithful_declared_conventions` | RUNNABLE: `ph2_` / `sysu_us_paper_faithful_declared_conventions` |
| OFFICIAL_REFERENCE | none | RUNNABLE: `ph2_` / `sysu_us_official_reference` |

- Track B: DSS-US Step I (per-image remapped Dice) EVALUATOR_READY. DSS-US Step II and all
  SGSCN paper metrics are BLOCKED_PROTOCOL (overlap measure, ties, HM/XOR are not defined).
- CAMUS, PH2, SYSU-US and the DINO ViT-S/8 checkpoint: BLOCKED_DATA. Native Track A: BLOCKED_ADAPTER.
- Details, declared conventions and fallbacks: `dss_us_sgscn_paper_faithful_runnable_20261002.md`.

## ACDC adaptation of DSS-US / SGSCN (2026-10-02)

Plumbing only, on the frozen v12 historical-224 ACDC contract; source profiles run unchanged.
Entrypoints: `baseline/{SGSCN,DSS_US}/scripts/run_acdc.py` (canonical environment required before
any data access), `scripts/run_acdc_native_tracks.py` (Track A / Track B; `enforce_official_entrypoint`),
`scripts/acdc_native_readiness.py`. Contract binding: `configs/acdc_native_tracks_v1.json`.

| | DSS-US | SGSCN |
|---|---|---|
| ACDC_PRODUCER_STATUS | PROTOCOL_READY | PROTOCOL_READY |
| ACDC_TRACK_A_STATUS | ADAPTER_READY (cardiac_adapter_v2) | ADAPTER_READY (cardiac_adapter_v2) |
| ACDC_TRACK_B_STATUS | EVALUATOR_READY (raw_id_majority_vote_v1) | EVALUATOR_READY (raw_id_majority_vote_v1) |

ACDC_DATA_STATUS BLOCKED_DATA (ACDC images; DINO checkpoint for DSS-US), ACDC_ENV_STATUS CANONICAL,
ACDC_FULL_RUN_STATUS NOT_STARTED. Details: `dss_us_sgscn_acdc_adaptation_20261002.md`.
