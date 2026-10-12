# Root validation receipt

Date: 2026-10-02. Audited base: `c31825a7f90df47f9f9382a9c9595e43f7f71236`.
Publication base after concurrent upstream integration:
`53fa6182ff8f5d262c2a45a8644e9b850a4d5c49` (ADNet few-shot lane added, no change
to audited Self-Audit/v3 model or v3 config). All runtime receipt source hashes
were rechecked against the integrated tree and match.
Scope: the combined cleanup in the isolated integration worktree. Root ran these
checks after code edits. This is software validation, not a scientific pass or
an independent final reviewer verdict.

## Combined regression

**451 passed, 2 warnings in 178.09 seconds.** Machine-readable cases and durations:
[validation_junit.xml](validation_junit.xml). The two warnings concern unsupported
MPS `pin_memory`; they are not CUDA or MPS performance measurements.

The environment was macOS arm64, Miniforge Python 3.11.16 / PyTorch 2.14.0.
`SELF_AUDIT_ALLOW_UNOFFICIAL_ENVIRONMENT=1` was explicitly used for local
portability checks; this does not certify the locked scientific environment
(Python 3.10 / PyTorch 2.4.1 / CUDA 12.1). No dependency installation was performed.

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
SELF_AUDIT_ALLOW_UNOFFICIAL_ENVIRONMENT=1 \
/Users/alvinluong/miniforge3/bin/python -m pytest -q \
  tests/test_checkpoint_binding.py tests/test_calibration_lineage.py \
  tests/test_unified_trainer.py tests/test_self_audit_pseudolabel_v3.py \
  tests/test_pseudolabel_api_compat.py tests/test_pseudolabel_v3_complete.py \
  tests/test_pseudolabel_v3_integration.py tests/test_pseudolabel_v3_regressions.py \
  tests/test_pseudolabel_full_pipeline.py tests/test_pseudolabel_system_v3.py \
  tests/test_wave3_downstream.py tests/shared_benchmark \
  --junitxml=reports/main_baseline_20261002/validation_junit.xml
```

The pre-edit calibration/unified subset had one sandbox-only subprocess failure
(`torch_shm_manager: Operation not permitted`); that test passed outside the
sandbox. The pre-integration combined run passed 439 tests. After integrating the new
ADNet lane, the entire final combined command above ran outside the sandbox
and passed all 451 tests. The retired CLI-acceptance test was intentionally removed; tests of
live calibration semantics and canonical legacy-flag rejection remain.

## Static and frozen-contract checks

- All **643** tracked Python files in the final integrated set parsed
  successfully; details in
  [static_validation.json](static_validation.json).
- Five extracted functions and two constants have identical ASTs to their
  original versions at the audited base.
- No changes to model, loss, data implementation or configuration files.
- No live imports/references to deleted runners in `scripts/` or `tests/`.
  Historical documentation deliberately names files in the pinned old revision.
- `bash -n scripts/run_full_pipeline.sh` and `git diff --check` passed.
  PowerShell execution was **NOT RUN**; its edits only change help/error strings.
- Active v12 freeze validation **PASSED unchanged**:
  `cardiac-benchmark-v12-historical-224-1d1ecbb72e49d999`, scientific payload SHA
  `1d1ecbb72e49d99942603625d2c601e13294b66604130afe993b071576848d11`;
  train=1526, dev=376, test=0. These are benchmark inventory counts, not a new
  quality experiment or the v3 full-cine cohort.

```bash
python scripts/validate_cardiac_benchmark_v12_historical_224_freeze.py
```

## Engineering measurements and limits

The standalone [probe](benchmark_student_runtime.py) was run after the test
process completed. Its [raw receipt](runtime_cpu.json) contains all CPU timing
samples, source/config/probe SHA-256 values, environment check, exact shapes,
warmups, repeats and random-weight status. It releases each prediction before
the next forward, synchronizes CUDA if explicitly requested, and records both
allocated and reserved peaks on CUDA. The reserved-memory value includes the
allocator's preceding warmup/profile history. The GPU path itself is **NOT RUN**.

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python \
  reports/main_baseline_20261002/benchmark_student_runtime.py \
  --device cpu --threads 1 --out reports/main_baseline_20261002/runtime_cpu.json
```

Use a fresh output path for reproduction; the probe refuses overwrites. The
probe has random weights and is not a complete trained-checkpoint benchmark.
No trained Dice, RTX 4080 Super latency/VRAM, end-to-end native-volume timing,
full-cine training, M&Ms transfer or final student evaluation was executed.
The configured local ACDC roots inspected by the coordinator were absent; this
is not a claim that the user has no dataset on another machine.

## Review and release scope

Three Orca report workers completed. The additional independent final review
could not execute because of readiness/authentication failures; see
[orchestration accounting](06_ORCHESTRATION.md). Root checked the migration,
decisive code paths, source distinctions, hashes and combined regressions.
No independent final-review PASS is claimed.

Publication is limited to the explicit cleanup/report allowlist. The unrelated
dirty main checkout and older audit branch source are left intact. A normal
fast-forward publication is required; no force-push or history rewrite is part
of this task.
