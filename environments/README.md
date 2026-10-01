# Official environments

Every official test, validation and experiment command runs in a declared,
fully pinned environment. There is one canonical numerical stack; arbitrary
Python / NumPy / PyTorch combinations are not official.

## `self-audit-canonical` v1 (canonical Self-Audit environment)

Declaration: [`self-audit-canonical/environment.json`](self-audit-canonical/environment.json)
(evidence, consumers, variants). Locks are fully resolved and hashed:

| Variant | Lock | PyTorch | Use |
|---|---|---|---|
| `cpu` | `requirements-cpu.lock` | 2.4.1+cpu, torchvision 0.19.1+cpu | tests, CPU validation, native CPU producers |
| `cu121` | `requirements-cu121.lock` | 2.4.1 (CUDA 12.1 runtime 12.1.105, cuDNN 9.1.0.70), torchvision 0.19.1 | GPU training and scientific runs |

Critical pins (both variants): Python 3.10 series (concrete lock 3.10.20, see
below), NumPy 1.26.4, SciPy 1.15.3, scikit-image 0.24.0, scikit-learn 1.7.2,
nibabel 5.4.2, timm 1.0.29, phate 2.0.0 (graphtools 2.1.0, PyGSP 0.6.1,
tasklogger 1.2.0), sewar 0.4.8, opencv-python 4.10.0.84, pillow 10.4.0,
PyYAML 6.0.3, matplotlib 3.10.9, tqdm 4.70.0, pytest 9.1.1.

Authoritative reference: the runtime-hardening reference environment (torch
2.4.1 / NumPy 1.26.4, 898 tests verified; declared server target Python 3.10,
torch 2.4.1, CUDA 12.1), with CUTS PHATE and native image-I/O pins from their
recorded evidence. Every pin cites its source file and commit in `environment.json`.

**Python contract: CPython 3.10 series.** Every declared target in the repository
(runtime runbook, reviewer gates, mask-free target runtime, the former
`environment.yaml`) specifies Python 3.10 only at series level. CPython 3.10.21
appears solely as the incidental interpreter of a macOS arm64 CPU verification
venv and is not an authoritative pin. The concrete lock is **3.10.20**: it is the
interpreter recorded by the project's Linux RTX 4070 training runs (tracked wandb
metadata) and the newest CPython 3.10 Linux x86_64 build available to the
reproducible installer. The validator enforces the 3.10 series and records the
exact micro version in provenance.

Consumers (active milestone): Self-Audit core, `shared_benchmark`, CUTS, DSS-US,
SGSCN and the native common infrastructure. DFC, STEGO and PiCIE are legacy and
out of scope (`reports/ACTIVE_MILESTONE_SCOPE.md`): they are not consumers, are not
verified compatible and do not enforce this environment. DSS-US and SGSCN were verified compatible (all
package tests pass), so no isolated upstream-reference environment is required;
their former ad-hoc Python 3.12 / torch 2.11 pins are superseded.

### Create

```bash
uv python install 3.10.20
uv venv --python 3.10.20 .runtime/envs/self-audit-canonical-cpu
uv pip sync --python .runtime/envs/self-audit-canonical-cpu/bin/python --require-hashes \
  --index-strategy unsafe-best-match environments/self-audit-canonical/requirements-cpu.lock
# or, with pip in a Python 3.10 venv:
python -m pip install -r environments/self-audit-canonical/requirements-cpu.lock
python scripts/check_environment.py --variant cpu      # must exit 0
```

GPU hosts use `requirements-cu121.lock` and `--variant cu121` (driver must
support CUDA 12.1). Conda users: `conda env create -f environment.yaml`.

### Enforcement

- `scripts/check_environment.py` fails (exit 1) on any mismatch in the Python
  series, a critical package or the PyTorch build variant.
- `src/environment_contract.require_official_environment()` raises before an
  official run starts and returns the environment identity (environment ID and
  version, lock SHA-256, Python, critical versions) for provenance. The SGSCN
  native producer records it as `provenance.environment_contract`.
- Runners whose source is bound by a benchmark freeze (CUTS/DFC) are launched
  through `scripts/run_in_official_environment.py --provenance-out RUN/environment.json -- <runner> ...`,
  which validates first, writes the identity (never overwriting it), then runs
  the unchanged runner.
- `SELF_AUDIT_ALLOW_UNOFFICIAL_ENVIRONMENT=1` exists only for portability runs:
  it prints a warning and records `official: false`; results are never official.

### Enforcement coverage (active milestone, 2026-10-01)

| Entrypoint | Rejects a non-canonical environment? |
|---|---|
| Self-Audit core: `scripts/train_self_audit.py`, `train_maskfree.py`, `src/self_audit/training/{train_annotation,train_auditor,finetune_joint}.py`, `scripts/evaluate_external_mnms.py`, `evaluate_maskfree_{epoch,reference}.py` | **yes**, before `main()` |
| shared_benchmark: `scripts/evaluate_cardiac_baseline_reference.py`, `evaluate_visualize_shared_benchmark.py` | **yes**, before `main()` |
| CUTS training (`cardiac_benchmark.train_stage1`) | **yes**, via `provenance.environment_identity()`; the contract identity is recorded in checkpoints/receipts |
| CUTS generation (`scripts/run_cuts_scientific.py`) | via `scripts/run_in_official_environment.py` only: the runner is bound by the active freeze v12, so it is not edited |
| `baseline/SGSCN/scripts/run_native.py` | **yes** (before heavy imports; identity in provenance) |
| `baseline/DSS_US/scripts/run_native.py` | all profiles BLOCKED_PROTOCOL; an executable profile must pass the check first |
| DFC, STEGO, PiCIE runners | out of scope (legacy); not enforced |

Sandboxed launches (bwrap) must bind the repository's `environments/` directory,
otherwise the check fails closed. `scripts/check_environment.py` must exit 0
before any official run.

### Regenerate locks

`environments/compile_locks.sh` (uv pip compile, Linux x86_64, CPython 3.10).
Change `requirements.in` only with new repository evidence; record it in
`environment.json` and bump `environment_version`.

## Portability environments (not official)

Cross-version checks (for example NumPy 1.24 / Python 3.8, NumPy 2.x / Python
3.12) may run regression tests only. Environments present on the server on
2026-10-01:

| Environment | Python / torch / NumPy | Status |
|---|---|---|
| `.runtime/envs/self-audit-canonical-cpu` | 3.10.20 / 2.4.1+cpu / 1.26.4 | **canonical (conforms)** |
| `.runtime/cuts_dfc_acdc` | 3.10.20 / 2.12.1+cu126 / 2.2.6 | non-conforming; portability only |
| `.runtime/cuts_dfc_acdc_py38` | 3.8.20 / 2.1.0+cu121 / 1.24.4 | non-conforming; portability only (recorded by the v10 CUTS/DFC full-run receipts; DFC is out of scope) |
| `~/miniconda3` base and other conda envs | various | outside this project; not official |

Historical environment records kept for provenance only (never used to create an
environment): `baseline/CUTS/benchmark/environments/cuts-cardiac-candidate.yaml`
(superseded candidate), `cuts_p0_local*.json` (non-scientific development
runtime), upstream baseline READMEs/environment files under `baseline/*` and
`baseline/CUTS/comparison/`.
