# Guidance runner and documentation maintenance — 2026-10-15

## Scope and unresolved input

Pulled `main` by fast-forward from `d4f503b7107eca4e8ac85f37a5dd01ec50479666`
to `4d58f3aebe4f205b58b3184c8971b1d838109d82`. Remote `main` was checked again
after the pull and still pointed to that revision.

The requested `astra_intergration` specification could not be located. Semble,
the working-tree literal search for `astra_intergration` / `astra_integration`,
and a filename check across fetched remote branch trees did not locate it.
Its path or branch was requested from the user. This maintenance change does
**not** claim to implement the missing specification.

## Implemented changes

- The W&B wrapper resolves the effective output directory through argparse,
  including repeated options, `--out=PATH`, and the child's `--ou` abbreviation.
  Previously it used the first literal `--out` while the child used the last,
  so tracking could be admitted inside the actual scientific output directory.
  Missing values now produce a CLI usage error instead of `IndexError`.
- Formatted the two guidance CLI entrypoints, the tracker and its tests.
  AST comparisons confirmed both entrypoints, the progress parser and JSON
  writer retain their behavior. The tracker output parser is the intentional
  behavior change.
- Added a documentation index and entrypoint table. Fixed the full runbook's
  stale feature-branch checkout instructions and labeled historical test
  receipts with their original context.

## Preservation

All 236 pre-existing modified/untracked local files were fingerprinted before
edits and verified unchanged. The existing changes in
`tests/test_maskfree_export.py` and `reports/maskfree150/` are excluded from
this maintenance commit.

There are no maintenance edits to `baseline/`, `benchmark_freezes/`, `src/`,
`configs/`, `run_configs/`, `splits/` or `experiments/`. Baseline recipes,
ablation code and historical evidence remain in place.

## Verification

Local environment: macOS arm64, Python 3.11.16, PyTorch 2.14.0, NumPy 2.4.6,
pytest 9.1.1; CUDA unavailable. This is local portability/software evidence,
not a run in the canonical Linux environment.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  /Users/alvinluong/miniforge3/bin/python -m pytest -q \
  tests/test_self_audit_*.py tests/test_pseudolabel_*.py \
  tests/test_guidance_final_metrics.py tests/test_guidance_wandb_runner.py
```

Final result: **466 passed, 1 warning in 15.45 s**, zero failures/errors/skips.
The warning is the existing tensor-to-scalar conversion in
`tests/test_self_audit_core.py:32`. The tracker suite tests
output resolution against the actual child CLI parser with execution mocked
at `run_full`; it never launches training or W&B. All changed Python files
compile, three CLI `--help` smoke checks pass, local documentation links resolve,
Ruff formatting and scoped `F,I` checks pass, and `git diff --check` passes.
Repository-wide lint is not claimed clean; the wider reviewer scan found
pre-existing formatting and style findings outside this patch.

No patient training, reference evaluation, GPU benchmark or W&B publication
was run for this maintenance task.

## Independent review

One read-only reviewer subagent inspected guidance logic and this maintenance
scope. Broader observations concerned checkpoint/manifest consistency,
reference-manifest provenance, environment receipts and nested artifact
inventories. These are retained as follow-up review topics, not silently
implemented as changes to a scientific recipe.

The reviewer also raised external authentication of evaluation artifacts.
The coordinator distinguishes this from the existing integrity contract:
hashes in a local manifest detect mutation relative to that manifest; they
do not authenticate an author who rewrites both payload and manifest. An
external trust anchor would be a separate design requirement.

Final patch review: **PASS, no blocker for the supported cleanup scope**.
The independent reviewer reran the tracker suite (**12 passed**) and the
stable-guidance/full-pipeline/final-metrics suites (**81 passed**), checked the
diff, confirmed output-option parity and confirmed baseline/model/config
preservation. Broader provenance observations remain deferred.
