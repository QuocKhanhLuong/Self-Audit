# SGSCN ACDC CPU ISA check — 2026-10-04

With `ONEDNN_MAX_CPU_ISA=AVX2` and `MKL_CBWR=AVX2,STRICT`, the two ACDC dev-slice raw partition hashes still differ from the old server, but three independent runs on the new server agree exactly. Use the new server (AMD Ryzen Threadripper 9960X) for all official SGSCN runs in this milestone; do not mix official SGSCN results from the two CPU platforms.

Hardware and environment:

- New server: AMD Ryzen Threadripper 9960X 24-Cores, 48 logical CPUs, `avx512f` present.
- Old server: AMD Threadripper 3960X, AVX2-only, as reported by the operator. The operator reports that the same ISA environment variables preserve the old-server hashes.
- Official environment: self-audit-canonical v1 CPU, CPython 3.10.20, torch 2.4.1+cpu, NumPy 1.26.4. Environment validator exit 0.
- Every check used CPU, `--threads 4`, `--split dev --limit 2`, the frozen v12 adaptation `baseline/SGSCN/config/acdc/acdc_ph2_paper_faithful_declared_conventions.json`, and `.runtime/acdc_self_audit_images_only`.
- Both variables were set before interpreter startup. PyTorch still reports its general ATen CPU capability as `AVX512`; this does not identify which kernels actually executed or establish the cause of the mismatch.
- Producer files, native profiles, environment locks and freezes were unchanged. Runs were generated against base HEAD `d72807d8afca93b3a9e4acfeb89ddabe574d0dc3` while the reference-pinning/test-import fixes were pending.

All three runs reached `RAW_COMPLETE`, samples=2. External receipts and raw seals were verified using `verify_acdc_seal_receipt()` in the canonical environment. No GT evaluation or full experiment was performed.

| Slice | Old server | New server: initial + repeat 1 + repeat 2 | Cross-server |
|---|---|---|---|
| z-0000 | `d9add82effae9ee043b123f997e4924f0e6b44f03cbd41449aa63786fa48edf5` | `a9baa08ffd28682abe9f15d0dc504dffbfad5f0a3afe696e4aa4524a00ebb766` | MISMATCH |
| z-0001 | `3e7c6e5a5ed5342f96649051593be156a00dbf8cd024f4fb1429e4e636cce09a` | `22ff4acda77f66c81d8036ec1942378f08524d21968fa8d4bfdf36dba8a2c36a` | MISMATCH |

The new-server values in the table are identical in all three runs. Cross-CPU bitwise reproduction was not obtained by restricting oneDNN/MKL with these two variables. CPU/numerical-kernel differences remain a possible explanation; the root cause has not been established. Repeatability was checked only for these two slices under this configuration and is not a claim about a full cohort.

Official SGSCN execution convention for this milestone: keep the selected new server, canonical CPU environment, `--threads 4`, and both checked environment variables fixed. Set them on each SGSCN invocation and record them with its run report/provenance. Changing this convention requires another reproducibility check. The frozen native profile and producer remain unchanged.

Example invocation from this checkout (substitute a fresh output and receipt path for each run):

```bash
ONEDNN_MAX_CPU_ISA=AVX2 MKL_CBWR=AVX2,STRICT \
  .runtime/envs/self-audit-canonical-cpu/bin/python baseline/SGSCN/scripts/run_acdc.py \
  --adaptation baseline/SGSCN/config/acdc/acdc_ph2_paper_faithful_declared_conventions.json \
  --image-root .runtime/acdc_self_audit_images_only \
  --split dev --limit 2 --threads 4 --device cpu \
  --output /workspace/acdc_runs/FRESH_RUN \
  --receipt /workspace/acdc_runs/receipts/FRESH_RUN.json
```

Local evidence (outside the checkout):

- `/workspace/acdc_runs/isa_check`, `isa_check_repeat1`, `isa_check_repeat2` contain the sealed raw runs.
- `/workspace/acdc_runs/receipts/isa_check.json`, `isa_check_repeat1.json`, `isa_check_repeat2.json` contain their external seal receipts.
- `/workspace/acdc_runs/native_fixes_20261004/isa_check_comparison.json` contains all exact raw hashes, run seals, paths, environment variables, and comparisons.
- `/workspace/acdc_runs/native_fixes_20261004/isa_numeric_provenance.json` and `isa_numpy_config.txt` record CPU capability, interpreter/library versions, environment variables, and numerical build configuration.
- `/workspace/acdc_runs/native_fixes_20261004/isa_check*.log` contains producer outputs.

Validation of the accompanying code fixes: the requested default-import-mode single-process command `PYTHONPATH=src .runtime/envs/self-audit-canonical-cpu/bin/python -m pytest -q tests/native_baselines baseline/SGSCN/tests baseline/DSS_US/tests` passed **169 tests and 11 subtests** (zero failures/errors/skips). One new test exercises `pin_references()` across every declared reference, including ADNet, without network access. The real `scripts/pin_native_references.py` CLI also completed and pinned ADNet at `c6bba85040c12ad1d2f351cdd8f72850daaaf3fb`. Freeze validators v10/v11/v12 all exited 0.
