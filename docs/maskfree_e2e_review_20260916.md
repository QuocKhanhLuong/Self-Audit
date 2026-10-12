# Mask-free end-to-end audit — 2026-09-16

## Scope and evidence boundary

Baseline production commit: `96c32b10fc7b8e09b48822e10ae9eb6cc149e253`.
Reviewed code lives in `src/self_audit_maskfree`, not the older supervised `src/self_audit` described by the top-level README. Optimizations are isolated on `audit/maskfree-e2e-20260916`; main and the user's running rental checkout were not changed.

The source was read component-by-component and exercised locally with Python 3.13.5, torch 2.10.0+cpu, NumPy 2.3.5, SciPy 1.17.0 and nibabel 5.4.2. These are CPU synthetic software tests, not real ACDC/M&Ms accuracy results or RTX timing. The user's B32 result (18.108305464 s/batch, 1.767 samples/s, 8.088 GiB reserved) remains a separate user-reported measurement. No real B64 timing breakdown was supplied.

## Correct end-to-end flow

```
image-only inventory and patient split
  -> study-wide fit/select/verify support partition and guards
  -> fitting-only normalization and physically masked 2.5-D input
  -> randomly initialized producer: contrastive + reconstruction + equivariance SSL
  -> detached producer features
  -> four candidate hypotheses, including the incumbent, and anatomical ontology
  -> frozen observation parameters fitted on O_fit
  -> global and regional comparisons on O_select
  -> identical-initialization, independent no-audit and audited students
  -> immutable exports and checkpoints
  -> held-out O_verify; optional manual-mask evaluation in a separate process
```

Support masks here are observation partitions, not manual segmentation annotations. The method is mask-free with explicit anatomical priors; it is not prior-free learning. The active observation implementation is spatial predictive. Temporal/cine claims need a separately implemented and tested observation model.

## Module-by-module findings

| Component | Checked contract | Finding / action |
| --- | --- | --- |
| Data and splits | Patient independence, study-wide withheld support, guards, fit-only normalization, geometry | Existing mutation/firewall/native-volume tests pass. Actual user data paths and cohort counts were not accessed. |
| Producer | Fitting-only SSL, random initialization, gradient isolation | Three SSL forwards plus a separate detached feature forward. Reuse is a possible later optimization, not enabled here. |
| Hypotheses | Four candidates, deterministic seeds, feature-bound bank identity | Only the first eight feature channels are consumed and hashed. Keep no pseudo-label cache across producer updates. |
| Candidate workers | Ordered chunks, seed identity, fail-closed worker and payload errors | Chunk size eight bounds IPC for bank generation; it does not constrain the subsequent audit batch to eight. |
| Ontology | Named anatomical rules, orientation metadata, unresolved states, bounded connected components | Preserve all ambiguity and fallback behavior. `study_continuity` exists but has no production caller in the reviewed package. |
| Observation | Fit/select separation, FP64, fixed capacity/iterations, immutable fits | `fit_many`, `score_many`, and prepared regional scores already exist. Removing integrity checks is not an acceptable speed optimization. |
| Auditor | Incumbent ties, candidate/round/region budgets, local observed disagreement | Fixed repeated full-image region bookkeeping; scoring formula, order and budgets unchanged. |
| Students | Shared initial state and bank, distinct target sources, no producer gradients | Existing gradient/state, scalar/bulk, and resume-equivalence tests pass. |
| Validation and export | Report-only epoch observation, frozen prediction lineage, verification after freeze | Separate validation/export costs from batch timing. Existing freeze-scope hashing already avoids repeated full-tree validation. |
| Launcher/profiler | Config identity, horizon, batch, completion | Added explicit B64 and 50-epoch profiling; identified but did not silently redefine the legacy full-training completion contract. |

## Implemented changes

### Indexed regional bookkeeping

The old regional path materialized one HxW boolean mask for every connected region, sorted areas through full mask reductions, and recomputed full-image challenger disagreement inside every region loop. The 32-region scoring cap did not bound this preliminary work.

The replacement retains the exact connected-component algorithm and non-convergence fallback, assigns integer region IDs, counts area/selection/disagreement once, and creates a region mask only when that region actually enters scoring. Stable area/role/component ordering, all unobserved/over-budget report rows, score-call limits, FP64 likelihoods, candidate choice and abstention semantics are retained. This removes the O(R*H*W) stored-mask representation; it does not remove the scientific regional scoring work.

### Consumed-feature transfer

The canonical generator consumes eight channels but the trainer copied all sixteen to CPU. The new helper slices before the device copy and releases the GPU feature reference before auditing and student updates. Custom generators still receive all channels. With B64, 224x224, FP32, the logical copied feature payload changes from 196 MiB to 98 MiB. This is a byte-count calculation, not a measured peak-VRAM reduction or GPU speedup.

### Explicit profiling interface

`profile_maskfree.py` and `benchmark_maskfree_rental.py` accept B64 and an explicit `--total-epochs 50`. A mismatch between the chosen horizon and loaded config is rejected rather than rewritten. Rental gates accept explicit dataset roots, including `--mnms-data-root /root/Self-Audit/data/MnM/extracted`. CPU audit and all numerical matching requirements remain unchanged.

## Tests actually executed

Baseline: **282 passed, 5 skipped**, 200.17 seconds.
Optimized: **307 passed, 5 skipped**, 193.88 seconds.

Command in each isolated source tree:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src \
PYTHONDONTWRITEBYTECODE=1 python -m pytest -v tests/test_maskfree*.py --disable-warnings
```

The 25 added cases cover exact old/new region masks and ordering, non-contiguous inputs, fragmented and nested regions, empty selection, non-convergence fallback, complete audit traces and score budgets, feature-channel identity/custom-generator compatibility, B64/50 config matching, and semantic name-permutation non-identifiability. Four skips require CUDA; the fifth concerns an unavailable historical snapshot. An earlier full-suite attempt timed out at 180 seconds; the subsequent complete baseline and optimized runs exited zero. Missing nibabel during initial collection was resolved before those complete runs.

## Measurements — retain the limitations and the outlier

All measurements below are local CPU/synthetic, with one Torch/OMP/MKL thread. The ordinary comparison used the same new harness, identical fixture path and sampling order, B8, 128x128, horizon 50, one warm-up and six measured batches per process, in before/after/after/before order.

| Run | Mean s/batch | Median s/batch | P95 s/batch |
| --- | ---: | ---: | ---: |
| Before 1 | 3.338353 | 3.316418 | 3.420767 |
| After 1 | 3.105019 | 3.117993 | 3.178576 |
| After 2 | 7.761864 | 3.102717 | 24.116442 |
| Before 2 | 3.238875 | 3.267567 | 3.292252 |

The first pair reduces mean batch time by about 7.0%, but the second optimized run contains a 31.112683-second batch (21.087522 seconds process CPU). Its cause has not been established. It is not discarded. Across all measured batches the optimized mean is worse, so **a sustained end-to-end speedup is not established**. Medians alone must not be substituted for this conclusion.

Both optimized runs pass the strict comparison against Before 1: score deltas are zero; candidate decisions, tensor hashes, budget counts, final model state and final RNG match. Before 2 also matches Before 1. No matching tolerance was relaxed.

A separate instrumented region-only comparison (B8/128, one warm-up, two measured batches, horizon 150) measured audit time 0.777984 -> 0.598192 s/batch, about 23.1% lower. This is diagnostic evidence, not sustained ordinary throughput.

An independent full `audit_banks` microbenchmark compared the unchanged baseline module with the optimized module, identical frozen synthetic inputs, warm-up, six alternating repetitions per version:

| Synthetic 64x64 fixture | Before mean | After mean | Ratio |
| --- | ---: | ---: | ---: |
| Nested regions | 20.271 ms | 19.889 ms | 1.02x |
| Highly fragmented regions | 447.615 ms | 52.868 ms | 8.47x |

Complete traces, margins and validity tensors match exactly. The fragmented case demonstrates the scaling defect; it is not a representative clinical speed estimate.

## Scientific and operational issues still open

1. **Semantic naming is not identified by appearance likelihood alone.** A new test swaps RV/LV names globally while preserving geometry. With the default symmetric class capacity and appearance score, the fitted likelihood ties within 2e-12. Anatomical ontology, independent geometric/volume evidence and abstention must remain explicit. A likelihood improvement is not proof of anatomical correctness.
2. **50-epoch full-training completion is inconsistent.** The trainer can finish and finalize 50 epochs, but marks only 150-epoch runs completed; the native CLI returns 2 for a finalized 50-epoch run, and the sequential shell verification requires 150. Changing only `TOTAL_EPOCHS=50` does not make that sequence correctly complete. This patch enables bounded profiling of that recipe, not an undocumented bypass of completion checks.
3. **Label ramp stays at 20 epochs.** That spans 40% of a 50-epoch horizon, versus about 13% of 150. Altering it is a recipe change requiring a separate experiment; no silent ramp/LR/budget change was made.
4. **CUDA equivalence remains unresolved.** The pre-existing `reports/maskfree150/performance_rental_4080s.md` reports 5.781 vs 5.189 s for one CPU/CUDA comparison, but `strict matched=False`. A successful CUDA preflight proves execution, not scientific equivalence or a large speedup. The earlier advice to treat CPU audit as a proven cause of three-hour epochs was too strong.
5. **No actual three-hour breakdown was read.** Training-unit count, loader/codec/cache time, main/worker CPU usage, validation and export costs need measurement on the target checkout. The earlier inferred roughly 19,000 units was an extrapolation, not an observed inventory.
6. The pre-existing general CI checkout fails on an orphaned `external/Medical-SAM3` gitlink; local test success must not be misrepresented as that CI passing. Temporary source/dependency/publication workflows used for this audit are removed from the final branch.

## Next optimizations, separated by risk

**Implementation-preserving candidates:** reuse the canonical producer reference features only after proving the same forward state/precision; retain a fallback for AMP/custom models. Profile raw NIfTI decoding and bounded frame-cache hit rates before adding an immutable, image-only decoded-frame cache. Consider a batch-owned prepared-score reduction API that validates immutable snapshots once, without deleting integrity protection. Benchmark CPU worker count against actual cgroup limits, not host RAM/CPU totals.

**Technique experiments, not automatic replacements:** fit-only study-level semantic continuity with frozen epoch snapshots and no sampling-order dependence; lower-resolution candidate proposals followed by full-resolution exact auditing; certified early-exit bounds for candidate score differences with FP64 fallback near decision thresholds. Each must be separately declared and evaluated against both label coverage/quality and compute. Do not cache stale pseudo-labels, drop difficult units, remove the no-audit control, or shrink the bank/fit/region budgets and call it the same experiment.

## Target-host diagnostic, without editing a running checkout

Use a separate worktree at this branch. Only run the gate when the selected GPU is free; the harness intentionally rejects competing compute jobs. It never starts full training.

```bash
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python scripts/benchmark_maskfree_rental.py \
  --data-root /root/Self-Audit/data \
  --mnms-data-root /root/Self-Audit/data/MnM/extracted \
  --batch-size 64 --total-epochs 50 \
  --candidate-workers 8 --candidate-chunk-size 8 \
  --output reports/maskfree150/review-b64-e50-diagnostic \
  --diagnostic
```

Choose a fresh output path on each invocation. Compare before/after with the same B64/50 recipe, data root, seed, workers, numerical backend and profiler harness. Do not use the old B32 report as an equivalence reference for B64. Do not resume an old-source or CPU-backend checkpoint into a changed source/CUDA run.
