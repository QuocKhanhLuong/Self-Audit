# Large physical batches with bounded candidate IPC

Base: `2f39a17656d74bd09be49ee291361f71987b7989`.

## Scope

The former candidate-worker `batch_size <= 8` restriction applied to the physical neural batch as well as the transport window. These are now separate:

- `batch_size`: neural/optimizer batch (8, 16, 32 are exposed by the benchmark).
- `candidate_workers`: 0, 2, 4, or 8 persistent CPU spawn workers.
- `candidate_chunk_size`: 1..8 units submitted and fully drained per IPC window.

For B32/W8/chunk8 the executor processes four ordered windows. Only consumed feature channels are cloned. Per-unit seeds are resolved for the full input sequence BEFORE slicing; default seeds do not restart at a window boundary. Candidate IDs, labels, probabilities, validity, ontology metadata, and audit budgets are not intentionally changed. No CUDA state is forked into workers.

The default remains B8/W0/prefetch0. Nothing automatically starts a full run. CPU FP64 audit remains the reference; existing CUDA matching failures/tolerances are not changed or waived by this patch. Neural precision remains FP32.

## Memory boundaries and accounting

Input tensor payload is bounded to 8 MiB/unit, at most `chunk_size * 8 MiB` in the submitted window. Candidate result tensors are bounded to 32 MiB/unit and validated inside the worker before return. The chunk iterator is drained before the next input window is prepared. Duplicate/out-of-range/missing result indices and worker failures fail closed; incomplete banks are never returned.

These are TENSOR transport limits, not a total RSS or VRAM promise. The final B banks are retained for the existing batched auditor; their memory grows with B. Worker interpreters, Python metadata, native temporary arrays, caches, allocator arenas and active neural tensors require additional memory. Runtime summary exposes candidate window counters, maximum transported input/result tensor bytes and retained result tensor bytes. Do not raise the cache or prefetch just to fill memory. Benchmark against the container's actual CPU/RAM limits. W8 is not assumed faster than W4.

## Runtime identity

`candidate_chunk_size` is a recorded runtime identity field, forwarded through config, CLI, launcher, preflight and checkpoint guards. A changed batch, worker count, chunk size, source or backend requires a fresh run under existing rules. A larger batch changes optimizer steps, augmentation/RNG grouping and training trajectory. Per-unit candidate equality assumes the SAME input features/seeds; it is not a claim that training at B8 and B32 is identical.

## Target-host gate

Stop the old job first; never edit its source underneath it. In the existing rental environment, with the correct GPU selected and no competing job:

```bash
cd /root/Self-Audit
TAG=$(date +%Y%m%d_%H%M%S)
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MASKFREE_PROGRESS=compact \
python scripts/benchmark_maskfree_rental.py \
  --batch-size 32 --candidate-workers 8 --candidate-chunk-size 8 \
  --prefetch-batches 0 \
  --output "reports/maskfree150/rental4080s/gate-b32-w8-${TAG}"
```

This gate runs ACDC and M&Ms preflights, then 5 warm-up + 30 measured real ACDC batches. `--print-plan` validates configuration without training. It does NOT prove real-data equivalence across worker counts without a matched reference: repeat the SAME B32 gate with workers0 (or a reviewed serial B32 report) and supply its `profile/profile_report.json` via `--reference-report` for W8. A B8 report must NOT be passed as an equality reference to a B32 run.

For throughput compare B8/W4, B16/W4, B32/W4 and B32/W8 by B/seconds, not just seconds/batch or VRAM. Do not run them concurrently. Retain the first valid configuration whose complete throughput and RAM use are acceptable; stop tests on OOM or worker errors. No measured RTX 4080S speedup is claimed here.

## Full run (after inspecting the gate)

```bash
cd /root/Self-Audit
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
RUN_FULL=1 WORKSPACE="$PWD" \
ACDC_DATA_ROOT="$PWD/data/ACDC" \
MNMS_DATA_ROOT="$PWD/data/MnM/extracted/M&M" \
AUDIT_DEVICE=cpu TIMING_MODE=production LOGGING_MODE=buffered \
DATA_CACHE_BYTES=67108864 PREFETCH_BATCHES=0 \
CANDIDATE_WORKERS=8 CANDIDATE_WORKER_THREADS=1 CANDIDATE_CHUNK_SIZE=8 \
BATCH_SIZE=32 ACCUM_STEPS=1 MASKFREE_FALLBACK='' \
TOTAL_EPOCHS=150 EPOCH_VALIDATION=1 MASKFREE_PROGRESS=compact \
MASKFREE_GPU_OVERRIDE='rental RTX 4080 SUPER; reviewed B32/W8 gate; CPU audit' \
PYTHON="$(command -v python)" bash scripts/run_maskfree_acdc_mnms.sh
```

Only use W8 when the real resource/throughput gate supports it. Set W4 in BOTH benchmark and full command to test the more conservative variant. Both datasets start independently. The launcher still requires each run's matching preflight.
