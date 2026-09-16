# Mask-free runtime refactor: RTX 4080S handoff

## Evidence and decision

Development base: `5c3113970693319bf8bf92b738e12787b473ea17`; branch
`perf/maskfree-4080s-runtime`. Development took place on macOS, with Python
3.11.16 / PyTorch 2.14.0 and no connected rental host. Rental measurements below
are **user-supplied**, not independently reproduced. Original rental tensor
artifacts were unavailable here. No full training job was launched.

| Complete real ACDC batch, 224 / 8 / FP32 | Mean s | p50 s | p95 s | Status |
|---|---:|---:|---:|---|
| Existing CPU audit | 5.781043008 | 5.781251429 | 6.453467085 | User-supplied baseline |
| Existing CUDA audit | 5.189159276 | 5.124224967 | 6.183280796 | Strict matched=False |
| This refactor, CPU audit | NOT RUN | NOT RUN | NOT RUN | Requires rental gate |

The previous CUDA ratio was 1.11406159, with about 2.096 GiB peak reserved VRAM.
`audit_score_tolerance` and `audit_tensor_hashes` failed. Their tolerance and
strict result have not been changed. Missing fit-parameter, score, regional-margin,
target and gradient tensors mean their maximum absolute/relative errors remain
**unknown**. CUDA audit is not promoted as a validated optimization.

At the supplied CPU mean, training alone costs approximately 4.22 hours/ACDC
epoch and 633 hours/150 epochs. Validation, export and checkpoint cost must be
added. New ETA and combined ACDC + M&Ms ETA remain unknown; the latter also needs
the actual M&Ms batch count. For measured latency `t`, training hours are
`2628*t/3600` per ACDC epoch, and `150*(2628+mnms_batches)*t/3600` for both datasets
only if their measured per-batch times agree.

## Implemented changes

| Stage | Existing behavior | Implemented behavior | Rental time / speedup |
|---|---|---|---|
| Stage timing | CUDA barriers around every stage, including host work | Production host timing; explicit synchronized diagnostic mode | NOT RUN |
| Scalar/gradient reporting | Individual device-to-host scalar copies | Batched copies; original per-parameter norm and Python summation order | NOT RUN |
| Target transfer | Per-unit target copies | Four detached stacked target copies per physical batch | NOT RUN |
| Lineage logging | Open, append, flush, fsync per unit | Persistent bounded buffer; durable flush before checkpoint offsets | NOT RUN |
| Candidate generation | Serial per unit | Optional persistent CPU spawn pool, 2 or 4 workers | Local fixture only, below |
| Image access | Shared bounded source-frame cache | Explicit per-dataset byte limit; optional FIFO input prefetch | NOT RUN |
| Producer SSL / feature forward | Separate forwards | Preserved: equivalence of reuse has not been established | NOT RUN |
| Observation fit / regional scoring | Existing batched/cached architecture | Preserved; CPU FP64 reference | NOT RUN |
| Student updates / optimizer | Existing equations and finite-gradient gate | Preserved; all active gradients checked before any step | NOT RUN |
| Epoch validation / final export | Existing isolated evaluation and freeze | Preserved | NOT RUN |

Code inspection establishes redundant barriers and per-unit durable writes, not
their fraction of rental wall time. Candidate creation is CPU tensor/topology
work; NIfTI decode is CPU/I/O work. Their remaining share, GPU utilization and
power distributions, CPU throttling, memory pressure, and the Amdahl bound are
unknown until the target diagnostic runs. Missing ordinary nested timings are
not zero. For measured affected fraction `f` and stage improvement `s`, the
maximum whole-batch speedup is `1 / ((1-f) + f/s)` before new overheads.

### Independently executed local candidate fixture

[Raw root measurements](rental4080s/local_candidate_probe/root_measured.json)
and [runner](rental4080s/local_candidate_probe/benchmark_candidate_pool.py):
8 synthetic units at 224, five measured repeats after pool setup; Darwin arm64,
parent Torch threads=8, each spawned worker threads=1.

| Candidate-only execution | Mean s | p50 s | Mean ratio |
|---|---:|---:|---:|
| Serial | 0.9236 | 0.9268 | 1.00x |
| 2 workers | 0.3882 | 0.3775 | 2.38x |
| 4 workers | 0.2178 | 0.2107 | 4.24x |

Candidate labels/digests matched exactly in this probe. Focused tests additionally
compare probabilities, validity, metadata, audit decisions, targets, losses,
gradients and RNG state on deterministic fixtures. Input pickle size was
19,679,605 bytes (4.62 ms to serialize), output 45,006,715 bytes. This is **not**
an end-to-end, CUDA, real-data, or annotation-quality result. Worker-supplied
earlier numbers are retained separately and are not the accepted timing evidence.

## Correctness and resource boundaries

- No changes to pooling, neural precision, scientific budgets, hypothesis or
  observation equations, topology, reference access, split or initialization.
  Canonical target profile remains 224, physical/effective batch 8, accumulation
  1, FP32 neural / FP64 observation, four candidates, two rounds, five fits,
  independent 150-epoch runs and per-epoch isolated evaluation.
- Candidate workers use `spawn`, detached CPU fitting inputs, explicit per-unit
  seeds and original-order results. Queues cover at most one physical batch of
  eight units, with an 8 MiB tensor payload bound per unit; only consumed feature
  channels are cloned for IPC. Worker exceptions propagate; owned processes are
  shut down, including abrupt process death. No CUDA context is forked.
- Input prefetch uses one thread, 0/1/2 batches and a 32 MiB queued-plus-inflight
  **output tensor** bound. It never runs the producer or caches pseudo-labels.
  Native decoded-frame temporaries and Python metadata are outside that bound.
  Default train cache is 64 MiB; the existing validation/global cache can retain
  a further 64 MiB. Process interpreters, allocator arenas, IPC outputs, manifests,
  and the active batch add memory. Thus these settings are not a total RSS cap.
- Partition, fit-only normalization, role-masked resampling and sealed verification
  access stay in the canonical loader. Cache counters and prefetch counters are
  emitted in runtime reports; optional prefetch may do unused input I/O after a
  checkpoint, but consumes no training RNG or optimizer updates.
- Buffered lineage retains every row in deterministic order. File and directory
  fsync precede checkpoint offsets and atomic checkpoint commit. Uncheckpointed
  rows can be discarded and replayed. Missing/truncated checkpointed logs fail
  closed instead of extending them with zeros. Sync mode remains available.
- Preflight writes both `audit_device` and `audit_device_identity`; the real run
  validator reads the latter. Old receipts are neither rewritten nor bypassed.
  Source, backend, precision and all new runtime options are exact-resume identity
  checks. Use a fresh run after changing them.
- Production timings are inclusive host wall time, including enqueue and dependency
  waits. Complete benchmark batches synchronize CUDA at outer boundaries.
  Diagnostic timings, cProfile, fsync/sync counters and a one-active-batch Torch
  trace are separate from ordinary throughput. Checkpoint time is recorded even
  in ordinary mode; do not sum nested stages or treat enqueue time as GPU compute.

Local validation: **56 focused tests passed; integration 267 passed, 4 skipped**.
After final queue/metadata review fixes, **45 affected-module tests passed**.
Commands, source hashes and results are in [validation.json](rental4080s/validation.json).
A 224px CPU diagnostic smoke completed with no validation errors and produced
cProfile plus a single active-batch Torch trace; its compact
[receipt](rental4080s/local_diagnostic_smoke.json) is retained, large local artifacts
are outside Git. CUDA-specific tests are unavailable on
this host. The multiprocessing tests require permission for `torch_shm_manager`
on macOS; an actual sandbox failure exposed and led to a fixed teardown hang.

## Target-host commands

Keep the current rental checkout and live experiment untouched. Inspect its local
diff and running processes before integrating this branch into a fresh checkout.
Retain rental-only pooling, AMP, audit-device, W&B and receipt fixes. Do not resume
an old-source checkpoint into this branch.

Read-only resource inspection (replace `12345` with the actual training PID):

```bash
cd /root/Self-Audit
python scripts/diagnose_maskfree_resources.py --pid 12345 \
  --data-root /root/Self-Audit/data/ACDC \
  --data-root '/root/Self-Audit/data/MnM/extracted/M&M' \
  --report-dir /root/Self-Audit/reports --interval 3 --count 10 \
  --output reports/maskfree150/rental4080s/resources-live.json
```

This resolves the target cgroup through procfs/mountinfo, distinguishes unknown
from unlimited, and records affinity, visible quota/cpuset upper bounds,
throttling, memory events/anon/file/pressure, RSS/PSS/swap/faults/I/O, filesystems,
disk space, shared memory, GPU identity/utilization/power and competing GPU jobs.
Hidden ancestor limits and contention cannot be inferred as allocated cores.
CPU-only competing jobs must also be inspected with `ps -eo pid,ppid,comm,pcpu,rss`.
Interpretation follows [Linux cgroup v2](https://docs.kernel.org/admin-guide/cgroup-v2.html)
and [procfs](https://www.kernel.org/doc/Documentation/filesystems/proc.txt).

With no other GPU benchmark/training process running, one bounded profiling/gate
entry point (existing Python environment; no installation):

```bash
cd /root/Self-Audit
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_maskfree_rental.py \
  --output reports/maskfree150/rental4080s/gate-production-v1
```

It runs ACDC preflight, M&Ms preflight, then 5 warm-up + 30 measured real ACDC
batches sequentially, at the locked profile and CPU audit. It records fresh
configs, command logs, process/cgroup/GPU resource trajectories and profile
artifacts. It refuses reused output, unavailable CUDA, unknown GPU competition,
or a competing compute job. `--print-plan` performs no training.

For attribution, use this same entry point with `--diagnostic` and a different
output: 1 warm-up + 3 measured batches, cProfile and one active Torch trace.
Run these sequentially. Then try `--prefetch-batches 1 --candidate-workers 2`
with another fresh output and `--reference-report <CPU-reference/profile_report.json>`.
Use 4 workers only within measured CPU/memory limits. Keep the strict matched
result visible. If host noise is large, repeat A/B/A. The old rental baseline
must be inspected before scheduling additional baseline work; compare immutable
source snapshots with the same harness and sample sequence when a new A/B is needed.

**Full run, only after reviewing the target equivalence and throughput gates:**

```bash
cd /root/Self-Audit
CUDA_VISIBLE_DEVICES=0 RUN_FULL=1 WORKSPACE=/root/Self-Audit \
ACDC_DATA_ROOT=/root/Self-Audit/data/ACDC \
MNMS_DATA_ROOT='/root/Self-Audit/data/MnM/extracted/M&M' \
AUDIT_DEVICE=cpu TIMING_MODE=production LOGGING_MODE=buffered \
DATA_CACHE_BYTES=67108864 PREFETCH_BATCHES=0 CANDIDATE_WORKERS=0 \
BATCH_SIZE=8 ACCUM_STEPS=1 TOTAL_EPOCHS=150 EPOCH_VALIDATION=1 \
MASKFREE_GPU_OVERRIDE='rental RTX 4080 SUPER; reviewed bounded gate' \
PYTHON=python bash scripts/run_maskfree_acdc_mnms.sh
```

This conservative command uses no unvalidated worker/prefetch choice. If a faster
optional setting passes the target gates, use exactly those tested settings in
a new run. Existing W&B configuration is retained. Final freeze/export and the
ACDC-completion gate before independent M&Ms remain mandatory.

## Review and orchestration provenance

Orca run `run_232f19c5c1d6`: configured Claude `opus` was discovered but failed
with organization subscription access disabled; no Opus work is claimed. Available
AGY workers performed disjoint data, candidate, diagnostics/runtime and read-only
review tasks. Root inspected and corrected worker code, including buffering,
pool-failure teardown, IPC storage bounds and resource namespace handling, and
ran the recorded tests/benchmark independently. Reviewer observations about
remaining coverage scalar copies are retained as a future measured optimization.
No remote performance or paper efficacy claim is authorized by this delivery.
