# MaskFree150 baseline evidence ledger (RTX 5070 Ti)

Status: **CPU ordinary and instrumented baseline receipts accepted (bounded
synthetic); GPU and real-data gates NOT RUN** (2026-09-15). This is the
before-side evidence ledger for paper correctness and provenance; it is not a
claim of RTX 5070 Ti throughput or clinical performance.

## Scope and evidence boundary

The user waived remote access. This macOS checkout has no CUDA device or
`nvidia-smi`, so RTX 5070 Ti, real ACDC, and real `4348dev` measurements are
**NOT RUN**. The user's separate observation of approximately **4.19 s/batch**
and **962 MiB** is retained as user-supplied context only: it is not a local
measurement, is not combined with CPU timing, and does not support a speedup,
memory, utilization, or ETA claim.

Earlier `local_baseline/profile_report.json` and every `corrected_*` profile are
diagnostic only. They used a small or synthetic cohort and/or a changing
source, instrumentation, or concurrent-test window; they are not accepted
before/after receipts and their timing must not enter a paper table.

## Immutable before source

The original immutable **32-file** source is recorded in the
[snapshot manifest](local_baseline/source_snapshot/snapshot_manifest.json):

| Identity | Value |
| --- | --- |
| Combined snapshot SHA-256 | `bee583199e1b4242efb0bc7cae5cc7261849d5a688019d1eabc042a98830327b` |
| Package combined SHA-256 | `36d2b02e32cf87206e6732e808432a8580431a4129041073f87c8ffc70366d0d` |
| Contents | FP32 templates and fixed even-grid pooling; no masks or reference inputs |
| Original `ontology.py` | `88b1a3c51c0f38d6a875a65703ab9cad59e816e57ed1e7695a220351f701c375` |
| Current `ontology.py` | `22c14e9984c7e5c4b27d3a7d6a6bd85a1c839744bc0d920656956cfb8eb2825c` |

The accepted before receipts import this snapshot through `--source-root` and
use the shared corrected synthetic fixture:

- [ordinary profile](performance_cpu/before_ordinary/profile_report.json)
- [instrumented profile](performance_cpu/before_instrumented/profile_report.json)

The current method is directly inspectable in the [FP64 observation
model](../../src/self_audit_maskfree/observation.py#L811), [direct evidence
auditor](../../src/self_audit_maskfree/auditor.py#L677), [FP32 model and
pooling path](../../src/self_audit_maskfree/models.py#L85), and
[connected-components path](../../src/self_audit_maskfree/ontology.py#L277).
The 150-epoch templates keep `amp: false` in
[`maskfree_acdc_150.yaml`](../../configs/maskfree_acdc_150.yaml#L67) and
[`maskfree_mnms_150.yaml`](../../configs/maskfree_mnms_150.yaml#L51).

## Paper-correctness evidence

These are bounded local software checks; none is a GPU or clinical result.

* Root ran
  `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src python -m pytest -q tests/test_maskfree*.py --basetemp=/private/tmp/selfaudit-root-density-final`:
  **164 passed, 2 CUDA skipped, 43.44 s**. This is the current combined
  working tree and includes a protected pre-existing edit to
  `tests/test_maskfree_export.py`; it is not a clean-clone publication count.
* Independent original/current ontology bytes matched labels and convergence/
  validity flags bitwise on **553 cases**: all 512 possible 3x3 masks (2^9),
  24 seeded 17x23 masks, 12 true 128x128 dense/sparse masks, one non-contiguous
  128x128 mask, two serpentine 33x35/128x128 non-converged masks, and
  empty/singleton cases.
* Final auditor fixtures compare outputs, detached targets, input and parameter
  gradients, post-step model tensors, and resume behavior. The same-source
  uninterrupted/resumed fixture in `tests/test_maskfree_trainer_performance.py`
  is **PASS**. Separately, before-versus-after cross-implementation checkpoint
  state equality at global step 25 is **PASS** for 15 model/optimizer/scaler,
  RNG, sampler, cursor, and counter fields; that is state equality, not a
  resume claim, and excludes source/config/run identity and timing/audit
  floats. Cross-source checkpoint migration is **NOT SUPPORTED**. Neural
  tensors remain FP32 and bitwise in the accepted fixture. FP64 observation
  reductions can differ at reduction-rounding scale (up to `1.14e-13` in the
  focused fixture), which is numeric rather than bitwise equivalence.
* The CPU connected-component certificate uses the same four-neighbour cross,
  maximum-raster-ID seed, and dilation cap `B-1`; it returns a certified result
  only when coverage is complete and otherwise falls back to the original
  synchronous recurrence. Scientific candidate, fit, score, and round budgets
  are unchanged.
* [`luna_data_validation_probe.py`](luna_data_validation_probe.py#L266) is a
  functional data-layer diagnostic: synthetic float32 NIfTI `[48,48,16]`
  reduced 16 unit reads to one cached load while values and exports stayed
  exact. Student, export, and reference objects are stubs; full
  epoch-validation throughput is **UNMEASURED**.

The method is image-only `spatial_predictive` v1 with `O_fit`/`O_select`/
`O_verify` separation, four candidates, two audit rounds, and at most five
nuisance-fit steps per candidate. No mask supervision enters generation,
selection, or training; reference evaluation is isolated until frozen
prediction evaluation.

## Accepted baseline timings

Both receipts use the same synthetic fixture, source/config/RNG contract,
five warm-up batches, and twenty measured `trainer._train_batch` batches. The
profiled full-scientific batch includes data load, neural, hypotheses, audit,
both students, and optimizer work; setup, progress, and checkpoint overhead
are reported separately. Checkpoint/event counters were zero in this measured
window (no such calls were observed), which does not mean production
filesystem logging is free. Epoch validation was enabled but not reached in
the partial epoch (25 of 128 batches); individual SSL sub-loss wall time,
real validation throughput, and GPU utilization are **UNMEASURED**. Ordinary
timing is the minimal whole-batch timer. Instrumented timing reports exclusive
stages and profiler overhead separately; its values must not be substituted
for ordinary throughput.

Benchmark metadata: HEAD
`19e45943fb87dd67e1c1b8bf5d2a1d8ed6133c5c` (dirty working tree), profiler
harness SHA-256
`2ed5496c0602bc7ba7a12c9793706b3726b87ec58b15fd1f87ddbb3760094cdf`,
Python `3.11.16`, torch `2.14`, CPU threads `8`, inter-op threads `12`.

| Mode | p50 (s) | p95 (s) | Mean (s) | Receipt |
| --- | ---: | ---: | ---: | --- |
| Ordinary | 6.7770277495 | 7.4723389934 | 6.8292987415 | [profile JSON](performance_cpu/before_ordinary/profile_report.json) |
| Instrumented | 6.8931343330 | 7.4557229187 | 6.8262708999 | [profile JSON](performance_cpu/before_instrumented/profile_report.json) |

Instrumented exclusive stage means over the 20 measured rows were. Percentages
use the instrumented full-batch mean `6.826270899947849 s`; they are shares,
not additive components.

| Stage | Mean seconds | Share of instrumented mean |
| --- | ---: | ---: |
| `audit` | 1.8389132048 | 26.9% |
| `bank_generation` | 3.8137257545 | 55.9% |
| `data_load` | 0.0840806480 | 1.2% |
| `feature_forward` | 0.0577258708 | 0.8% |
| `producer_step` | 0.5858466854 | 8.6% |
| `student_no_audit_step` | 0.1934020043 | 2.8% |
| `student_audited_step` | 0.1977399728 | 2.9% |
| `optimizers.step` | 0.0104072937 | 0.2% |

Nested stage values overlap and must not be summed into whole-batch latency.
Measured accounting (20 rows) is **640 fits**, **3,200 fit steps**, **10,664
score calls** (640 primary and 10,024 regional), **2,868/16,349 regions**, and
**40 accepted edits**; **159/160** synthetic units were semantic-unresolved.
These are workload receipts, not evidence that the synthetic cohort represents
ACDC or M&Ms.

## Unrun gates and targets

| Gate or target | Result |
| --- | --- |
| RTX 5070 Ti 50–100 real batches | **NOT RUN**; no CUDA host/data |
| Real ACDC | **NOT RUN** |
| Real `4348dev` | **NOT RUN** |
| VRAM and periodic GPU utilization | **NOT RUN**; remain null until sampled on the target host |
| Per-epoch validation throughput | **UNMEASURED**; probe uses stubs only |
| Target-GPU 1 s/batch | **UNVERIFIED** |

No CPU-to-GPU extrapolation is valid. The before/after CPU comparison, including
the ordinary p50 ratio (**2.9480586503x**, below a 3x target), is documented in
the [optimized ledger](performance_optimized_5070ti.md) and independently
gated by the [equivalence ledger](performance_equivalence.md).

## Commands

The [orchestration runbook](performance_orchestration.md#conditional-profiling-and-full-run-commands)
contains the canonical profiler commands, source-root pairing rules, bounded
CUDA profile, and the one manual full-run launcher. `--kind` is an artifact
label only; a before command must import the immutable snapshot via
`--source-root`, while an after command must import the final optimized
snapshot and supply the matching same-mode `--reference-report`.
