# MaskFree150 optimized comparison ledger (RTX 5070 Ti)

Status: **Matched CPU ordinary and instrumented pairs accepted (bounded
synthetic); GPU and real-data gates NOT RUN** (2026-09-15). The ordinary CPU
p50 ratio is **2.9480586503x** (below a 3x target); the instrumented p50 ratio
is reported separately and is not ordinary throughput. No RTX 5070 Ti,
ACDC, or `4348dev` result is available in this checkout.

## What the accepted after result means

The optimized source combines prepared regional reductions, fixed even-grid
pooling, a bounded exact connected-component certificate, and data caching.
These are CPU implementation changes in the same method, not a second
implementation claim. The constrained observation auditor remains CPU FP64,
the neural path remains FP32, and all candidate, fit, score, round, and
physical scientific budgets are unchanged.

The cache prepares likelihood features **once per candidate for all regional
queries**. Primary `score_many` and the subsequent per-candidate
`prepare_score` each still compute likelihood; therefore this is not a claim
that all selection likelihoods are evaluated once globally. Raw neural work is
about **1.039 s** per measured batch (about 47% of the final instrumented
mean), while instrumented exclusive means leave about **0.607 s** in audit and
**0.452 s** in bank generation. These remaining costs and the duplicate
primary/regional setup are future measurement opportunities, not evidence of
another optimization.

The user's separate observation of approximately **4.19 s/batch** and **962
MiB** on an RTX 5070 Ti is context only. It is not merged with these CPU
receipts and does not support a GPU speedup, memory, utilization, or ETA claim.

## Source and provenance gate

The immutable before source is the [32-file snapshot
manifest](local_baseline/source_snapshot/snapshot_manifest.json):

| Side | Package SHA-256 | Snapshot SHA-256 / path |
| --- | --- | --- |
| Before | `36d2b02e32cf87206e6732e808432a8580431a4129041073f87c8ffc70366d0d` | `bee583199e1b4242efb0bc7cae5cc7261849d5a688019d1eabc042a98830327b` ([manifest](local_baseline/source_snapshot/snapshot_manifest.json)) |
| After | `dc19f207f4b5f3ac61a6913bb5b1e6e0fca3a81cb65ffaa4fb795d019f831dfe` | `46c0d90f835bb7e18abfd0e54c8540d72ca8b218e999025656939de41ccaa862` ([manifest](performance_cpu/after_density_ordinary/source_snapshot/snapshot_manifest.json)) |

The original/current ontology hashes are
`88b1a3c51c0f38d6a875a65703ab9cad59e816e57ed1e7695a220351f701c375` and
`22c14e9984c7e5c4b27d3a7d6a6bd85a1c839744bc0d920656956cfb8eb2825c`.
The [original ontology bytes](local_baseline/source_snapshot/files/src/self_audit_maskfree/ontology.py)
and [current implementation](../../src/self_audit_maskfree/ontology.py#L277)
are linked for review. A changed source/config/RNG identity is a new
comparison; cross-source checkpoint migration is not supported.

## Correctness evidence before timing

Root's current-production command on the combined working tree reported
**164 passed, 2 CUDA skipped, 43.44 s**:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src python -m pytest -q tests/test_maskfree*.py --basetemp=/private/tmp/selfaudit-root-density-final
```

The count includes a protected pre-existing local edit to
`tests/test_maskfree_export.py`, which is not part of this report's scope and
must not be read as a clean-clone publication count. Independent ontology
comparison matched labels and convergence/validity flags bitwise on **553
cases** (all 512 possible 3x3 masks, 24 seeded 17x23 masks, 12 true 128x128
dense/sparse masks, one non-contiguous mask, two serpentine non-converged
masks, and empty/singleton cases). Auditor checks cover outputs, detached
targets, input/parameter gradients, post-step model tensors, and resume
behavior; the same-source uninterrupted/resumed fixture and a separate
per-step neural fixture are **PASS**. Before-versus-after cross-implementation
checkpoint state equality at global step 25 is also **PASS** for 15
model/optimizer/scaler, RNG, sampler, cursor, and counter fields; this is
state equality rather than a resume claim and excludes source/config/run
identity and timing/audit floats. Cross-source migration is **NOT SUPPORTED**.
FP64 observation reductions differ only at reduction-rounding scale (up to
`1.14e-13` in the focused fixture); neural tensors remain FP32.

The connected-component fast path is a certificate with the original
four-neighbour cross, maximum-raster-ID seed, dilation cap `B-1`, and fallback
to the original synchronous recurrence whenever coverage is incomplete. The
functional data probe uses synthetic float32 NIfTI `[48,48,16]` and reduced 16
reads to one cached load with exact values/exports; student/export/reference
objects are stubs and full epoch-validation throughput is **UNMEASURED**.

## Accepted matched timings

Each mode uses five warm-up and twenty measured batches on the same corrected
synthetic fixture, source/config/RNG contract, and batch declarations. The
profiled full-scientific batch is `trainer._train_batch` and includes data
load, neural, hypotheses, audit, both students, and optimizer work; setup,
progress, and checkpoint overhead are reported separately. Checkpoint/event
counters were zero in the measured window (no such calls were observed), which
does not mean production filesystem logging is free. Epoch validation was
enabled but not reached in the partial epoch (25 of 128 batches); individual
SSL sub-loss wall time, real validation throughput, and GPU utilization are
**UNMEASURED**. The ordinary timer covers the whole scientific batch;
instrumented stage values are a separate run and include profiler-wrapper
overhead.

Benchmark metadata: HEAD
`19e45943fb87dd67e1c1b8bf5d2a1d8ed6133c5c` (dirty working tree), profiler
harness SHA-256
`2ed5496c0602bc7ba7a12c9793706b3726b87ec58b15fd1f87ddbb3760094cdf`,
Python `3.11.16`, torch `2.14`, CPU threads `8`, inter-op threads `12`.

| Mode | Before p50 / p95 / mean (s) | After p50 / p95 / mean (s) | p50 ratio | Verdict |
| --- | --- | --- | ---: | --- |
| Ordinary | 6.7770277495 / 7.4723389934 / 6.8292987415 | 2.2988103540 / 2.4132523857 / 2.3100720415 | **2.9480586503x** | Accepted bounded CPU pair; 3x target **NOT MET** |
| Instrumented | 6.8931343330 / 7.4557229187 / 6.8262708999 | 2.2109471040 / 2.2524264274 / 2.1991256999 | **3.1177291942x** | Accepted instrumentation pair; not ordinary throughput |

Raw and matched artifacts:

| Mode | Before | After | Match |
| --- | --- | --- | --- |
| Ordinary | [profile](performance_cpu/before_ordinary/profile_report.json) | [profile](performance_cpu/after_density_ordinary/profile_report.json) | [matched report](performance_cpu/after_density_ordinary/matched_report.json) |
| Instrumented | [profile](performance_cpu/before_instrumented/profile_report.json) | [profile](performance_cpu/after_density_instrumented/profile_report.json) | [matched report](performance_cpu/after_density_instrumented/matched_report.json) |

Instrumented exclusive stage means over the 20 measured rows were:

| Stage | Before (s) | After (s) | Speedup (before/after) | Share of mean (before -> after) |
| --- | ---: | ---: | ---: | ---: |
| `audit` | 1.8389132048 | 0.6066393897 | **3.031x** | 26.9% -> 27.6% |
| `bank_generation` | 3.8137257545 | 0.4520310557 | **8.437x** | 55.9% -> 20.6% |
| `data_load` | 0.0840806480 | 0.0509529772 | **1.650x** | 1.2% -> 2.3% |
| `feature_forward` | 0.0577258708 | 0.0602991479 | **0.957x** | 0.8% -> 2.7% |
| `producer_step` | 0.5858466854 | 0.5832315667 | **1.004x** | 8.6% -> 26.5% |
| `student_no_audit_step` | 0.1934020043 | 0.1979551376 | **0.977x** | 2.8% -> 9.0% |
| `student_audited_step` | 0.1977399728 | 0.1972897083 | **1.002x** | 2.9% -> 9.0% |
| `optimizers.step` | 0.0104072937 | 0.0117399083 | **0.886x** | 0.2% -> 0.5% |

Nested stages overlap; their values must not be summed into whole-batch
latency. Process-CPU means were **10.8550 s -> 6.2607 s** (ordinary) and
**10.7190 s -> 6.0259 s** (instrumented); CPU percentages were **158.9475% ->
271.0176%** and **157.0257% -> 274.0121%**, respectively. These are host
process measurements, not GPU utilization.
Ratios below one in the stage table indicate a slightly slower after stage;
they are not hidden zero or missing measurements.

The following nested measurements use the instrumented mean full-batch values
`6.826270899947849 s` (before) and `2.199125699896831 s` (after). Shares are
relative to those means, and nested values are not additive to the top-level
timer.

| Nested operation | Before (s) | After (s) | Speedup (before/after) | Share of mean (before -> after) | Accounting |
| --- | ---: | ---: | ---: | ---: | --- |
| Connected components (`CC`) | 3.95032314779819 | 0.150964373242459 | **26.167x** | 57.9% -> 6.9% | 237.85 calls/batch |
| Observation fit | 0.19003755414742 | 0.181292247955571 | **1.048x** | 2.8% -> 8.2% | 32 scalar fits -> `fit_many` 32 items |
| Primary + regional scoring | 1.021064253282384 | 0.2479158316680696 | **4.119x** | 15.0% -> 11.3% | after: `score_many` 0.0727611376089044 + `prepare_score` 0.0952828085079091 + regional reductions 0.0798718855512561; 501.2 regional calls/batch |

The primary and regional before values are not isolated wall-time partitions;
the after row reports the measured nested components explicitly. Raw neural
work is `1.0387755604169798 s` after versus `1.0347145332867513 s` before,
about 47% of the **after instrumented mean**, not of ordinary timing.

## Work accounting and equivalence

Filtering `batches[]` to `phase == "measured"` preserves the scientific work:
**640 fits**, **3,200 fit steps**, **10,664 score calls** (640 primary and
10,024 regional), **2,868/16,349 regions**, and **40 accepted edits** across
20 batches; **159/160** synthetic units were semantic-unresolved. The matched
reports mark all 44 equivalence checks true in both modes. Final model bytes
and RNG state matched; the same-source uninterrupted/resumed fixture passed,
and the separate cross-implementation checkpoint equality at global step 25
passed for the 15 declared fields. The ordinary/instrumented pair is a combined
end-to-end cache-enabled result, not an isolated connected-component or
primitive ablation.

The [equivalence ledger](performance_equivalence.md) records strict discrete
identity and numeric tolerances (including score NLL max
`3.637978807091713e-12`, gain max `1.5543122344752192e-15`, and normalized/total
max `1.3322676295501878e-15`).

## Unrun GPU and real-data gates

| Gate | Result |
| --- | --- |
| RTX 5070 Ti 50–100 real batches | **NOT RUN**; remote access waived and no CUDA host/data |
| Target-GPU 1 s/batch | **UNVERIFIED** |
| Real ACDC | **NOT RUN** |
| Real `4348dev` | **NOT RUN** |
| VRAM / periodic GPU utilization | **NOT RUN**; null until sampled on target host |
| Per-epoch validation throughput | **UNMEASURED**; functional stubs only |

No CPU result is extrapolated to GPU or to real cohorts. A full 150-epoch run
is not authorized by these local receipts.

## Commands

The [orchestration runbook](performance_orchestration.md#conditional-profiling-and-full-run-commands)
contains the exact before/after source-root pairing, same-mode reference
reports, bounded CUDA profile, and one gated full-run launcher. `--kind`
labels an artifact; it never selects implementation bytes.
