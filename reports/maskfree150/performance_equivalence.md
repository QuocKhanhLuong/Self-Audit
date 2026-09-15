# MaskFree150 correctness and performance-equivalence gate

Status: **CPU ordinary and instrumented matched-equivalence gates PASS on the
bounded synthetic fixture; RTX 5070 Ti and real-data gates NOT RUN**
(2026-09-15). This receipt separates exact software checks from the CPU timing
comparison and makes no GPU, clinical, novelty, or second-implementation
claim.

## Claims covered by this gate

The accepted before/after profiles use the same corrected fixture, physical and
effective batch/accumulation settings, seeds, declared batches, and source/config
contract. Before imports the immutable 32-file snapshot; after imports the
final optimized snapshot. Ordinary timing and instrumented timing are separate
windows, each with five warm-up and twenty measured batches. The profiled
full-scientific batch is `trainer._train_batch` and includes data load, neural,
hypotheses, audit, both students, and optimizer work; setup, progress, and
checkpoint overhead are reported separately. Checkpoint/event counters were
zero in the measured window (no such calls were observed), which does not mean
production filesystem logging is free. Epoch validation was enabled but not
reached in the partial epoch (25 of 128 batches); individual SSL sub-loss wall
time, real validation throughput, and GPU utilization are **UNMEASURED**.
Existing
`local_baseline/profile_report.json` and `corrected_*` profiles remain
diagnostic only and are not part of this gate.

## Source and provenance

| Side | Package SHA-256 | Snapshot SHA-256 |
| --- | --- | --- |
| Before | `36d2b02e32cf87206e6732e808432a8580431a4129041073f87c8ffc70366d0d` | `bee583199e1b4242efb0bc7cae5cc7261849d5a688019d1eabc042a98830327b` ([manifest](local_baseline/source_snapshot/snapshot_manifest.json)) |
| After | `dc19f207f4b5f3ac61a6913bb5b1e6e0fca3a81cb65ffaa4fb795d019f831dfe` | `46c0d90f835bb7e18abfd0e54c8540d72ca8b218e999025656939de41ccaa862` ([manifest](performance_cpu/after_density_ordinary/source_snapshot/snapshot_manifest.json)) |

The original/current ontology hashes are
`88b1a3c51c0f38d6a875a65703ab9cad59e816e57ed1e7695a220351f701c375` and
`22c14e9984c7e5c4b27d3a7d6a6bd85a1c839744bc0d920656956cfb8eb2825c`.
The [original ontology bytes](local_baseline/source_snapshot/files/src/self_audit_maskfree/ontology.py)
and [current implementation](../../src/self_audit_maskfree/ontology.py#L277)
are linked. Root also verified all 17 original observation scalar
functions/methods unchanged, with `losses.py`, `hypotheses.py`, `config.py`,
`contracts.py`, `export.py`, and `runtime.py` bytewise unchanged from the
original source snapshot.

## Bounded correctness evidence

* The current-production command on the combined working tree was
  `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src python -m pytest -q tests/test_maskfree*.py --basetemp=/private/tmp/selfaudit-root-density-final`:
  **164 passed, 2 CUDA skipped, 43.44 s**. The count includes a protected
  pre-existing local edit to `tests/test_maskfree_export.py`; it is not a
  clean-clone publication count.
* Independent original/current ontology comparison matched labels and
  convergence/validity flags bitwise on **553 cases**: all 512 possible 3x3
  masks (2^9), 24 seeded 17x23 masks, 12 true 128x128 dense/sparse masks, one
  non-contiguous 128x128 mask, two serpentine 33x35/128x128 non-converged
  masks, and empty/singleton cases.
* Auditor fixtures compare strict outputs, detached student targets,
  input/parameter gradients, post-step model tensors, and resume behavior. The
  same-source uninterrupted/resumed fixture and separate per-step neural
  fixture are **PASS**; neural tensors are FP32 and bitwise. Separately,
  before-versus-after cross-implementation checkpoint state equality at global
  step 25 is **PASS** for 15 model/optimizer/scaler, RNG, sampler, cursor, and
  counter fields; this is state equality rather than a resume claim and
  excludes source/config/run identity and timing/audit floats. Cross-source
  checkpoint migration is **NOT SUPPORTED**. FP64 observation reductions can
  differ only at reduction-rounding scale (up to `1.14e-13` in the focused
  fixture).
* The connected-component fast path is a certificate: four-neighbour cross,
  maximum-raster-ID seed, masked dilation through cap `B-1`, exact return only
  on complete coverage, and original synchronous recurrence otherwise. No
  candidate, fit, score, round, or physical scientific budget is reduced.
* The data-layer probe is functional only: synthetic float32 NIfTI `[48,48,16]`
  reduced 16 reads to one cached load with exact values/exports. Its
  student/export/reference objects are stubs; full epoch-validation throughput
  is **UNMEASURED**.

## Equivalence receipt

The two matched JSON reports mark all 44 checks true in their respective
timing modes: [ordinary matched report](performance_cpu/after_density_ordinary/matched_report.json)
and [instrumented matched report](performance_cpu/after_density_instrumented/matched_report.json).

Benchmark metadata for both pairs: HEAD
`19e45943fb87dd67e1c1b8bf5d2a1d8ed6133c5c` (dirty working tree), profiler
harness SHA-256
`2ed5496c0602bc7ba7a12c9793706b3726b87ec58b15fd1f87ddbb3760094cdf`,
Python `3.11.16`, torch `2.14`, CPU threads `8`, inter-op threads `12`.

| Gate | Result and evidence |
| --- | --- |
| Matched ordinary CPU timing | **PASS** — before p50/p95/mean `6.7770277495 / 7.4723389934 / 6.8292987415 s`; after `2.2988103540 / 2.4132523857 / 2.3100720415 s`; p50 ratio **2.9480586503x** |
| Matched instrumented CPU timing | **PASS** — before `6.8931343330 / 7.4557229187 / 6.8262708999 s`; after `2.2109471040 / 2.2524264274 / 2.1991256999 s`; p50 ratio **3.1177291942x**; instrumentation is not ordinary throughput |
| Neural outputs, detached targets, gradients, post-step tensors | **PASS** — strict bitwise fields in bounded fixtures; final model SHA and RNG identity match |
| FP64 fit/score values | **PASS** — score NLL max absolute delta `3.637978807091713e-12` (field tolerance `3e-10`); normalized/total max `1.3322676295501878e-15`; focused observation reduction delta up to `1.14e-13` |
| Labels, flags, order, thresholds, decisions | **PASS** — exact equality; no epsilon pruning |
| Logical and physical work | **PASS** — exact budgets and explicit logical-to-physical cache mapping |
| Same-source uninterrupted/resumed fixture | **PASS** — `tests/test_maskfree_trainer_performance.py` |
| Cross-implementation checkpoint state equality | **PASS** — 15 model/optimizer/scaler, RNG, sampler, cursor, and counter fields exact at global step 25; excludes source/config/run identity and timing/audit floats |
| Cross-source checkpoint migration | **NOT SUPPORTED** — source identity forbids migration |
| RTX 5070 Ti, real ACDC, real `4348dev` | **NOT RUN** — remote access waived; no GPU/real-data evidence |

The matched numeric maxima include audit gain `1.5543122344752192e-15` and
accepted-gain `4.440892098500626e-16`. Across 20 measured batches, accounting
is **640 fits**, **3,200 fit steps**, **10,664 score calls** (640 primary and
10,024 regional), **2,868/16,349 regions**, and **40 accepted edits**, with
**159/160** synthetic units semantic-unresolved. These counts are identical in
the accepted before/after receipts; they do not establish real-cohort quality.

The after cache prepares likelihood features once per candidate for all
regional queries, but primary `score_many` and subsequent per-candidate
`prepare_score` each compute likelihood. Therefore no claim is made that all
selection likelihood is evaluated once globally. Raw neural work is about
`1.039 s` per batch (about 47% of the after instrumented mean), with
instrumented audit and bank-generation means about `0.607 s` and `0.452 s`;
remaining duplicate setup is future work.

The full nested bottleneck table (connected components, observation fitting, and
combined primary/regional scoring with shares and ratios) is retained in the
[optimized ledger](performance_optimized_5070ti.md#accepted-matched-timings).

The exact cross-implementation state comparison excludes expected source/config/
run identity and timing/audit floats. It is not a resume claim and does not
authorize cross-source resume.

## Unrun claims and commands

The target-GPU **1 s/batch** objective is **UNVERIFIED**; the CPU ordinary 3x
target is **NOT MET** at `2.9480586503x`. GPU memory and utilization remain null
until a verified host supplies periodic samples. No CPU-to-GPU extrapolation,
clinical claim, or full 150-epoch run follows from these receipts.

Use the [orchestration runbook](performance_orchestration.md#conditional-profiling-and-full-run-commands)
for the exact before/after source-root pairing, same-mode reference reports,
bounded CUDA profile, and one gated full-run launcher. The profiler's `--kind`
option is an artifact label only; it never selects implementation bytes.
