# MaskFree150 performance and correctness orchestration ledger

Status: **CPU ordinary and instrumented pairs accepted; GPU and real-data gates
NOT RUN** (2026-09-15). This runbook records evidence boundaries, source
identity, and gate ordering. Root owns plans, reviews, and integration; all
workers are configured **Luna Max**. This worker owns only the four performance
reports in this directory and made no source, test, data, GPU, commit, or push
change.

## Scope decision

The user explicitly waived remote access. The local macOS checkout has no CUDA
device or `nvidia-smi`; RTX 5070 Ti, real ACDC, and real `4348dev` results are
therefore **NOT RUN**. The user's separate observation (`~4.19 s/batch`,
`~962 MiB`) is context only and is never combined with CPU timing or used to
infer GPU speedup, VRAM, utilization, or ETA.

Earlier `local_baseline/profile_report.json` and `corrected_*` runs are
diagnostic only because they used a small or synthetic cohort and/or a changing
source, instrumentation, or concurrent-test window. They are not accepted
before/after evidence. The accepted receipts below use one shared corrected
fixture, explicit source roots, and one same-mode reference report per pair.

## Immutable source and method proof

The original immutable 32-file source is recorded in the
[before snapshot manifest](local_baseline/source_snapshot/snapshot_manifest.json):

| Identity | Value |
| --- | --- |
| Before combined snapshot | `bee583199e1b4242efb0bc7cae5cc7261849d5a688019d1eabc042a98830327b` |
| Before package combined | `36d2b02e32cf87206e6732e808432a8580431a4129041073f87c8ffc70366d0d` |
| After package combined | `dc19f207f4b5f3ac61a6913bb5b1e6e0fca3a81cb65ffaa4fb795d019f831dfe` |
| After snapshot combined | `46c0d90f835bb7e18abfd0e54c8540d72ca8b218e999025656939de41ccaa862` ([manifest](performance_cpu/after_density_ordinary/source_snapshot/snapshot_manifest.json)) |
| Original/current ontology | `88b1a3c51c0f38d6a875a65703ab9cad59e816e57ed1e7695a220351f701c375` / `22c14e9984c7e5c4b27d3a7d6a6bd85a1c839744bc0d920656956cfb8eb2825c` |

Method source links are the [FP64 observation model](../../src/self_audit_maskfree/observation.py#L811),
[direct auditor](../../src/self_audit_maskfree/auditor.py#L677), [FP32
neural/pooling path](../../src/self_audit_maskfree/models.py#L85), [data
discovery](../../src/self_audit_maskfree/data/discovery.py#L348), and
[profiling CLI](../../scripts/profile_maskfree.py#L2519). The method is
image-only `spatial_predictive` v1: `O_fit` for generation/fitting, adaptive
`O_select` for evidence selection, post-freeze `O_verify`, four candidates,
two rounds, and at most five fitting steps per candidate. No mask supervision
enters generation, selection, or training; reference evaluation is isolated.

## Evidence ledger

| Evidence | Accepted result | Boundary |
| --- | --- | --- |
| Current-production suite | **164 passed, 2 CUDA skipped, 43.44 s** | Combined working tree; includes protected pre-existing `tests/test_maskfree_export.py` edit, not a clean-clone publication count |
| Ontology independent comparison | **553 cases, labels/flags bitwise** | Original `88b1a3c5` vs current `22c14e99`; all 512 possible 3x3 masks (2^9), seeded rectangular, dense/sparse, non-contiguous, non-converged, empty/singleton cases |
| Auditor/model/resume | strict bitwise neural outputs/targets/gradients/states; same-source uninterrupted/resumed fixture **PASS**; separate cross-implementation checkpoint state equality (15 fields at global step 25) **PASS** | FP64 observation reduction deltas up to `1.14e-13`; cross-source migration **NOT SUPPORTED** |
| Connected components | certified 4-cross/max-raster seed, dilation cap `B-1`, exact return else original recurrence | no scientific budgets reduced |
| Data probe | synthetic `[48,48,16]` NIfTI, 16 reads vs one cached load, exact values/exports | student/export/reference stubs; epoch-validation throughput **UNMEASURED** |
| Source audit | all 17 original observation scalar functions/methods unchanged; `losses.py`, `hypotheses.py`, `config.py`, `contracts.py`, `export.py`, `runtime.py` bytewise unchanged | bounded source comparison only |
| CPU ordinary pair | **PASS** — before `6.7770277495 / 7.4723389934 / 6.8292987415 s`; after `2.2988103540 / 2.4132523857 / 2.3100720415 s`; p50 ratio **2.9480586503x** | five warm-up + 20 measured synthetic batches; 3x target **NOT MET** |
| CPU instrumented pair | **PASS** — before `6.8931343330 / 7.4557229187 / 6.8262708999 s`; after `2.2109471040 / 2.2524264274 / 2.1991256999 s`; p50 ratio **3.1177291942x** | separate instrumented window; not ordinary throughput |
| Target GPU and real data | **NOT RUN** | remote access waived; no host/data |

The current-suite command was:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src python -m pytest -q tests/test_maskfree*.py --basetemp=/private/tmp/selfaudit-root-density-final
```

The accepted raw receipts and matched reports are:

| Mode | Before | After | Match |
| --- | --- | --- | --- |
| Ordinary | [profile](performance_cpu/before_ordinary/profile_report.json) | [profile](performance_cpu/after_density_ordinary/profile_report.json) | [matched](performance_cpu/after_density_ordinary/matched_report.json) |
| Instrumented | [profile](performance_cpu/before_instrumented/profile_report.json) | [profile](performance_cpu/after_density_instrumented/profile_report.json) | [matched](performance_cpu/after_density_instrumented/matched_report.json) |

Across 20 measured rows, logical work is **640 fits**, **3,200 fit steps**,
**10,664 scores** (640 primary and 10,024 regional), **2,868/16,349 regions**,
and **40 accepted edits**; **159/160** synthetic units are semantic-unresolved.
The matched JSON marks all 44 equivalence checks true in each mode, with score
NLL max delta `3.637978807091713e-12`, gain max
`1.5543122344752192e-15`, and normalized/total max
`1.3322676295501878e-15`.

The profiled full-scientific batch is `trainer._train_batch` and includes data
load, neural, hypotheses, audit, both students, and optimizer work; setup,
progress, and checkpoint overhead are reported separately. Checkpoint/event
counters were zero in the measured window (no such calls were observed), which
does not mean production filesystem logging is free. Epoch validation was
enabled but not reached in the partial epoch (25 of 128 batches); individual
SSL sub-loss wall time, real validation throughput, and GPU utilization are
**UNMEASURED**.

Benchmark metadata: HEAD
`19e45943fb87dd67e1c1b8bf5d2a1d8ed6133c5c` (dirty working tree), profiler
harness SHA-256
`2ed5496c0602bc7ba7a12c9793706b3726b87ec58b15fd1f87ddbb3760094cdf`,
Python `3.11.16`, torch `2.14`, CPU threads `8`, inter-op threads `12`.

## Gate order

1. **CPU ordinary and instrumented pairs — PASS.** Each mode has the same
   fixture, batch declarations, model/RNG/config/source contract, and five
   warm-up plus twenty measured rows. Ordinary p50 is
   `6.7770277495 -> 2.2988103540 s` (**2.9480586503x**, below 3x); instrumented
   p50 is `6.8931343330 -> 2.2109471040 s` (**3.1177291942x**) and is not an
   ordinary-throughput claim.
2. **Independent equivalence — PASS.** Strict discrete labels, flags, order,
   thresholds, budgets, per-batch accounting, final model/RNG hashes, and
   same-source uninterrupted/resumed fixture passes. Separately, the
   before-versus-after cross-implementation checkpoint state equality passes
   for 15 model/optimizer/scaler, RNG, sampler, cursor, and counter fields at
   global step 25; that comparison excludes expected source/config/run
   identity and timing/audit floats. Neither result authorizes cross-source
   migration.
3. **Real-GPU gate — unavailable.** A verified physical UUID/name and image-only
   data root are prerequisites for a new bounded 50–100-real-batch profile and
   separate per-epoch validation measurement. GPU utilization remains null until
   periodic sampling. Target-GPU **1 s/batch is UNVERIFIED**.
4. **Manual full run — not authorized.** Run the exact sequential launcher only
   after the real-GPU gate and independent equivalence receipt are accepted.
   No automatic full run is permitted.

The after cache prepares likelihood features **once per candidate for all
regional queries**. Primary `score_many` and subsequent per-candidate
`prepare_score` each still compute likelihood, so this is not a claim that all
selection likelihood is evaluated once globally. Raw neural work is about
`1.039 s` per batch (about 47% of the after instrumented mean); instrumented
exclusive audit and bank-generation means are about `0.607 s` and `0.452 s`.
These remaining costs and duplicate setup are future measurement opportunities,
not hidden speedup attribution.

The quantitative nested bottleneck table (connected components, observation fit,
and combined primary/regional scoring with ratios and mean shares) is kept in
the [optimized ledger](performance_optimized_5070ti.md#accepted-matched-timings).

## ETA contract (not an estimate)

No new numeric ETA is defensible while target-GPU timing (`t_GPU`, hence
`t_ACDC`/`t_MNMS`), real ACDC/M&Ms batch counts, and validation/export durations
are unmeasured. Keep the formulas
visible so a future real receipt can fill them without extrapolating CPU data:

| Quantity | Formula | Current result |
| --- | --- | --- |
| New ACDC epoch | `2628 * t_ACDC / 3600 + validation/export hours` | **UNKNOWN** (`t_ACDC` and validation/export unmeasured) |
| 150 ACDC epochs | `150 * (new ACDC epoch)` | **UNKNOWN** |
| Combined ACDC + M&Ms 150-epoch run | `150 * (new ACDC epoch + N_MNMS_batches * t_MNMS / 3600 + M&Ms validation/export hours)` | **UNKNOWN** (`t_MNMS`, `N_MNMS_batches`, and validation/export unmeasured) |

The user's 4.19 s/batch observation is not substituted into these formulas;
no CPU-to-GPU or synthetic-to-real ETA follows.

## Conditional profiling and full-run commands

Inspect the live CLI/source before any future benchmark. The profiler's
`--kind` flag is an artifact label only; implementation bytes come from
`--source-root`. Every after invocation must provide the same-mode reference
report and reference snapshot.

```bash
cd /home/linhdang/workspace/quockhanh_workspace/SpecMamba
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src python scripts/profile_maskfree.py --help
```

The accepted local before ordinary command shape is:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src \
python scripts/profile_maskfree.py \
  --synthetic \
  --synthetic-root reports/maskfree150/local_baseline/corrected_fixture \
  --device cpu \
  --source-root reports/maskfree150/local_baseline/source_snapshot \
  --output reports/maskfree150/performance_cpu/before_ordinary \
  --warmup 5 --measured 20 --kind baseline --timing-mode ordinary \
  --run-id root-before-ordinary
```

The accepted local before instrumented command is identical except for its
mode, output, and run id:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src \
python scripts/profile_maskfree.py \
  --synthetic \
  --synthetic-root reports/maskfree150/local_baseline/corrected_fixture \
  --device cpu \
  --source-root reports/maskfree150/local_baseline/source_snapshot \
  --output reports/maskfree150/performance_cpu/before_instrumented \
  --warmup 5 --measured 20 --kind baseline --timing-mode instrumented \
  --run-id root-before-instrumented
```

For the accepted local after ordinary pair, use the final optimized snapshot,
the original source as reference, and the same-mode before report:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src \
python scripts/profile_maskfree.py \
  --synthetic \
  --synthetic-root reports/maskfree150/local_baseline/corrected_fixture \
  --device cpu \
  --source-root reports/maskfree150/performance_cpu/after_density_ordinary/source_snapshot \
  --reference-snapshot reports/maskfree150/local_baseline/source_snapshot \
  --reference-report reports/maskfree150/performance_cpu/before_ordinary/profile_report.json \
  --output reports/maskfree150/performance_cpu/after_density_ordinary \
  --warmup 5 --measured 20 --kind optimized --timing-mode ordinary \
  --run-id root-after-density-ordinary
```

The after instrumented command uses the same source and fixture, but references
the instrumented before report and writes the instrumented output:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src \
python scripts/profile_maskfree.py \
  --synthetic \
  --synthetic-root reports/maskfree150/local_baseline/corrected_fixture \
  --device cpu \
  --source-root reports/maskfree150/performance_cpu/after_density_ordinary/source_snapshot \
  --reference-snapshot reports/maskfree150/local_baseline/source_snapshot \
  --reference-report reports/maskfree150/performance_cpu/before_instrumented/profile_report.json \
  --output reports/maskfree150/performance_cpu/after_density_instrumented \
  --warmup 5 --measured 20 --kind optimized --timing-mode instrumented \
  --run-id root-after-density-instrumented
```

The command blocks above document historical accepted receipts. Any rerun must
use a **new output directory and unique `--run-id`** for every mode and side;
never replace the published raw JSON or its source snapshot.

If a verified CUDA host and image-only data root become available, run only a
new bounded 50–100-real-batch profile in each mode. The before GPU shape is:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-33f45df8-0bbb-fd21-48a0-af323522bf89 \
MASKFREE_GPU_UUID=GPU-33f45df8-0bbb-fd21-48a0-af323522bf89 \
MASKFREE_ALLOW_GPU_NAME="RTX 5070 Ti" \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src \
python scripts/profile_maskfree.py \
  --config <verified-resolved-config> --device cuda \
  --source-root reports/maskfree150/local_baseline/source_snapshot \
  --timing-mode ordinary --warmup 5 --measured 50 \
  --kind baseline --output <new-before-output> --run-id <unique-before-id>
```

The after GPU shape must import the final optimized snapshot and match the
same-mode before report:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-33f45df8-0bbb-fd21-48a0-af323522bf89 \
MASKFREE_GPU_UUID=GPU-33f45df8-0bbb-fd21-48a0-af323522bf89 \
MASKFREE_ALLOW_GPU_NAME="RTX 5070 Ti" \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src \
python scripts/profile_maskfree.py \
  --config <verified-resolved-config> --device cuda \
  --source-root <final-optimized-snapshot> \
  --reference-snapshot reports/maskfree150/local_baseline/source_snapshot \
  --reference-report <matching-before-ordinary-report.json> \
  --timing-mode ordinary --warmup 5 --measured 50 \
  --kind optimized --output <new-after-output> --run-id <unique-after-id>
```

Repeat the after command with `--timing-mode instrumented` and the matching
before-instrumented report. Replace angle-bracket fields only with verified
values; `--measured 100` is the largest allowed alternative. The conditional
CUDA profile is **NOT RUN** here and cProfile/nested wrappers remain
diagnostics.

The exact full sequential launcher is printed once below. It is conditional on
the real-GPU gate and independent equivalence receipt; it was **NOT RUN**:

```bash
cd /home/linhdang/workspace/quockhanh_workspace/SpecMamba
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-33f45df8-0bbb-fd21-48a0-af323522bf89 \
MASKFREE_GPU_UUID=GPU-33f45df8-0bbb-fd21-48a0-af323522bf89 \
MASKFREE_ALLOW_GPU_NAME="RTX 5070 Ti" \
RUN_FULL=1 TOTAL_EPOCHS=150 BATCH_SIZE=8 ACCUM_STEPS=1 \
EPOCH_VALIDATION=1 bash scripts/run_maskfree_acdc_mnms.sh
```

ACDC must complete and pass its own freeze/completion gate before independent
M&Ms starts; no weights, optimizer state, banks, or pseudo-label versions may
cross datasets. The current macOS checkout remains **NOT RUN** for all real
data/GPU work.
