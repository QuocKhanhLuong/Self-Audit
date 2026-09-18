# DFC cardiac benchmark re-audit

Audit date: 2026-09-18.  This was a read-only code/test audit.  No real images, GT, masks, or GT-derived metrics were accessed.

## Gate and upstream evidence

The checked revision is `93fd8236fb40b6a6283eaf31385a973f186d5bb2`, equal to `origin/main`; the initial worktree was clean.  The frozen benchmark validator passed as `cardiac-benchmark-v1-2200633e7f727d37`; the real ACDC/M&Ms image-only manifests remain deferred because roots are unavailable.

The local source identifies the upstream reference as Kanezaki's `pytorch-unsupervised-segmentation-tip`.  `baseline/DFC/src/cardiac_benchmark/provenance.py:13-16` records `UPSTREAM_DFC_SHA = 181318ad40dbfb5c0add8580a05c30e5e3a7ad58`.  The source tree is not a separate Git checkout/submodule, so this audit verified parity with checked-in `baseline/DFC/demo.py`, not an external byte-for-byte checkout at that SHA.  External upstream identity is **UNRESOLVED**; local reference parity is **READY LOCAL**.

## Architecture and direct-fitting fidelity

`MyNet` in `baseline/DFC/src/cardiac_benchmark/dfc_runner.py:42-56` is a direct parameterized transcription of `demo.py:43-66`:

* Conv(input -> 100, 3x3, pad 1) -> ReLU -> BN;
* with `nConv=2`, one Conv(100 -> 100, 3x3, pad 1) -> ReLU -> BN;
* final Conv(100 -> 100, 1x1) -> BN.

The local tests report/verify 101,800 parameters for the one-channel primary model and 103,600 for a supported three-channel variant.  The primary immutable configuration is `baseline/DFC/config/cardiac/dfc_direct_2d_minl3.yaml` and `DFCConfig` (`dfc_runner.py:13-39`): input 1, nChannel 100, nConv 2, maxIter 1000, minLabels 3, SGD lr 0.1/momentum 0.9/no weight decay, loss weights both 1, no scheduler, no AMP, float32, train-mode fresh final forward.

`run_dfc` (`dfc_runner.py:89-115`) creates a fresh CPU model, fresh BN state and fresh SGD optimizer on every call immediately after applying the sample seed.  It accepts no checkpoint or prior model state.  There is no model/optimizer/BN/previous-frame carry-over in this callable path.  This preserves per-image transductive direct fitting.

Each update performs forward -> final-BN response -> per-pixel argmax pseudo-label -> mean CE -> independently mean-reduced vertical and horizontal absolute-difference terms -> backward -> `optimizer.step()`.  It counts *pre-update* active labels, then evaluates the stop condition after the update (`dfc_runner.py:99-109`).  The final map comes from a fresh, graph-enabled, train-mode post-update forward (`dfc_runner.py:110-114`).  `minLabels=3` is a stop threshold, not a command to produce exactly three labels; `DFCResult` records final active labels separately.  There is no KMeans.

The per-sample seed is stable: `derive_sample_seed` (`provenance.py:40-52`) canonicalizes `["dfc-sample-seed-v1", 42, sample_id]`, SHA-256 hashes UTF-8 compact JSON, takes the first eight digest bytes big-endian, and masks to 63 bits.  It is independent of path, manifest-row order and scheduling provided callers pass the shared canonical sample ID.  `seed_everything` seeds Python, NumPy, Torch and CUDA immediately before model construction (`provenance.py:55-61`).

## Shared data, primary mode and normalization

`load_primary_2d` (`baseline/DFC/src/cardiac_benchmark/dataset.py:41-61`) obtains the shared canonical context then reads only item 1, so DFC primary has a genuinely central-only input.  It uses float32, p0.5/p99.5 clipping, float64 population mean/std (`ddof=0`), a `max(std, 1e-6)` denominator, then `resize_values_to_grid`.  The shared routine uses FreeMask masked-area normalized resampling over full FOV, yielding the frozen 224x224 grid.

The exact order is **decode shared context -> select central slice -> percentile clip -> population z-score -> shared masked spatial resampling -> `[1,1,224,224]`**.  This is appropriately distinct from CUTS normalization and has no neighbour-slice leakage.

There is one literal discrepancy with the stated intended normalization recipe: `normalize_central_slice` (`dataset.py:20-29`) does not set `hi = lo + 1e-6` when `hi <= lo` before clipping.  A constant image still produces zeros because the population-standard-deviation floor is applied, but the code is not the declared exact algorithm.  Classification: **SCIENTIFIC DEVIATION (minor but must be resolved/frozen)**.

`cardiac_benchmark.manifest.validate_scientific_run` is only a thin common-validator wrapper (`manifest.py:19-21`).  The local `make_fixture_manifest` is explicitly fixture-only (`manifest.py:24-76`).  There is no DFC-local split construction, ED/ES restriction, foreground/mask eligibility filter, or real scientific manifest consumer loop.  The data primitive therefore has shared-contract parity, but a complete scientific execution entry point is absent.

The only registered primary is `DFC-Direct-2D-Default-MinL3`.  The parameter object could represent other `profile_id` values for tests, including three-channel input, but no declared/registered 2D-MinL4, 2.5D-MinL3 or 2.5D-MinL4 sensitivity configurations/runners were found.  They must not be presented as completed benchmark modes.

## Raw output, adapter boundary and firewall

`DFCResult.raw_cluster_map` is an anonymous contiguous little-endian int32 `[H,W]` partition (`dfc_runner.py:111-115`), accompanied in memory by update count, forward count, stop reason, pre-stop and final active cluster counts and losses.  `write_raw_artifact` (`output.py:13-23`) writes `raw_cluster_map.npy` plus shape, dtype, raw-content hash and file hash.  `create_freeze`/`validate_freeze` (`freeze.py:11-48`) provide terminal accounting and can bind a config file, manifest, code and environment identity supplied by a caller.

This is useful infrastructure but is not a complete scientific raw producer.  There is no callable/CLI that consumes every shared-manifest record, derives its seed, executes DFC, constructs the mandatory source-image/grid/config/code/runtime metadata, writes the artifact, retries/resumes via `can_resume`, and freezes the complete inventory.  Required fields can be omitted by a caller, and runtime/memory provenance is not automatically collected.  Raw partition freeze design: **NEEDS FIX**.

The DFC cardiac modules never import `shared_benchmark.adapter` or call `adapt_partition`.  There is no semantic BG/RV/MYO/LV inference in DFC itself, which is correct, but there is also no actual raw-freeze -> shared-adapter -> separate semantic artifact pipeline.  Adapter boundary: **MISSING**.

The callable cardiac path exposes only image tensors/records and no GT or semantic hints.  The common manifest validator verifies image-source identity and the shared firewall/adapter rejects GT/mask/label/oracle/Hungarian/predictive evidence.  `baseline/DFC/demo.py:17-19,75-90,138-142` has a legacy `--scribble` route, but it is outside `src/cardiac_benchmark` and is not imported by the cardiac runner.  It is **LEGACY UNUSED**, not evidence that the cardiac callable path uses GT.  The code-level firewall is **READY LOCAL**; physical GT isolation is **DEFERRED** until real roots are mounted separately.

## Tests and runtime/server readiness

Commands observed:

```text
python -m pytest -q --basetemp .pytest-dfc baseline/DFC/tests/cardiac  # 8 passed
python -m compileall baseline/DFC/src/cardiac_benchmark                # passed
```

Tests cover architecture and local-reference initialization/loss/loop parity, parameter counts, seed derivation, fixture data loading, raw artifact/hash helper, freeze validation and a bounded fresh-model reproducibility check.  They do not cover a real shared scientific manifest, repeated raw-output hashes across complete runs, worker-order/concurrent process behaviour, an inventory producer, physical GT isolation, adapter-after-freeze sequencing, or an actual resume path.

`run_dfc` exposes per-sample iteration/loss information but does not record wall-clock duration or GPU memory; it defaults to CPU unless a caller supplies a device.  There is no concurrency coordinator, worker-safe process contract, CLI, partial-output recovery loop, or server preflight that runs 50–100 real image-only slices.  No code measures a baseline `T_base` or evaluates `1.25 * T_base <= 72 hours`.  Consequently, a planned DFC runtime gate cannot currently be enforced and no runtime number is inferred by this audit.

## Verdicts

| DFC category | Verdict | Basis |
|---|---|---|
| Upstream fidelity | READY LOCAL | Checked-in wrapper is parity-tested against checked-in `demo.py`; external SHA checkout remains unresolved. |
| Shared manifest integration | READY LOCAL | Common validator/data record is used, but no production inventory consumer exists. |
| Shared-grid integration | READY LOCAL | Central 2D input calls shared masked-resize full-FOV grid. |
| GT firewall | READY LOCAL | Cardiac callable path is image-only; physical isolation deferred. |
| Raw output contract | NEEDS FIX | Good primitives, missing mandatory metadata assembly and production materializer. |
| Reproducibility | READY LOCAL | Fresh per-call state and stable canonical seed are tested; complete scientific run stability is untested. |
| Adapter boundary | MISSING | No executable handoff/persistence after raw artifact freeze. |
| Real-data readiness | BLOCKED | Real manifests/physical isolation deferred; no scientific inventory runner. |
| Runtime readiness | BLOCKED | No bounded real preflight, runtime/memory recorder, concurrency/resume controller, or T-budget gate. |

## Required work before real scientific execution

1. Materialize separately mounted image-only ACDC/M&Ms shared manifests and enforce physical GT isolation.
2. Add a production inventory runner that uses the canonical record and seed, logs runtime/memory, creates mandatory raw metadata, supports terminal partial retries/resume, and seals all records.
3. Resolve and freeze the exact constant-image percentile behaviour.
4. Add declared sensitivity configurations only if they are intended; otherwise retain the single primary profile.
5. Wire the existing shared adapter only after sealed raw-artifact validation, persist semantic and validity artifacts separately, and test the ordering.
6. Implement a 50–100 slice image-only preflight and use it to measure `T_base` before applying the 72-hour budget calculation.
7. Independently pin/attest the upstream DFC source corresponding to `181318ad...`.
