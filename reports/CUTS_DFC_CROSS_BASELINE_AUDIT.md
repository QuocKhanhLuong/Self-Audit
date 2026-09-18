# CUTS / DFC cross-baseline cardiac benchmark audit

Audit date: 2026-09-18.  Read-only scope: executable benchmark code, frozen artifacts, and local unit/fixture tests.  No GT, labels, segmentation metrics, real training, or real scientific execution was run.

## Common benchmark gate

Repository state was clean at `93fd8236fb40b6a6283eaf31385a973f186d5bb2`, equal to `origin/main`.  `python scripts/validate_cardiac_benchmark_freeze.py` passed:

| Frozen item | Observed identity |
|---|---|
| Freeze ID | `cardiac-benchmark-v1-2200633e7f727d37` |
| Scientific payload | `2200633e7f727d379058c11713308a4fd1dace76e8c573cb7ce4d23dba326391` |
| Adapter spec | `34b1faeb7b40f77c2d4d9789a6e1a342e891fcb8ff8db4edd30957b2ae9d114a` |
| Synthetic fixtures | `cdf03cd9f2d312a5156a8ae444b116a1e2231eff8d05f575804a2b398fe6a823` |
| Shared grid | `7c9d33fed0facbbabe736a5216bc599b65f1b465e3e26a3d19c624cf9be57949` (embedded in `resolved_shared_contract.json`) |

The validator correctly reports ACDC and M&Ms manifests / physical image-only roots as deferred.  It does not establish real-data readiness.

## Shared identity, inventory, and spatial parity

`src/shared_benchmark/manifest.py:28-194` creates canonical sample IDs, preserves patient-level common splits, sorts lexicographically, enforces endpoint-replicated context, and records `generation_membership = all_authoritatively_discovered_frames_x_all_z_slices`.  `validate_manifest` (`manifest.py:222-263`) validates that inventory and each record's fields.  No baseline-specific reinterpretation of patient, split, frame, target Z or target grid was found in either cardiac loader.

| Contract item | CUTS | DFC |
|---|---|---|
| Source record | `ImageOnlyCardiacDataset` consumes a shared-manifest record (`baseline/CUTS/src/cardiac_benchmark/dataset.py:34-73`) | `load_primary_2d(record, root)` consumes the same record (`baseline/DFC/src/cardiac_benchmark/dataset.py:41-61`) |
| Slice/context | Decodes canonical three-slice `[z-1,z,z+1]`; 2D retains centre after preprocessing; 2.5D retains all | Decodes canonical context but selects central `[1]` before preprocessing |
| Intensity preprocess | Stack p0.5/p99.5 -> [0,1] | Central p0.5/p99.5 clip -> float64 population z-score |
| Spatial transform | Shared full-FOV `resize_values_to_grid` | Same shared full-FOV `resize_values_to_grid` |
| Input channels | Primary CUTS-2D: 1; sensitivity CUTS-2.5D: 3 | Primary DFC: 1 |
| Target grid | Shared 224x224 | Shared 224x224 |
| Raw output | Intended int64 anonymous `[H,W]` PHATE/KMeans partition | int32 little-endian anonymous `[H,W]` direct-fit partition |

`src/shared_benchmark/spatial.py:125-136` calls FreeMask `masked_resize` with all-true support: both loaders use the required masked-area-normalized whole-FOV resampling, not independent bilinear resizing.  This parity is **READY LOCAL**.

One primary-contract discrepancy is material: CUTS normalizes its full three-slice stack before taking the central channel, while DFC selects and normalizes only the central slice.  Therefore CUTS-2D output values can change when a neighbouring source slice changes.  The two methods may have different allowed intensity preprocessing, but this contradicts a strict CUTS-2D “central slice only/no neighbouring-slice leakage” requirement.  CUTS needs a decision/fix before scientific execution.

## GT firewall evidence

The common scientific manifest validation verifies declared source files/hashes (`src/shared_benchmark/manifest.py:263-278`); shared `firewall.py` rejects forbidden record keys/path tokens; the semantic adapter contract has an additional recursive forbidden-evidence check (`semantic_contract.py:41-119`).  Both callable cardiac loaders access source image values through `read_context_stack`, not label/mask paths.  No GT, Dice, IoU, Hausdorff, Hungarian mapping, oracle mapping, label hint or point hint was reachable from the CUTS or DFC cardiac partition generators.

Search hits were trace-classified rather than treated as proof:

| Location / hit family | Classification | Reachability |
|---|---|---|
| CUTS analysis and legacy helper scripts with labels/metrics | LEGACY UNUSED | Not imported by `baseline/CUTS/src/cardiac_benchmark` or `scripts_cardiac` generation path. |
| CUTS patch `range_aware_ssim` | SAFE | Image-to-image similarity for positive patch selection; no GT input. |
| `baseline/DFC/demo.py --scribble` | LEGACY UNUSED | Original demo only; not called by `src/cardiac_benchmark/dfc_runner.py`. |
| Shared adapter firewall tests | SAFE TEST | Fixture checks; no GT metric executed. |

Code-level GT-free generation is **READY LOCAL**.  Physical filesystem isolation and real image-only manifests are **DEFERRED**, so the server-level firewall has not been demonstrated.  Because neither baseline has a dedicated production command, command-level prevention of an operator selecting legacy scripts is also absent.

## Adapter integration and output freezes

The sole semantic implementation is `src/shared_benchmark/adapter.py:152-216`, governed by the frozen `adapter_v1_spec.json`.  It accepts a raw partition and clean shared record, returns uint8 semantic map / bool validity map, records raw partition, source-image, manifest, adapter-spec, graph and output hashes, and contains no method-name switch.  Its fixture, permutation, topology, determinism, and firewall tests pass.

Neither `baseline/CUTS/src/cardiac_benchmark` nor `baseline/DFC/src/cardiac_benchmark` imports `adapt_partition`.  Neither persists a semantic/validity artifact after raw output is sealed.  Thus the intended ordering is designed but not implemented in either baseline:

```text
baseline partition -> raw artifact / immutable raw freeze -> shared cardiac_adapter_v1 -> separately frozen semantic artifact
```

Current actual executable paths stop at raw-partition helpers.  Adapter boundary verdict for both baselines: **MISSING**.  This is not a request to add semantic logic to either core; the needed handoff must remain shared and post-freeze.

### Raw partition freeze design

| Baseline | Verdict | Evidence |
|---|---|---|
| CUTS | NEEDS FIX | Latent metadata and raw-bundle helpers carry many hashes, but checkpoint persistence accesses nonexistent `manifest["freemask_source_sha"]` and raw metadata accesses nonexistent `provenance["image_checksum"]`; real raw artifacts cannot currently complete. |
| DFC | NEEDS FIX | Raw i4 writer, attempt ledger, freeze validator, and resume predicate exist, but no production manifest loop automatically constructs mandatory input/source/grid/config/code/runtime provenance or invokes resume. |
| Semantic output | MISSING | Adapter returns structured in-memory result, but neither baseline nor a shared persistence layer writes/binds a semantic-output freeze containing raw hash, adapter code/spec identity, map hashes, coverage, assignment and VOID reasons. |

## Test evidence and missing scientific tests

Observed commands:

```text
python -m pytest -q --basetemp .pytest-shared tests/shared_benchmark                 # 38 passed
python -m pytest -q --basetemp .pytest-cuts baseline/CUTS/tests/cardiac/test_p0.py    # 8 passed
python -m pytest -q --basetemp .pytest-dfc baseline/DFC/tests/cardiac                 # 8 passed
python scripts/validate_cardiac_benchmark_freeze.py                                   # passed
python -m compileall src/shared_benchmark baseline/CUTS/src/cardiac_benchmark baseline/DFC/src/cardiac_benchmark  # passed
git diff --check                                                                       # passed
```

The shared suite covers frozen adapter fixtures, raw-label permutation invariance, determinism and forbidden evidence.  CUTS tests cover small fixtures/core helper behavior; DFC tests cover local-reference loop parity, fresh-model behavior and output/freeze primitives.  Missing tests are material:

* current scientific-manifest consumer parity for CUTS and DFC;
* fixture-vs-scientific mode separation in complete entry points;
* repeat full-run raw partition/artifact hash stability;
* DFC process/worker-order isolation and production no-carry-over enforcement;
* CUTS checkpoint/model-state isolation plus successful current-manifest checkpoint/export/cluster run;
* enforced raw-freeze-before-adapter sequencing and semantic persistence;
* physical filesystem GT isolation;
* bounded real image-only preflight and inventory completion/recovery.

## Server / runtime readiness

CUTS has a synthetic resource probe (`baseline/CUTS/src/cardiac_benchmark/preflight.py`) and caller-selected checkpoint/output paths, but no production end-to-end CLI, resume path, complete artifact terminal ledger, or bounded real-slice preflight.  It is **NEEDS FIX** for runtime/server execution.

DFC has a per-call transductive runner and returns iteration/loss counts, but no wall-clock or GPU-memory recording, no concurrency coordinator, no production partial-output/retry controller, and no 50–100 real-slice preflight.  No code measures `T_base` or enforces the conceptual `1.25 * T_base <= 72 hours` gate.  It is **BLOCKED** for runtime/server execution; no runtime number was fabricated.

## Final readiness matrix

| Category | CUTS | DFC |
|---|---|---|
| Upstream fidelity | UNRESOLVED | READY LOCAL |
| Shared manifest integration | READY LOCAL | READY LOCAL |
| Shared grid integration | READY LOCAL (with 2D neighbour-normalisation deviation) | READY LOCAL |
| GT firewall | READY LOCAL | READY LOCAL |
| Raw output contract | NEEDS FIX | NEEDS FIX |
| Reproducibility | NEEDS FIX | READY LOCAL |
| Adapter boundary | MISSING | MISSING |
| Real-data readiness | BLOCKED | BLOCKED |
| Runtime readiness | NEEDS FIX | BLOCKED |

## Exact remaining work before scientific execution

1. Materialize ACDC/M&Ms image-only manifests, preserve their shared identities/inventory, and physically isolate GT.
2. Fix CUTS current-manifest/provenance key mismatches and decide/fix its primary central-only preprocessing.
3. Add production, non-GT runners for both baselines with complete mandatory raw provenance, terminal accounting, hash validation, retry/resume and bounded real-slice preflight.
4. Freeze raw partitions before invoking the unchanged common adapter; persist a separate semantic-output freeze and test boundary ordering.
5. Add the missing scientific/reproducibility/firewall tests listed above.
6. Establish external source identity attestations for CUTS and DFC, and measure DFC `T_base` before deciding server inventory feasibility.
