# CUTS cardiac benchmark re-audit

> Historical 224/FreeMask audit. Superseded by the Self-Audit-aligned v3
> static preflight in `reports/SERVER_PREFLIGHT_STATE_CUTS_DFC.md`.

Audit date: 2026-09-18.  Scope was executable code and local tests only; no real dataset, ground truth, or segmentation metric was opened or run.

## Gate and evidence

The checked revision is `93fd8236fb40b6a6283eaf31385a973f186d5bb2`; it equals `origin/main`, and the audit began with a clean worktree.  `python scripts/validate_cardiac_benchmark_freeze.py` passed with freeze `cardiac-benchmark-v1-2200633e7f727d37` and scientific payload `2200633e7f727d379058c11713308a4fd1dace76e8c573cb7ce4d23dba326391`.  Real ACDC and M&Ms roots/manifests remain deferred by the validator.

The frozen adapter specification hash is `34b1faeb7b40f77c2d4d9789a6e1a342e891fcb8ff8db4edd30957b2ae9d114a`, fixture hash is `cdf03cd9f2d312a5156a8ae444b116a1e2231eff8d05f575804a2b398fe6a823`, and the resolved-contract embedded shared-grid hash is `7c9d33fed0facbbabe736a5216bc599b65f1b465e3e26a3d19c624cf9be57949`.

## Reference fidelity and method

The local CUTS source is the core under `baseline/CUTS/src/`; the cardiac wrapper calls `model.CUTSEncoder`, `data_utils.extend.ExtendedDataset`, `data_utils.patch_sampler.PatchSampler`, `utils.losses.NTXentLoss`, and `utils.scheduler.LinearWarmupCosineAnnealingLR`.  No separately verifiable upstream Git remote/submodule or immutable upstream CUTS commit is recorded by the cardiac implementation.  `current_cuts_sha()` in `baseline/CUTS/src/cardiac_benchmark/provenance.py:44-46` runs `git` at `baseline/CUTS`; in this checkout it resolves the enclosing Self-Audit repository HEAD, not an independently pinned CUTS source revision.  Upstream byte-for-byte provenance is therefore **UNRESOLVED**.

The locally called Stage-1 algebra is faithful to the checked-in CUTS core:

* `CUTSEncoder` (`baseline/CUTS/src/model/CUTS_model.py:10-156`) has four replicate-padded 5x5 convolutional blocks: C->16->32->64->128, each Conv -> BatchNorm2d -> LeakyReLU.  There is no pooling, so the latent is 128 x H x W.
* `PatchRecon` (`CUTS_model.py:159-195`) is Linear(128, C*P*P), LeakyReLU, Linear(C*P*P, C*P*P), yielding `[B,S,C,P,P]`; cardiac uses P=5.
* `PatchSampler` (`data_utils/patch_sampler.py:12-125`) samples S=8 anchors inside valid patch borders, seeks a positive within a five-pixel neighbourhood for at most 20 attempts using range-aware SSIM > 0.5, then uses the specified nearest-neighbour fallback.  It explicitly has no separately sampled negatives: the other pairs in `NTXentLoss` are negatives.  Its per-call reseeding to 42 is preserved rather than changed.
* `NTXentLoss` (`utils/losses.py:6-66`) uses temperature 0.1, L2-normalized anchor/positive embeddings, in-image negatives, and a batch mean.  `train_stage1._losses` (`train_stage1.py:105-110`) combines MSE patch reconstruction and contrastive loss as `0.999 * reconstruction + 0.001 * contrastive`.

The primary recipe in `Stage1Config` and `build_optimization` (`train_stage1.py:25-50, 90-110`) is AdamW (lr 1e-3, weight decay 1e-4), 200 epochs, warmup-10 cosine scheduler, batch size 16, seed 42, and no AMP.  It trains one global model on train patients and uses dev loss for checkpoint selection (`train_stage1.py:145-174`); it is not per-image or transductive training.  CUDA is used when available, otherwise CPU.  No checkpoint resume implementation exists.

### Deviations from the original/general CUTS workflow

| Finding | Classification | Evidence and consequence |
|---|---|---|
| Shared patient manifest, image-only loader, whole-FOV 224 grid | BENCHMARK-INTEGRATION ONLY | `dataset.py:34-73`, `train_stage1.py:73-87`; required common benchmark adaptation. |
| One-channel CUTS-2D and three-channel CUTS-2.5D profiles | METHOD-PRESERVING ADAPTATION / sensitivity | `train_stage1.py:90-94`.  The 2.5D profile is not an official upstream CUTS mode and must remain sensitivity-only. |
| PHATE then KMeans K=10 raw partition | METHOD-PRESERVING ADAPTATION | `cluster_kmeans.py:19-57`: PHATE(3, knn=100, landmark=500, t=2), then KMeans(10); no K=4 path is implemented. |
| CUTS-2D normalizes a three-slice context before selecting the central channel | SCIENTIFIC DEVIATION | `dataset.py:58-62` calls `_normalize_image_only(read_context_stack(...))`, then takes `resized[1:2]`.  Neighbour intensity values alter the primary central slice scaling.  This contradicts the requested central-slice-only/no-neighbour-leakage interpretation. |
| `source_cuts_sha` is monorepo HEAD, not pinned upstream CUTS source identity | UNRESOLVED | `provenance.py:44-46`; insufficient independently attributable upstream provenance. |

## Shared data and spatial contract

The wrapper does not create a split: it reads the common manifest through `cardiac_benchmark.manifest`, and `ImageOnlyCardiacDataset` preserves record patient, split, frame, slice, canonical context, source hash, transform, grid hash and sample ID (`dataset.py:46-72`).  `Stage1Config.scientific_run=True` calls the common `validate_scientific_manifest` gate (`train_stage1.py:73-79`).  No random split, `train_test_split`, manual cardiac cohort split, ED/ES selector, foreground filter, or label-based eligibility was found in the cardiac executable path.

Spatially, the loader reads canonical `[z-1,z,z+1]`, performs CUTS-specific stack percentile [0.5,99.5] unit-interval normalization, then calls `shared_benchmark.spatial.resize_values_to_grid` (`dataset.py:26-31,58-62`).  That shared function delegates to FreeMask `masked_resize` with all-true support (`src/shared_benchmark/spatial.py:125-136`), yielding the frozen 224x224 whole FOV rather than ordinary bilinear resize.  The intended order is therefore **context decode -> CUTS normalization -> shared masked spatial resampling -> profile channel selection**.

For `CUTS-2D`, the output tensor has one central channel but has the normalisation leakage noted above.  For `CUTS-2.5D`, all three shared-context channels are retained, including endpoint replication.  There is no code that marks 2.5D as sensitivity-only in an emitted run artifact/config; that designation is presently policy/documentation, not an executable guard.

## Partition and raw-artifact path

The intended raw path is frozen encoder latent export (`export_latents.py:18-46`) -> PHATE/KMeans (`cluster_kmeans.py:19-57`) -> int64 anonymous `[H,W]` map -> raw bundle (`freeze.py:17-49`).  The core never imports `shared_benchmark.adapter`, and no semantic BG/RV/MYO/LV inference occurs in the CUTS cardiac modules.  This correctly keeps semantic logic out of Stage 1 and clustering.

However, the executable raw path is not ready:

1. A successful checkpoint write tries `manifest["freemask_source_sha"]` at `train_stage1.py:167`.  `shared_benchmark_manifest.v1` has `source_manifest`, not that top-level field.  A scientific training run will raise `KeyError` when first saving the best checkpoint.
2. Raw partition metadata tries `latent_metadata["provenance"]["image_checksum"]` at `cluster_kmeans.py:83`, whereas the common-image provenance carries `source_image_hash` (`dataset.py:67`).  The raw `.npy` is written first, but metadata writing then fails; the raw bundle cannot be scientifically sealed.
3. The metadata contains useful checkpoint, manifest, config, environment and partition hashes, but omits a reliable source-image field under the expected name and has no proven successful end-to-end producer.  The raw bundle itself hashes file contents but does not independently bind all required scientific fields.
4. `scripts_cardiac/p0_local_smoke.py` still constructs an obsolete manifest (`manifest_kind`, `freemask_source_sha`) while the current writer validates `shared_benchmark_manifest.v1`; this documented smoke command is stale and not a runnable production preflight.

Raw-output freeze design verdict: **NEEDS FIX**.  It has the correct separation and sealing primitives, but the two key mismatches block actual scientific raw artifact creation.

## GT firewall and adapter boundary

The callable cardiac data path takes source-image records only and loads via `read_context_stack`; the scientific manifest validator verifies declared image hashes.  It does not expose GT, label, Dice, IoU, Hausdorff, Hungarian, oracle, or hint inputs.  The shared firewall and adapter contract independently reject those evidence classes.  This is **READY LOCAL** for code-level image-only generation, while physical GT isolation and real roots are **DEFERRED**.

The larger CUTS checkout contains legacy/analysis code with labels and metrics (for example `baseline/CUTS/src/scripts_analysis/run_metrics.py` and legacy data helpers).  Those hits are **LEGACY UNUSED** for `baseline/CUTS/src/cardiac_benchmark` and `scripts_cardiac`; the only patch-sampler dependency is image-to-image `range_aware_ssim`, not a label metric.  They should nevertheless not be used as the cardiac entry point.

`src/shared_benchmark/adapter.py` is the only shared adapter implementation and is method-independent, but the CUTS cardiac code imports/calls neither `adapt_partition` nor an artifact-to-adapter handoff.  The required sequence `raw partition freeze -> cardiac_adapter_v1 -> distinct semantic artifact` is therefore **MISSING**, not merely untested.  No CUTS-specific semantic logic was found.

## Tests and runtime readiness

Commands observed during this audit:

```text
python -m pytest -q --basetemp .pytest-cuts baseline/CUTS/tests/cardiac/test_p0.py  # 8 passed
python -m compileall baseline/CUTS/src/cardiac_benchmark                              # passed
```

The eight tests cover small shared-manifest fixture loading, core architecture/loss checks, latent export and clustering helpers, bundle checks, and synthetic preflight-related contracts.  They do **not** execute scientific training/checkpoint persistence against a current shared manifest, raw partition metadata export, model/checkpoint resume isolation, run-to-run raw hash stability, adapter-after-freeze sequencing, real-manifest parity, or physical filesystem GT isolation.  Thus passing tests do not invalidate the blockers above.

`baseline/CUTS/src/cardiac_benchmark/preflight.py` has a synthetic batch/PHATE resource probe, including GPU memory when CUDA is available.  It is not a bounded 50–100 real-slice runner and does not measure an end-to-end checkpoint/export/partition workload.  Training can target a GPU and write a checkpoint at a caller-selected path, but has no resume/partial-output recovery orchestrator and no production CLI/config that makes the complete server invocation reproducible.

## Verdicts

| CUTS category | Verdict | Basis |
|---|---|---|
| Upstream fidelity | UNRESOLVED | Local core algebra is faithful; upstream commit/source identity is not independently pinned. |
| Shared manifest integration | READY LOCAL | Common manifest is consumed and validated in scientific mode. |
| Shared-grid integration | READY LOCAL | Shared masked-resize whole-FOV grid is called; 2D stack-normalisation leakage remains a separate deviation. |
| GT firewall | READY LOCAL | Cardiac generation path is image-only; physical isolation is deferred. |
| Raw output contract | NEEDS FIX | Checkpoint and raw metadata key mismatches block production artifacts. |
| Reproducibility | NEEDS FIX | Explicit seeds exist, but no completed artifact pipeline or demonstrated scientific hash stability. |
| Adapter boundary | MISSING | No executable raw-freeze-to-adapter invocation/persistence. |
| Real-data readiness | BLOCKED | Real manifests are deferred and raw path currently errors. |
| Runtime readiness | NEEDS FIX | Synthetic preflight exists; no end-to-end server runner, resume, or bounded real-image preflight. |

## Required work before real scientific execution

1. Materialize ACDC and M&Ms image-only shared manifests with physical GT isolation.
2. Repair the two incompatible shared-manifest/provenance field accesses and add an end-to-end non-GT scientific-artifact test.
3. Define/fix CUTS-2D preprocessing so neighbouring images cannot affect its primary central-slice values, or explicitly freeze a different primary contract.
4. Add a dedicated production runner/config with checkpoint/resume, terminal accounting, raw-hash reproducibility, and a 50–100 image-only-slice preflight.
5. Wire sealed raw partitions to the existing shared adapter, persist distinct semantic/validity outputs, and test that the adapter runs only after raw freeze.
6. Pin or otherwise attest the actual CUTS upstream source identity, not merely the enclosing monorepo revision.

## Repair and replacement-freeze evidence (2026-09-18)

The historical CUTS-2D deviation above was repaired.  The old executable order
was `read_context_stack -> normalize all three planes -> shared resize ->
select resized central channel`; this allowed neighbouring slices to change a
nominally 2D input.  `ImageOnlyCardiacDataset.__getitem__` now selects
`stack[1:2]` before `_normalize_image_only` for `CUTS-2D`, then applies the
shared whole-FOV spatial transform.  `CUTS-2.5D` retains the explicit
three-plane stack path and is still sensitivity-only by policy.

Checkpoint persistence in `train_stage1.py` now binds
`manifest_hash`, `source_manifest.logical_sha256`, `shared_grid_hash`, split
identity/seed, CUTS mode, config hash, repository identity, and benchmark seed.
Latent and raw partition metadata carry the same fields plus canonical
`provenance.source_image_hash`; the obsolete lowercase
`freemask_source_sha`/`image_checksum` accesses were removed.  The local smoke
fixture now constructs `shared_benchmark_manifest.v1` directly.  Remaining
occurrences of those two legacy strings are test assertions that verify their
absence; the uppercase `FREEMASK_SOURCE_SHA` declaration is only a legacy
reference constant and is not read by the repaired scientific path.

The new regression coverage proves bit-identical CUTS-2D tensors for radically
different neighbours, proves CUTS-2.5D remains neighbour-sensitive, validates
checkpoint provenance, and validates raw-export source hashes.  Results:
`tests/shared_benchmark` 38 passed, CUTS cardiac tests 11 passed, DFC cardiac
tests 8 passed.  The old freeze was intentionally stale after the repair; its
validator reached the changed `baseline/CUTS/src/cardiac_benchmark/dataset.py`
binding.  It was then regenerated in place as
`cardiac-benchmark-v1-56969c44eba76815` (scientific payload
`56969c44eba76815c911ad9050049c4799f5e5b2cb31c907e9a2fb8f7d427a30`).
Adapter, fixture, and shared-grid hashes remain unchanged.  Raw artifact
orchestration, executable adapter handoff, real manifests, and physical GT
isolation remain unresolved.
