# CUTS + DFC — static preflight state (ACDC only)

## Verdict

CODE_AND_STATIC_PREFLIGHT_PASS + CUTS_STAGE1_LIMITED_PASS — WAITING_FOR_USER_DECISION.

This status is deliberately not READY_FOR_FULL_RUN: only the authorized CUTS
one-epoch train/dev timing probe ran. No 200-epoch scientific training, DFC
optimization, CUTS generation/PHATE/KMeans, or DFC benchmark sample was run.
M&Ms is untouched; full-scientific runtime gates remain NOT_RUN.

## Source identity and protocol trace

The source of truth is the checked-in Self-Audit path, not the legacy
self_audit_maskfree discovery module:

| Protocol fact | Repository evidence | Frozen interpretation |
|---|---|---|
| Cohort | configs/self_audit_full.yaml:12-20; splits/acdc_patient_split_seed42.json:2-15 | ACDC training cohort, 100 patients / 200 ED+ES volumes; seed 42. |
| Patient split | splits/acdc_patient_split_seed42.json:4-16,98-118,120-... | Explicit 80 train / 20 val(dev), patient-disjoint; no test member. |
| Frame selection | scripts/preprocess_acdc.py:40-60 | Exactly the Info.cfg ED and ES rank-3 image frames; no arbitrary frame sampling. |
| Slice inventory | scripts/preprocess_acdc.py:82-84,110-115; src/self_audit/data/common.py:506-518 | Every acquired Z slice; foreground_only=false in configs/self_audit_full.yaml:27-30; context is [z-1,z,z+1] with endpoint replication (common.py:231-244). |
| Normalization | scripts/preprocess_acdc.py:19-25; src/self_audit/data/common.py:188-204,529-546 | One image-volume 0.5/99.5 percentile clip followed by population z-score. The source paired loader also reads masks for supervised training; the benchmark projection is image-only and never opens them. |
| Network grid | configs/self_audit_full.yaml:19-20; src/self_audit/data/common.py:247-257,556-565 | In-plane 256x256, bilinear values (align_corners=False), nearest center mask; no through-plane resampling. |
| Entrypoint/evaluation | scripts/train_self_audit.py:77-84; src/self_audit/training/_utils.py:542-558; src/self_audit/training/unified_trainer.py:951-973,123-130 | One unified config, ACDC train+val loaders only; calibration/diagnostics use val; the project explicitly records no independent test. |

src/self_audit/data/acdc.py:261-365 confirms that an explicit manifest
governs membership and rejects missing/extra cases; its fallback at
:404-425 is not used for this protocol. The image-only receipt and builder
implemented for this preflight are
scripts/build_self_audit_acdc_selection.py:30-116 and
src/shared_benchmark/self_audit_protocol.py:86-264: they bind the checked-in
split/ED-ES receipt, hash selected image bytes, and create all-Z records
without reading Info.cfg, masks, or annotations inside bwrap.

## Why 70/15/15 became 82/18/50

The old ACDC path in scripts/prepare_shared_benchmark_manifest.py imported
self_audit_maskfree.data.discovery (the retained explicit legacy branch is now
at lines 47-58). That module
declares base ratios 70/15/15 at src/self_audit_maskfree/data/discovery.py:38-43.
Its assign_splits implementation preserves the official ACDC testing folder
first, then renormalizes the remaining 100 training patients to train/dev at
discovery.py:252-313 (dev = 0.15/(0.70+0.15), remainder train). Therefore:

* 50 official testing patients stay test;
* the remaining 100 become 82 train and 18 dev after rounding;
* all frames/slices of those patients yield the previous 82/18/50 manifest.

That is a valid explanation of the old numbers, but it is not Self-Audit's
original protocol. The old v1/v2 FreeMask-derived freezes remain historical
artifacts; v3 is the only current CUTS/DFC scientific contract. The old
entrypoint now defaults ACDC to --protocol self_audit; --protocol maskfree is
explicit legacy compatibility only.

## Upstream normalization audit

The two baseline repositories do not provide a cardiac MRI normalization that
can justify the previous adapter transforms:

| Code path | Upstream behavior | v3 adapter decision |
|---|---|---|
| CUTS | `baseline/CUTS/src/datasets/brain_tumor_nifti.py:66-69` maps its unrelated brain-tumor NIfTI image per image to `[-1,1]`; `baseline/CUTS/src/model/CUTS_model.py:79-156` has no intensity transform. | Do not transplant the brain-tumor rule or the former percentile-to-unit-interval adapter. Apply the Self-Audit volume transform once, then select central channel for CUTS-2D or the endpoint stack for CUTS-2.5D. |
| DFC | `baseline/DFC/demo.py:68-70` reads BGR `uint8` and divides by `255`; it has no MRI, percentile, mean/std, crop, or geometry preprocessing. | Do not apply `/255` to float NIfTI or invent central-slice statistics. Apply the Self-Audit volume transform once, then pass only the central plane to direct 2D DFC. |

The implementation is `src/shared_benchmark/spatial.py:221-302`, used by
`baseline/CUTS/src/cardiac_benchmark/dataset.py:17-100` and
`baseline/DFC/src/cardiac_benchmark/dataset.py:10-88`. The final scientific
normalization version is
`self_audit.volume_percentile_clip_0p5_99p5_zscore.v1` for both methods;
their inputs still differ by channel selection (CUTS-2D/DFC central plane vs
CUTS-2.5D triplet). Legacy identity normalization is retained only for old
synthetic fixture manifests and is rejected by the v3 scientific runner.
No train/dev statistics are fit, and no GT/ROI values enter preprocessing.

## Grid and provenance audit

| Field | v1/v2 historical FreeMask contract | v3 Self-Audit contract | Consequence |
|---|---|---|---|
| Size | 224x224 | 256x256 | Scientific input geometry changes; this is why old v2 cannot be reused for CUTS/DFC. |
| FOV/crop | whole FOV, crop=null | whole FOV, crop=null | No crop semantics changed. |
| Forward values | masked-area normalized convolution | PyTorch bilinear, align_corners=False | Scientific interpolation semantics changed and is explicitly versioned. |
| Masks/labels | nearest; inverse probabilities bilinear-renormalized | same declarations (masks are not exposed to baseline) | No GT is mounted in this preflight. |
| Normalization | FreeMask config did not define the Self-Audit source transform | Self-Audit source is one 0.5/99.5 clip + volume z-score; CUTS/DFC add no method-specific normalizer | Shared source preprocessing is consistent; channel/context inputs still differ by baseline. |
| Split/membership | maskfree 70/15/15 + official test => 82/18/50 | checked-in ACDC training 80/20, no test; ED+ES x all Z | This is the corrected cohort. |
| Provenance | configs/maskfree_*_150.yaml, old hashes | four Self-Audit phase/unified configs, hashes in v3 grid | v3 hash 57858ddf831decee0ce40c0fcc66f68b794e9c94b8f1eebe0a3a146502e535bf. |

audit_device:auto was introduced in mask-free commit 5c31139 and changes
the mask-free trainer's resolved audit placement/backend (therefore it is not
universally “metadata only”). For CUTS/DFC, however, the runners never import
that trainer or read audit_device; the old grid loader consumed only
image_size. Thus it cannot change a CUTS/DFC benchmark run, but it remains a
real execution-identity change for the separate mask-free trainer. We did not
silently treat that difference as equivalent; we replaced the wrong source
protocol instead.

## v3 freeze, manifest and isolation evidence

- Current freeze: benchmark_freezes/cardiac_benchmark_v3/FREEZE_MANIFEST.json, ID cardiac-benchmark-v3-9fbfcf0aee952e89, payload 9fbfcf0aee952e89ada5b58a673f2144cdef0577db657dc137b6c3a05e7c374a.
- v3 validator: PASS; source and fixture regeneration checks pass.
- Shared manifest logical hash: a334cc20bfab90d93fa05579a9c508e419297f20647dbae655156b00669935aa; repeated source and builder runs reproduce the same identity. The bwrap copy has the same logical hash; only host-local receipt paths differ.
- Counts (actual materialized records):

| split | patients | ED/ES volumes | all-Z samples |
|---|---:|---:|---:|
| train | 80 | 160 | 1,526 |
| dev | 20 | 40 | 376 |
| test | 0 | 0 | 0 |

- Selection receipt: 200 image frames (160 train / 40 dev), no test field; hardlinks point to original image inodes. Image-only tree has 200 files, no Info.cfg/*_gt, and no source data was changed.
- Bwrap entrypoint test: old shared builder invoked with --protocol self_audit produced the same manifest hash. Namespace receipt PASS_BWRAP_IMAGE_ONLY: 200 visible image files, zero forbidden files, original workspace/data paths absent; /mnt/selection.json visible. The namespace binds only selected code/config, the dedicated environment, image root, receipt and output.
- Estimated manifest output: 5.7 MiB (source manifest); freeze tree about 5.9 MiB. Disk was observed near 97% used; this is recorded as a future output-capacity need only, not a reason to change cohort or block static completion.

## CUTS ↔ DFC protocol alignment

| Dimension | Self-Audit source | CUTS | DFC |
|---|---|---|---|
| Cohort/split | ACDC training, 80/20 patient split, no test | Consumes the same v3 manifest; stage-1 creates train/dev loaders only (baseline/CUTS/src/cardiac_benchmark/train_stage1.py:94-108) | Consumes the same v3 manifest; CLI rejects non-v3 split/grid. |
| Selection/context | ED+ES, every Z, endpoint [z-1,z,z+1] | 2D uses center channel; 2.5D keeps the shared triplet (baseline/CUTS/src/cardiac_benchmark/dataset.py:37-49,76-90) | Direct-2D takes only the center channel (baseline/DFC/src/cardiac_benchmark/dataset.py:41-60). |
| Grid/FOV | 256, whole FOV, no crop, bilinear values | shared grid is mandatory; no target override | shared grid is mandatory; no target override |
| Method normalization | volume clip + z-score in source loader | source volume clip + z-score, then central/stack selection; no CUTS-specific transform (CUTS dataset.py:17-100) | source volume clip + z-score, then central selection; no DFC-specific transform (DFC dataset.py:10-88) |
| Algorithm schedule | source schedule is not changed | full scientific CUTS remains 200 epochs and final-epoch checkpoint (train_stage1.py:25-49,166-218); the limited Stage 1 timing probe is one epoch only; PHATE/KMeans scope unchanged | MinL3 / maxIter=1000, fresh model per image remain (dfc_direct_2d_minl3.yaml:2-21) |
| Generation/evaluation | Self-Audit reports train/val diagnostics, no independent test | future generation is only from explicit manifest split; no generation run here | future optimization is only from explicit manifest split; no optimization run here |

## Environment/GPU static checks

- Runtime env: `.runtime/cuts_dfc_acdc_py38` has its own site-packages
  (Python 3.8.20, nibabel 5.2.1, scikit-learn 1.3.2, phate 1.0.11 and
  `sewar`); its `pyvenv.cfg` uses the existing `corgs` interpreter/base
  packages (PyTorch 2.1.0+cu121, CUDA 12.1) read-only. No package was
  installed or modified in `corgs`; existing environments/jobs were not
  changed.
- The first namespace launch failed before algorithm import because only the
  base interpreter was mounted and `sewar` from the venv site-packages was not
  visible; no data or algorithm workload ran in that attempt. The corrected
  venv+base read-only mount passed the import/CUDA probe and produced the
  limited result below.
- CUDA kernel probe passed on both RTX A4000 (compute 8.6): 16,908,288 allocated / 20,971,520 reserved bytes, matmul result 1024. Current snapshot: GPU0 free 14,645 MiB, util 0%; GPU1 free 5,961 MiB, util 100% (unrelated PID 3650019 resident). No job was stopped.
- Disk: about 30 GiB free at inspection; no output scope was reduced and no files were removed.

## Tests and gates

- compileall: PASS.
- shared synthetic/static suite: 55 passed.
- CUTS cardiac suite: 14 passed (one upstream scheduler-order warning).
- DFC cardiac suite: 11 passed.
- Synthetic source-normalization test: PASS; source-volume statistics and
  endpoint context are checked without any ACDC algorithm workload.
- v3 freeze validator: PASS.
- Self-Audit selection/manifest repeated identity: PASS.
- GT isolation/bwrap and source-hash verification: PASS_STATIC.
- CUTS Stage 1 limited train+dev pass: PASS_RUNTIME_LIMITED; 215.102 s measured in bwrap on GPU0, peak allocated 4,103,912,448 bytes (3.82 GiB), peak reserved 5,345,640,448 bytes (4.98 GiB). Output is `.runtime/cuts_stage1_20260920_0250_retry2` and is explicitly not a scientific checkpoint.
- Full ACDC train, DFC optimization, CUTS generation/PHATE/KMeans, benchmark 1/10/50–100, and DFC `T_base`: NOT_RUN.
- M&Ms: NOT_RUN / DEFERRED.

## Runtime gates deliberately left open

| Gate | Status | Reason |
|---|---|---|
| CUTS limited benchmark (one train epoch + dev pass) | PASS_RUNTIME_LIMITED | 215.102 s measured; peak allocated 3.82 GiB / reserved 4.98 GiB; stopped after this pass. |
| DFC bounded sample pilot | NOT_RUN | Not authorized in the CUTS-only decision. |
| CUTS 200 epochs / final checkpoint | NOT_RUN | Full ACDC decision gate; do not start in the limited phase. |
| CUTS generation/PHATE/KMeans | NOT_RUN | Explicit user restriction. |
| DFC 1/10/50–100 optimization and T_base | NOT_RUN / NOT_COMPUTABLE | Explicit user restriction; no T_budget=1.25×T_base≤72h claim. |
| Real peak VRAM/throughput/ETA | NOT_MEASURED | CUDA probe is not an algorithm measurement. |
| Full ACDC | WAITING_FOR_USER_DECISION | Requires explicit authorization and a new runtime plan. |

## Two-stage runtime runbook (prospective only)

Stage 1 is a separately authorized **limited benchmark**. The CUTS portion
was executed with the validated bwrap path mappings below: one train epoch and
one dev pass, then stopped. Its measured 215.102 s is an observed train+dev
time; a linear 200-epoch extrapolation is about 11 h 57 min and is not a
scientific ETA. DFC was not run under the CUTS-only authorization. Stage 1
must not launch CUTS 200 epochs.

Inside bwrap, bind only:

| In-container path | Host content | Mode |
|---|---|---|
| `/opt/self_audit/src` | `src/` | read-only |
| `/opt/self_audit/cuts` | `baseline/CUTS/src/` | read-only |
| `/opt/self_audit/dfc` | `baseline/DFC/src/` | read-only |
| `/opt/self_audit/scripts` | `scripts/run_*.py` and static configs | read-only |
| `/images` | `.runtime/acdc_self_audit_images_only` | read-only |
| `/manifest.json` | v3 manifest | read-only |
| `/selection.json` | selection receipt | read-only |
| `/outputs` | fresh stage-1 output | read-write |

The CUTS training entrypoint inside that namespace is
`/env/bin/python -c 'from cardiac_benchmark.train_stage1 import Stage1Config,train_stage1; ...'`
with `PYTHONPATH=/opt/self_audit/src:/opt/self_audit/cuts`, `manifest_path=/manifest.json`,
`image_root=/images`, `scientific_run=False`, and a one-epoch benchmark config
only. Scientific v3 training remains the separate 200-epoch entrypoint and is
not part of Stage 1. CUTS dev generation/evaluation uses
`/env/bin/python /opt/self_audit/scripts/run_cuts_scientific.py` with
`--manifest /manifest.json --image-root /images --split dev`; generation on
train, if requested for diagnostics, is labeled train-only and never reported
as validation. DFC uses
`/env/bin/python /opt/self_audit/scripts/run_dfc_scientific.py` with
`--manifest /manifest.json --image-root /images --split dev`; each image gets
its own model/optimizer/BN state. GT is mounted only in a separate evaluator
after raw outputs and mapping/config are frozen; it is never visible to either
baseline or used for cluster mapping/hyperparameter choice.

Stage 2 is **full ACDC**, including CUTS-2D 200 epochs/final checkpoint and the
full DFC protocol, and requires a new explicit user decision. The one-epoch
checkpoint above must not be reused as a scientific checkpoint. No Stage-2
command is executed or authorized by this handoff. Do not reuse this limited
runtime as a scientific result, do not mount `data/ACDC`, and do not start
M&Ms.
