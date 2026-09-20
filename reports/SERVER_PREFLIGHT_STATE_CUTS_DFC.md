# CUTS + DFC — static preflight state (ACDC only)

## Verdict

CODE_AND_STATIC_PREFLIGHT_PASS + CUTS_RUNTIME_TRAINING_PASS +
CUTS_GENERATION_BENCHMARK_10_PASS; full dev generation is pending.
DFC optimization and M&Ms remain untouched.

The authorized CUTS-2D scientific run completed all 200 epochs on GPU0 and
wrote the final checkpoint. A separate image-only bwrap smoke then generated
one dev raw partition with PHATE/KMeans and one `cardiac_adapter_v1` semantic
artifact. No full 376-sample dev generation or GT evaluation has run.

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

- Current freeze: benchmark_freezes/cardiac_benchmark_v3/FREEZE_MANIFEST.json, ID cardiac-benchmark-v3-3549ca2a4564c44c, payload 3549ca2a4564c44cd8bd1d1859bcd20809e216ad5d0bffe78e46cffb0ccee8b2.
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

## Runtime evidence after CUTS authorization

- Full CUTS-2D ACDC training: `PASS_RUNTIME_FULL_TRAINING`; output
  `.runtime/cuts_acdc_200ep_20260920`, 200/200 epochs, elapsed 42,990.679 s
  (~11 h 56 min), peak allocated 4,103,912,448 bytes (3.82 GiB), peak
  reserved 5,345,640,448 bytes (4.98 GiB). The final checkpoint hash is
  `1b8c9f6edd4c2d8400dbb1373b337d66c9c93f3b3d86641655b7a04815008b13` and
  `resumed_from=null`; train/dev totals are 0.00120940 / 0.00080524.
- CUTS generation smoke: `PASS_RUNTIME_SMOKE`; session
  `self-audit-runtime` completed with exit 0 and then ended normally. The
  image-only bwrap run used one `dev` sample (patient004, frame 0001, z 0000)
  from the final checkpoint. Raw PHATE/KMeans generation took 91.158 s and
  adapter v1 took 7.636 s (total 98.841 s); per-sample CUDA peak was
  87,130,624 bytes allocated / 146,800,640 bytes reserved. Environment was
  Python 3.8.20, PyTorch 2.1.0+cu121, PHATE 1.0.11, scikit-learn 1.3.2 on
  NVIDIA RTX A4000. Output is
  `.runtime/cuts_acdc_generation_smoke_20260920` (904 KiB).
- Smoke provenance: raw partition hash
  `d7b54b9e97b08d646f21e04c5f2a4f8c5ec12a272d0610ca883127aac514f65b`,
  semantic map hash
  `67469ce0ac88c179c69adccb4782231d16eb8cce364941ba2224bbbddc162bc3`,
  validity map hash
  `57e52fe4c361eb45354ccfa7b13c3774c880738f29987015054047b2c32ae82a`.
  The run used PHATE `(n_components=3, knn=100, n_landmark=500, t=2)`,
  KMeans `K=10`, clustering seed 1, manifest hash
  `a334cc20bfab90d93fa05579a9c508e419297f20647dbae655156b00669935aa`, and
  grid hash `57858ddf831decee0ce40c0fcc66f68b794e9c94b8f1eebe0a3a146502e535bf`.
- The smoke adapter output is diagnostic only: image-only coverage was
  0.22717, with BG resolved and LV/MYO/RV unresolved on this single sample.
  This is not a validation metric; no GT path was mounted or opened.
- CUTS 10-sample generation benchmark: `PASS_RUNTIME_BENCHMARK_10`; all 10
  raw and semantic artifacts completed with exit 0 in the reused
  `self-audit-runtime` session. The deterministic `--limit 10` selection was
  patient004/frame 0001, z 0000–0009 (one volume/frame, so diversity is not
  claimed). Raw generation was 834.492 s total (83.449 s/sample mean, range
  68.702–94.105 s); adapter was 61.716 s total (6.172 s/sample mean); total
  was 896.765 s (89.677 s/sample mean, range 74.366–100.430 s). Peak CUDA
  memory was 87,130,624 bytes allocated / 146,800,640 bytes reserved.
- The 10-sample output is 8,620,015 bytes (8.5 MiB), about 0.302 GiB for
  376 samples by linear extrapolation. The corresponding linear runtime
  extrapolation is 33,718 s (~9.37 h) for 376 samples; this is an estimate,
  not a measured full-dev ETA. No GT path was mounted or opened.

## CUTS ↔ DFC protocol alignment

| Dimension | Self-Audit source | CUTS | DFC |
|---|---|---|---|
| Cohort/split | ACDC training, 80/20 patient split, no test | Consumes the same v3 manifest; stage-1 creates train/dev loaders only (baseline/CUTS/src/cardiac_benchmark/train_stage1.py:94-108) | Consumes the same v3 manifest; CLI rejects non-v3 split/grid. |
| Selection/context | ED+ES, every Z, endpoint [z-1,z,z+1] | 2D uses center channel; 2.5D keeps the shared triplet (baseline/CUTS/src/cardiac_benchmark/dataset.py:37-49,76-90) | Direct-2D takes only the center channel (baseline/DFC/src/cardiac_benchmark/dataset.py:41-60). |
| Grid/FOV | 256, whole FOV, no crop, bilinear values | shared grid is mandatory; no target override | shared grid is mandatory; no target override |
| Method normalization | volume clip + z-score in source loader | source volume clip + z-score, then central/stack selection; no CUTS-specific transform (CUTS dataset.py:17-100) | source volume clip + z-score, then central selection; no DFC-specific transform (DFC dataset.py:10-88) |
| Algorithm schedule | source schedule is not changed | full scientific CUTS remains 200 epochs and final-epoch checkpoint (train_stage1.py:25-49,166-218); the limited Stage 1 timing probe is one epoch only; PHATE/KMeans scope unchanged | MinL3 / maxIter=1000, fresh model per image remain (dfc_direct_2d_minl3.yaml:2-21) |
| Generation/evaluation | Self-Audit reports train/val diagnostics, no independent test | one image-only dev smoke is recorded; full generation/evaluation remains pending | future optimization is only from explicit manifest split; no optimization run here |

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
- CUTS cardiac suite: 15 passed (two upstream scheduler-order warnings).
- DFC cardiac suite: 11 passed.
- Synthetic source-normalization test: PASS; source-volume statistics and
  endpoint context are checked without any ACDC algorithm workload.
- v3 freeze validator: PASS.
- Self-Audit selection/manifest repeated identity: PASS.
- GT isolation/bwrap and source-hash verification: PASS_STATIC.
- Epoch-boundary checkpoint/resume synthetic test: PASS; `checkpoint_last.pt`
  binds optimizer, scheduler, RNG, loader state, manifest/grid/config identity.
- CUTS Stage 1 limited train+dev pass: PASS_RUNTIME_LIMITED; 215.102 s measured in bwrap on GPU0, peak allocated 4,103,912,448 bytes (3.82 GiB), peak reserved 5,345,640,448 bytes (4.98 GiB). Output is `.runtime/cuts_stage1_20260920_0250_retry2` and is explicitly not a scientific checkpoint.
- Full CUTS-2D scientific run: PASS_RUNTIME_FULL_TRAINING; 200/200 epochs and final `checkpoint.pt` are present in `.runtime/cuts_acdc_200ep_20260920`.
- CUTS generation smoke (1 dev sample, PHATE/KMeans + adapter): PASS_RUNTIME_SMOKE; full 376-sample dev generation and GT evaluation remain NOT_RUN.
- DFC optimization, DFC `T_base`, and M&Ms: NOT_RUN.
- M&Ms: NOT_RUN / DEFERRED.

## Runtime gates deliberately left open

| Gate | Status | Reason |
|---|---|---|
| CUTS limited benchmark (one train epoch + dev pass) | PASS_RUNTIME_LIMITED | 215.102 s measured; peak allocated 3.82 GiB / reserved 4.98 GiB; stopped after this pass. |
| DFC bounded sample pilot | NOT_RUN | Not authorized in the CUTS-only decision. |
| CUTS 200 epochs / final checkpoint | PASS_RUNTIME_FULL_TRAINING | 200/200 completed; final checkpoint and runtime receipt are present. |
| CUTS generation smoke / PHATE/KMeans | PASS_RUNTIME_SMOKE | One dev sample completed in image-only bwrap; 91.158 s raw + 7.636 s adapter, peak 83 MiB allocated / 140 MiB reserved. |
| CUTS generation benchmark (10 samples) | PASS_RUNTIME_BENCHMARK_10 | 10/10 raw + semantic complete; 896.765 s total, 89.677 s/sample mean; 8.62 MB output. |
| Full CUTS dev generation/evaluation | NOT_RUN | 10-sample benchmark only; full 376-sample generation and isolated GT evaluation await the next decision. |
| DFC 1/10/50–100 optimization and T_base | NOT_RUN / NOT_COMPUTABLE | Explicit user restriction; no T_budget=1.25×T_base≤72h claim. |
| Real peak VRAM/throughput/ETA | PARTIAL_PASS | Full training and 10-sample generation receipts are recorded; full dev generation remains an extrapolation. |
| Full ACDC | CUTS TRAINING PASS / BENCHMARK PASS / GENERATION PENDING / DFC PENDING | CUTS checkpoint and 10-sample benchmark exist; DFC still requires a separate decision. |

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

Stage 2 CUTS-2D training is complete and the final checkpoint is retained.
The one-sample smoke and the deterministic 10-sample generation benchmark
passed in the reused tmux session `self-audit-runtime`; the session ended
after exit 0. Full 376-sample dev generation and isolated GT evaluation are
still pending. DFC remains a separate, pending decision; do not mount
`data/ACDC` outside the image-only namespace and do not start M&Ms.

## Adapter v2 / active v6 static remediation (2026-09-20)

This remediation did **not** run ACDC loading, CUTS training, DFC
optimization, PHATE/KMeans, generation, GT evaluation, or M&Ms. It supersedes
only the semantic-adapter provenance of the eleven previous CUTS semantic
outputs.

| Contract | Self-Audit source | CUTS | DFC | v2 conclusion |
|---|---|---|---|---|
| Cohort/split | v3: train=1,526, dev=376, test=0 | train only; dev is validation-only | dev only when authorized | unchanged; no test split |
| Frame/slice/context | ED+ES/all-Z, endpoint [z-1,z,z+1] | 2D central channel; 2.5D triplet | central channel | unchanged |
| Normalization | `spatial.py:222-280`: full-volume 0.5/99.5 clip then population z-score | `CUTS/.../dataset.py:38-102`: no extra normalizer | `DFC/.../dataset.py:28-83`: identity after source transform | v2 central image is this normalized center plane |
| Grid/FOV | `spatial.py:283-306`: full FOV, 256x256 bilinear, `align_corners=False` | same | same | exact frozen central-image contract |
| Semantic topology | anonymous image-only partition | raw CUTS map only | raw DFC map only | unchanged topology; intensity/orientation disabled |

Changes and evidence:

- `adapter.py:35-49,148-166` keeps real reasons for assigned components
  (`unique_border_background`, `unique_enclosure_outer`,
  `unique_enclosure_inner`, `unique_adjacent_rv`) rather than the v1 fallback
  `unassigned_component`; maps, tie policy, role rules, and VOID behavior are
  unchanged.
- adapter v2 seals complete `adapter_metadata`; `artifacts.py:686-840`
  validates its digest and recomputes from verified raw partition + record +
  image-only central image + frozen spec on every semantic read. Trace/reason,
  `central_image_sha256`, coverage, and other metadata tampering is rejected
  even after a forged inner/outer re-hash.
- implementation identity includes adapter, graph, contract, artifacts,
  firewall, spatial, provenance, and `self_audit_maskfree/data/firewall.py`.
- new v2 artifacts are isolated in `semantic-cardiac_adapter_v2`; historical
  v1 artifacts remain in `semantic`, preventing overwrite.

Active identity: adapter spec SHA-256
`be5b46f1c0b7bc71e3c49117b649d43d208b42b4d237c95ea06d43de9b6ea290`;
freeze `cardiac-benchmark-v6-5a27a40b4f28f211`, payload SHA-256
`5a27a40b4f28f211041fa6a582ef5181eb773a769c19332270e76f74acc540da`.
CUTS/DFC runner defaults bind v6. v4/v5 are retained, unconsumed static
predecessors created while closing runner and non-overwrite provenance; do not
use them for runtime.

Static/synthetic checks: shared suite **60 passed** (trace, full-metadata
tamper/re-hash, dependency-cache, map fixture, freeze-mutation coverage); CUTS
**15 passed** (two existing scheduler-order warnings); DFC **11 passed**; v6
freeze validator and both runner CLI imports PASS; 200 synthetic raw-ID
bijections preserve semantic map, validity map, and scientific result hash.

The existing 200-epoch checkpoint and raw partitions were not modified. The
checkpoint `.runtime/cuts_acdc_200ep_20260920/checkpoint.pt` SHA-256 is
`1b8c9f6edd4c2d8400dbb1373b337d66c9c93f3b3d86641655b7a04815008b13`; it is
conditionally reusable only after original training and v3 input provenance
verification. The 1 smoke + 10 benchmark semantic outputs are marked
`SUPERSEDED_PENDING_AUTHORIZED_REGENERATION` in
`.runtime/cuts_adapter_v2_supersession_20260920.json`; their raw artifacts are
preserved. A future authorized handoff must verify raw seals, reconstruct the
image-only v2 central image, and use `run_adapter_after_raw`; never re-seal v1
metadata or loosen a v1 verifier.

**Status: CODE_AND_STATIC_PREFLIGHT_PASS — WAITING_FOR_USER_DECISION.** All
runtime gates for this remediation are **NOT_RUN**.

## CUTS v6 full dev generation launch (2026-09-20)

Authorized full CUTS-2D dev generation was launched in tmux
`self-audit-runtime` using the image-only bwrap namespace, v6 adapter spec,
the preserved final epoch-200 checkpoint, and no `--limit`. No GT, DFC, M&Ms,
or evaluator is mounted. Output is isolated at
`.runtime/cuts_acdc_generation_full_v6_20260920`.

The first dev sample is **RAW_COMPLETE + SEMANTIC_COMPLETE** in the distinct
`semantic-cardiac_adapter_v2` stage: raw 87.828 s, adapter 8.156 s, CUDA peak
allocated/reserved 87,130,624 / 146,800,640 bytes. The full 376-sample job is
**RUNNING**; these are startup measurements only, not a completed benchmark or
scientific result.

## DFC v6 one-sample smoke (2026-09-20)

An authorized image-only DFC dev smoke completed with `exit_status=0` in the
`dfc-smoke` window of tmux `self-audit-runtime`; CUTS remained running in its
separate window. The sample produced **RAW_COMPLETE + SEMANTIC_COMPLETE** in
`semantic-cardiac_adapter_v2`. Its unmodified primary profile ran MinL3,
`maxIter=1000`, 1,000 updates, stopped at `max_iter`, and used a fresh
model/optimizer/BN state.

Measured raw optimization was 17.691 s; adapter 4.177 s; total 21.922 s;
CUDA peak allocated/reserved was 376,495,616 / 538,968,064 bytes. A linear
one-sample extrapolation gives 2.284 h for 376 dev samples and 2.855 h for
`1.25×T_base`, below 72 h. This is a smoke-derived extrapolation, not the
50–100-sample DFC benchmark or authorization to start full DFC.

## DFC v6 full dev launch (2026-09-20)

The user separately authorized full DFC dev generation on GPU0 while the
already-authorized CUTS generation is CPU-heavy. It runs in the additional
`dfc-full-v6` window of the existing tmux session `self-audit-runtime`; CUTS
continues unchanged in window 0. The full runner has no `--limit`, selects the
frozen 376-sample dev split, makes a fresh MinL3 model/SGD/BN state per image,
and is contained in the image-only bwrap namespace. It exposes no GT,
evaluator, CUTS checkpoint, M&Ms data, or write path except
`.runtime/dfc_acdc_full_v6_20260920/{raw,semantic}`.

At the stability check, **13/376** raw artifacts and their v2 semantic
artifacts were `*_COMPLETE`, with zero failed artifact states. Across those 13
completed independent DFC optimizations, measured mean raw optimization time
was **15.485 s**, mean adapter time **3.788 s**, mean end-to-end time **19.328
s** (maximum 21.685 s). The execution receipts record CUDA peak
allocated/reserved **376,393,216 / 538,968,064 bytes** per DFC sample. This is
the per-process PyTorch measurement, not total GPU0 usage; it is low as
expected for one float32 `[1,1,256,256]` image, batch 1, and a two-convolution
network. The 13-sample linear operational extrapolation is about **2.02 h**
for 376 dev samples and **2.52 h** after the `1.25×T_base` allowance; it is an
in-progress extrapolation, not a completed scientific result. The job remains
**RUNNING** in tmux and survives a Codex disconnect.
