# CUTS + DFC — server handoff (ACDC static preflight)

## Scope and stop condition

Scope is CUTS and DFC on the original Self-Audit ACDC data protocol. M&Ms is
not run. The completed state is
CODE_AND_STATIC_PREFLIGHT_PASS + CUTS_STAGE1_LIMITED_PASS — WAITING_FOR_USER_DECISION.
Only the authorized CUTS one-epoch train/dev timing probe ran; full scientific
runtime is not ready.

## Source-of-truth protocol

- ACDC/training only: 100 patients and 200 ED/ES rank-3 frames.
- Checked-in seed-42 patient split: 80 train / 20 dev, no independent test.
- ED/ES comes from Info.cfg; every acquired Z slice is represented; context is
  endpoint-replicated [z-1,z,z+1].
- Self-Audit network grid: 256x256, whole FOV, no crop, value interpolation
  bilinear align_corners=False; canonical source preprocessing clips 0.5/99.5
  and volume-z-scores.
- The old FreeMask-derived 70/15/15 policy is historical compatibility only.
  It explains the former 82/18/50 manifest (50 official testing patients plus
  82/18 train/dev after renormalization), but it is not CUTS/DFC authority.

Primary line evidence is recorded in
reports/SERVER_PREFLIGHT_STATE_CUTS_DFC.md.

## Freeze and artifacts

- HEAD/origin/main at audit: 817f2ba1427506732ce6bb30bc5980e955710533.
- Current freeze: benchmark_freezes/cardiac_benchmark_v3,
  ID cardiac-benchmark-v3-9fbfcf0aee952e89.
- Scientific payload: 9fbfcf0aee952e89ada5b58a673f2144cdef0577db657dc137b6c3a05e7c374a.
- Shared grid: 57858ddf831decee0ce40c0fcc66f68b794e9c94b8f1eebe0a3a146502e535bf.
- Shared manifest logical hash:
  a334cc20bfab90d93fa05579a9c508e419297f20647dbae655156b00669935aa.
- Counts: train 80 patients/160 volumes/1,526 samples; dev 20/40/376;
  test 0/0/0.
- v1 and v2 freeze directories are retained as historical artifacts; v3 is
  generated from current Self-Audit sources and does not edit old expected
  hashes.

## Normalization boundary

Upstream CUTS has no cardiac MRI normalizer; its unrelated NIfTI loader maps
brain-tumor images per image to `[-1,1]`. Upstream DFC's direct path only does
BGR-`uint8` `/255`. Neither justifies the former cardiac adapter transforms.
Both scientific v3 loaders now use the Self-Audit source transform exactly
once: volume-wide 0.5/99.5 clipping plus population z-score. CUTS selects the
central channel (2D) or endpoint-replicated triplet (2.5D); DFC selects only the
central channel. There is no method-specific intensity fit, and the inputs are
not claimed byte-identical because channel/context selection differs.

## CUTS and DFC protocol boundary

CUTS remains CUTS-2D primary, CUTS-2.5D sensitivity, and the full scientific
protocol remains max_epochs=200 with a final-epoch checkpoint, then the
existing PHATE/KMeans path. DFC remains
DFC-Direct-2D-Default-MinL3 with maxIter=1000, MinL3, fresh model/optimizer/BN
per image. Only the shared cohort/grid/manifest authority changed; algorithms,
losses and optimization schedules were not shortened or retuned.

Both runners now reject a non-v3 Self-Audit split/grid and default their adapter
spec path to the v3 freeze. The shared manifest entrypoint defaults ACDC to
--protocol self_audit; maskfree is explicit legacy mode.

## Environment and isolation

The `.runtime/cuts_dfc_acdc_py38` venv has its own site-packages (Python
3.8.20, nibabel 5.2.1, scikit-learn 1.3.2, phate 1.0.11, `sewar`) and uses
the existing `corgs` interpreter/base packages (PyTorch 2.1.0+cu121, CUDA
12.1) read-only. No package was installed or modified in `corgs` or any
existing job environment.
An initial bwrap invocation failed before algorithm import because it mounted
only the base interpreter and hid venv-only `sewar`; no workload ran there.
The corrected venv+base read-only mapping passed the import/CUDA probe.
CUDA allocation/matmul/synchronize passed on both RTX A4000 (compute 8.6).
The bwrap image-only receipt reports 200 visible image files and zero
Info.cfg/*_gt; original workspace/data paths were absent. Source hashes were
verified inside the namespace. Original data and all unrelated processes were
preserved.

GPU snapshot at the last static probe: GPU0 free 14,645 MiB/util 0%; GPU1 free
5,961 MiB/util 100% with an unrelated resident process. Do not stop it. Disk
was near 97% used (~30 GiB free); this is capacity information for the user,
not a reason to alter scope or mark static code incomplete.

## Static test evidence

- compileall: PASS.
- shared synthetic/static tests: 55 passed.
- CUTS cardiac tests: 14 passed (one upstream scheduler-order warning).
- DFC cardiac tests: 11 passed.
- v3 freeze validator: PASS.
- selection/manifest repeatability and bwrap builder identity: PASS.
- source-normalization synthetic test: PASS.
- CUTS Stage 1 limited train+dev: PASS_RUNTIME_LIMITED in bwrap, 215.102 s;
  peak allocated 4,103,912,448 bytes (3.82 GiB), reserved 5,345,640,448 bytes
  (4.98 GiB), GPU0 RTX A4000. Checkpoint SHA-256 is
  `55a6aa889b9b9c2b011a718374a1e20689776274d3a550772dfd77bdab3776f8`.
  It records `max_epochs=1`, `scientific_run=false`, and must not be reused.
- Linear 200-epoch extrapolation from this measured train+dev pass is about
  11 h 57 min; this is extrapolation, not a scientific ETA. DFC T_base and
  1.25*T_base<=72h remain NOT_MEASURED/NOT_COMPUTABLE.
- DFC optimization, CUTS generation/PHATE, benchmark 1/10/50–100, and full
  ACDC remain NOT_RUN.

## Two-stage next decision

Stage 1 is a separately authorized limited benchmark. The CUTS-only portion
has completed: one train epoch/dev loader pass with measured time/VRAM, then
stop. The DFC pilot was not authorized. The in-bwrap paths are
`/opt/self_audit/src`, `/opt/self_audit/cuts`, `/opt/self_audit/dfc`,
`/opt/self_audit/scripts`, `/images`, `/manifest.json`, `/selection.json`, and
`/outputs`; no `data/ACDC`, `Info.cfg`, or GT is exposed. The CUTS train module
entrypoint is `/env/bin/python -c 'from cardiac_benchmark.train_stage1 import
Stage1Config,train_stage1; ...'` with `PYTHONPATH=/opt/self_audit/src:/opt/self_audit/cuts`.
Dev generation/evaluation uses
`/env/bin/python /opt/self_audit/scripts/run_cuts_scientific.py --manifest
/manifest.json --image-root /images --split dev` only with a valid scientific
checkpoint. DFC uses
`/env/bin/python /opt/self_audit/scripts/run_dfc_scientific.py --manifest
/manifest.json --image-root /images --split dev`. Any train generation is
diagnostic-only and cannot be reported as validation; GT is evaluator-only
after raw/config freeze.

Stage 2 is full ACDC (CUTS-2D 200 epochs/final checkpoint and unchanged DFC)
and requires the user's next explicit decision. Do not reuse the one-epoch
checkpoint as a scientific checkpoint. Never invent a test split or start
M&Ms.
