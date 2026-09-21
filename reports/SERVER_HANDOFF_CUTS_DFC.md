# CUTS + DFC — server handoff (ACDC static preflight)

## Scope and stop condition

Scope is CUTS and DFC on the original Self-Audit ACDC data protocol. M&Ms is
not run. The current state is
CODE_AND_STATIC_PREFLIGHT_PASS + CUTS_FULL_TRAINING_PASS +
CUTS_GENERATION_BENCHMARK_10_PASS; full dev generation/evaluation is pending.
DFC remains unrun.

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
  ID cardiac-benchmark-v3-3549ca2a4564c44c.
- Scientific payload: 3549ca2a4564c44cd8bd1d1859bcd20809e216ad5d0bffe78e46cffb0ccee8b2.
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
- CUTS cardiac tests: 15 passed (two upstream scheduler-order warnings).
- Epoch-boundary checkpoint/resume synthetic test: PASS; `checkpoint_last.pt`
  carries optimizer/scheduler/RNG/loader state and identity guards.
- DFC cardiac tests: 11 passed.
- v3 freeze validator: PASS.
- selection/manifest repeatability and bwrap builder identity: PASS.
- source-normalization synthetic test: PASS.
- CUTS Stage 1 limited train+dev: PASS_RUNTIME_LIMITED in bwrap, 215.102 s;
  peak allocated 4,103,912,448 bytes (3.82 GiB), reserved 5,345,640,448 bytes
  (4.98 GiB), GPU0 RTX A4000. Checkpoint SHA-256 is
  `55a6aa889b9b9c2b011a718374a1e20689776274d3a550772dfd77bdab3776f8`.
  It records `max_epochs=1`, `scientific_run=false`, and must not be reused.
- Full CUTS-2D: PASS_RUNTIME_FULL_TRAINING; 200/200 epochs completed and
  `.runtime/cuts_acdc_200ep_20260920/checkpoint.pt` is present. Elapsed time
  was 42,990.679 s; peak allocated/reserved memory was 3.82/4.98 GiB.
- CUTS generation smoke: PASS_RUNTIME_SMOKE in tmux `self-audit-runtime`,
  one dev sample only. Raw PHATE/KMeans took 91.158 s and adapter v1 took
  7.636 s; peak was 83/140 MiB allocated/reserved. The session ended after
  exit 0 and the output is `.runtime/cuts_acdc_generation_smoke_20260920`.
- CUTS 10-sample generation benchmark: PASS_RUNTIME_BENCHMARK_10 in the same
  tmux name (reused after the smoke session ended); all 10 raw and semantic
  artifacts completed. Total was 896.765 s (89.677 s/sample mean), peak was
  83/140 MiB allocated/reserved, and output size was 8,620,015 bytes. The
  deterministic limit selected patient004/frame 0001/z 0000–0009 only, so
  the 376-sample estimate (~9.37 h and ~0.302 GiB) is linear extrapolation,
  not a measured full-dev result.
- Full 376-sample dev generation, isolated GT evaluation, DFC T_base,
  DFC optimization, and M&Ms remain NOT_RUN.

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

Stage 2 CUTS-2D training is complete and the final checkpoint is retained.
The one-sample image-only generation smoke and the 10-sample benchmark passed
in tmux `self-audit-runtime`; full dev generation/evaluation remains pending.
The DFC half remains pending a separate decision. Do not reuse the one-epoch
checkpoint as a scientific checkpoint. Never invent a test split or start
M&Ms.

## Adapter/provenance handoff update — active v6

The active semantic contract is `cardiac_adapter_v2` in
`benchmark_freezes/cardiac_benchmark_v6` (freeze ID
`cardiac-benchmark-v6-5a27a40b4f28f211`). It preserves the Self-Audit v3 ACDC
protocol exactly: train=1,526, dev=376, test=0; ED+ES/all-Z; whole-FOV 256
bilinear grid. CUTS and DFC runner defaults bind this v6 spec.

The change fixes assignment-trace provenance, seals complete adapter metadata,
recomputes v2 semantic metadata from verified raw+image-only input on reads,
and closes cache identity over actual dependencies. It does not alter topology,
optimization schedules, source normalization, checkpoint/raw bytes, or GT
isolation. New artifacts use `semantic-cardiac_adapter_v2`; historical v1
artifacts remain in `semantic`.

Evidence: shared static suite 60 passed; CUTS 15 passed; DFC 11 passed; v6
validator and 200-case synthetic raw-ID permutation test passed. Eleven old
CUTS semantic outputs (1 smoke + 10 benchmark) are superseded but retained;
their raw artifacts and checkpoint provenance are recorded in
`.runtime/cuts_adapter_v2_supersession_20260920.json`. No ACDC workload ran
for this update. **CODE_AND_STATIC_PREFLIGHT_PASS — WAITING_FOR_USER_DECISION.**

## Runtime handoff — CUTS v6 full dev generation is running

The authorized job is in tmux `self-audit-runtime`, output
`.runtime/cuts_acdc_generation_full_v6_20260920`, under image-only bwrap with
adapter v2 and no limit. Its first sample completed raw+semantic v2
successfully (87.828 s + 8.156 s; 87/147 MiB allocated/reserved). The remaining
full 376-sample generation is still RUNNING; no GT evaluation, DFC, or M&Ms
was started. Closing Codex does not stop this tmux session.

## DFC smoke result

One image-only DFC v6 dev smoke completed successfully on GPU0: MinL3,
maxIter=1000, 1,000 updates, fresh model/optimizer/BN state, raw 17.691 s,
adapter 4.177 s, peak CUDA 376/539 MB allocated/reserved, and semantic v2
completion. Its linear one-sample `1.25×T_base` estimate for 376 dev samples
is 2.855 h (<72 h), but this is not yet the required multi-sample DFC budget
benchmark and full DFC was not started.

## Runtime handoff — DFC v6 full dev generation is running

The user subsequently authorized full DFC dev generation on GPU0. It is in
window `dfc-full-v6` of the existing tmux session `self-audit-runtime`; CUTS
continues in window 0. The new wrapper
`.runtime/run_dfc_full_v6_tmux.sh` is image-only bwrap: it binds only GPU0,
the v6 freeze/spec, source code, the image-only manifest/root, and its own
new raw/semantic output directories. It does not mount GT, evaluator data,
CUTS checkpoints, or M&Ms.

The runner has no `--limit` and therefore processes all 376 dev samples as
independent fresh-model MinL3 optimizations. Its initial stable receipt sample
contained 13/376 completed raw + semantic-v2 artifacts and zero failed states.
Measured means were 15.485 s DFC raw optimization, 3.788 s adapter, and
19.328 s total per image (maximum 21.685 s); peak PyTorch CUDA
allocated/reserved per DFC sample was 376,393,216 / 538,968,064 bytes. The
corresponding in-progress linear estimate is 2.02 h full dev and 2.52 h with
the 1.25× budget allowance. These measurements are operational, not a final
scientific claim. The job is RUNNING and persists after Codex disconnects.
