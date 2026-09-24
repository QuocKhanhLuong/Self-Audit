# CUTS + DFC — preflight plan after Self-Audit protocol correction

This plan covers ACDC only. M&Ms remains deferred. The authorized CUTS-only
Stage 1 limited timing probe and the full CUTS-2D 200-epoch training have
completed. A one-sample image-only generation smoke and a 10-sample generation
benchmark also passed in tmux `self-audit-runtime`; full dev
generation/evaluation and DFC remain gated separately.

## Current authority

1. Read reports/SERVER_HANDOFF_CUTS_DFC.md and this state file.
2. Preserve the dirty worktree and all unrelated jobs.
3. Validate benchmark_freezes/cardiac_benchmark_v3 with
   scripts/validate_cardiac_benchmark_freeze.py.
4. Use the checked-in Self-Audit source identity:
   configs/self_audit_full.yaml, splits/acdc_patient_split_seed42.json,
   scripts/preprocess_acdc.py, src/self_audit/data/acdc.py, and
   src/self_audit/data/common.py. Do not infer membership from
   self_audit_maskfree.

## Static gates already completed

- Self-Audit source trace and 70/15/15 versus 82/18/50 explanation.
- v3 grid: 256x256, whole-FOV, crop=null, PyTorch bilinear align_corners=False,
  source config hashes bound; old v1/v2 contracts retained as history.
- Selection receipt: ACDC/training, seed 42, 80 train / 20 dev patients, ED+ES
  only, no test.
- Image-only hardlink tree and bwrap namespace: no Info.cfg, *_gt, original
  workspace/data path, or annotation locator.
- Shared manifest builder: source-hash validation and counts 1,526/376/0
  records (train/dev/test), 80/20/0 patients.
- Dependency/CUDA probes, compileall, synthetic shared/CUTS/DFC tests and v3
  freeze validator.
- CUTS epoch-boundary checkpoint/resume guard: `checkpoint_last.pt` is written
  after each completed epoch and final export remains epoch-200-only.
- Upstream normalization audit: CUTS has no cardiac normalizer (its unrelated
  brain-tumor NIfTI loader uses per-image `[-1,1]`), while direct DFC only uses
  BGR-`uint8` `/255`. v3 therefore applies the Self-Audit source volume
  transform once and adds no method-specific intensity transform.

## Before any runtime authorization

- Recheck Git HEAD/origin and worktree; do not checkout, reset, stash, merge,
  or overwrite current user changes.
- Recheck GPU processes/utilization and choose only an authorized free device;
  do not stop or reconfigure the resident job.
- Recheck image-only mount namespace and source-hash receipt. Never expose
  data/ACDC or any GT path to a baseline process.
- Check output capacity and record it. Disk pressure must not change cohort,
  split, or scientific scope; ask the user for an output location if runtime
  artifacts do not fit.

## Two-stage prospective runtime runbook (not run)

The two stages are separately authorized. Stage 1 is a limited benchmark that
reports measured time/peak VRAM and then stops; Stage 2 is full ACDC only after
the user's next decision. CUTS 200 epochs is not part of Stage 1.

### Stage 1 — limited benchmark (CUTS completed; DFC pending)

Use a fresh output root, the v3 manifest, and the validated image-only bwrap
mapping: `/opt/self_audit/src` ← `src/`, `/opt/self_audit/cuts` ←
`baseline/CUTS/src/`, `/opt/self_audit/dfc` ← `baseline/DFC/src/`,
`/opt/self_audit/scripts` ← the two runners/configs, `/images` ←
`.runtime/acdc_self_audit_images_only`, `/manifest.json` ← v3 manifest,
`/selection.json` ← selection receipt, and `/outputs` read-write. No
`data/ACDC`, `Info.cfg`, or GT path is mounted.

Inside bwrap, the limited CUTS train entrypoint is:

    /env/bin/python -c 'from cardiac_benchmark.train_stage1 import Stage1Config, train_stage1; train_stage1(Stage1Config(profile="CUTS-2D", dataset="acdc", manifest_path="/manifest.json", image_root="/images", scientific_run=False, max_epochs=1), "/outputs/cuts_limited/checkpoint.pt", device="cuda")'

with `PYTHONPATH=/opt/self_audit/src:/opt/self_audit/cuts`. This is a timing
probe only; it cannot produce the scientific 200-epoch checkpoint. The CUTS
dev generation/evaluation entrypoint is:

    /env/bin/python /opt/self_audit/scripts/run_cuts_scientific.py --manifest /manifest.json --image-root /images --output-root /outputs/cuts_dev --checkpoint /outputs/cuts_limited/checkpoint.pt --split dev --mode 2d --device cuda

The command above is only valid after a provenance-qualified scientific
checkpoint exists; otherwise dev generation remains `NOT_RUN`. If train
generation is used for diagnostics, label it `train_diagnostic` and never
report its metrics as validation. DFC's bounded dev pilot is:

    /env/bin/python /opt/self_audit/scripts/run_dfc_scientific.py --manifest /manifest.json --image-root /images --output-root /outputs/dfc_dev --split dev --device cuda

The pilot keeps MinL3, `maxIter=1000`, and fresh model/optimizer/BN per image.
The CUTS timing result is 215.102 s for train+dev, with peak allocated 3.82
GiB and reserved 4.98 GiB; a linear 200-epoch extrapolation is about 11 h 57
min. These are measured-versus-extrapolated values, not a scientific result.
DFC remains unrun. GT is only mounted in a separate evaluator after raw output
and mapping/config freeze; it cannot affect generation, mapping, or tuning.

### Stage 2 — full ACDC (CUTS training complete; generation/evaluation pending; DFC pending)

The unchanged CUTS-2D 200-epoch final-epoch protocol completed successfully;
the final checkpoint is in `.runtime/cuts_acdc_200ep_20260920`. The authorized
one-sample dev generation smoke and deterministic 10-sample benchmark also
completed in image-only bwrap using PHATE/KMeans and adapter v1. Full
376-sample dev generation and isolated GT evaluation have not run. DFC
`T_budget=1.25*T_base` remains unmeasured and unrun. M&Ms remains deferred.

The above commands are prospective only; they are not runtime evidence or a
readiness claim.

## Adapter-v2 prerequisite for any future runtime decision

Use the active v6 adapter spec:
`benchmark_freezes/cardiac_benchmark_v6/configs/adapter_v2_spec.json`. It
binds the actual Self-Audit runtime central image: full-volume 0.5/99.5 clip,
population z-score, then whole-FOV 256x256 bilinear resize with
`align_corners=False`. This is shared source preprocessing, not an added CUTS
or DFC normalizer; adapter intensity/orientation resolution remain disabled.

The old v1 semantic stage is preserved. v2 writes only to
`semantic-cardiac_adapter_v2`, so an authorized regeneration must verify each
raw artifact, reconstruct the image-only central plane, and call the v2
raw-to-adapter handoff. Do not re-seal old metadata. Static v6 validation and
synthetic tests passed; this remediation added no runtime workload.

## Baseline-only v10 plan (supersedes the v6 runtime commands above)

The v10 plan is still prospective. It preserves the Self-Audit cohort and
does not create a test split. The image-only bwrap namespace must expose only
the v10 source tree, baseline code, v10 manifest/freeze, read-only `/images`,
and a run-specific writable `/outputs`; no `data/ACDC`, `Info.cfg`, `*_gt`,
evaluator tree, or M&Ms path may be visible.

Before runtime, resolve the environment gate: CUTS and DFC must share one
environment and one scikit-image version for the 224 resampling. The inspected
Py3.8 CUTS env has scikit-image 0.21.0 and `sewar`; the Py3.10 DFC env has
scikit-image 0.25.2 and lacks `sewar`. This is not solved by changing the
protocol or using separate numerical preprocessors.

After a separate user authorization, the two stages remain distinct:

1. Limited benchmark: run the authorized CUTS/DFC subset with v10, measure
   wall time and peak VRAM, report measured values versus extrapolation, then
   stop. CUTS training in any timing probe must use the explicit v10
   `input_normalization` value and must not be treated as a scientific
   checkpoint. DFC remains MinL3, `maxIter=1000`, fresh model/optimizer/BN per
   image, with `T_budget=1.25×T_base` checked against 72 hours.
2. Full ACDC: only after the next user decision, start CUTS-2D from epoch 0
   for 200 epochs and retain the final epoch checkpoint; then run DFC per-image
   under the same v10 manifest. The v6 checkpoint and all v6 raw/semantic
   outputs are rejected by manifest/grid/input identity guards.

No stage was run during this static correction. Current result is
**CODE_AND_STATIC_PREFLIGHT_PASS — WAITING_FOR_USER_DECISION**, not
`READY_FOR_FULL_RUN`.

## Authorized execution now in progress

CUTS v10 has been authorized and launched in the existing
`self-audit-runtime:cuts-v10-200ep` window. The first epoch completed with a
valid epoch-boundary checkpoint, so the job is stable and may continue after
the Codex session ends. Full completion, final-checkpoint hash, peak VRAM, and
ETA are still pending. No DFC, generation, PHATE/KMeans, evaluation, or M&Ms
workload has been started by this launch.
