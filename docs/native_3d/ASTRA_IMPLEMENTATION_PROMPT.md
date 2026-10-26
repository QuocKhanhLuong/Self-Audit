# Astra implementation task: NativeStudent + EMA, from 2.5D to controlled 3D experiments

Implement this task in the Self-Audit repository. Deliver working code, opt-in configurations, meaningful tests, bounded verification receipts and runnable commands. Do not stop at a proposal, pseudocode, interface stubs or a list of future work. Implement all modules below; later research arms may stay disabled by default, but their configured paths must execute. This task authorizes implementation and bounded checks, not long patient training or a new accuracy claim.

## 1. Establish the exact starting point

- Fetch origin and read `AGENTS.md`, `ACTIVE_PIPELINE.json`, `docs/METHOD.md`, this directory's `README.md`, and the two architecture figures. Inspect current branch, HEAD, worktrees, remotes and dirty files first. Preserve other work and active server jobs.
- This handoff is on `origin/docs/native-3d-astra-handoff-20261026`, based on `fa7a8664621f5540e755e7869051fa25b3926cba` (`experiments/teacher-memory-full-20261024`). Create a separate implementation worktree/branch `feat/native-volume-ema-pilot` from the handoff branch. If that implementation branch already exists, inspect and continue it without resetting it. Never force-push, reset user work or merge into main automatically.
- The executed A0–A6 model source was based on `8252760092e5dc97e0c441bd2d3d306146dcff64`; use the completed report's `ASSET_LINEAGE.json` and source hashes to check this. Read `reports/teacher_memory_full_20261024/completed_20261025/REPORT.md` and available numeric evidence.
- The research baseline is **A0: NativeStudent + fixed-window + ema_full**, L8, batch24, crop256, FP32, 10k outer iterations, 1k supervised-only delay. Do not substitute the canonical `joint_guard` defaults or batch52 from a different recipe. Preserve A0, fixed-window, adaptive/meta-Dice and A1–A6 historical paths and artifacts.
- Hardware target: the user's PRO4000 with 24 GB VRAM and approximately 129 GB host RAM. Query the actual device, free memory and software versions; do not infer the exact GPU generation from its nickname.
- Inspect existing data paths and the configured `KhanhDrive:` rclone remote read-only if needed. Prefer existing verified assets. Do not download large data/model weights, accept new licenses, overwrite datasets, or change running jobs in this implementation task. If an asset is absent, finish synthetic/software work and report the exact missing asset and prepared retrieval command. Do not expose credentials.

## 2. Scientific and compatibility contracts

This is semi-supervised learning with dense labels from the declared TRAIN-L patient budget. No U/VAL/TEST masks may enter training, pseudo-target generation, region construction, threshold selection, sampling decisions or model selection. Audit labels remain part of L. EMA is not independent truth. External CineMA/SAM/TotalSegmentator targets and memory are disabled in every new primary arm; keep existing external-teacher arms reproducible.

Current labels and evaluation are already volumetric: a frame mask has Z×H×W support, and the metric computes volume confusion per frame, then means over frames/patients. The model currently consumes z−1,z,z+1 at fixed t and predicts center z. Do not describe 3D as repairing a proven alignment bug. Do not confuse Z with time T, spatial3D with 4D, supervoxels with semantic labels, or CT VISTA3D results with cine-MRI efficacy.

Keep `train.py` as the shared training entry point and `self_audit/` as the implementation package. New architecture/data/loss/optimizer modes must be explicit, validated and opt-in. Version configuration/checkpoint schemas where necessary; old configs, state_dicts, initialization RNG, sampler schedules and disabled-path outputs must retain their behavior. No silent migrations, fallback weights or implicit downloading.

For ordinary full EMA retain the timeline supervised update -> fixed teacher targets -> pseudo update if supported -> EMA once. A single combined-loss update is a separate explicitly named factor, not a silent refactor. Existing guarded transactions must retain their own contracts. Record optimizer and EMA update counts separately.

## 3. G0: Implement geometry and volume data contracts first

Add a versioned image-geometry sidecar/schema without contaminating U records with label references. Record source hashes, native shape/axis order, affine, units in mm, spacing, orientation, slice order and temporal frame mapping. Validate mask/frame geometry, frame/cine geometry and voxel correspondence. Reject inconsistent or missing geometry when a physical-space operation requires it; do not guess from array shape or filenames.

Implement a volume/slab loader returning image, optional labels, geometric-valid support, label-valid support, and reversible transform metadata. These masks are distinct. Unknown labels use 255, never background. Keep the existing image-only intensity normalization and original v2 loader intact. Patient splits remain patient-level, with all frames/slices from a patient in the same role.

Support deterministic contiguous slabs from a single t, native-Z and explicitly configured resampling, shared XY crop across the slab, and coupled spatial transforms. Choose/resample FOV in physical units where configured. Use appropriate continuous image/probability interpolation, nearest-neighbor categorical labels, conservative validity handling and explicit pixel-center conventions. Evaluation maps probabilities back to the native reference grid before argmax under the declared rule. Never claim resampling invents through-plane information.

## 4. Implement the architecture ladder with controls

### G1: Volume-batched 2.5D control

Run the unchanged NativeStudent on each z of a contiguous slab and stack outputs. Preserve center-slice target identity and edge-neighbor behavior. First use the legacy loss reduction so the data-grouping factor can be isolated; expose volume/frame loss as a separate switch. Log label/valid voxel exposure and patient/frame/slab IDs.

### G2: Hybrid 2D encoder and 3D decoder/fusion

Reuse the exact DINOv3 ConvNeXt-Tiny initialization and current input adapter per z. Arrange multiscale features as B×C×Z×h×w and implement real learned cross-Z residual fusion/decoding producing B×4×Z×H×W. Retain native-resolution detail support. Keep fixed-window attention within each slice for the first implementation so 3D attention is not another simultaneous factor.

Provide an explicit depth-mixing-off control, plus a capacity-matched 2D/additive control as close as feasible; report residual parameter/FLOP differences. Ensure the new residual path can initialize to the old per-slice function. A simple slice loop with no cross-Z dependency is not the hybrid arm. Test that changing a neighbor can affect the relevant center output in the enabled path but not the depth-mixing-off control beyond the original 2.5D neighborhood.

### G3/G8: Full 3D CNN and anisotropy

Implement a compact residual 3D U-Net-style student under the same trainer and EMA interface. Expose anisotropic kernel/stride planning: preserve Z with 1×3×3 kernels/1×2×2 strides where appropriate and allow 3×3×3 mixing at declared scales. Support a specified depth and physical FOV; do not default to cubic128 or isotropic1mm for every cine.

Implement a scratch initialization baseline. A 2D-to-3D inflation option, if offered, must list exactly mapped/unmapped tensors, scaling and trainability, and have tests; no claim of checkpoint parity. Do not silently add a new pretrained model. Keep scratch/inflated/pretrained policy separate from dimensionality. Include supervised-only and EMA configurations.

## 5. Loss, optimization and teacher behavior

### G4: Explicit Dice estimators

Retain legacy CE + batch-pooled present-foreground Dice exactly. Add CE + frame/slab-volume foreground Dice with an explicit grouping and absence policy. Ignore padded/unknown voxels in every numerator and denominator. CE must still address false positives for absent foreground classes. Distinguish mean-of-ratios from ratio-of-sums, crop/slab Dice from whole-frame Dice, and class mean from background-inclusive Macro4.

For gradient accumulation, implement and document the chosen logical-batch estimator. Do not claim that averaging microbatch Dice losses is the same as pooled Dice. Test reference gradients on a small complete logical batch, or explicitly mark a mean-per-volume alternative as a different objective. Expose boundary/asymmetric losses only as separate optional factors if implemented; do not bundle them into the default recipe.

### Recipe modernization

Add opt-in scheduler-only and optimizer-recipe arms: legacy AdamW-body/SGD-head with a declared LR decay; then grouped AdamW for encoder, decoder/detail and head with explicit rates. Preserve legacy clipping, weight-decay and update behavior unless independently configured. Do not copy the SGD head LR into AdamW blindly. Separate LR warmup from supervised-only pseudo-delay. Define whether the scheduler indexes outer iterations or actual optimizer updates and handle skipped pseudo steps consistently.

In ema_full the legacy `pseudo_lr` does not control the ordinary pseudo optimizer. Make the new configuration unambiguous without changing old behavior. Add optional pseudo-weight ramp and log both loss contributions and actual update/gradient norms; AdamW loss scaling is not guaranteed proportional parameter-update scaling. Save/resume every scheduler/optimizer/scaler/RNG state.

### G5: Volume EMA

Deep-copy the actual student architecture for the teacher; update it once after the committed final student update. Retain same-grid weak/strong views as the first control. Any spatially different views require an explicit transform of probabilities and valid support. Pseudo-targets are detached. Unsupported/empty acceptance skips the pseudo update according to the declared legacy-compatible policy, with no hidden optimizer or EMA steps.

Log per-class acceptance/coverage, pseudo loss, teacher-student disagreement and FP/FN/calibration only on allowed labeled development support. Provide separate student/EMA evaluation outputs, with prediction-model choice fixed before comparisons. Do not silently replace student scores with the better of student/EMA.

## 6. Implement later research modules, disabled by default

### G6: Skip and detail ablations

Add delayed nonlinear fusion at one specified fine scale first: preserve separate skip/context representations through a spatial nonlinear block before compression. Compare against additive fusion with matched capacity and retain the existing native detail branch. Concat followed immediately by one affine1×1 can be equivalent to the current two projections plus addition; do not label that a novel information-preserving operator. Add controlled diagnostic hooks for branch reliance and skip stop-gradient during training, distinguishing frozen-checkpoint interventions from retraining ablations.

### G7: Region learning from U

Implement three selectable paths: no regions; image-only 3D supervoxels/regions; detached student/EMA-derived regions/features. Supply a real deterministic region builder, explicit physical-distance/support handling and an auxiliary region-consistency objective. Reject or downweight regions crossing confident class boundaries under a declared image/prediction-only rule. Do not read U masks. Regions are class-agnostic structural supervision, not new RV/MYO/LV ground truth. Implement teacher/region refresh cadence, cache hashes and stop-gradient tests. Do not add a hidden SAM or external-teacher dependency. This is inspired by a data-organization idea, not a reproduction or established novelty claim for VISTA3D.

### Temporal consistency (separate from spatial Z)

Add a disabled temporal arm using adjacent t volumes and explicit temporal correspondence/warps with valid/occlusion masks. Implement an operational, documented registration/correspondence backend and detached-target consistency path, with a synthetic known-motion test. Identity alignment is allowed only as a separately named diagnostic or for verified aligned fixtures, never as a silent substitute for cardiac motion. Do not copy ED/ES reference masks onto other phases as ground truth. Log temporal supervision and compute separately.

### G9: Annotation/correction workflow

Provide a minimal executable 3D correction workflow (CLI/standard JSON+NIfTI inputs is sufficient; no elaborate UI required): point/correction inputs in declared coordinates, a prompt-conditioned refinement path, sparse label support, and export to native geometry with provenance. Preserve manual corrections; distinguish original labels, new corrections, pseudo-labels and unknown regions. Supply a replayable point/correction fixture and a bounded interactive test. Synthetic simulated clicks can use known fixture labels; real TRAIN-L click simulation must remain within L and be labeled simulated. Do not read U/VAL/TEST masks to create training clicks. An evaluation that uses reference-driven clicks is an additional-supervision interactive protocol, never automatic inference. Record clicks, corrected slices and annotation time if actually measured; do not fabricate human timing.

## 7. Ablation registry and runtime tooling

Create a machine-readable registry/config set covering these comparisons. Implement all paths, but do not launch this matrix:

| ID | Factor / comparison |
|---|---|
| V00 | Exact historical A0 reference |
| V01 | Volume-batched2.5D with legacy reduction |
| V02 | V01 + frame/slab Dice only |
| V03 | Hybrid depth-mixing-off, same grouping/loss/init |
| V04 | Hybrid with learned depth mixing |
| V05 | Capacity-matched per-slice/additive control for V04 |
| V06 | Full3D anisotropic CNN, scratch |
| V07 | V06 with declared alternative Z mixing/planning only |
| V08 | Native-Z versus configured resampling control |
| O01 | Scheduler-only versus matching constant-LR recipe |
| O02 | Grouped AdamW recipe versus O01; label optimizer+LR confound honestly |
| O03 | Pseudo-weight ramp versus matching no-ramp recipe |
| S01/S02 | Delayed nonlinear skip versus matched-capacity additive fusion |
| R00/R01/R02 | No-region / image-only region / EMA-derived region |
| T00/T01 | Spatial-only / motion-aligned temporal consistency |
| I00/I01 | Automatic-only / interactive correction, separate supervision budget |
| C00/C01 | Supervised-only / supervised-extra update-budget controls |

Avoid an uncontrolled Cartesian product. Each row must name its parent, changed factors, initialization, label exposure, loss reduction, optimizer/EMA counts, geometry and prediction model. Batch differences, postprocessing, AMP and checkpoint rules are explicit factors. Historical A0 FP32/batch24 remains immutable; new throughput optimization cannot rewrite its receipt.

Implement preflight, bounded synthetic smoke, optional bounded TRAIN smoke, profile, resume and frozen evaluation commands. Default commands must not launch full training. Hardware profiling on an idle available GPU is allowed for short synthetic/bounded checks; never stop someone else's job. Use mixed precision only after finite-gradient and objective checks. Profile candidate slab depths, H/W, microbatch and accumulation under a conservative VRAM headroom; pick measured samples/valid-voxels per second, not simply the largest batch. Keep host caching bounded and avoid worker/mmap oversubscription. Record warmup exclusion, synchronized timing, peak allocated/reserved VRAM and effective label exposure. Cap the initial GPU verification at 20 minutes total and any real TRAIN smoke at 10 outer iterations per selected path; no long efficacy run.

## 8. Required tests and acceptance gates

- Old config/checkpoint/disabled-path regression; no unexpected baseline initialization or RNG changes.
- Phantom geometry with asymmetric landmarks, unequal spacing, nonidentity affine, flipped/permuted axes and known frame IDs; image-label/point alignment through crops/warps/resampling/native export. Check designed invariants, not bitwise recovery after lossy interpolation.
- No U/VAL/TEST mask access during any training/preparation/region/click path; disjoint patient IDs; missing-label and padded-support tests.
- Z=1, short/variable Z, odd H/W, absent foreground, empty pseudo acceptance, partial labels and invalid geometry.
- Reference loss/gradient tests for pooled versus grouped Dice, mixed valid support and the declared accumulation estimator.
- Enabled cross-Z influence, disabled control behavior, nonzero finite gradients into intended decoder/encoder parameters, and no teacher/region target gradients.
- Correct supervised/pseudo/EMA update order/counts; clipping/AMP and skipped-update behavior; uninterrupted versus checkpoint-resumed state equivalence within declared platform tolerance.
- Identity/known-motion temporal test, occlusion support, and coordinate-correct interactive fixture with preserved manual edits.
- Student-only deployment/export, native-grid output shape/classes, and no external teacher/cache/network access required for new primary arms.
- Run repository-required checks: `python scripts/check_repo.py` and `python -m pytest -q`, plus relevant new tests. Use the intended environment; report exact commands/results. CPU/synthetic PASS is not GPU/patient accuracy evidence.

First establish numerical/geometry correctness; there is no meta-window hypergradient in these new primary arms. Preserve historical meta-gradient tests without adding expensive bilevel optimization to the 3D baseline.

## 9. Deliverables and completion

Commit and push the implementation branch with allowlist staging. Do not delete/rewrite historical reports, large data, checkpoints or either figure in this handoff. Update docs with actual callable commands, flags and schemas, not guessed examples. Include:

1. Working opt-in implementation for every module above, configuration/ablation registry and tests.
2. `IMPLEMENTATION_STATUS` mapping each requirement to files, test evidence and IMPLEMENTED/VERIFIED/BLOCKED/NOT_RUN status. Missing assets may block a real-data run, not justify leaving executable software as a stub.
3. Source/config/weight/data hashes; geometry and label-access receipts; environment and bounded runtime profile if executed.
4. A concise report of actual checks, known limitations, unresolved confounds and exact server commands for a future approved pilot/full run.
5. Separate reporting of software correctness, synthetic results, real TRAIN smoke, real development Dice and unrun work. No guaranteed Dice gain, no SOTA claim and no novelty claim based merely on combining known components.

After implementation and bounded checks, stop before long training. Keep VAL/TEST already inspected as development evidence; new seeds do not restore an independent holdout. Explain any hard blocker precisely while finishing independent work. Do not ask permission again for ordinary reversible implementation steps already authorized here.

## Reference context

- [Current architecture](figures/native-student-current.png) and [current EMA training](figures/ema-full-training.png) describe existing A0, not the proposed3D system.
- [VISTA3D](https://arxiv.org/html/2406.05285v3): data/annotation inspiration; do not import CT teacher semantics as MRI ground truth.
- [ACDC acquisition](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html): inspect each actual file's geometry before planning3D.
- [3D U-Net](https://arxiv.org/abs/1606.06650), [MedNeXt](https://arxiv.org/abs/2303.09975), [Primus](https://arxiv.org/html/2503.01835v2): prior art and design comparisons, not proof of benefit under this L8 protocol.
