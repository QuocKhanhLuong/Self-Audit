# Native-volume EMA pilot: implementation and bounded verification

Implemented the pinned handoff 486fae40c490e382d58896442215476d7d32095e on
feat/native-volume-ema-pilot, based on fa7a8664621f5540e755e7869051fa25b3926cba.
This is an executable software pilot. No patient efficacy, novelty or GPU
performance claim is established, and no long training was run.

## Delivered

- Strict native physical geometry sidecars, raw/prepared voxel correspondence,
  separate geometric/label validity, deterministic same-time slabs and global
  2.5D neighbors, reversible physical resampling/FOV, and native probability
  inversion before argmax.
- Unchanged per-slice NativeStudent volume batching; pinned DINO hybrid with
  learned multiscale cross-Z residuals, exact zero-initialized old function,
  depth-off and planar capacity controls; learned anisotropic scratch 3D U-Net.
- Exact legacy CE+pooled present-FG Dice and separate per-volume estimator;
  reference accumulation gradients; legacy and grouped optimizer recipes,
  outer-index scheduler/warmup, pseudo ramp, fixed teacher targets and one EMA.
- Full checkpoint/RNG/sampler/cache/optimizer/scheduler/scaler identity;
  planned horizon distinct from bounded stop, intermediate checkpoints,
  exact CPU resume and explicit historical-checkpoint compatibility.
- Delayed nonlinear/matched additive skip controls and frozen interventions;
  image/EMA region consistency with bounded per-sample cache; operational
  detached temporal registration/warping with validity and occlusion.
- Image-guided 3D correction, native exports, manual/unknown/provenance
  persistence, strict TRAIN-L click binding and frame validation. I00 uses
  zero clicks; I01 consumes a separate additional-supervision budget.
- All 23 registry rows, preflight, bounded smoke/profile, shared train.py,
  frozen evaluation and teacher/cache-independent student export.

Implementation mapping: [IMPLEMENTATION_STATUS](../../docs/native_3d/IMPLEMENTATION_STATUS.md).
Commands and schemas: [RUNBOOK](../../docs/native_3d/RUNBOOK.md).

## Actual verification

- Final repository checks: check_repo PASS; **422 pytest tests passed**,
  one existing TorchScript deprecation warning.
  [Post-CLI gate](POST_CLI_GATE.json).
- Stable core gate: 59 successful subprocess commands; all 23 preflights,
  22 executable two-step manufactured smokes plus V00 historical-reference
  verification, six CPU profiles, actual train/stop/resume/evaluate/export.
  [Gate](verified/FINAL_VALIDATION.json), [commands](verified/COMMANDS.json).
- Actual CLI planned horizon 4, stop 2, resume 4: student, EMA, optimizers,
  schedulers, trainer/sampler/cache and RNG payloads exactly equal to an
  uninterrupted run. Intermediate save_every output verified.
  Image-only frozen inference opened zero labels. Export reload logits were
  bitwise equal with no teacher, external cache or original pretrained weight
  required. [Roundtrip](verified/ROUNDTRIP.json).
- Final legacy CLI replay matches the handoff baseline exactly for student,
  EMA, optimizer, RNG and every step row in a three-step CPU fixture.
  This is compatibility evidence, not canonical batch24 efficacy.
  [Parity](V2_FINAL_PARITY.json).
- Geometry/gradient/support, forbidden-read, missing-class/partial-label,
  cache partition/resume, corruption rejection, correction lineage and
  nonidentity motion witnesses were independently reviewed by three Astra
  ultra workers. Luna and Gemini 3.8 Flash workers implemented code through
  Orca. [Geometry/aux review](review_geometry_aux.md),
  [model/trainer review](review_models_trainer.md), [tooling review](review_tooling.md).
  Owner appendices are distinguished from independent reviewer evidence.

## Short CPU profile

Local macOS arm64, 12 CPU cores, 24 GiB RAM, PyTorch2.14.0; one Torch thread,
FP32, manufactured inputs, explicit tiny backbones/widths, H/W32. Each row
excludes two warmup transactions and measures ten supervised/pseudo/EMA
transactions: counts 10/10/10. Examples/s counts L and U volume examples.
Micro2/accum1 and micro1/accum2 have the same logical batch 2 and voxel exposure.
Native V06 uses Z5; the explicit matched-Z3 row isolates that shape difference.

| Path | Z | Micro / accum | Examples/s | Valid voxels/s | Measured seconds |
|---|---:|---:|---:|---:|---:|
| V01 2.5D | 3 | 1 / 1 | 37.00 | 30471 | 0.541 |
| V04 hybrid | 3 | 1 / 1 | 33.28 | 27413 | 0.601 |
| V06 matched Z3 | 3 | 1 / 1 | 43.15 | 35538 | 0.464 |
| V06 configured Z5 | 5 | 1 / 1 | 32.03 | 38282 | 0.624 |
| V04 micro2 / accum1 | 3 | 2 / 1 | 46.30 | 38881 | 0.864 |
| V04 micro1 / accum2 | 3 | 1 / 2 | 36.47 | 30631 | 1.097 |

[Profile receipts](verified/) retain effective label/geometric exposure, source
and config hashes, overrides and counters. These sub-second/few-second
measurements are diagnostic and noisy. They do not select a PRO4000 batch or
predict DINO/GPU throughput. GPU VRAM/timing are NOT RUN, represented as null.

## Identity and preservation

Executed core source SHA-256: 3668c8cdd9523d2440a7e93ab2ed0564ba08ddd6bf8998ef88210ff6c96632ac.
[Per-file source/config/test hashes](SOURCE_INVENTORY.json).
Expected DINO source commit:
6876159a11b4df116f30f667f8c9888617df0751; expected weight SHA-256:
21b726bb286e037f00a23fb4699fa9bda9c75b6a615bd57ce3013cec1b528d54.
These are retained historical pins; real weights were not loaded locally.
New primary arms disable external teachers and banks; bank hash is not
applicable. Fixture/geometry/checkpoint/export hashes are in the roundtrip
receipt. [Missing assets and prepared discovery](ASSET_REQUIREMENTS.json).

Historical A0 JSON remains
efa1817634320e0aab29d25ccbe9dfec4633a09f9db54b9c735865fbf9938bef.
All canonical historical self_audit modules remain byte-identical; only
train.py gained explicit opt-in routing. Original checkout HEAD and all 236
pre-existing dirty/untracked files remain unchanged.
[Preservation](FINAL_PRESERVATION.json).
Baseline/fixed/meta paths, historical A0–A6 outcomes and both handoff figures
are preserved. The existing negative pairing result is not relabeled positive.

## Limits and NOT RUN

The user confirmed the target PRO4000 is not rented. Target device generation,
CUDA/AMP behavior, 24 GB VRAM headroom, approximately 129 GB server RAM and
KhanhDrive assets cannot be verified on that absent server. Local rclone has no
configured remote; no large data/weight download, license action or job change
was performed.

Real DINO forward/training, real-patient geometry/TRAIN smoke, patient
development/test Dice, human annotation time/utility, long training and
novelty/efficacy are NOT RUN. Mocked DINO construction verifies an interface,
not real pretrained behavior. Previously inspected VAL/TEST remain development
evidence; new seeds do not restore an independent holdout.

Temporal registration is a bounded translation backend, not established
deformable cine registration. Regions are structural targets, not semantic
ground truth. Full-volume benefit, pairing/region/skip benefit, and all
scientific comparisons need matched real-data experiments. V01 differs from
historical A0 in bounded exposure/schedule/prediction settings; O02 changes
optimizer and learning rates; CNN scratch initialization and depth are
declared confounds. Timers are cooperative, not kernel-preemptive. Mmap cache
bounds retained references; live caller views may retain mappings.

Earlier failed/raced review snapshots and the first verification harness input
error remain in reports. The accepted final receipts supersede those results;
no failure was converted to patient success.

## Next pilot

Follow RUNBOOK section 4 on the eventual server after checking actual idle
hardware and existing hash-pinned assets. Prepare physical metadata from
verified raw NIfTI, then run the bounded V04 stop4 -> resume8 commands (or V06
scratch) with explicit manifest/geometry hashes. Keep the declared recipe,
effective batch, voxel exposure and update counts fixed across comparisons.
Do not launch the full experiment matrix or long training as part of this
delivery. The replay runner can reproduce the software gate into a new output
directory without patient data.
