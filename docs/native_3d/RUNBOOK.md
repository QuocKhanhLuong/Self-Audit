# Native-volume pilot runbook

This runbook describes bounded software checks only.  The registry is a
comparison manifest, not a request to launch the full matrix.  Every command
below is synthetic and CPU-only unless `--device cuda` is explicitly supplied
after an idle-device check.  A GPU profile is capped at 20 minutes and a
TRAIN smoke is capped at 10 outer steps; patient efficacy and GPU claims remain
`NOT_RUN` in this handoff.

## 1. Preflight one arm

```sh
python scripts/volume_preflight.py --config configs/native_volume/V04.json
```

Preflight loads `configs/native_volume/REGISTRY.json`, validates the parent and
changed-factor declaration, checks the versioned `self_audit.volume.v1` schema,
rejects forbidden U/VAL/TEST mask or external-teacher references, and reports
the config and registry SHA-256.  It does not load image data or instantiate a
trainer.  Replace `V04` with any of `V00..V08`, `O01..O03`, `S01..S02`,
`R00..R02`, `T00..T01`, `I00..I01`, or `C00..C01`.

`V00` is the immutable historical A0 descriptor (fixed-window/ema_full,
batch 24, crop 256, FP32, 10k outer iterations and 1k supervised-only
delay).  The descriptor is exempt from executable runtime caps and is never
instantiated; smoke returns `HISTORICAL_REFERENCE_ONLY`.  The bounded eight
step cap applies only to new executable synthetic arms.

## 2. Synthetic smoke

```sh
python scripts/volume_smoke.py --config configs/native_volume/V04.json --steps 2 --crop-size 64 --batch 1 --device cpu
```

The smoke creates a deterministic, temporary `VolumeTrainingData` fixture with
separate labeled TRAIN-L and image-only TRAIN-U records, then executes the
actual `VolumeTrainer` supervised -> fixed-target -> pseudo -> one-EMA
transaction.  It checks the `B,3,Z,H,W -> B,4,Z,H,W` contract, finite gradients,
update counts, detached teacher, label-read provenance, and fixture hashes; it
never uses U masks.  Region rows require a nonzero region-loss/cache witness,
temporal rows require a nonzero consistency/support witness, C01 requires the
extra supervised update witness, and an explicit I00/I01 correction protocol
executes point-conditioned correction plus native NIfTI/JSON export.  V00 returns
`HISTORICAL_REFERENCE_ONLY` after canonical hash verification and never builds
or runs the 10k-step historical model.  Use
`--output reports/native_volume_pilot/V04-smoke.json` to retain the JSON
receipt.  The production recipe remains canonical DINO/fixed-window;
`--crop-size`, `--steps`, and `--batch` are explicit synthetic overrides
recorded under `runtime_overrides` (the tiny synthetic backbone and decoder
width [96,128,192,256] -> [8,16,24,32] substitutions are likewise recorded),
never silent production edits.  I00/I01 correction
witnesses must pass an explicit protocol file, for example
`--correction-config configs/native_volume/I01_correction.json`; the workflow
does not infer an algorithm from the experiment filename.

## 3. Synchronized profile

```sh
python scripts/volume_profile.py --config configs/native_volume/V04.json --iterations 10 --warmup 2 --crop-size 64 --microbatch 1 --device cpu
```

The profile excludes warmup transactions, synchronizes CUDA timing when
requested, and measures the same end-to-end fixture `VolumeTrainer` L/U
transactions rather than a model-only forward estimate.  Receipts include
effective batch volumes/slices/valid voxels, supervised/pseudo/EMA update
counts (including measured-only deltas), wall and CUDA-synchronized time,
microbatch/accumulation, model parameters, fixture hashes, and (on CUDA) peak
allocated/reserved VRAM, free VRAM from `cuda.mem_get_info`, and headroom.
`--max-minutes` cannot exceed 20.
Candidate slab depths and microbatches should be profiled separately; choose a
measured valid-voxel rate with conservative headroom rather than the largest
batch.  Torch CPU threads use training.threads. The array loader retains at most eight mmap arrays by default (max_cached_arrays); live caller views can retain their backing mapping after eviction. Region targets use a separate bounded per-sample LRU whose capacity and order survive resume.

## 4. Explicit bounded TRAIN smoke (future, approved assets only)

No real TRAIN smoke is executed in this delivery because the PRO4000 server and
verified assets are unavailable.  The current shared entry point requires
explicit hash-pinned paths.  Its argparse contract includes `--steps` for a
bounded resume continuation and `--resume/--resume-sha256`; these are never
inferred from a filename.  After an approved asset and idle-device check, use:

```sh
python train.py --config configs/native_volume/V04.json --data /approved/train_manifest.json --data-sha256 <manifest-sha256> --geometry /approved/geometry.json --geometry-sha256 <geometry-sha256> --out reports/native_volume_pilot/V04-train.json --device cuda
```

The planned horizon is 8 outer steps.  To exercise a genuine continuation,
first stop at step 4 (the checkpoint must be emitted by the callable trainer),
then resume that exact step-4 checkpoint directory to target step 8; a step-8 checkpoint cannot
be resumed with `--steps 8` as evidence of continuation:

```sh
python train.py --config configs/native_volume/V04.json --data /approved/train_manifest.json --data-sha256 <manifest-sha256> --geometry /approved/geometry.json --geometry-sha256 <geometry-sha256> --steps 4 --out /approved/V04/checkpoints/step0004 --device cuda
python train.py --config configs/native_volume/V04.json --data /approved/train_manifest.json --data-sha256 <manifest-sha256> --geometry /approved/geometry.json --geometry-sha256 <geometry-sha256> --resume /approved/V04/checkpoints/step0004 --resume-sha256 <step0004-manifest-sha256> --steps 8 --out reports/native_volume_pilot/V04-resume --device cuda
```

For a hash-pinned continuation of a prepared checkpoint (bounded horizon only):

```sh
python train.py --config configs/native_volume/V04.json --data /approved/train_manifest.json --data-sha256 <manifest-sha256> --geometry /approved/geometry.json --geometry-sha256 <geometry-sha256> --resume /approved/V04/checkpoints/step0004 --resume-sha256 <step0004-manifest-sha256> --steps 8 --out reports/native_volume_pilot/V04-resume-prepared --device cuda
```

Frozen evaluation and export are separate hash-pinned commands:

```sh
python scripts/volume_evaluate.py --checkpoint /approved/V04/checkpoints/step0004 --checkpoint-sha256 <step0004-manifest-sha256> --manifest /approved/eval_manifest.json --manifest-sha256 <manifest-sha256> --geometry /approved/geometry.json --geometry-sha256 <geometry-sha256> --prediction-model student --device cuda --output reports/native_volume_pilot/V04-eval.json
python scripts/volume_export.py --checkpoint /approved/V04/checkpoints/step0004 --checkpoint-sha256 <step0004-manifest-sha256> --out reports/native_volume_pilot/V04-export --device cuda
```

These commands require the pinned DINO source/weight provenance in the selected
config (`source_commit=6876159a11b4df116f30f667f8c9888617df0751`,
`weights_sha256=21b726bb286e037f00a23fb4699fa9bda9c75b6a615bd57ce3013cec1b528d54`),
matching manifest and geometry hashes, and a clean prepared CUDA environment.
The pilot horizon is at most 8 bounded outer steps after integration; stop on
any manifest/geometry/weight mismatch, invalid support, non-finite update, or
the 20-minute profile cap.  No GPU command was run in this delivery.

The command must fail closed unless the configured patient-level TRAIN manifest
and geometry sidecar are hash-pinned and contain no U/VAL/TEST masks.  Do not
replace this with a full run or by passing a development/test mask.  Record
actual source/config/weight/data hashes and separate software, synthetic,
TRAIN-smoke, development Dice, and `NOT_RUN` evidence.

## 5. Registry discipline

`REGISTRY.json` gives every row one parent and one or more changed factors.  A
  row is not a Cartesian-product launch: geometry, loss reduction, optimizer,
AMP, postprocessing and prediction-model choices are explicit in its JSON.
The historical A0 receipt is never rewritten by throughput settings.  New
  arms retain canonical DINO provenance/fixed-window production settings,
  FP32, `steps <= 8` bounded software defaults, no external teacher, and
  student prediction unless a row explicitly declares otherwise.  DINO decoder
  rows use `[96,128,192,256]`; V06/V07 are separate four-stage anisotropic
  CNN3D scratch topologies `[16,32,64,128]`, never DINO decoder widths. V06
  uses conservative thin-cine kernels `[(1,3,3),(1,3,3),(3,3,3),(3,3,3)]`
  and V07 uses `[(3,3,3)]*4`; both use strides
  `[(1,1,1),(1,2,2),(1,2,2),(1,2,2)]`. CPU smoke uses only explicit
  family-aware tiny overrides, recorded in receipts: DINO `[8,16,24,32]`,
  CNN3D `[8,16,24,32]`, plus crop/steps/batch overrides.

All pseudo targets must be detached; all region/temporal targets are detached;
the shared trainer owns the single supervised -> fixed target -> pseudo -> one
EMA timeline.  Rejected or empty updates must not change live optimizer or EMA
state.  CPU/synthetic PASS is software evidence only and is not patient Dice,
GPU throughput, anatomy, clinical, or publication evidence.


## 6. Physical preparation and manual correction

The prepared TRAIN manifest retains historical v2 normalization and patient
label budget. A separate physical-metadata JSON maps patient IDs to affine in
mm, orientation, slice order, temporal frame mapping, and hash-pinned raw
source_image. Only TRAIN-L entries additionally carry mask_sources. Raw
cine/frame/mask geometry and prepared voxel correspondence are checked.

    NV_TRAIN_SHA=$(sha256sum /approved/train_manifest.json | cut -d ' ' -f 1)
    python scripts/volume_prepare.py --manifest /approved/train_manifest.json --manifest-sha256 "$NV_TRAIN_SHA" --physical-metadata /approved/physical_metadata.json --out /approved/native_geometry.json
    python scripts/volume_correct.py --image /approved/frame.nii.gz --prompts /approved/prompts.json --mode geodesic_3d --prior-probabilities /approved/student_probabilities.npy --out-labels /approved/corrected.nii.gz --out-provenance /approved/corrected_provenance.nii.gz --out-probabilities /approved/corrected_probabilities.nii.gz --out-json /approved/correction_receipt.json

NIfTI inputs must declare recognized spatial units and valid geometry.
NPY/NPZ inputs instead require an explicit geometry sidecar and patient ID.
Prompts declare native voxel XYZ or world mm; arrays use ZYX. Replay supplies
both prior-labels and prior-provenance to preserve manual edits and lineage.
BG/RV/MYO/LV IDs are 0/1/2/3; unknown is 255. Provenance
original/manual/propagated/pseudo IDs are 1/2/3/4; unknown is 255.
Reference-driven simulated clicks additionally require the exact TRAIN-L
manifest/hash, patient, frame, mask identity and declared label budget.
I00 uses zero clicks; I01 is additional supervision. Human time is NOT RUN.

## 7. Reproduce the software verification

    python reports/native_volume_pilot/verification_runner.py /tmp/self-audit-volume-verification-new

The destination must be new. The runner executes repository checks, bounded
manufactured smokes, matched CPU profiles, and actual CLI train/stop/resume,
image-only evaluation and student-export roundtrips. V00 is a historical
descriptor only. Temporary arrays/checkpoints are discarded; receipts and
command logs remain. No patient experiment matrix or long training is run.

Temporal correspondence is a bounded image-derived translation backend with
support/occlusion rejection; adequacy for deforming cardiac motion is
unverified. Region/correction methods are executable research controls.
Real DINO execution, patient geometry, CUDA AMP, target GPU throughput/VRAM,
patient Dice and human correction utility remain NOT RUN. Timers are
cooperative: they reject an overrun after an operation completes and cannot
preempt a running kernel.
