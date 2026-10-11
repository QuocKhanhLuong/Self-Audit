# Stable SSL guidance: native-data full pipeline and preflight

This opt-in research pipeline is on `feat/stable-ssl-guidance` (draft PR #33).
Legacy v3/CQA code and split files remain unchanged. Runnable software and a
successful preflight do not establish that the anatomical readout is accurate.

## Full execution contract

```text
mandatory real preflight
  -> all TRAIN native cine images -> frozen CineMA-MAE -> fixed guidance bank
  -> fixed-budget student training -> freeze last-epoch checkpoint
  -> export and freeze ALL validation predictions
  -> export and freeze ALL test predictions
  -> lock BOTH prediction manifests
  -> separate validation reference evaluator
  -> separate test reference evaluator
```

There is no mock feature or preprocessed-data fallback in the production CLI.
Neither teacher nor student reads target masks, GT-derived crops, ED/ES metadata,
or GT class mapping. References cannot select a threshold or checkpoint.
Validation here means final held-out scoring, NOT Dice-guided early stopping.
The selected checkpoint is the last epoch of the recipe locked before training.

Feature learning continues when semantic support is missing. A representation-only
checkpoint outputs UNKNOWN, is scored honestly, and is not reported as successful
segmentation. Upstream CineMA supervision/preprocessing lineage remains
SSL-declared, not independently certified in its entirety.

## 1. Checkout and isolated environment

Do not change source in a checkout used by an active run. Python 3.11 and git
must be installed. In a clean checkout:

```bash
cd /workspace/Self-Audit
git fetch origin feat/stable-ssl-guidance
git switch feat/stable-ssl-guidance
git pull --ff-only origin feat/stable-ssl-guidance

bash scripts/setup_stable_guidance.sh
source .venv-stable-guidance/bin/activate
```

The setup script refuses to overwrite an existing environment. Activate an
already compatible isolated environment instead when one exists. It pins
PyTorch 2.6.0/torchvision 0.21.0 cu124, the inspected CineMA source revision and
HF Hub. Editable CineMA installation preserves upstream subpackage imports.
The resolved dependency snapshot is written inside the environment.
`TORCH_VARIANT=cpu` installs the CPU pair. No driver is installed or modified.
Actual CUDA execution is tested in preflight; availability is not inferred from
`nvidia-smi` alone. No silent CPU fallback is permitted.

Offline execution requires exact cached weights/config, or BOTH
`--checkpoint /path/cinema.safetensors --teacher-config /path/config.yaml`.
Only the pinned MAE weight SHA256 is accepted; segmentation-finetuned weights
cannot be substituted by renaming a file.

## 2. Three disjoint patient splits

The original `splits/acdc_patient_split_seed42.json` contains 80 TRAIN and 20
validation patients, with NO test split. The full runner rejects it rather than
reusing validation as test. Create a separate manifest once:

```bash
python scripts/make_stable_guidance_split.py \
  --source splits/acdc_patient_split_seed42.json \
  --out splits/acdc_guidance_80_10_10_seed42.json \
  --seed 42
```

This keeps all 80 original TRAIN identities, and partitions the old validation
pool into 10 validation and 10 test identities by deterministic SHA256 ranking.
It does not read images, diagnosis metadata or masks; the old file is untouched.
The output file cannot be overwritten silently.

IMPORTANT: the new test is an exploratory INTERNAL holdout drawn from validation
used in earlier experiments. It is NOT a pristine test and NOT the official
hidden ACDC challenge test. Use genuinely untouched external patients for a
stronger final generalization claim. No change of file name erases prior use.
An existing valid three-way manifest can be passed directly instead.

## 3. Run the complete real-data workflow

```bash
RUN="$PWD/runs/stable_guidance/full_$(date +%Y%m%d_%H%M%S)"

python scripts/run_stable_guidance_full.py \
  --dataset acdc \
  --root "$PWD/data/ACDC/training" \
  --split-manifest splits/acdc_guidance_80_10_10_seed42.json \
  --config configs/stable_guidance_experimental.json \
  --out "$RUN" \
  --device cuda \
  --epochs 50 \
  --batch-size 8 \
  --tile-batch 1 \
  --width 32 \
  --lr 0.0003 \
  --threads 4 \
  --seed 42 \
  --evaluation acdc \
  --reference-root "$PWD/data/ACDC/training"
```

Preparation uses `patients=0`: ALL TRAIN patients and ALL frames/slices, not the
old four-patient pilot. Fifty epochs is an initial declared budget, not an
optimized convergence or accuracy promise. Student slice batch 8 and frozen
teacher 3-D tile batch 1 are separate. No graph cap is silently reduced to fit
VRAM. Fresh output is required; stage failures are recorded in RUN_STATUS.json.

For preflight only, append `--preflight-only`. To continue, use the SAME full
command/output with `--resume` and without `--preflight-only`. Do not assign a new
RUN timestamp when resuming. There is no skip-preflight flag. Resume checks the
source/config/split recipe, reruns preflight and reuses only completed frozen
stages. IN-EPOCH TRAINING RESUME IS NOT IMPLEMENTED. An incomplete stage directory
is refused, not overwritten or accepted as a completed model; preserve its logs
and use a fresh run or recover deliberately.

## Preflight scope

1. Real imports: nibabel, CineMA/torchvision/timm, safetensors and HF Hub.
2. All native image arrays across the locked cohort: finite voxels, native 4-D
   shape/affine, unique patient/file identities, no split overlap or byte-identical
   duplicated cine, graph node limits and an uncompressed bank disk estimate.
3. Real official hash-pinned MAE inference on two genuine TRAIN frames, all Z,
   from the deterministic largest in-plane TRAIN image. Repeat feature extraction
   must agree. No anatomy quality or GT is used to choose this memory probe.
4. Anchor/graph/solver execution on those frames. Low anatomy support is reported,
   not misrepresented as high-quality segmentation.
5. Two actual feature-distillation backward/optimizer steps at the requested
   student batch, finite nonzero gradients and encoder change, followed by a
   checkpoint save/reload comparison. Repeated samples solely to fill a memory
   test batch are disclosed. The probe weights are discarded before training.

The pass status is `PASS_EXECUTION_ONLY_QUALITY_UNVERIFIED`. It is a bounded
execution test, not a benchmark of every cohort shape or anatomical accuracy.
Failures write `BLOCKED`, the exact failed stage and exception. No fake provider
is used to turn an import/checkpoint/CUDA failure into a pass.

## Independent reference scoring

Both val and test prediction sets are frozen BEFORE any GT is opened. The ACDC
adapter requires exactly two `patientXXX_frameYY_gt.nii[.gz]` files per held-out
patient; filename indices are converted from 1-based to 0-based. Info.cfg is
never opened. The model predicts the entire cine; only annotated frames are scored.

Class order: BG=0, RV=1, MYO=2, LV=3, UNKNOWN=255. Scoring requires exact native
XYZ shape and affine; no mask resizing, remapping or GT-based Hungarian matching.
Every exported patient must have references; there is no silently reduced cohort.

Metrics include whole-volume class Dice/IoU, patient-macro foreground Dice,
coverage/UNKNOWN, all-pixel accuracy and native-mm HD95 when units are available.
UNKNOWN causes foreground false negatives: it is neither ignored in the Dice
denominator nor reinterpreted as background. A class absent on both sides has
Dice=null and is excluded, not awarded Dice=1. Empty surfaces have null HD95 and
an explicit failure reason. HD95 is the 95th percentile of pooled bidirectional
native surface distances under the full affine, including voxel spacing.
Accepted-pixel accuracy is secondary and never replaces full-volume Dice.

For M&Ms/other layouts, use `--evaluation manifest --reference-manifest FILE`:

```json
{
  "schema": "self_audit.references.v1",
  "entries": [
    {"patient_id": "CASE01", "frame_index": 0, "path": "refs/CASE01_frame0.nii.gz"}
  ]
}
```

Each entry points to one genuinely annotated 3-D frame. Paths may be relative to
the manifest. A shared manifest may contain both val/test but must cover every
exported subject. Never treat unannotated zero-filled frames of a 4-D reference
as dense GT: document the annotated frames explicitly. `--evaluation none`
exports without scoring and reports COMPLETED_UNSCORED.

## Outputs and status

```text
RECIPE.json
PREFLIGHT_<timestamp>.json
logs/
bank/FROZEN.json
student/last.pt
student/training.json
predictions_val/        # NPZ TZYX + native NIfTI XYZT
predictions_test/
PREDICTIONS_LOCKED.json
evaluation_val/METRICS.json
evaluation_test/METRICS.json
RUN_STATUS.json
```

Native labels preserve full XYZT shape and affine. A completed zero-support run
ends with `COMPLETED_SCORED_NO_SEMANTIC_SUPPORT` and exit 2, not an accuracy claim.
Partial support is also explicit. Keep these diagnostic outcomes; do not force
nonzero predictions or weaken gates just to obtain an exit code 0.

## Verification actually performed

Code commit: `7638907167bab166a50ddfdd0ea15b5241b88218`.

- Local CPU: 65 passed, 6 native-IO tests skipped because genuine nibabel was
  unavailable. A real local production preflight was attempted and BLOCKED at
  dependency imports. The user's filesystem and CUDA device are not mounted here.
- GitHub Actions run `38111798541`: native-contracts job `114388723344` passed
  71 tests, 0 skips, using genuine nibabel. Manufactured NIfTI roundtrip covers
  prepare/train/export/evaluate with an explicitly synthetic feature provider.
- The same run's real-checkpoint-smoke job `114388723191` installed the isolated
  upstream environment, loaded the genuine official MAE, extracted repeatable
  features from explicitly manufactured NIfTI images, solved the bank and ran
  student backward/save/reload. It returned PASS_EXECUTION_ONLY_QUALITY_UNVERIFIED.
- Existing self-audit-ci push run `38111798515` also succeeded.

These are CPU software/integration results, NOT ACDC/M&Ms recovery or target-GPU
measurements. The identical real preflight must pass on the user's raw data and
GPU before the full training command proceeds. See reports/stable_full_20261011/
for local and CI receipts; no clinical quality or publication novelty is asserted.
