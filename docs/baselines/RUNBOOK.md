# Run Candidate A or A-3D

The `--config` file selects the track. A recipes use
`self_audit.candidate_a.v2`; A-3D recipes use `self_audit.candidate_a3d.v1`.
The command dispatcher rejects a track/checkpoint/schema mismatch. The A-3D
reference protocol consumes one native full-volume frame as `[1,1,D,H,W]` and
requires runtime `shape_zyx`, `spacing_zyx`, `affine_zyx` and `context=full_volume`
(the prepared manifest stores physical spacing as `spacing_zyx_mm`).
It rejects crop/resample/implicit augmentation. See
[CANDIDATE_A3D.md](CANDIDATE_A3D.md) for the complete A-3D command sequence.

For the main A-3D contribution check, use the one-seed plan in
[EXPERIMENTS.md](EXPERIMENTS.md) with `--seeds 42` and the four arms
`full teacher_free fixed_queries global_only_queries`. That plan creates two
shared teacher stages plus four matched student runs per dataset. The larger
21-arm catalogue is reserved for secondary diagnostics.

The dedicated few-shot protocol is documented in
[FEWSHOT.md](FEWSHOT.md). Its canonical profile runs all 21 A-3D arms at
`k=2,5,8,10` labeled TRAIN patients, with `full` first at each `k`; this is 92
jobs per dataset and 184 jobs for ACDC plus M&Ms. The dedicated entrypoint is
`scripts/candidate_a_fewshot.py`; it creates one top-level `FEWSHOT_PLAN.json`
plus four child plans and refuses to resume an existing output directory. Use
the commands in `FEWSHOT.md` rather than adding an ad-hoc label flag to the
main ablation planner.
For ACDC plus M&Ms priority ordering, pass both root plans to
`scripts/candidate_a_campaign.py`; its `campaign_state.json` runs all teacher
stages and `full` arms first, then the remaining arms. The child runner alone
executes one k at a time and does not provide that global ordering.

## Optional A-3D semi-supervised run

The opt-in profile at `configs/candidate_a3d/semi_supervised.json` adds only
detached EMA consistency on image-only TRAIN U. It keeps the native full-volume
geometry, typed measurement bank, code-only decoder and all labeled
segmentation/measurement/K1 losses. No U mask, U GT-derived SDF target or U K1
utility target is read; detached EMA soft SDF consistency is allowed. The EMA helper is present for every arm, including
`teacher_free`; it is a student self-EMA, not the dual-branch teacher.

Plan the same compact four-arm comparison with:

```bash
SEMI_CONFIG=configs/candidate_a3d/semi_supervised.json
python scripts/candidate_a_ablations.py plan --track candidate_a3d \
  --dataset acdc --train /prepared/acdc/train.json \
  --development /prepared/acdc/development.json \
  --model-config /private/mednext3d.json --semi-config "$SEMI_CONFIG" \
  --out /runs/acdc-a3d-semi-s42 --seeds 42 \
  --arms full teacher_free fixed_queries global_only_queries \
  --epochs 100 --pretrain-epochs 100 --warmup-epochs 10
python scripts/candidate_a_ablations.py run \
  --plan /runs/acdc-a3d-semi-s42/PLAN.json --execute --device cuda
```

For the few-shot planner, pass the same `--semi-config` to
`scripts/candidate_a_fewshot.py plan` and keep the root/child output paths
new. `steps_per_epoch: 0` resolves to
`max(number_of_labeled_frames, number_of_TRAIN_patients)` (80 on prepared
ACDC), subject to covering every labeled frame and U patient, and is reused
across `k=2,5,8,10`. For original M&Ms, use and record that dataset's own
resolved count; do not reuse 80 by convention. Each logical update pairs one
labeled frame step with one U-patient visit; the U cycle visits every U patient
once per cycle and chooses one native cine frame cyclically for that visit.
Labeled frames repeat in balanced cycles. Intensity-only gamma/voxel-noise is
applied to labeled and U views without changing geometry. Record logical sample
updates separately from optimizer updates, including labeled/U and
teacher/student counts. This means 100 x 80 logical updates at ACDC for every
`k`, replacing the old `k=2` shorthand of 100 x 4 optimizer updates; it is not
compute-matched across label budgets. The EMA state resets at each new stage,
and this workflow does not resume interrupted optimizer/RNG state.

`teacher_free` and `teacher_frozen` omit the dual-branch teacher U objective by
factor, so their optimizer counts and wall time can differ. A previous
labeled-only run is not a causal semi-supervised control because its schedule
and augmentation differ; a matched U-disabled control must use this same
resolved schedule and augmentation.

Install `requirements.txt` in the intended CPU/CUDA environment. Read
[DATA_PREPARATION.md](DATA_PREPARATION.md) to prepare ACDC or original M&Ms with
native geometry and locked patient splits. Copy `configs/backbone.example.json`
to a local JSON and supply the actual DINOv3 source commit, local weight path and
SHA256. A local compatible pretrained teacher is not bundled.

All hashes below refer to the corresponding `MANIFEST.json` or data JSON file.
Use a new output directory outside source data. For a patient cohort, more than
128 planned updates requires `--allow-full-run`.

```bash
python train.py --config configs/candidate_a/teacher_pretrain.json \
  --model-config /private/backbone.json --data /prepared/acdc/train.json \
  --data-sha256 "$TRAIN_SHA" --validation-data /prepared/acdc/development.json \
  --validation-data-sha256 "$DEV_SHA" --out /runs/pretrain --device cuda --allow-full-run
python train.py --config configs/candidate_a/teacher_warmup.json \
  --model-config /private/backbone.json --data /prepared/acdc/train.json \
  --data-sha256 "$TRAIN_SHA" --validation-data /prepared/acdc/development.json \
  --validation-data-sha256 "$DEV_SHA" --teacher-checkpoint /runs/pretrain/final \
  --teacher-checkpoint-sha256 "$PRETRAIN_SHA" --out /runs/warmup --device cuda --allow-full-run
python train.py --config configs/candidate_a/full.json \
  --model-config /private/backbone.json --data /prepared/acdc/train.json \
  --data-sha256 "$TRAIN_SHA" --validation-data /prepared/acdc/development.json \
  --validation-data-sha256 "$DEV_SHA" --teacher-checkpoint /runs/warmup/final \
  --teacher-checkpoint-sha256 "$WARMUP_SHA" --out /runs/joint --device cuda --allow-full-run
```

For M&Ms use the planner to generate the same recipes with `dataset.name=mnms`;
do not relabel ACDC data as M&Ms. `validation_every=1` logs development every epoch;
set it to 0 explicitly for TRAIN-only logging. `checkpoint_every_epochs` controls
periodic checkpoints; the final checkpoint is always saved. This is stage transfer,
not optimizer/RNG resume.

`metrics.csv` and `metrics.jsonl` contain epoch, split, model branch, class,
Dice/F1/precision/recall, TP/FP/FN, support and absent-class status. TRAIN rows use
online predictions across that epoch. Validation rows use frozen epoch-end model
predictions. These metric columns are pooled over voxels and agree with the row's
counts (`aggregation=pooled_voxels`); validation also records the separate
`macro_patient_dice/f1/precision/recall` averages. `epochs.jsonl` summarizes epoch coverage/loss/timing;
`updates.jsonl` records the exact patient/frame/slice and code size for each update.
Teacher-only stages evaluate teacher branches; joint stages evaluate the student.
Undefined values are null/empty, never silently perfect scores. Dice and F1 are
equal for the one-vs-rest binary definition used per class.

```bash
python scripts/export_candidate_a.py --checkpoint /runs/joint/final \
  --checkpoint-sha256 "$JOINT_SHA" --out /exports/student
python evaluate.py --checkpoint /exports/student --checkpoint-sha256 "$EXPORT_SHA" \
  --data /prepared/acdc/development.json --data-sha256 "$DEV_SHA" \
  --out /runs/frozen-evaluation --device cuda
python scripts/candidate_a_infer.py --checkpoint /exports/student \
  --checkpoint-sha256 "$EXPORT_SHA" --image /inputs/context_3xHxW.npy \
  --spacing-yx "$DY_MM" "$DX_MM" --out /predictions/example --device cuda
```

Exports have student weights only, require no teacher checkpoint, and load against
the pinned source. Evaluation freezes predictions before opening reference masks.
Code bytes, query counts and measured inference time are reported with their scope.
Packet roundtrip checks are included in the measured reference path.

Use [EXPERIMENTS.md](EXPERIMENTS.md) for hash-pinned ablation plans on either dataset.
Do not treat CPU fixture checks as patient/GPU performance or a 96-Dice result.
