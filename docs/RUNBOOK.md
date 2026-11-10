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
