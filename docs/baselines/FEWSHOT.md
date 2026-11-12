# A-3D few-shot label-budget sweep

This campaign measures the A-3D method as the number of labeled TRAIN patients
changes. The execution priority is to inspect the `full` arm at
`k ∈ {2, 5, 8, 10}` first, then run the remaining catalogue arms at the same
four budgets. The canonical few-shot profile contains all 21 A-3D arms, one
seed (`42`) and fixed schedules. The existing compact four-arm contribution
check (`full`, `teacher_free`, `fixed_queries`, `global_only_queries`) remains a
separate experiment described in [EXPERIMENTS.md](EXPERIMENTS.md).

## Fixed protocol

Prepare one locked ACDC or original M&Ms TRAIN/development split with at least
10 TRAIN patients. Keep the development and test cohorts unchanged for every
value of `k`. For a new preparation, set the dense label budget to 10 and keep
the preparation seed at 42; do not prepare four unrelated random cohorts. The
sweep derives four training manifests from the same prepared TRAIN pool:

```text
K2  ⊂  K5  ⊂  K8  ⊂  K10
```

The nested subsets are selected by a deterministic permutation of TRAIN patient
IDs using seed `42`, then taking the first `k` IDs. Selection uses manifest IDs
and metadata only. It must not open masks from a patient before that patient is
selected. Every selected patient contributes all of its annotated frames to the
dense labeled TRAIN budget. A nonselected labeled patient becomes an image-only
`unlabeled` record in the derived manifest; its mask paths, mask checksums and
labels are removed from that manifest and are never opened by training. Existing
image-only TRAIN records remain image-only. Development and test mask records
are not copied into TRAIN and are not used for target generation, subset
selection, checkpoint choice or early stopping.

Each `k` has its own teacher pretrain, boundary warm-up and 21 student arms. A
teacher checkpoint trained with one `k` must not be reused for another `k`,
including a larger-budget checkpoint reused for a smaller-budget run. All arms
use the same A-3D model, bank, geometry protocol, packet budget, augmentation
policy, optimizer schedule and seed within a `k`; only the named ablation factor
changes. Epoch counts stay fixed, so updates and labeled data increase with
`k`. Report this as a label-budget curve rather than a compute-matched scaling
study.

The optional A-3D semi-supervised variant is enabled with
`--semi-config configs/candidate_a3d/semi_supervised.json`. It keeps the same
21-arm catalogue and seed, but adds image-only TRAIN consistency under one
hash-pinned schedule. The planner accepts either the complete profile file or a
JSON object containing its `semi_supervised` block and stores the normalized
block plus file hash in the root and child plans. The labeled A-3D losses and
K1 utility targets remain unchanged; U never contributes a mask, a GT-derived
SDF target or a K1 utility label. Detached EMA soft SDF consistency is allowed.

For semi-supervised runs, `steps_per_epoch: 0` resolves to
`max(number_of_labeled_frames, number_of_TRAIN_patients)` (80 on the prepared
ACDC split), subject to covering every labeled frame and every U patient. A
logical update consists of one labeled-frame step plus one optional U-frame
step. The U side visits each U patient once per logical update cycle and chooses
one native cine frame cyclically for that visit; it does not promise to visit
every cine frame in one epoch. Labeled frames repeat in a deterministic,
balanced cycle. Report `logical_sample_updates` separately from optimizer
updates and include labeled/U/teacher-EMA/student-EMA counts; 80 is not itself
an optimizer-step count. The same resolved schedule and intensity-only gamma
plus voxel-noise augmentation apply across the 21 semi arms. `teacher_free`
and `teacher_frozen` remove the dual-branch teacher U objective by factor, so
their student self-EMA U objective remains in joint training, but optimizer
counts need not match other arms; report measured counts and time rather than
assuming equal compute.

## Jobs and outputs

For each dataset, the canonical plan contains 92 jobs:

```text
k2:  teacher_pretrain → teacher_warmup → 21 student arms (full first)
k5:  teacher_pretrain → teacher_warmup → 21 student arms (full first)
k8:  teacher_pretrain → teacher_warmup → 21 student arms (full first)
k10: teacher_pretrain → teacher_warmup → 21 student arms (full first)
```

That is 23 jobs per `k` (`2` shared teacher stages plus `21` arms), or 184
jobs for both ACDC and original M&Ms. The four `s42_full` outputs are the
priority comparison; the remaining 20 arms at each `k` are the secondary
component sweep. Use the campaign supervisor below to enforce this global phase
ordering across all requested datasets and k values.

Use a new output directory for each dataset and sweep. The few-shot planner
records, for every `k`, the derived TRAIN manifest hash, the
ordered selected patient IDs, the source manifest hash, seed, config hash,
teacher checkpoint hashes and student checkpoint hash. A plan is invalid if
the source manifest, selected IDs or derived manifest changes after planning.

## Server commands

Prepare the source pool once, then create a new few-shot campaign directory.
The preparation command must leave at least ten labeled TRAIN patients in
`train.json`; the few-shot planner derives the smaller manifests itself.

For ACDC:

```bash
python prepare_acdc.py \
  --root /data/ACDC/training \
  --split /data/splits/acdc_locked.json \
  --out /data/prepared/acdc \
  --labels 10 --seed 42 --evaluation
```

For original M&Ms, use the existing explicit index and source ontology:

```bash
python prepare_mnms.py \
  --index /data/mnms/index.csv --root /data/mnms \
  --out /data/prepared/mnms --labels 10 --seed 42 \
  --source-class-map '{"0":"background","1":"LV","2":"MYO","3":"RV"}' \
  --frame-base 0 --axis-order XYZT --mask-axis-order XYZ \
  --dataset-release original_2020 --evaluation
```

Create an expanded few-shot plan for one dataset as follows. The default
`configs/fewshot/candidate_a3d.json` selects `k=2,5,8,10`, seed 42, fixed
training schedules and all 21 canonical A-3D arms; no `--arms` or seed sweep is
used. The `full` arm is first in every child plan, but this child runner executes
one k at a time and therefore is not the global priority scheduler.

```bash
DATASET=acdc
DATA=/data/prepared/$DATASET
MODEL=/server/config/mednext3d.json
OUT=/server/runs/${DATASET}-a3d-fewshot-s42
SEMI_CONFIG=configs/candidate_a3d/semi_supervised.json

python scripts/candidate_a_fewshot.py plan \
  --dataset "$DATASET" \
  --train "$DATA/train.json" \
  --development "$DATA/development.json" \
  --model-config "$MODEL" \
  --semi-config "$SEMI_CONFIG" \
  --out "$OUT"
```

Omit `--semi-config` for the labeled-only campaign. The semi config should be
the repository profile `configs/candidate_a3d/semi_supervised.json` (or an
equivalent hash-pinned copy on the server), not an untracked ad-hoc flag.

To select the profile explicitly, add
`--config configs/fewshot/candidate_a3d.json` to the `plan` command. Omit
`--execute` for plan-only validation. The plan and every child plan require new
output directories and refuse to overwrite or resume partial runs. For M&Ms,
set `DATASET=mnms`, use a new `OUT`, and create a second plan with the same
command sequence.

After both plans validate, use the campaign supervisor to run the priority
phase (`teacher_pretrain → teacher_warmup → full` for every k and dataset), then
the remaining 20 arms:

```bash
ACDC_PLAN=/server/runs/acdc-a3d-fewshot-s42/FEWSHOT_PLAN.json
MNMS_PLAN=/server/runs/mnms-a3d-fewshot-s42/FEWSHOT_PLAN.json
STATE=/server/runs/a3d-fewshot/campaign_state.json
LOGS=/server/runs/a3d-fewshot/logs

python scripts/candidate_a_campaign.py \
  --plan "$ACDC_PLAN" --plan "$MNMS_PLAN" \
  --state "$STATE" --log-dir "$LOGS" \
  --device cuda --min-free-gb 15 --execute
```

The supervisor records durable state and a GPU lock. Re-run without
`--execute` for schedule validation. It rejects changed plan hashes, partial
outputs and invalid checkpoint provenance; use `--restart-failed` only after
reviewing a failed job, because it archives the failed output and creates a new
retry directory. The native command below remains a child-by-child fallback and
does not provide the global full-first ordering:

```bash
python scripts/candidate_a_fewshot.py run \
  --plan "$ACDC_PLAN" --execute --device cuda
```

The top-level plan has four child plans and 92 stages per dataset:

```text
$OUT/manifests/k2/train.json       $OUT/k2/PLAN.json
$OUT/manifests/k5/train.json       $OUT/k5/PLAN.json
$OUT/manifests/k8/train.json       $OUT/k8/PLAN.json
$OUT/manifests/k10/train.json      $OUT/k10/PLAN.json
```

Each child plan contains `s42_pretrain`, `s42_warmup`, `s42_full` and the
remaining 20 catalogue arms. The root plan is `$OUT/FEWSHOT_PLAN.json`; the
supervisor state is the authoritative execution order and completion record.
A non-null
external teacher weight is rejected for this sweep because one externally
pretrained checkpoint cannot satisfy the exact per-`k` training provenance;
`weights: null` means the teacher is initialized locally and then trained in
each `k` branch. This is not evidence of a pretrained SOTA teacher.

The profile uses `checkpoint_every_epochs: 0`, so only each final checkpoint is
retained by default. `decoder_chunk_size: null` uses the profile default; an
operator may pass `--decoder-chunk-size N` to the planner when GPU memory
requires it, but the chosen value becomes part of the hash-pinned child plans
and must be held fixed across the comparison.

After development scoring has written a frozen prediction inventory and a
terminal metrics marker, inspect storage candidates with:

```bash
python scripts/candidate_a_retention.py \
  --state "$STATE" \
  --receipt /server/runs/a3d-fewshot/RETENTION_RECEIPT.json
```

The default is a dry run. Adding `--execute` removes only validated frozen
prediction arrays, while retaining metrics, inventories, code packets and
checkpoints. The receipt records every removed file and its hash; never run it
against an unverified or active evaluation directory.

## Metrics and interpretation

Save the existing per-class Dice, F1, recall and precision for TRAIN and frozen
development predictions. Include pooled voxel counts, macro-patient columns,
realized packet bytes, query counts and measured inference time. Also record the
number of labeled patients, annotated frames/volumes, logical sample updates,
and optimizer updates for each `k`. For semi-supervised jobs, break the update
receipt down into labeled-frame visits, U-patient visits, teacher/student
labeled optimizer steps and teacher/student U-consistency steps. Compare the
same development cohort across all four points. Do not select a `k`, checkpoint
or stopping epoch from test masks.

The semi consistency statistic is a soft probability discrepancy over the
confidence-selected voxels. Its normalizer is fixed as
`C_foreground + background_weight * C_background`; it does not create hard
pseudo masks. The labeled and U views use only intensity gamma/voxel-noise
augmentation. A labeled-only run from the earlier campaign is not a causal
semi-supervised control because its update count and augmentation differ. A
matched control must rerun the same arm, seed, resolved schedule and
augmentation with U consistency disabled.

The sweep has one seed, so it reports point estimates only and does not support
across-seed uncertainty. It does not establish SOTA, 96 Dice, teacher quality,
novelty or clinical benefit. Patient/GPU results remain `NOT_RUN` until the
server campaign is executed with verified data and teacher weights. A plan or
synthetic smoke receipt is not a real-run result.
