# Baseline recovery and one shared ACDC + M&Ms student

This is a **code-only delivery**. Patient training, DINO inference, historical
payload recovery, PRO4000 measurements and patient quality gates were NOT RUN.
CPU manufactured tests are software evidence only. Do not expand the head audit
matrix, increase L, or extend the schedule when a quality gate fails.

## Locked recipes and gates

- Recover the historical **no-window** baseline: ACDC L8/U72, seed 42,
  10,000 outer iterations including 1,000 full-model supervised warmup iterations,
  supervised batch 8 and U batch 8, crop 256. AdamW updates encoder, input adapter,
  decoder, detail and fuse; SGD updates the segmentation head. Supervised loss is
  CE + present-foreground Dice. EMA runs once after the committed student update.
- DINO supplies the encoder initialization. The newly initialized segmentation
  path must train. `update_scope=head` is a legacy pseudo-guard setting; it does
  **not** freeze the `ema_full` supervised model. Startup and first-step receipts
  record actual optimizer ownership, gradients and parameter changes.
- Before enabling U, a fixed TRAIN-L probe requires valid predictions for RV,
  MYO and LV. It excludes padding/ignore=255 and runs after warmup, before the next
  supervised update. U sampling, augmentation and pseudo-labeling never read U masks.
- Tiny-fit diagnostic: fixed TRAIN patients `patient007` and `patient010`, fixed
  selected slices, at most 200 updates, FG Dice >= .95 and all foreground classes.
  This diagnostic checkpoint is never used to initialize the shared experiment.
- Baseline comparison: final step-10,000 raw student, foreground patient-macro
  >= **.8118620630** (historical .8318620630 minus .02), complete 50-patient /
  100-frame coverage and all foreground classes active. The already observed
  TEST50 is a **development recovery benchmark**, not fresh confirmation.
- Shared experiment: **ACDC 4 + M&Ms 4 = total L8**, fresh pinned SSL DINO, one
  student/EMA pair, 50/50 alternating domains. Same 10,000/1,000 schedule and L8/U8
  batches initially. Development thresholds per dataset: FG >= .75 and each
  RV/MYO/LV >= .60 with complete coverage. Stop if either dataset fails.

An outer experiment-release receipt permits shared training only after baseline
verification. Offline `gate-check` rehashes artifacts and rescores frozen
predictions against references. Learner startup verifies the pinned receipt
without opening reference mask arrays. It does not use query labels for updates,
sampling, pseudo-targets or per-step acceptance.

## Environment and paths

Use the committed branch and an environment with PyTorch, NumPy, pytest and
matplotlib. The DINO source checkout and licensed weights must already be present;
these commands do not download them. CUDA is required for the intended PRO4000
24GB run and never silently falls back to CPU.

Prepare `model.json` with the five keys in `configs/backbone.example.json`:

```json
{
  "backbone": "dinov3_convnext_tiny",
  "source_repo": "/absolute/path/to/dinov3",
  "source_commit": "6876159a11b4df116f30f667f8c9888617df0751",
  "weights": "/absolute/path/to/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth",
  "weights_sha256": "21b726bb286e037f00a23fb4699fa9bda9c75b6a615bd57ce3013cec1b528d54"
}
```

The shell examples below assume Linux and existing prepared data. Replace paths
and `<...>` SHA placeholders with actual values printed by `sha256sum FILE`.
Use a new output directory for every command. Never overwrite raw/prepared data.

```bash
python scripts/check_repo.py
python -m pytest -q
```

## 1. Prefer recovery of the existing historical checkpoint

The manifest declarations and recipe were verified from Git. The physical
historical payload was unavailable in this code-only session. Check it first:

```bash
python scripts/run_baseline_recovery.py verify-history \
  --manifest reports/window_pilot_test50_20261024/baseline/CHECKPOINT_MANIFEST.json \
  --payload /root/self-audit-pilot-artifacts/runs/baseline/final/weights.pt \
  --require-payload --out runs/history_verification.json
```

Expected payload SHA:
`b36bcafe9613325d85ee70dabd6a36fef7c17429882967831a151c0cce6d8437`.
Without `--require-payload`, the command verifies declarations only and reports
`physical_payload_verified=false`; that is not a recovered model.

Inference of an old checkpoint must execute its exact source. The current loader
deliberately rejects a different source closure. Do not edit the old checkpoint
manifest or use `strict=False` to bypass this check.

```bash
git worktree add --detach ../Self-Audit-history-0f54455 0f54455b67bb11a72fbfe2a8ac6fae4073f3e9ca
cd ../Self-Audit-history-0f54455
python evaluate.py \
  --checkpoint /root/self-audit-pilot-artifacts/runs/baseline/final \
  --checkpoint-sha256 03c1730f88941ada898bed7fcf1b9f13cdd8f08bcce4390539fec665c82e97de \
  --data /absolute/path/to/acdc_test50.json \
  --data-sha256 22a9ea63821f266bd59e810bcd8f9a6958e6f92cc363e4a360b5ab9d4eb1e25c \
  --out /absolute/path/to/runs/history_fresh_eval --device cuda
```

The original DINO paths declared in the historical manifest must be available.
If those pinned assets are unavailable, record BLOCKED. Do not change their
identity. Return to this delivery checkout for the diagnostic and gate steps.
An intact historical payload plus valid fresh inference avoids retraining 10,000 steps.

## 2. Tiny-fit and bounded profile

The locked TRAIN SHA is
`e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4`.
It contains exactly patient007, 010, 011, 024, 026, 041, 077 and 099 as L.

```bash
python scripts/run_baseline_recovery.py tiny-fit \
  --data /absolute/path/to/acdc_train.json \
  --data-sha256 e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4 \
  --model-config model.json --out runs/tiny_fit --device cuda \
  --max-steps 200 --target-dice 0.95

python scripts/run_baseline_recovery.py profile \
  --config configs/baseline_recovery/baseline_l8_10k.json \
  --model-config model.json --data /absolute/path/to/acdc_train.json \
  --data-sha256 e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4 \
  --out runs/profile_baseline --device cuda --steps 4
```

`PROFILE_REPORT.json` records synchronized latency, cold total/first commit cost,
marginal iteration cost, optimized throughput, sampled/optimized slices, actual
optimizer calls, EMA count and allocated/reserved VRAM. The baseline's initial
four iterations measure supervised warmup, **not post-warmup U throughput**.
The profiler calls the canonical trainer and cannot certify segmentation quality.
No microbatch accumulation change is implemented: if batch 8 does not fit, stop
and resolve equivalent loss/optimizer semantics before declaring a reproduction.

## 3. Retrain only when historical recovery is unavailable

Lock the source, resolved model/config, TRAIN manifest, reference protocol and
final-step selection before the run. Keep the generated protocol immutable.

```bash
python scripts/run_baseline_recovery.py predeclare \
  --config configs/baseline_recovery/baseline_l8_10k.json --model-config model.json \
  --train-manifest /absolute/path/to/acdc_train.json \
  --evaluation-manifest /absolute/path/to/acdc_test50.json \
  --out runs/baseline_protocol.json --protocol-type baseline
sha256sum runs/baseline_protocol.json

python scripts/run_baseline_recovery.py train-baseline \
  --config configs/baseline_recovery/baseline_l8_10k.json --model-config model.json \
  --data /absolute/path/to/acdc_train.json \
  --data-sha256 e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4 \
  --protocol runs/baseline_protocol.json --protocol-sha256 <protocol_sha> \
  --out runs/baseline_10k --device cuda --allow-full-run
```

`train.py` is the sole trainer. `STARTUP.json` records parameter names, optimizer
groups, L/U batch sizes and initialization. The first row of `steps.jsonl` records
nonzero module gradients and updates. Rows log completed optimizer/EMA counts,
slice counts and actual patient/domain exposure; `REPORT.json` records totals.
10000 outer iterations can include 10000 supervised plus up to 9000 U updates.
The canonical semi-v2 checkpoint format still declares `resume=NOT_SUPPORTED`;
this delivery does not promise exact resume for that trainer. An interrupted run
is incomplete evidence, not a new budget or an automatically extended run.

```bash
python scripts/evaluate_shared_student.py \
  --checkpoint runs/baseline_10k/final --checkpoint-sha256 <final_MANIFEST_sha> \
  --train-manifest /absolute/path/to/acdc_train.json \
  --train-manifest-sha256 e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4 \
  --data-acdc /absolute/path/to/acdc_test50.json \
  --data-acdc-sha256 22a9ea63821f266bd59e810bcd8f9a6958e6f92cc363e4a360b5ab9d4eb1e25c \
  --out runs/baseline_eval --device cuda
```

## 4. Verify the outer baseline release gate

For a newly trained baseline:

```bash
python scripts/run_baseline_recovery.py gate-check \
  --checkpoint-manifest runs/baseline_10k/final/MANIFEST.json \
  --checkpoint-payload runs/baseline_10k/final/weights.pt \
  --train-manifest /absolute/path/to/acdc_train.json \
  --train-config runs/baseline_10k/CONFIG.json --protocol runs/baseline_protocol.json \
  --evaluation-manifest /absolute/path/to/acdc_test50.json \
  --frozen-predictions runs/baseline_eval/PREDICTIONS_FROZEN.json \
  --metrics runs/baseline_eval/METRICS.json --tinyfit-receipt runs/tiny_fit/TINY_FIT_REPORT.json \
  --out runs/baseline_gate.json
sha256sum runs/baseline_gate.json
```

For historical reuse, use its physical `final/MANIFEST.json` and `weights.pt`,
the archived `baseline/TRAIN_CONFIG.json`, the archived `PROTOCOL.json`, and the
fresh original-evaluator output from step 1. Keep the same TRAIN/evaluation pins.
The tiny-fit report is generated with this delivery's source and the same pinned
DINO model configuration. The gate supports the original probability-array format
and the new raw-label format explicitly; it never rewrites historical provenance.

Missing files/identity are `BLOCKED_MISSING_EVIDENCE`; incorrect ownership/lineage
is `implementation_error`; failed tiny-fit is `not_converged`; a complete baseline
below the declared threshold is `empirical_failure`. Preserve failures and stop.

## 5. Prepare a real loader-supported 4+4 manifest

The inputs are existing locked prepared TRAIN manifests. Selection is seeded
from their available labeled candidate pools; this restriction is disclosed as
`restricted_manifest_labeled`, not falsely described as selection over all TRAIN.
Demoted L records become image-only U without opening their masks. U records have
only `id`, `image`, `image_sha256`. No U/reference labels are used for selection.

Supply pinned image catalogs covering every declared evaluation ID in each domain.
The preparation command checks actual byte/voxel overlaps before opening selected
TRAIN masks. Keep ACDC 055/076 and the eight recorded M&Ms duplicate IDs quarantined.
It rejects overlap/incomplete catalogs instead of changing locked cohorts.

```bash
python scripts/prepare_shared_training.py \
  --acdc-manifest /absolute/path/to/acdc_train.json --acdc-manifest-sha256 <acdc_train_sha> \
  --mnm-manifest /absolute/path/to/mnm_train.json --mnm-manifest-sha256 <mnm_train_sha> \
  --mnm-receipt reports/baseline_recovery/MNM_PREPARATION_SOURCE.json \
  --mnm-receipt-sha256 82f6a05b38fa8e69607b0fd6fcecf0062809353dfb47cc8107cbbfb865bf030b \
  --acdc-eval-manifest /absolute/path/to/acdc_eval_image_catalog.json --acdc-eval-manifest-sha256 <acdc_catalog_sha> \
  --mnm-eval-manifest /absolute/path/to/mnm_eval_image_catalog.json --mnm-eval-manifest-sha256 <mnm_catalog_sha> \
  --acdc-budget 4 --mnm-budget 4 --seed 42 --out /absolute/path/to/shared_prepared
```

Catalogs use `patients` with pinned `id/image/image_sha256` entries and declared
evaluation IDs. They may cover locked VAL and TEST images together; no masks are
needed for overlap checks. The historical M&Ms receipt is valid only for its bound
manifest hash. For other prepared inputs, provide explicit verified ontology
metadata or a manifest-bound preparation receipt. A raw mapping plan alone does
not prove an applied transformation. Raw M&Ms mapping is 1→3, 2→2, 3→1 exactly once;
canonical outputs are never remapped again. Labels are BG0/RV1/MYO2/LV3/ignore255.

`shared_prepared/train.json` records the exact eight IDs, domain map, source hashes,
class mapping, quarantine, preprocessing and lineage. `PREPARATION.json` records
the loader roundtrip. These real manifests were not generated in this code-only
session because their physical inputs were unavailable.

Copy `configs/baseline_recovery/shared_4plus4_10k.json` to `shared_run.json` and set
`baseline_recovery.baseline_certificate_path` and `baseline_certificate_sha256` to
the verified gate file/pin. Use `model.json` via `--model-config` at training.
For `check-shared-prereq`, put those same resolved five model fields directly in
`shared_run.json`; it validates the resolved configuration.

```bash
python scripts/run_baseline_recovery.py check-shared-prereq \
  --certificate runs/baseline_gate.json --expected-cert-sha <gate_sha> \
  --shared-manifest /absolute/path/to/shared_prepared/train.json --shared-config shared_run.json
python scripts/run_baseline_recovery.py profile \
  --config shared_run.json --data /absolute/path/to/shared_prepared/train.json \
  --data-sha256 <shared_train_sha> --out runs/shared_profile --device cuda --steps 4
python train.py --config shared_run.json \
  --data /absolute/path/to/shared_prepared/train.json --data-sha256 <shared_train_sha> \
  --out runs/shared_4plus4 --device cuda --allow-full-run
```

Do not resume or initialize from the ACDC-L8 segmentation checkpoint. Shared
startup requires the exact SSL DINO hash; the label budget remains exactly 4+4.

## 6. One checkpoint, two datasets, frozen evaluation and plots

```bash
python scripts/evaluate_shared_student.py \
  --checkpoint runs/shared_4plus4/final --checkpoint-sha256 <shared_final_MANIFEST_sha> \
  --train-manifest /absolute/path/to/shared_prepared/train.json --train-manifest-sha256 <shared_train_sha> \
  --data-acdc /absolute/path/to/acdc_val.json --data-acdc-sha256 <acdc_val_sha> \
  --data-mnm /absolute/path/to/mnm_val.json --data-mnm-sha256 <mnm_val_sha> \
  --out runs/shared_eval --device cuda
python scripts/render_recovery_report.py --train-log runs/shared_4plus4/steps.jsonl \
  --evaluation runs/shared_eval --manifest /absolute/path/to/acdc_val.json /absolute/path/to/mnm_val.json \
  --out runs/shared_figures --max-overlays 3
```

All predictions for both domains are saved and hashed before any reference array
opens. Raw student multiclass argmax is primary. Confusion is pooled within each
frame volume, then Dice averages available frames per patient/class and foreground
classes within each patient, then patients. Undefined values remain undefined.
Reports include per-class/per-dataset scores, patient/frame coverage, prediction
counts, absent-class FP rates with fixed valid-voxel denominators and failure cases.
BG-inclusive Dice and equal-domain mean are secondary. This is multi-dataset
training, not external-domain generalization.

Optional student-only packaging uses the existing exporter (despite its legacy
filename, it also supports the A0/no-memory baseline):

```bash
python scripts/export_teacher_memory_student.py \
  --checkpoint runs/shared_4plus4/final --checkpoint-sha256 <shared_final_MANIFEST_sha> \
  --out runs/shared_student_only
```

The export contains student tensors only. `load_exported_student(folder, sha)`
loads it under the pinned source. For an image-only forward, load a prepared
float32 `[T,Z,Y,X]` cine in [0,1], use `TrainingData.context(cine,t,z)` for the
three adjacent Z slices with edge repetition, then `model(context).argmax(1)`.
Preprocessing is per-cine image-only .5/99.5 percentile normalization, native
orientation/grid, no resize; padding validity excludes artificial pixels.
Do not advertise any exported patient model as complete until the fixed gates pass.

## Software receipts and outstanding evidence

See `reports/baseline_recovery/RECIPE_COMPARISON.md`, `HISTORICAL_EVIDENCE.json`,
`VALIDATION.json` and the manufactured integration artifacts. The initial audit
matrix remains deferred. Actual patient checkpoint bytes, PRO4000 VRAM/latency,
tiny-fit convergence and ACDC/M&Ms quality are still **NOT RUN** in this delivery.
