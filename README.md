# SpecUMamba — Self-Audit

This repository contains cardiac segmentation research with distinct training
contracts. **The current research target is fully no-GT training**, with masks
used only by an independent evaluator after predictions are frozen. The target
deployment device is an **RTX 4080 Super 16GB**; Dice >=90% and GPU speed remain
unverified objectives.

## Current No-GT Research Flow

`scripts/run_full_pipeline_v3.py` coordinates the v3 path in
`src/self_audit_pseudolabel/`:

```text
image-only full cine -> appearance/registration teacher -> anatomical seeds
-> conservative pseudo-labels (UNKNOWN=255) -> hashed freeze
-> compact student trained on training-patient pseudo-labels
```

The frozen teacher and student have an independent native-grid evaluator. With
`--train-student`, the runner trains the student, freezes native validation predictions,
and evaluates them separately. Checkpoint-only test export is also available.
No real-data quality or target-GPU result is established by these entrypoints.
Freeze the recipe and checkpoint-selection rules without GT feedback before final evaluation.
See the [05-Oct checkup fixes and verification](docs/pseudolabel_v3_checkup_20261005.md)
and [v3 runbook](docs/pseudolabel_v3_review_runbook.md).

The student uses a shared encoder and A0 head, with optional Dynamic Window
refinement: compact/balanced/accurate execute 0/1/3 internal refinement passes.
It has **80,462 resident parameters** at the checked-in default configuration.
No diffusion runs in this student's forward path. CUTS retains PHATE clustering
and a separate diffusion-condensation route as comparison methods.

### W&B tracking for v3

The v3 orchestrator owns one optional W&B run for the teacher, frozen evaluation,
and optional student stages. W&B is disabled unless `--wandb` is passed. Supply
the key through the environment; never put it in source, YAML, Git history, or a
command-line argument:

```bash
read -r -s -p "W&B API key: " WANDB_API_KEY
printf '\n'
export WANDB_API_KEY
wandb login
```

Online full ACDC teacher + evaluation + student tracking:

```bash
python scripts/run_full_pipeline_v3.py \
  --dataset acdc \
  --root "$PWD/data/ACDC/training" \
  --split-manifest splits/acdc_patient_split_seed42.json \
  --config configs/pseudolabel_v3.json \
  --out "$PWD/runs/pseudolabel_v3/full_$(date +%Y%m%d_%H%M%S)" \
  --teacher-epochs 3 --student-epochs 10 --batch-size 1 --threads 4 \
  --profile balanced --train-student --device cuda \
  --wandb --wandb-mode online \
  --wandb-project self-audit-v3 \
  --wandb-run-name pseudolabel-v3-acdc
```

When the server has no outbound network, use `--wandb-mode offline`; the run
is written below the run directory and can be synced later with
`wandb sync <offline-run-directory>`. W&B telemetry is best-effort and cannot
change the v3 freeze artifacts or evaluation result.

See the [main flow, measurements and research decisions](reports/main_baseline_20261002/00_DECISION.md)
and [primary-source research with the 4080 Super evaluation protocol](reports/main_baseline_20261002/03_RESEARCH.md).
The separate `src/self_audit_maskfree/` implementation and benchmark baselines
remain active dependencies. Historical multi-config runners have been retired;
[reproduction instructions](docs/legacy_reproduction.md) pin their Git revision.
Mask-supported comparisons such as ADNet few-shot have a separate supervision
contract; their scores do not establish fully no-GT performance.

## Supervised Reference Baseline

The commands below train the **supervised reference**, not the no-GT v3 path.
Its authoritative contract is [`docs.md`](docs.md), and its implementation is
under `src/self_audit/`. Some model components are also reused by v3.

## Active Repository Layout

```text
src/self_audit/data/         ACDC/M&Ms loaders and common 2.5-D contract
src/self_audit/models/       ConvNeXt/FPN annotation and dynamic-window expert
src/self_audit/audit/        counterfactual targets, generator, and gate
src/self_audit/losses/       annotation and transition-audit losses
src/self_audit/training/     unified schedule, trainer, config, and utilities
src/self_audit/evaluation/   volume inference, calibration, and reporting metrics
configs/self_audit_full.yaml canonical baseline configuration (Schema Version 1)
tests/                       contract, smoke, and downstream integration tests
```

## Canonical Unified Training Pipeline

Self-Audit is trained via a single canonical command driven by a unified configuration:

```bash
python scripts/train_self_audit.py --config configs/self_audit_full.yaml
```

Training runs as a contiguous 130-epoch curriculum across a single optimization loop:
- **Interval 0 (Epochs 0-99) — `annotation_bootstrap`**:
  - **Objective**: `weighted_a0_a3`
  - **Trainable**: `annotation` (ConvNeXt-Tiny encoder + FPN segmentation heads)
  - **Auditor**: Frozen (`auditor_lr = 0.0`)
  - **Learning Rates**: `encoder_lr = 3e-5`, `annotation_lr = 3e-4`
  - **Losses**: `annotation_weight = 1.0`, `audit_weight = 0.0`
  - **Rollout**: `propagate_no_audit`, `transition_population = "none"`
  - **Data Augmentation**: Enabled
- **Interval 1 (Epochs 100-119) — `auditor_training`**:
  - **Objective**: `counterfactual_audit`
  - **Trainable**: `auditor` (Dynamic-window self-audit network)
  - **Encoder & Heads**: Frozen (`encoder_lr = 0.0`, `annotation_lr = 0.0`)
  - **Learning Rates**: `auditor_lr = 3e-4`
  - **Losses**: `annotation_weight = 0.0`, `audit_weight = 1.0`
  - **Rollout**: `annotation_eval`, `transition_population = "adjacent_and_synthetic"`
  - **Optimizer**: Fresh AdamW instance initialized at epoch 100 boundary
  - **Data Augmentation**: Disabled
- **Interval 2 (Epochs 120-129) — `joint_self_audit`**:
  - **Objective**: `retained_final_annotation`
  - **Trainable**: `all` (Encoder, FPN heads, and auditor)
  - **Learning Rates**: Differentiated fine-tuning (`encoder_lr = 1e-6`, `annotation_lr = 1e-5`, `auditor_lr = 1e-5`)
  - **Losses**: `annotation_weight = 1.0`, `audit_weight = 1.0`
  - **Rollout**: `threshold_gate`, `transition_population = "active_attempted"`
  - **Optimizer**: Fresh AdamW instance initialized at epoch 120 boundary
  - **Data Augmentation**: Enabled

### Recipe Changes vs. Interface Changes

- **Live Model Weights Across Stages**: The canonical unified runner (`scripts/train_self_audit.py` / `UnifiedTrainer`) carries live model weights across interval boundaries while resetting optimizer/scheduler states. The retired single-process runner did the same; the retired multi-process shell workflow reloaded saved stage-best checkpoints across processes. These were different recipes; see [historical reproduction](docs/legacy_reproduction.md).
- **Unified Single-Config Interface**: Historical multi-phase flags (`--config_a`, `--config_b`, `--config_c`, `--epochs_a`, `--epochs_b`, `--epochs_c`, `--start_phase`) and multi-phase checkpoint names (`phase_c_best.pt`) are removed from the canonical CLI. The entire curriculum is governed by `configs/self_audit_full.yaml` executed as a contiguous 130-epoch schedule with a single unified W&B run.

### Checkpoint Selection & Calibration Lineage

- **Saved Checkpoints**: The run saves `best.pt` and `last.pt` in `output_dir` (`weights/self_audit_full`).
- **Selection Boundary**: Checkpoint selection for `best.pt` occurs **strictly during the profile's gated schedule interval** (the interval whose `rollout` is `threshold_gate`), and the boundary is therefore per profile, not a global `120`. For the staged profiles (`self_audit_full{,_mnms}.yaml`) the gated interval is `[120, 130)`, so epochs `< 120` cannot be selected. For the joint-from-epoch-1 profiles (`self_audit_joint_from_start{,_mnms}.yaml`) the single interval `[0, 130)` is gated, so selection is open from epoch 0. `checkpoint.best_selection_min_epoch` outside the profile's own gated interval is rejected by the strict loader in both directions.
- **Integrity**: A missing `best.pt` is not a successful completed pipeline; silent fallback to `last.pt` is strictly prohibited. Bounded smoke runs (`--max_steps` or `--max_val_batches`) cannot certify completion or publish calibration artifacts.
- **Bound Post-Training Calibration**: Upon completion, `best.pt` is strictly bound into `run_post_training_calibration`, emitting `calibration.json` with cryptographic lineage verification and final diagnostics (`headroom` and `decomposition`) evaluated at the selected $\tau_{calibrated}$.

### Run Tracking & Experiment Identities

- **Single W&B Run**: A single W&B run tracks the full 130-epoch curriculum using monotonic global logging keys (`train/loss`, `val/macro_dice`, `modes/final_macro_dice`, `opt_step`). Accidental phase prefixes (`phase_a/`, `phase_b/`, `phase_c/`) are stripped.
- **Canonical Identities**:
  - `self_audit_full`: Canonical ACDC unified curriculum.
  - `self_audit_full_mnms`: Canonical M&Ms native supervision curriculum.

### Resuming Training

```bash
python scripts/train_self_audit.py \
  --config configs/self_audit_full.yaml \
  --resume weights/self_audit_full/last.pt
```

Resuming strictly restores model weights, optimizer/scheduler states, AMP scaler, RNG state seeds, and best metric tracking. The checkpoint must match the configuration signature; checkpoints from incomplete epochs or restricted validation are rejected.

## Exact Execution Commands

### 1. ACDC Unified Training

```bash
python scripts/train_self_audit.py --config configs/self_audit_full.yaml
```

### 2. External M&Ms Evaluation

Evaluate the frozen ACDC-trained `best.pt` checkpoint on the external M&Ms testing split without in-domain adaptation, calibration, or training:

```bash
python scripts/evaluate_external_mnms.py \
  --config configs/self_audit_acdc_to_mnms.yaml \
  --checkpoint weights/self_audit_full/best.pt \
  --data-root preprocessed_data/mnm \
  --split testing \
  --tau-accept 0.0 \
  --device cuda \
  --output reports/external_mnms.json
```

`--checkpoint` must point at the frozen ACDC `best.pt` that was actually produced by the source run — `weights/self_audit_full/best.pt` for the canonical staged run, `runs/<ACDC_RUN>/weights/best.pt` for a native Candidate C runner run, `weights/self_audit_joint_from_start/best.pt` for a joint-profile run — and the path above is only the canonical example, not a default that fits every run. `--tau-accept 0.0` is a fixed decision threshold stated for this evaluation; it is not the source run's calibrated $\tau$ and is not derived from `calibration.json`.

`data/ACDC`, `preprocessed_data/ACDC`, and `preprocessed_data/mnm` are local dataset paths ignored by Git. The external evaluation requires four-class `preprocessed_data/mnm`; the two-class `mnm_binary` derivative is incompatible with the four-class checkpoint and is rejected. The external command never trains, calibrates, or selects a threshold from M&Ms.

### 3. Native M&Ms Supervision Training

Train the full unified Self-Audit curriculum directly on M&Ms native data.
Defaults differ per entrypoint and are stated where each one is described:
`scripts/train_self_audit.py` and `scripts/run_full_pipeline.sh` default to the
staged `configs/self_audit_full.yaml`, so the joint-from-epoch-1 curriculum must
be named explicitly there, while `scripts/run_acdc_mnms_candidate_c.sh` defaults
to the joint-from-epoch-1 profiles (see below):

```bash
# Staged M&Ms curriculum (annotation_bootstrap -> auditor_training -> joint).
python scripts/train_self_audit.py --config configs/self_audit_full_mnms.yaml

# Joint from epoch 1 on M&Ms: one interval [0, 130), trainable=all,
# objective=retained_final_annotation, threshold gate live from the first step.
python scripts/train_self_audit.py --config configs/self_audit_joint_from_start_mnms.yaml
```

`scripts/run_full_pipeline.sh` still defaults to `configs/self_audit_full.yaml`;
passing `--config configs/self_audit_joint_from_start_mnms.yaml` is the only way
to reach the joint profile through it.

**The profile YAML alone is the baseline network, not Candidate C.** Both
`configs/self_audit_joint_from_start_mnms.yaml` and `configs/self_audit_full_mnms.yaml`
declare `model.window_mode: "current"`, so the commands above train the
current-window baseline under the stated curriculum. Candidate C is selected by
the native runner, which generates a run-local config with
`model.window_mode: "candidate_c"` and leaves the checked-in profiles untouched:

```bash
# ACDC then M&Ms, both Candidate C, both joint from epoch 1 (the runner default).
bash scripts/run_acdc_mnms_candidate_c.sh

# Same two runs under the staged curriculum instead.
CURRICULUM=staged bash scripts/run_acdc_mnms_candidate_c.sh
```

Each invocation writes its generated run configs into its own fresh
`run_configs/<curriculum>_<stamp>_XXXXXX/` directory and keeps them as the
provenance record of what was executed, so a second invocation can never rewrite
the config a running invocation's pending M&Ms leg is about to read. The two
legs are independent runs: no checkpoint, run directory, or W&B project is
shared, and neither leg resumes the other.

Native M&Ms supervision requires an unambiguously paired cohort: startup
validation fails if any `volumes/` entry has no matching `masks/` entry or vice
versa, if two files normalize to one case key (`case.npy` beside `case.npz`), if
one image matches more than one candidate mask, or if one mask would be claimed
by two images — rather than training on a silently smaller or guessed cohort.
Evaluation-only discovery keeps its
tolerant behaviour for trees that intentionally carry unlabelled volumes.

### 4. Proposal-1 Frozen Transition Bank Export

Export the Proposal-1 transition bank using the canonical unified config and bound `best.pt` checkpoint. Generation is GT-free and on-policy:

```bash
python scripts/export_transition_bank.py \
  --config configs/self_audit_full.yaml \
  --checkpoint weights/self_audit_full/best.pt \
  --output reports/transition_bank.json \
  --device cuda
```

## Legacy Reproduction

For historical multi-phase (Phase A -> Phase B -> Phase C) triple-config training and reproduction commands, see [`docs/legacy_reproduction.md`](docs/legacy_reproduction.md).

## Verification

```bash
python -m pytest tests/test_wave3_downstream.py -v
python -m pytest tests/test_unified_trainer.py -v
python -m py_compile $(find src scripts tests -name "*.py")
```

## Audited cardiac adapters

The existing model contract is unchanged: each sample is [3,H,W] with
[z-1,z,z+1] spatial slices, replicated boundary slices, center-slice target,
in-plane resize, and volume-wise 0.5/99.5 percentile clipping plus z-score.
The unified labels are 0 Background, 1 RV, 2 MYO, 3 LV.

The audited configurations are:

- configs/cmr_multi.yaml
- configs/cmrxmotion.yaml
- configs/self_audit_mixed_four_datasets.yaml

Dataset source paths are configuration values. The source directories
data/CMR-MULTI and data/CMRxMotion are read-only. Run the audits before
training:

    python scripts/audit_cmr_multi.py --data-root data/CMR-MULTI --output-dir reports
    python scripts/audit_cmrxmotion.py --data-root data/CMRxMotion --output-dir reports

Generate subject-level manifests (never slice/frame-level splits):

    python scripts/generate_subject_splits.py --dataset cmr_multi --data-root data/CMR-MULTI --output splits/cmr_multi_subject_split_seed42.json
    python scripts/generate_subject_splits.py --dataset cmr_motion --data-root data/CMRxMotion --output splits/cmrxmotion_subject_split_seed42.json

Then build the evidence report and visual QC:

    python scripts/build_dataset_compatibility_report.py
    python scripts/generate_dataset_qc.py --cmr-multi-root data/CMR-MULTI --cmrxmotion-root data/CMRxMotion --output-dir reports/qc

Adapter/model smoke verification:

    python scripts/verify_dataset_adapters.py --config configs/cmr_multi.yaml --device cpu --num-workers 0
    python scripts/verify_dataset_adapters.py --config configs/cmrxmotion.yaml --device cpu --num-workers 0
    python scripts/verify_dataset_adapters.py --config configs/self_audit_mixed_four_datasets.yaml --device cpu --num-workers 0 --allow-missing-sources

Mixed training uses dataset/subject-balanced sampling. ACDC and M&Ms must be
provided through their configured roots and subject-safe manifests; the smoke
command reports missing roots instead of fabricating masks. CMR-MULTI cases
with uncertain Z/T inference are excluded. CMRxMotion volumes without GT are
excluded from supervised training, and affine-mismatch cases require explicit
allow_affine_mismatch plus visual QC.

The adapter layer changes file discovery, native format interpretation,
subject metadata, Z/T reconstruction, and source label mapping only. It does
not change the model, loss, optimizer, scheduler, or training-loop logic.
