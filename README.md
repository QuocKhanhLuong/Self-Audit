# SpecUMamba — Self-Audit

This repository contains the locked Self-Audit baseline for cardiac semantic
annotation and counterfactual self-audit.

The authoritative contract is [`docs.md`](docs.md). The active implementation
is isolated under `src/self_audit/`; the former S3R, distillation, and teacher
namespaces have been removed.

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

- **Live Model Weights Across Stages**: The canonical unified runner (`scripts/train_self_audit.py` / `UnifiedTrainer`) and the historical single-process runner (`scripts/train_self_audit_legacy.py`) both carry live model weights across phase/interval boundaries while resetting optimizer/scheduler states. In contrast, the standalone multi-process legacy shell workflow (`scripts/run_full_pipeline_legacy.sh` / separate `train_annotation.py`, `train_auditor.py`, `finetune_joint.py` entrypoints) reloaded saved stage-best checkpoints from disk across disconnected processes.
- **Unified Single-Config Interface**: Historical multi-phase flags (`--config_a`, `--config_b`, `--config_c`, `--epochs_a`, `--epochs_b`, `--epochs_c`, `--start_phase`) and multi-phase checkpoint names (`phase_c_best.pt`) are removed from the canonical CLI. The entire curriculum is governed by `configs/self_audit_full.yaml` executed as a contiguous 130-epoch schedule with a single unified W&B run.

### Checkpoint Selection & Calibration Lineage

- **Saved Checkpoints**: The run saves `best.pt` and `last.pt` in `output_dir` (`weights/self_audit_full`).
- **Selection Boundary**: Checkpoint selection for `best.pt` occurs **strictly during the gated schedule interval** (`[120, 130)`). Early epochs (< 120) cannot be selected as `best.pt`.
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

`data/ACDC`, `preprocessed_data/ACDC`, and `preprocessed_data/mnm` are local dataset paths ignored by Git. The external evaluation requires four-class `preprocessed_data/mnm`; the two-class `mnm_binary` derivative is incompatible with the four-class checkpoint and is rejected. The external command never trains, calibrates, or selects a threshold from M&Ms.

### 3. Native M&Ms Supervision Training

Train the full unified Self-Audit curriculum directly on M&Ms native data:

```bash
python scripts/train_self_audit.py --config configs/self_audit_full_mnms.yaml
```

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
