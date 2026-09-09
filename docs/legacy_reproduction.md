# Historical / Legacy Reproduction Guide

This document records the historical multi-phase (Phase A -> Phase B -> Phase C) training workflows and their corresponding legacy entrypoints. 

> [!NOTE]
> For all active and production training, evaluation, calibration, and bank exports, use the **Canonical Unified Pipeline** documented in [`README.md`](../README.md) via `scripts/train_self_audit.py --config configs/self_audit_full.yaml`.

---

## 1. Historical Multi-Phase Architecture

Historically, Self-Audit was trained using three separate sequential processes:
1. **Phase A (Annotation Bootstrap)**: Trained the ConvNeXt encoder and FPN segmentation decoder on supervised annotation losses ($A_0 \dots A_3$).
2. **Phase B (Auditor Training)**: Froze the encoder and annotation head, training only the dynamic-window auditor on counterfactual transitions.
3. **Phase C (Joint Fine-Tuning)**: Unfroze all components with differentiated learning rates, fine-tuning the model under active threshold-gated rollouts.

### Recipe vs. Interface Differences
- **Live Weights Across Stages vs. Standalone Shell Reloads**:
  - Both the canonical unified runner (`scripts/train_self_audit.py` / `UnifiedTrainer`) and the historical single-process Python runner (`scripts/train_self_audit_legacy.py`) carry *live model weights* across stage/interval boundaries while resetting optimizers at boundaries.
  - In contrast, the standalone multi-process legacy shell workflow (`scripts/run_full_pipeline_legacy.sh` / separate phase scripts `train_annotation.py`, `train_auditor.py`, `finetune_joint.py`) executed disconnected processes and reloaded prior-stage checkpoints from disk.
  - The canonical unified runner replaces the fragmented triple-config architecture with a contiguous 130-epoch global schedule (0-99, 100-119, 120-129) governed by a single Schema Version 1 configuration (`configs/self_audit_full.yaml`), a single unified W&B run, and model selection (`best.pt`) strictly during the final gated interval [120, 130).
- **Run Tracking & Logging**:
  - Historical execution produced three disjoint W&B runs and disconnected run directories.
  - Canonical execution maintains a single W&B run across the entire 130 epochs.

---

## 2. Historical Individual Entrypoints

```bash
# Phase A: Annotation bootstrap (epochs 0..100)
python src/self_audit/training/train_annotation.py \
  --config configs/self_audit_annotation.yaml

# Phase B: Auditor training (epochs 100..120)
python src/self_audit/training/train_auditor.py \
  --config configs/self_audit_auditor.yaml \
  --annotation_checkpoint weights/self_audit/phase_a_annotation.pt

# Phase C: Joint fine-tuning (epochs 120..130)
python src/self_audit/training/finetune_joint.py \
  --config configs/self_audit_joint.yaml \
  --checkpoint weights/self_audit/phase_b_auditor.pt
```

---

## 3. Historical Triple-Config Runner

The historical runner orchestrated the three stages sequentially:

```bash
python scripts/train_self_audit_legacy.py \
  --config_a configs/self_audit_annotation.yaml \
  --config_b configs/self_audit_auditor.yaml \
  --config_c configs/self_audit_joint.yaml \
  --output_dir weights/self_audit_legacy
```

Historical shell scripts:
- `scripts/run_full_pipeline_legacy.sh`
- `scripts/run_full_pipeline_legacy.ps1`

Historical checkpoint filenames produced:
- `phase_a_best.pt` / `phase_a_last.pt`
- `phase_b_best.pt` / `phase_b_last.pt`
- `phase_c_best.pt` / `phase_c_last.pt`

In the canonical unified pipeline, only `best.pt` (selected during gated schedule) and `last.pt` (final step) are produced.
