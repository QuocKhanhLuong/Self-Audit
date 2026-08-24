# SpecUMamba — Self-Audit

This repository contains the locked Self-Audit baseline for cardiac semantic
annotation and counterfactual self-audit.

The authoritative contract is [`docs.md`](docs.md). The active implementation
is isolated under `src/self_audit/`; the former S3R, distillation, and teacher
namespaces have been removed.

## Active layout

```text
src/self_audit/data/         ACDC/M&Ms loaders and common 2.5-D contract
src/self_audit/models/       ConvNeXt/FPN annotation and dynamic-window expert
src/self_audit/audit/        counterfactual targets, generator, and gate
src/self_audit/losses/       annotation and transition-audit losses
src/self_audit/training/     Phase A, B, and C entrypoints
src/self_audit/evaluation/   volume inference and reporting metrics
configs/self_audit_*.yaml    locked baseline configurations
tests/test_self_audit_*.py   synthetic contract and smoke tests
```

## Training phases

```bash
python src/self_audit/training/train_annotation.py \
  --config configs/self_audit_annotation.yaml

python src/self_audit/training/train_auditor.py \
  --config configs/self_audit_auditor.yaml \
  --annotation_checkpoint weights/self_audit/phase_a_annotation.pt

python src/self_audit/training/finetune_joint.py \
  --config configs/self_audit_joint.yaml \
  --checkpoint weights/self_audit/phase_b_auditor.pt
```

The ACDC-only → M&Ms domain-shift configuration is
`configs/self_audit_acdc_to_mnms.yaml`. M&Ms is not merged into first-baseline
training data.

## Verification

```bash
python -m pytest -q tests/test_self_audit_*.py
python -m py_compile $(find src scripts tests -name "*.py")
```

Real dataset loaders fail clearly when data is absent; they do not fabricate
fixtures or inference results.
