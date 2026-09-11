# Unified runtime commands and GPU verification boundary

This runbook accompanies the runtime-hardening audit. The final integrated reviewer gate and exact test counts are recorded separately; these commands are not evidence that a GPU training run was executed.

## Canonical ACDC training

```bash
python scripts/train_self_audit.py --config configs/self_audit_full.yaml --device cuda
```

The default outputs are weights/self_audit_full and reports/self_audit_full. Resume a valid completed-epoch state with the same resolved execution recipe/data identity:

```bash
python scripts/train_self_audit.py --config configs/self_audit_full.yaml --device cuda --resume weights/self_audit_full/last.pt
```

Retain the selected_best directory with last.pt; immutable selection references are part of the current checkpoint contract. A copied standalone best.pt is sufficient for frozen evaluation; last.pt plus its referenced selected snapshot are needed for resumable training/selection recovery. Do not remove referenced snapshots to save space. Historical checkpoints whose source/config/worker state cannot establish the new exact-resume contract must fail clearly rather than silently migrate.

## Frozen ACDC to external M&Ms evaluation

This command fixes tau at the existing protocol value 0.0. It performs no training, checkpoint selection or threshold tuning on M&Ms labels:

```bash
python scripts/evaluate_external_mnms.py --config configs/self_audit_acdc_to_mnms.yaml --checkpoint weights/self_audit_full/best.pt --data-root preprocessed_data/mnm --split testing --tau-accept 0.0 --device cuda --output reports/external_mnms.json
```

If using the calibrated ACDC threshold, replace 0.0 with tau_accept from the ACDC calibration artifact after verifying its checkpoint binding matches this best.pt. Keep that value frozen before looking at M&Ms results. The external CLI accepts a fixed scalar; it does not calibrate from external labels.

## Separate native M&Ms supervised experiment

```bash
python scripts/train_self_audit.py --config configs/self_audit_full_mnms.yaml --device cuda
```

This is a separate experiment with its own train/val membership and outputs. It requires prepared paired 3D volumes/masks and valid disjoint M&Ms-native train/val discovery or split metadata. It is not an extension of ACDC training. The four-class mapping remains raw 0 → BG 0, raw 1 → LV 3, raw 2 → MYO 2, raw 3 → RV 1; binary mnm derivatives are refused.

## Short real-GPU first-checkpoint canary (not executed by this audit)

On the declared server, confirm Python 3.10, torch 2.4.1, CUDA 12.1 and RTX 4070 first. Use fresh canary output paths:

```bash
python scripts/train_self_audit.py --config configs/self_audit_full.yaml --device cuda --max_steps 2 --max_val_batches 1 --output_dir weights/runtime_gpu_canary --report_dir reports/runtime_gpu_canary --no_wandb --no_tqdm
```

This bounded canary intentionally writes an incomplete, nonresumable last.pt and skips calibration. It checks actual CUDA initialization, bf16 forward/backward and first-save portability without a long retraining run. It does not establish full-curriculum GPU resume, peak-memory sufficiency, real-data final calibration/diagnostics, backend W&B delivery or production safety. Those remain real-server verification work. An exception must still exit nonzero and preserve a truthful failure artifact where the filesystem permits.

## Interpretation

Training schedule and objectives are preserved: zero-based epochs 0–99 annotation bootstrap, 100–119 auditor, 120–129 joint. Persistent workers are disabled in both canonical configs so epoch-boundary worker RNG can be reconstructed; this changes worker lifetime/throughput and can change the old run stochastic trajectory, not the augmentation distribution or loss objective. Undefined/nonfinite JSON metrics are null, never numeric zero. Required final research-report commit precedes success telemetry finalization; a post-finalization telemetry refresh is optional and reports its own degradation. CUDA nondeterministic kernels remain a separate issue from exact RNG/state restoration.
