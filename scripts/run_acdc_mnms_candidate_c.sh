#!/usr/bin/env bash
# Sequential native Candidate C training: ACDC from epoch 0, then M&Ms from epoch 0.
# The two runs are independent; M&Ms never loads the ACDC checkpoint.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p runs logs run_configs

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
ACDC_RUN="${ACDC_RUN:-acdc_candidate_c_4070_${STAMP}}"
MNMS_RUN="${MNMS_RUN:-mnms_candidate_c_4070_${STAMP}}"

ACDC_CONFIG="run_configs/acdc_candidate_c.yaml"
MNMS_CONFIG="run_configs/mnms_candidate_c.yaml"

# Generate run-specific configs from the canonical configs so this script is
# self-contained and the canonical baseline YAML files remain unchanged.
python - <<'PY'
from pathlib import Path
import yaml

pairs = [
    (
        Path("configs/self_audit_full.yaml"),
        Path("run_configs/acdc_candidate_c.yaml"),
        "self_audit_acdc_candidate_c",
    ),
    (
        Path("configs/self_audit_full_mnms.yaml"),
        Path("run_configs/mnms_candidate_c.yaml"),
        "self_audit_mnms_candidate_c",
    ),
]

for src, dst, experiment_name in pairs:
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
    cfg["experiment"]["name"] = experiment_name
    cfg["model"]["window_mode"] = "candidate_c"
    cfg["training"]["rollout"]["predicted_history_exposure"] = True
    cfg["training"]["rollout"]["predicted_history_weight"] = 0.1
    dst.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(
        f"[config] {dst}: window_mode={cfg['model']['window_mode']} "
        f"predicted_history_exposure={cfg['training']['rollout']['predicted_history_exposure']}"
    )
PY

echo "=============================================================================="
echo "Self-Audit sequential Candidate C native training"
echo "GPU visibility : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "ACDC run       : ${ACDC_RUN}"
echo "M&Ms run       : ${MNMS_RUN}"
echo "=============================================================================="

# -----------------------------------------------------------------------------
# 1) Native ACDC — independent full 130-epoch run from initialization.
# -----------------------------------------------------------------------------
echo
printf '>>> START ACDC: %s\n' "$ACDC_RUN"

python -u scripts/train_self_audit.py \
  --config "$ACDC_CONFIG" \
  --data_root preprocessed_data/ACDC/training \
  --device cuda \
  --wandb \
  --wandb_mode online \
  --wandb_project self-audit-acdc-candidate-c \
  --wandb_run_name "$ACDC_RUN" \
  --output_dir "runs/$ACDC_RUN/weights" \
  --report_dir "runs/$ACDC_RUN/reports" \
  --no_tqdm \
  2>&1 | tee "logs/${ACDC_RUN}.log"

printf '>>> ACDC COMPLETE: %s\n' "$ACDC_RUN"

# -----------------------------------------------------------------------------
# 2) Native M&Ms — a new independent full 130-epoch run from initialization.
#    No --resume and no ACDC checkpoint are supplied here by design.
# -----------------------------------------------------------------------------
echo
printf '>>> START M&Ms: %s\n' "$MNMS_RUN"

python -u scripts/train_self_audit.py \
  --config "$MNMS_CONFIG" \
  --data_root preprocessed_data/mnm \
  --device cuda \
  --wandb \
  --wandb_mode online \
  --wandb_project self-audit-mnms-candidate-c \
  --wandb_run_name "$MNMS_RUN" \
  --output_dir "runs/$MNMS_RUN/weights" \
  --report_dir "runs/$MNMS_RUN/reports" \
  --no_tqdm \
  2>&1 | tee "logs/${MNMS_RUN}.log"

printf '>>> M&Ms COMPLETE: %s\n' "$MNMS_RUN"

echo
echo "=============================================================================="
echo "ALL NATIVE RUNS COMPLETE"
echo "ACDC : runs/$ACDC_RUN"
echo "M&Ms : runs/$MNMS_RUN"
echo "=============================================================================="
