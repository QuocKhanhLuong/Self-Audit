#!/usr/bin/env bash
# ==============================================================================
# Self-Audit: Canonical Unified Training Pipeline Runner
#
# Executes the single-process unified training pipeline (Schema Version 1)
# via configs/self_audit_full.yaml across the 130-epoch curriculum.
#
# For historical multi-stage (A -> B -> C) execution, see:
#   scripts/run_full_pipeline_legacy.sh
# ==============================================================================

set -euo pipefail

CONFIG="configs/self_audit_full.yaml"
DATA_ROOT=""
SPLIT_MANIFEST=""
DEVICE=""
NUM_WORKERS=""
MAX_STEPS=""
MAX_VAL_BATCHES=""
OUTPUT_DIR=""
REPORT_DIR=""
RESUME=""
TAU_ACCEPT=""
SKIP_CALIBRATION=false
WANDB_ENABLED=""
WANDB_MODE=""
WANDB_PROJECT=""
WANDB_ENTITY=""
WANDB_RUN_NAME=""
NO_TQDM=false
SMOKE=false

show_help() {
  cat << 'HELP_EOF'
Usage: bash scripts/run_full_pipeline.sh [OPTIONS]

Canonical Unified Pipeline Options:
  --config <path>                         Path to unified YAML (default: configs/self_audit_full.yaml)
  --smoke                                 Run bounded smoke verification (CPU safe, 2 steps)
  --device <cuda|cpu>                     Target compute device (default: cuda)
  --data_root <path>                      Dataset root directory override
  --split_manifest <path>                 Custom split manifest JSON path override
  --num_workers <int>                     DataLoader worker count
  --max_steps <int>                       Global optimizer step budget
  --max_val_batches <int>                 Maximum validation batches to evaluate
  --output_dir <dir>                      Checkpoint output directory (default: weights/self_audit)
  --report_dir <dir>                      Report output directory (default: reports)
  --resume <path>                         Path to checkpoint (last.pt) to resume from
  --tau_accept <float>                    Override decision threshold
  --skip_calibration                      Skip post-training calibration
  --wandb                                 Enable Weights & Biases logging
  --no_wandb                              Disable Weights & Biases logging
  --wandb_mode <offline|online|disabled>  WandB run mode (default: offline)
  --wandb_project <name>                  WandB project name (default: self-audit)
  --wandb_entity <name>                   WandB entity/user/team name
  --wandb_run_name <name>                 WandB run name
  --no_tqdm                               Disable tqdm progress bars
  -h, --help                              Show this help message

For historical multi-phase (A -> B -> C) pipeline execution, use:
  bash scripts/run_full_pipeline_legacy.sh
HELP_EOF
  exit 0
}

# Parse Arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --smoke) SMOKE=true; shift ;;
    --device) DEVICE="$2"; shift 2 ;;
    --data_root) DATA_ROOT="$2"; shift 2 ;;
    --split_manifest) SPLIT_MANIFEST="$2"; shift 2 ;;
    --num_workers) NUM_WORKERS="$2"; shift 2 ;;
    --max_steps) MAX_STEPS="$2"; shift 2 ;;
    --max_val_batches) MAX_VAL_BATCHES="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --report_dir) REPORT_DIR="$2"; shift 2 ;;
    --resume) RESUME="$2"; shift 2 ;;
    --tau_accept) TAU_ACCEPT="$2"; shift 2 ;;
    --skip_calibration) SKIP_CALIBRATION=true; shift ;;
    --wandb) WANDB_ENABLED=true; shift ;;
    --no_wandb|--no-wandb) WANDB_ENABLED=false; shift ;;
    --wandb_mode) WANDB_MODE="$2"; shift 2 ;;
    --wandb_project) WANDB_PROJECT="$2"; shift 2 ;;
    --wandb_entity) WANDB_ENTITY="$2"; shift 2 ;;
    --wandb_run_name) WANDB_RUN_NAME="$2"; shift 2 ;;
    --no_tqdm) NO_TQDM=true; shift ;;
    --config_a|--config_b|--config_c|--config_annotation|--config_auditor|--config_joint|--start_phase|--epochs_a|--epochs_b|--epochs_c)
      echo "Error: Legacy multi-phase option '$1' is not supported by canonical run_full_pipeline.sh." >&2
      echo "Use scripts/run_full_pipeline_legacy.sh for historical multi-phase execution." >&2
      exit 2
      ;;
    -h|--help) show_help ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ -n "$OUTPUT_DIR" ]]; then mkdir -p "$OUTPUT_DIR"; fi
if [[ -n "$REPORT_DIR" ]]; then mkdir -p "$REPORT_DIR"; fi

CMD_ARGS=(
  --config "$CONFIG"
)

if [[ -n "$DEVICE" ]]; then CMD_ARGS+=(--device "$DEVICE"); fi
if [[ -n "$OUTPUT_DIR" ]]; then CMD_ARGS+=(--output_dir "$OUTPUT_DIR"); fi
if [[ -n "$REPORT_DIR" ]]; then CMD_ARGS+=(--report_dir "$REPORT_DIR"); fi
if [[ -n "$DATA_ROOT" ]]; then CMD_ARGS+=(--data_root "$DATA_ROOT"); fi
if [[ -n "$SPLIT_MANIFEST" ]]; then CMD_ARGS+=(--split_manifest "$SPLIT_MANIFEST"); fi
if [[ -n "$NUM_WORKERS" ]]; then CMD_ARGS+=(--num_workers "$NUM_WORKERS"); fi
if [[ -n "$RESUME" ]]; then CMD_ARGS+=(--resume "$RESUME"); fi
if [[ -n "$TAU_ACCEPT" ]]; then CMD_ARGS+=(--tau_accept "$TAU_ACCEPT"); fi
if [[ "$SKIP_CALIBRATION" == "true" ]]; then CMD_ARGS+=(--skip_calibration); fi

if [[ "$SMOKE" == "true" ]]; then
  CMD_ARGS+=(--max_steps 2 --max_val_batches 2)
  if [[ -z "$DEVICE" ]]; then CMD_ARGS+=(--device cpu); fi
else
  if [[ -n "$MAX_STEPS" ]]; then CMD_ARGS+=(--max_steps "$MAX_STEPS"); fi
  if [[ -n "$MAX_VAL_BATCHES" ]]; then CMD_ARGS+=(--max_val_batches "$MAX_VAL_BATCHES"); fi
fi

if [[ "$WANDB_ENABLED" == "true" ]]; then
  CMD_ARGS+=(--wandb)
  if [[ -n "$WANDB_MODE" ]]; then CMD_ARGS+=(--wandb_mode "$WANDB_MODE"); fi
  if [[ -n "$WANDB_PROJECT" ]]; then CMD_ARGS+=(--wandb_project "$WANDB_PROJECT"); fi
  if [[ -n "$WANDB_ENTITY" ]]; then CMD_ARGS+=(--wandb_entity "$WANDB_ENTITY"); fi
  if [[ -n "$WANDB_RUN_NAME" ]]; then CMD_ARGS+=(--wandb_run_name "$WANDB_RUN_NAME"); fi
elif [[ "$WANDB_ENABLED" == "false" ]]; then
  CMD_ARGS+=(--no_wandb)
fi

if [[ "$NO_TQDM" == "true" ]]; then CMD_ARGS+=(--no_tqdm); fi

echo "=============================================================================="
echo "          Self-Audit: Canonical Unified Training Pipeline"
echo "=============================================================================="
echo " Config:          $CONFIG"
echo " Device:          ${DEVICE:-<from config>}"
echo " Smoke Mode:      $SMOKE"
echo " Output Dir:      ${OUTPUT_DIR:-<from config>}"
echo " Report Dir:      ${REPORT_DIR:-<from config>}"
echo " WandB:           ${WANDB_ENABLED:-<from config>}"
echo " TQDM Bars:       $([[ "$NO_TQDM" == "true" ]] && echo "disabled" || echo "enabled")"
echo "=============================================================================="

exec python scripts/train_self_audit.py "${CMD_ARGS[@]}"
