#!/usr/bin/env bash
# ==============================================================================
# Self-Audit: Full End-to-End Pipeline Execution Script
#
# Runs the full cardiac segmentation & self-audit pipeline locally:
#   Phase 0: Runtime Preflight Sanity Check
#   Phase 1: Phase A - Supervised Annotation Network Training
#   Phase 2: Phase B - Counterfactual Transition Auditor Training
#   Phase 3: Phase C - Threshold-Controlled Joint Fine-tuning
#   Phase 4: Validation Transition Cache Collection
#   Phase 5: Decision Threshold (tau_accept) Calibration
#
# Usage:
#   bash scripts/run_full_pipeline_legacy.sh [options]
#
# Examples:
#   # Standard full run on GPU
#   bash scripts/run_full_pipeline_legacy.sh --device cuda
#
#   # Fast local smoke verification (1-2 steps, CPU safe)
#   bash scripts/run_full_pipeline_legacy.sh --smoke --device cpu
#
#   # Enable offline WandB tracking
#   bash scripts/run_full_pipeline_legacy.sh --wandb --wandb_mode offline
#
#   # Resume from Phase B using an existing Phase A checkpoint
#   bash scripts/run_full_pipeline_legacy.sh --start_phase B
# ==============================================================================

set -euo pipefail

# ------------------------------------------------------------------------------
# Default Settings
# ------------------------------------------------------------------------------
CONFIG_A="configs/self_audit_annotation.yaml"
CONFIG_B="configs/self_audit_auditor.yaml"
CONFIG_C="configs/self_audit_joint.yaml"

DATA_ROOT=""
DEVICE="cuda"
BATCH_SIZE=""
NUM_WORKERS=""
IMAGE_SIZE=""
EPOCHS_A=""
EPOCHS_B=""
EPOCHS_C=""

WANDB_ENABLED=false
WANDB_MODE="offline"
WANDB_PROJECT="self-audit"
WANDB_ENTITY=""
NO_TQDM=false

SMOKE=false
START_PHASE="A"
SKIP_PREFLIGHT=false

SPLIT_MANIFEST=""
OUTPUT_DIR="weights/self_audit"
REPORT_DIR="reports"

# ------------------------------------------------------------------------------
# Help / Usage
# ------------------------------------------------------------------------------
show_help() {
  cat << 'EOF'
Usage: bash scripts/run_full_pipeline_legacy.sh [OPTIONS]

Pipeline Options:
  --start_phase <A|B|C|cache|calibrate>   Start pipeline from specific stage (default: A)
  --smoke                                 Run in ultra-fast smoke/sanity mode (1-2 steps)
  --skip_preflight                        Skip Phase 0 preflight check

Configuration Options:
  --config_a, --config_annotation <path>  Config YAML for Phase A (default: configs/self_audit_annotation.yaml)
  --config_b, --config_auditor <path>     Config YAML for Phase B (default: configs/self_audit_auditor.yaml)
  --config_c, --config_joint <path>       Config YAML for Phase C (default: configs/self_audit_joint.yaml)
  --split_manifest <path>                 Custom split manifest JSON path

Data & Hardware Options:
  --data_root <path>                      Dataset root directory
  --device <cuda|cpu>                     Target compute device (default: cuda)
  --batch_size <int>                      Batch size override for all phases
  --num_workers <int>                     DataLoader worker count
  --image_size <int>                      Spatial resolution (e.g. 256 or 64)

Epoch Overrides:
  --epochs_a <int>                        Epochs for Phase A (default from config)
  --epochs_b <int>                        Epochs for Phase B (default from config)
  --epochs_c <int>                        Epochs for Phase C (default from config)

Tracking Options:
  --wandb                                 Enable Weights & Biases logging
  --wandb_mode <offline|online|disabled>  WandB run mode (default: offline)
  --wandb_project <name>                  WandB project name (default: self-audit)
  --wandb_entity <name>                   WandB entity/user/team name
  --no_tqdm                               Disable tqdm progress bars

Path Options:
  --output_dir <dir>                      Checkpoint output directory (default: weights/self_audit)
  --report_dir <dir>                      Reports/cache directory (default: reports)
  -h, --help                              Show this help message
EOF
  exit 0
}

# ------------------------------------------------------------------------------
# Parse Arguments
# ------------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --start_phase) START_PHASE="$2"; shift 2 ;;
    --smoke) SMOKE=true; shift ;;
    --skip_preflight) SKIP_PREFLIGHT=true; shift ;;
    --config_a|--config_annotation) CONFIG_A="$2"; shift 2 ;;
    --config_b|--config_auditor) CONFIG_B="$2"; shift 2 ;;
    --config_c|--config_joint) CONFIG_C="$2"; shift 2 ;;
    --split_manifest) SPLIT_MANIFEST="$2"; shift 2 ;;
    --data_root) DATA_ROOT="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --num_workers) NUM_WORKERS="$2"; shift 2 ;;
    --image_size) IMAGE_SIZE="$2"; shift 2 ;;
    --epochs_a) EPOCHS_A="$2"; shift 2 ;;
    --epochs_b) EPOCHS_B="$2"; shift 2 ;;
    --epochs_c) EPOCHS_C="$2"; shift 2 ;;
    --wandb) WANDB_ENABLED=true; shift ;;
    --no_wandb) WANDB_ENABLED=false; shift ;;
    --wandb_mode) WANDB_MODE="$2"; shift 2 ;;
    --wandb_project) WANDB_PROJECT="$2"; shift 2 ;;
    --wandb_entity) WANDB_ENTITY="$2"; shift 2 ;;
    --no_tqdm) NO_TQDM=true; shift ;;
    --visualize) VISUALIZE=true; shift ;;
    --vis_samples) VIS_SAMPLES="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --report_dir) REPORT_DIR="$2"; shift 2 ;;
    -h|--help) show_help ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

VISUALIZE=${VISUALIZE:-false}
VIS_SAMPLES=${VIS_SAMPLES:-4}

# Normalize start phase to uppercase
START_PHASE=$(echo "$START_PHASE" | tr '[:lower:]' '[:upper:]')

mkdir -p "$OUTPUT_DIR" "$REPORT_DIR"

echo "=============================================================================="
echo "          Self-Audit: Cardiac Segmentation & Audit Pipeline"
echo "=============================================================================="
echo " Device:          $DEVICE"
echo " Start Phase:     $START_PHASE"
echo " Smoke Mode:      $SMOKE"
echo " Output Dir:      $OUTPUT_DIR"
echo " Report Dir:      $REPORT_DIR"
echo " WandB:           $WANDB_ENABLED (mode: $WANDB_MODE, project: $WANDB_PROJECT)"
echo " TQDM Bars:       $([[ "$NO_TQDM" == "true" ]] && echo "disabled" || echo "enabled")"
echo "=============================================================================="

# Build common arguments array
COMMON_ARGS=()
if [[ -n "$DATA_ROOT" ]]; then COMMON_ARGS+=(--data_root "$DATA_ROOT"); fi
if [[ -n "$DEVICE" ]]; then COMMON_ARGS+=(--device "$DEVICE"); fi
if [[ -n "$BATCH_SIZE" ]]; then COMMON_ARGS+=(--batch_size "$BATCH_SIZE"); fi
if [[ -n "$NUM_WORKERS" ]]; then COMMON_ARGS+=(--num_workers "$NUM_WORKERS"); fi
if [[ -n "$IMAGE_SIZE" ]]; then COMMON_ARGS+=(--image_size "$IMAGE_SIZE"); fi
if [[ "$WANDB_ENABLED" == "true" ]]; then
  COMMON_ARGS+=(--wandb --wandb_mode "$WANDB_MODE" --wandb_project "$WANDB_PROJECT")
  if [[ -n "$WANDB_ENTITY" ]]; then COMMON_ARGS+=(--wandb_entity "$WANDB_ENTITY"); fi
fi
if [[ "$NO_TQDM" == "true" ]]; then COMMON_ARGS+=(--no_tqdm); fi
if [[ "$VISUALIZE" == "true" ]]; then
  COMMON_ARGS+=(--visualize --vis_dir "$REPORT_DIR/visualizations" --vis_samples "$VIS_SAMPLES")
fi

# ------------------------------------------------------------------------------
# Phase 0: Preflight Sanity Check
# ------------------------------------------------------------------------------
if [[ "$SKIP_PREFLIGHT" == "false" && ("$START_PHASE" == "A" || "$START_PHASE" == "PREFLIGHT") ]]; then
  echo ""
  echo ">>> [Phase 0] Running Preflight Sanity Check..."
  PREFLIGHT_ARGS=(--config "$CONFIG_A")
  if [[ -n "$DATA_ROOT" ]]; then PREFLIGHT_ARGS+=(--data_root "$DATA_ROOT"); fi
  python scripts/self_audit_preflight.py "${PREFLIGHT_ARGS[@]}"
  echo ">>> [Phase 0] Preflight Check Completed Successfully."
fi

# ------------------------------------------------------------------------------
# Phase 1: Phase A Supervised Annotation Network Training
# ------------------------------------------------------------------------------
CHECKPOINT_A="$OUTPUT_DIR/phase_a_annotation.pt"
if [[ "$START_PHASE" == "A" ]]; then
  echo ""
  echo ">>> [Phase 1/5] Phase A: Training Annotation Network..."
  PHASE_A_ARGS=(
    --config "$CONFIG_A"
    --output "$CHECKPOINT_A"
    "${COMMON_ARGS[@]}"
  )
  if [[ -n "$EPOCHS_A" ]]; then PHASE_A_ARGS+=(--epochs "$EPOCHS_A"); fi
  if [[ "$SMOKE" == "true" ]]; then
    PHASE_A_ARGS+=(--epochs 1 --max_steps 2 --max_val_batches 2 --no_pretrained)
  fi

  python src/self_audit/training/train_annotation.py "${PHASE_A_ARGS[@]}"
  echo ">>> [Phase 1/5] Phase A Completed. Checkpoint saved: $CHECKPOINT_A"
fi

# ------------------------------------------------------------------------------
# Phase 2: Phase B Transition Auditor Training
# ------------------------------------------------------------------------------
CHECKPOINT_B="$OUTPUT_DIR/phase_b_auditor.pt"
if [[ "$START_PHASE" == "A" || "$START_PHASE" == "B" ]]; then
  echo ""
  echo ">>> [Phase 2/5] Phase B: Training Transition Auditor..."
  if [[ ! -f "$CHECKPOINT_A" && ! -f "$OUTPUT_DIR/best.pt" ]]; then
    echo "ERROR: Phase A checkpoint not found at $CHECKPOINT_A. Run Phase A first." >&2
    exit 1
  fi
  ANNOTATION_CKPT="$CHECKPOINT_A"
  if [[ ! -f "$ANNOTATION_CKPT" && -f "$OUTPUT_DIR/best.pt" ]]; then
    ANNOTATION_CKPT="$OUTPUT_DIR/best.pt"
  fi

  PHASE_B_ARGS=(
    --config "$CONFIG_B"
    --annotation_checkpoint "$ANNOTATION_CKPT"
    --output "$CHECKPOINT_B"
    "${COMMON_ARGS[@]}"
  )
  if [[ -n "$EPOCHS_B" ]]; then PHASE_B_ARGS+=(--epochs "$EPOCHS_B"); fi
  if [[ "$SMOKE" == "true" ]]; then
    PHASE_B_ARGS+=(--epochs 1 --max_steps 2 --max_val_batches 2)
  fi

  python src/self_audit/training/train_auditor.py "${PHASE_B_ARGS[@]}"
  echo ">>> [Phase 2/5] Phase B Completed. Checkpoint saved: $CHECKPOINT_B"
fi

# ------------------------------------------------------------------------------
# Phase 3: Phase C Joint Fine-Tuning
# ------------------------------------------------------------------------------
CHECKPOINT_C="$OUTPUT_DIR/phase_c_joint.pt"
if [[ "$START_PHASE" == "A" || "$START_PHASE" == "B" || "$START_PHASE" == "C" ]]; then
  echo ""
  echo ">>> [Phase 3/5] Phase C: Joint Fine-Tuning..."
  AUDITOR_CKPT="$CHECKPOINT_B"
  if [[ ! -f "$AUDITOR_CKPT" && -f "$OUTPUT_DIR/best.pt" ]]; then
    AUDITOR_CKPT="$OUTPUT_DIR/best.pt"
  fi
  if [[ ! -f "$AUDITOR_CKPT" ]]; then
    echo "ERROR: Phase B checkpoint not found at $CHECKPOINT_B. Run Phase B first." >&2
    exit 1
  fi

  PHASE_C_ARGS=(
    --config "$CONFIG_C"
    --checkpoint "$AUDITOR_CKPT"
    --output "$CHECKPOINT_C"
    "${COMMON_ARGS[@]}"
  )
  if [[ -n "$EPOCHS_C" ]]; then PHASE_C_ARGS+=(--epochs "$EPOCHS_C"); fi
  if [[ "$SMOKE" == "true" ]]; then
    PHASE_C_ARGS+=(--epochs 1 --max_steps 2 --max_val_batches 2)
  fi

  python src/self_audit/training/finetune_joint.py "${PHASE_C_ARGS[@]}"
  echo ">>> [Phase 3/5] Phase C Completed. Checkpoint saved: $CHECKPOINT_C"
fi

# ------------------------------------------------------------------------------
# Phase 4: Cache Validation Transitions
# ------------------------------------------------------------------------------
TRANSITIONS_CACHE="$REPORT_DIR/validation_transitions.pt"
if [[ "$START_PHASE" == "A" || "$START_PHASE" == "B" || "$START_PHASE" == "C" || "$START_PHASE" == "CACHE" ]]; then
  echo ""
  echo ">>> [Phase 4/5] Caching Validation Transitions..."
  JOINT_CKPT="$CHECKPOINT_C"
  if [[ ! -f "$JOINT_CKPT" && -f "$OUTPUT_DIR/best.pt" ]]; then
    JOINT_CKPT="$OUTPUT_DIR/best.pt"
  fi
  if [[ ! -f "$JOINT_CKPT" ]]; then
    echo "ERROR: Joint checkpoint not found at $CHECKPOINT_C." >&2
    exit 1
  fi

  CACHE_ARGS=(
    --config "$CONFIG_C"
    --checkpoint "$JOINT_CKPT"
    --output "$TRANSITIONS_CACHE"
  )
  if [[ -n "$DATA_ROOT" ]]; then CACHE_ARGS+=(--data_root "$DATA_ROOT"); fi
  if [[ -n "$SPLIT_MANIFEST" ]]; then CACHE_ARGS+=(--split_manifest "$SPLIT_MANIFEST"); fi
  if [[ -n "$IMAGE_SIZE" ]]; then CACHE_ARGS+=(--image_size "$IMAGE_SIZE"); fi
  if [[ -n "$DEVICE" ]]; then CACHE_ARGS+=(--device "$DEVICE"); fi
  if [[ -n "$BATCH_SIZE" ]]; then CACHE_ARGS+=(--batch_size "$BATCH_SIZE"); fi
  if [[ "$NO_TQDM" == "true" ]]; then CACHE_ARGS+=(--no_tqdm); fi

  python scripts/cache_validation_transitions.py "${CACHE_ARGS[@]}"
  echo ">>> [Phase 4/5] Transitions Cached: $TRANSITIONS_CACHE"
fi

# ------------------------------------------------------------------------------
# Phase 5: Threshold Calibration
# ------------------------------------------------------------------------------
CALIBRATION_JSON="$REPORT_DIR/calibration_result.json"
if [[ "$START_PHASE" == "A" || "$START_PHASE" == "B" || "$START_PHASE" == "C" || "$START_PHASE" == "CACHE" || "$START_PHASE" == "CALIBRATE" ]]; then
  echo ""
  echo ">>> [Phase 5/5] Calibrating Threshold (tau_accept)..."
  if [[ ! -f "$TRANSITIONS_CACHE" ]]; then
    echo "ERROR: Transitions cache not found at $TRANSITIONS_CACHE." >&2
    exit 1
  fi

  python scripts/calibrate_threshold.py \
    --transitions "$TRANSITIONS_CACHE" \
    --output "$CALIBRATION_JSON"

  echo ">>> [Phase 5/5] Calibration Complete. Result saved: $CALIBRATION_JSON"
fi

# ------------------------------------------------------------------------------
# Phase 6: Visual Report Generation (Optional)
# ------------------------------------------------------------------------------
if [[ "$VISUALIZE" == "true" && -f "$OUTPUT_DIR/phase_c_joint.pt" ]]; then
  echo ""
  echo ">>> [Phase 6/6] Generating Visual Inspection Reports..."
  python scripts/visualize_predictions.py \
    --checkpoint "$OUTPUT_DIR/phase_c_joint.pt" \
    --config "$CONFIG_C" \
    --num_samples "$VIS_SAMPLES" \
    --output_dir "$REPORT_DIR/visualizations" \
    --device "$DEVICE"
  echo ">>> [Phase 6/6] Visual Inspection Images saved to: $REPORT_DIR/visualizations"
fi

echo ""
echo "=============================================================================="
echo "                 Self-Audit Pipeline Execution Complete!"
echo "=============================================================================="
echo " Phase A Checkpoint:  $CHECKPOINT_A"
echo " Phase B Checkpoint:  $CHECKPOINT_B"
echo " Phase C Checkpoint:  $CHECKPOINT_C"
echo " Transitions Cache:   $TRANSITIONS_CACHE"
echo " Calibration Report:  $CALIBRATION_JSON"
if [[ "$VISUALIZE" == "true" ]]; then
  echo " Visualizations:      $REPORT_DIR/visualizations"
fi
echo "=============================================================================="
