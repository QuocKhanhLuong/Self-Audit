#!/usr/bin/env bash
#
# The documented top-level entry point: full native ACDC, then full independent
# native M&Ms, sequentially.
#
#   CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
#   RUN_FULL=1 TOTAL_EPOCHS=150 BATCH_SIZE=8 \
#   bash scripts/run_maskfree_acdc_mnms.sh
#
# The two runs are INDEPENDENT. Each starts from random initialization at global
# epoch 0, writes to its own run directory, and gets its own run id. No weight,
# no optimizer state, no candidate bank, no pseudo-label version and no resume
# argument crosses between them. ACDC 150 + M&Ms 150 is 300 dataset-global
# epochs, not 150 shared.
#
# ACDC must finish and pass its own completion check - all 150 epochs, the final
# checkpoints, a freeze manifest that validates and enumerates every compared
# method - before M&Ms is allowed to start. A failed or partial ACDC run stops
# the sequence; it does not silently hand the GPU to the next dataset.
#
# Without RUN_FULL=1 this prints both resolved plans and exits 0 without
# touching the GPU, the data or the workspace. Real execution is currently
# deferred by the user (the RTX 4070 is occupied), so at the time of writing no
# part of this sequence has been run.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FULL="scripts/run_maskfree_full.sh"

usage() {
  cat <<'HELPTEXT'
scripts/run_maskfree_acdc_mnms.sh - sequential independent ACDC then M&Ms runs.

USAGE
  scripts/run_maskfree_acdc_mnms.sh
  scripts/run_maskfree_acdc_mnms.sh --datasets acdc,mnms
  scripts/run_maskfree_acdc_mnms.sh --help

  Each dataset is delegated to scripts/run_maskfree_full.sh, which owns the
  stages (resolve -> inventory -> preflight gate -> 150 epochs -> completion
  check). Run `scripts/run_maskfree_full.sh --help` for the full environment
  reference; every variable listed there applies here and is forwarded as-is,
  with two exceptions:

    RUN_ID          Per-dataset ids are derived as <RUN_ID>_<dataset> so the two
                    runs can never share a directory. A deterministic base id
                    is generated when RUN_ID is unset.
    DATA_ROOT       Ignored here, because one root cannot serve two datasets.
                    Use ACDC_DATA_ROOT and MNMS_DATA_ROOT instead.

  ACDC_DATA_ROOT=<dir>   Override the ACDC image root for this invocation.
  MNMS_DATA_ROOT=<dir>   Override the M&Ms image root for this invocation.
  EPOCH_VALIDATION=1     Report-only frozen-student dev Dice after every
                         completed epoch; set 0 for an explicit opt-out.
  ACDC_EPOCH_REFERENCE_CONFIG=<path>
                          Optional control-plane path forwarded only to the
                          ACDC epoch evaluator; training never opens it.
  MNMS_EPOCH_REFERENCE_CONFIG=<path>
                          Optional control-plane path forwarded only to the
                          M&Ms epoch evaluator; training never opens it.
  ACDC_REFERENCE_CONFIG=<path>
                          Optional explicit reference config, evaluated only
                          after both dataset freezes validate.
  MNMS_REFERENCE_CONFIG=<path>
                          Optional explicit reference config, evaluated only
                          after both dataset freezes validate.
  --datasets a,b         Restrict or reorder the sequence. Default: acdc,mnms.

WHAT THIS COMMAND PERFORMS WHEN RUN_FULL=1
  inventory and preflight, two actual 150-epoch runs with both student arms,
  final native-grid exports, the E0-E5 and negative-control reports, the
  prediction freeze, and image-only verification after the freeze. The optional
  isolated reference evaluation is a separate process
  (scripts/evaluate_maskfree_reference.py) and runs only where a separate
  configuration supplies legitimate masks; no external reference is required
  for this pipeline to succeed.
HELPTEXT
}

DATASETS="acdc,mnms"
while [ "$#" -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --datasets) DATASETS="${2:-}"; shift 2 ;;
    *) echo "ERROR: unknown argument: $1" >&2; echo "Try --help." >&2; exit 2 ;;
  esac
done

[ -f "$FULL" ] || { echo "ERROR: missing $FULL" >&2; exit 1; }

RUN_FULL="${RUN_FULL:-0}"
WORKSPACE="${WORKSPACE:-/home/linhdang/workspace/quockhanh_workspace/SpecMamba}"
BASE_RUN_ID="${RUN_ID:-maskfree150_seq_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
  if command -v python3 >/dev/null 2>&1; then PYTHON="python3"; else PYTHON="python"; fi
fi

log() { printf '[maskfree-seq] %s\n' "$*"; }
die() { printf '[maskfree-seq] ERROR: %s\n' "$*" >&2; exit 1; }

IFS=',' read -r -a SEQUENCE <<< "$DATASETS"
[ "${#SEQUENCE[@]}" -gt 0 ] || die "empty --datasets list"

case "$BASE_RUN_ID" in
  */*|.|..) die "RUN_ID must be a single path component, got '$BASE_RUN_ID'" ;;
esac

seen_datasets=""
for dataset in "${SEQUENCE[@]}"; do
  case "$dataset" in
    acdc|mnms) ;;
    *) die "unknown dataset in --datasets: '$dataset'" ;;
  esac
  case " $seen_datasets " in
    *" $dataset "*) die "dataset '$dataset' appears more than once; duplicate runs are not independent" ;;
  esac
  seen_datasets="$seen_datasets $dataset"
done

log "sequence: ${SEQUENCE[*]}"
log "workspace: $WORKSPACE"
if [ "$RUN_FULL" != "1" ]; then
  log "RUN_FULL is not 1: printing each dataset's resolved plan, executing nothing."
fi

for dataset in "${SEQUENCE[@]}"; do
  log "=== dataset: $dataset ==="

  dataset_run_id=""
  dataset_run_id="${BASE_RUN_ID}_${dataset}"

  dataset_data_root=""
  dataset_epoch_reference_config=""
  case "$dataset" in
    acdc)
      dataset_data_root="${ACDC_DATA_ROOT:-}"
      dataset_epoch_reference_config="${ACDC_EPOCH_REFERENCE_CONFIG:-}"
      ;;
    mnms)
      dataset_data_root="${MNMS_DATA_ROOT:-}"
      dataset_epoch_reference_config="${MNMS_EPOCH_REFERENCE_CONFIG:-}"
      ;;
  esac
  log "epoch validation: enabled=${EPOCH_VALIDATION:-1} reference_config=${dataset_epoch_reference_config:-<none>}"

  # DATA_ROOT and RUN_ID are set per dataset and deliberately not inherited from
  # the caller's environment: one root or one id shared by two datasets would
  # make the runs dependent, which is exactly what this sequence must prevent.
  env_args=(env -u DATA_ROOT -u RUN_ID -u EPOCH_REFERENCE_CONFIG)
  if [ -n "$dataset_data_root" ]; then env_args+=("DATA_ROOT=$dataset_data_root"); fi
  if [ -n "$dataset_epoch_reference_config" ]; then
    env_args+=("EPOCH_REFERENCE_CONFIG=$dataset_epoch_reference_config")
  fi
  env_args+=("RUN_ID=$dataset_run_id" "WORKSPACE=$WORKSPACE" "PYTHON=$PYTHON")
  "${env_args[@]}" bash "$FULL" --dataset "$dataset"

  if [ "$RUN_FULL" = "1" ]; then
    # Independent re-check of the completion receipt before the next dataset is
    # allowed to start. run_maskfree_full.sh already checked it; this second,
    # explicit gate is what makes "ACDC finished" a precondition of M&Ms rather
    # than an assumption.
    last_run_file="$WORKSPACE/logs/maskfree150/$dataset_run_id/last_run_dir.txt"
    [ -f "$last_run_file" ] \
      || die "$dataset finished without recording its run directory; refusing to start the next dataset"
    run_dir="$WORKSPACE/runs/maskfree150/$dataset/$dataset_run_id"
    recorded_run_dir="$(< "$last_run_file")"
    [ "$recorded_run_dir" = "$run_dir" ] \
      || die "$dataset recorded unexpected run directory '$recorded_run_dir' (expected '$run_dir')"
    log "gate: re-verifying $dataset completion receipt at $run_dir"
    bash "$FULL" --verify-run "$run_dir"
    log "gate: $dataset PASSED; the next dataset may start"
  fi
done

if [ "$RUN_FULL" = "1" ]; then
  # Reference masks are an opt-in, post-freeze process. Requiring both dataset
  # freezes here prevents a partial --datasets invocation from being mistaken
  # for a complete paired evaluation, and each dataset receives only its own
  # explicitly supplied reference configuration.
  if [ -n "${ACDC_REFERENCE_CONFIG:-}" ] || [ -n "${MNMS_REFERENCE_CONFIG:-}" ]; then
    has_acdc=0
    has_mnms=0
    case " $seen_datasets " in *" acdc "*) has_acdc=1 ;; esac
    case " $seen_datasets " in *" mnms "*) has_mnms=1 ;; esac
    [ "$has_acdc" -eq 1 ] && [ "$has_mnms" -eq 1 ] \
      || die "reference evaluation requires both acdc and mnms freezes in this sequence"
    for reference_dataset in acdc mnms; do
      reference_config=""
      case "$reference_dataset" in
        acdc) reference_config="${ACDC_REFERENCE_CONFIG:-}" ;;
        mnms) reference_config="${MNMS_REFERENCE_CONFIG:-}" ;;
      esac
      [ -n "$reference_config" ] || continue
      [ -f "$reference_config" ] || die "reference config does not exist: $reference_config"
      reference_run_id="${BASE_RUN_ID}_${reference_dataset}"
      reference_run_dir="$WORKSPACE/runs/maskfree150/$reference_dataset/$reference_run_id"
      frozen_manifest="$reference_run_dir/exports/freeze_manifest.json"
      [ -f "$frozen_manifest" ] || die "validated freeze missing before reference evaluation: $frozen_manifest"
      reference_output="$reference_run_dir/reference"
      log "reference: evaluating $reference_dataset with explicit config $reference_config"
      "$PYTHON" scripts/evaluate_maskfree_reference.py \
        --frozen-manifest "$frozen_manifest" \
        --reference-config "$reference_config" \
        --output "$reference_output" \
        --print-summary
    done
  fi
  log "sequence complete for: ${SEQUENCE[*]}"
  log "Runtime completion and scientific outcome are separate fields of each run receipt."
  log "Read label_generation_objective in each receipt before describing the result as a success."
else
  log "plan printed for: ${SEQUENCE[*]}; nothing was executed."
  log "Full execution status: NOT STARTED."
fi
