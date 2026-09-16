#!/usr/bin/env bash
#
# One mask-free 150-epoch run for ONE dataset: inventory -> preflight gate ->
# 150 global epochs with both student arms -> native-grid export, freeze and
# image-only verification -> completion check.
#
# This script owns operator concerns only. It parses nothing scientific: the
# resolved config is produced by W5's strict MaskfreeConfig loader
# (src/self_audit_maskfree/config.py) and written read-only, and every stage is
# an ordinary invocation of the owning worker's CLI. There is no second config
# parser here and no hidden training argument.
#
# EXECUTION IS OPT-IN. Without RUN_FULL=1 this prints the fully resolved plan
# and the exact commands, and exits 0 without touching the GPU, the data or the
# workspace. That is the current state of this project: the user has deferred
# real execution because the RTX 4070 is occupied, so nothing here has been run
# against a GPU and no GPU claim in this repository is backed by a run.
#
# Usage: scripts/run_maskfree_full.sh --help

set -euo pipefail

# --------------------------------------------------------------------------- #
# defaults
# --------------------------------------------------------------------------- #
DEFAULT_WORKSPACE="/home/linhdang/workspace/quockhanh_workspace/SpecMamba"
UNSUPPORTED_GPU_PATTERNS="5070 Ti"

DATASET=""
VERIFY_RUN=""

usage() {
  cat <<'HELPTEXT'
scripts/run_maskfree_full.sh - one mask-free 150-epoch run for one dataset.

USAGE
  scripts/run_maskfree_full.sh <acdc|mnms>
  scripts/run_maskfree_full.sh --dataset <acdc|mnms>
  scripts/run_maskfree_full.sh --verify-run <run_dir>
  scripts/run_maskfree_full.sh --help

STAGES (in order; each must succeed before the next starts)
  1. resolve      dataset, workspace containment, fresh run id, GPU identity,
                  physical/effective batch. Writes one unique read-only
                  resolved config through W5's schema.
  2. inventory    scripts/prepare_maskfree_data.py - image-only discovery and
                  manifest. Never opens a mask tree.
  3. preflight    scripts/train_maskfree.py --preflight - bounded real
                  fit/score/edit-or-rejection/backward, checkpoint and tiny
                  export. Emits the batch gate receipt. Cannot mark completion.
  4. train        scripts/train_maskfree.py - the single 150-epoch global
                  timeline, both student arms, exports, freeze, verification.
  5. verify       completion check of the run receipt and required artifacts,
                  including strict validate_freeze() over both primary and
                  full-input deployment freeze manifests, image-only summary
                  reports, hardware qualification and per-unit verification.

ENVIRONMENT
  RUN_FULL=1            Actually execute. Anything else prints the plan only.
  WORKSPACE=<dir>       Run root. Default:
                        /home/linhdang/workspace/quockhanh_workspace/SpecMamba
                        All run artifacts are contained under
                        $WORKSPACE/runs/maskfree150/<dataset>/<run_id>.
  DATA_ROOT=<dir>       Override the config's image root for this dataset.
  RUN_ID=<id>           Override the generated run id. Must not already exist.
  TOTAL_EPOCHS=150      Global epochs. 150 is the contracted timeline.
  BATCH_SIZE=8          Requested PHYSICAL batch. Default 8.
  ACCUM_STEPS=1         Gradient accumulation. Default 1.
  EPOCH_VALIDATION=1    Report-only frozen-student dev Dice after every
                        completed epoch. Set 0 for an explicit operational
                        opt-out; it never selects a checkpoint.
  EPOCH_REFERENCE_CONFIG=<path>
                        Optional control-plane path forwarded to the isolated
                        epoch evaluator. Training never opens it; reference
                        masks stay in the separate evaluator.
  MASKFREE_FALLBACK=    Explicit pre-run OOM fallback, resolved BEFORE the run:
                        "4x2" or "2x4" for effective batch 8. There is no
                        mid-run OOM retry, batch change, LR rescale or
                        resolution change. The fallback changes optimizer
                        microbatch grouping; W4 contrastive negatives remain
                        within-image and are recorded in the run receipt.
  MASKFREE_GPU_UUID=    Explicit physical GPU UUID. If unset, a single numeric
                        CUDA_VISIBLE_DEVICES value (for example "1") is
                        resolved against the full nvidia-smi inventory and
                        replaced with that GPU's real UUID before Python starts.
                        A comma-separated or otherwise ambiguous selector is a
                        hard error on a full run.
  MASKFREE_GPU_OVERRIDE=Explicit rented-GPU identity record for a host whose
                        product name is not RTX 4070. It is written verbatim to
                        reports/gpu_identity.json and the run receipt. The
                        legacy MASKFREE_ALLOW_GPU_NAME substring is also
                        accepted for an explicit product-name override.
  MASKFREE_ALLOW_GPU_NAME=
                        Legacy explicit product-name substring override. It is
                        required when selecting a GPU matching the unsupported
                        list (currently: 5070 Ti), unless MASKFREE_GPU_OVERRIDE
                        supplies an explicit rented-GPU identity record.
  ALLOW_CPU=1           Run on CPU. Skips the GPU gate. For software checks
                        only; a CPU run is not the contracted experiment.
  PYTHON=<path>         Interpreter. Default: python3 if present, else python.
  MASKFREE_PROGRESS=    compact (default): tqdm and one summary per epoch.
                        verbose: print detailed diagnostic events and metrics.
  AUDIT_DEVICE=cpu     Validated CPU observation reference; cuda is experimental.
                        When unset, preserve the template's intentional setting.
  TIMING_MODE=production|diagnostic
  LOGGING_MODE=buffered|sync
  LOG_BUFFER_BYTES=262144
  DATA_CACHE_BYTES=67108864
  PREFETCH_BATCHES=0   0, 1 or 2; input-only bounded lookahead, no pseudo-label cache.
  PREFETCH_MAX_BYTES=33554432
  CANDIDATE_WORKERS=0  0, 2 or 4; optional CPU spawn pool, benchmark before use.
  CANDIDATE_WORKER_THREADS=1
                        Runtime values above override the strict config only
                        when explicitly set. They are recorded in preflight,
                        startup reports and exact-resume identity.

NOT DONE BY THIS SCRIPT
  No commit, no push, no checkpoint migration, no cross-dataset resume, no
  reference/ground-truth reading. The isolated reference evaluator is a
  separate process (scripts/evaluate_maskfree_reference.py) that runs only
  after a validated freeze. Epoch validation follows the same isolation rule:
  every epoch freezes both students on image-only development inputs, and any
  mask-based Dice is computed by its child evaluator without tuning or
  checkpoint selection.
HELPTEXT
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --dataset) DATASET="${2:-}"; shift 2 ;;
    --verify-run) VERIFY_RUN="${2:-}"; shift 2 ;;
    acdc|mnms) DATASET="$1"; shift ;;
    *) echo "ERROR: unknown argument: $1" >&2; echo "Try --help." >&2; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-}"
# Flush progress immediately inside GUI terminals and the tee log pipelines.
export PYTHONUNBUFFERED=1
if [ -z "$PYTHON" ]; then
  if command -v python3 >/dev/null 2>&1; then PYTHON="python3"; else PYTHON="python"; fi
fi

log() { printf '[maskfree] %s\n' "$*"; }
die() { printf '[maskfree] ERROR: %s\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------------------- #
# completion check - also the wrapper's gate between the two datasets
# --------------------------------------------------------------------------- #
verify_run_dir() {
  local run_dir="$1"
  [ -d "$run_dir" ] || die "run directory does not exist: $run_dir"
  # W5 writes the run's own completion record at reports/pipeline_report.json.
  local report="$run_dir/reports/pipeline_report.json"
  [ -f "$report" ] || die "no reports/pipeline_report.json in $run_dir; the run never reached its own finalization"

  MASKFREE_RUN_DIR="$run_dir" MASKFREE_REPO="$ROOT" "$PYTHON" - <<'PY'
import json
import os
import sys
from pathlib import Path

run_dir = Path(os.environ["MASKFREE_RUN_DIR"])
report = json.loads((run_dir / "reports" / "pipeline_report.json").read_text(encoding="utf-8"))

problems = []
epochs = report.get("epochs_completed")
last = report.get("last_completed_epoch")
total = report.get("total_epochs")
if total != 150:
    problems.append(f"total_epochs={total!r}; this launcher only accepts the 150-epoch timeline")
if epochs != 150:
    problems.append(f"epochs_completed={epochs!r}, required 150")
if last != 149:
    problems.append(f"last_completed_epoch={last!r}, required 149")
if report.get("completed") is not True:
    problems.append(f"completed={report.get('completed')!r}, status={report.get('status')!r}")
if report.get("status") != "completed":
    problems.append(f"status={report.get('status')!r}, required 'completed'")
if report.get("bounded_run"):
    problems.append("bounded_run is set: a --max-steps/--max-epochs probe is not a completed run")
if report.get("hardware_qualified") is not True:
    problems.append(
        "hardware_qualified is not true; a CPU 150-epoch software run cannot pass the contracted GPU gate"
    )
finalization = report.get("finalization")
finalization_available = finalization.get("available") if isinstance(finalization, dict) else None
if not isinstance(finalization, dict) or finalization.get("available") is not True:
    problems.append(f"finalization.available={finalization_available!r}; required true")

# Artifact names are W5's (runtime.RunPaths). This list is the operator-visible
# contract between the launcher and the trainer.
required_files = [
    "resolved_config.json",
    "source_provenance.json",
    "data_manifest.json",
    "reports/epoch_metrics.jsonl",
    "checkpoints/last.pt",
    "checkpoints/producer_final.pt",
    "checkpoints/student_no_audit_final.pt",
    "checkpoints/student_audited_final.pt",
    "exports/freeze_manifest.json",
    "exports/deployment/freeze_manifest.json",
    "reports/image_only_summary.json",
    "reports/image_only_summary.csv",
    "reports/image_only_summary.md",
]
for relative in required_files:
    if not (run_dir / relative).is_file():
        problems.append(f"missing required artifact: {relative}")

primary_manifest = None
freeze_path = run_dir / "exports" / "freeze_manifest.json"
if freeze_path.is_file():
    sys.path.insert(0, str(Path(os.environ["MASKFREE_REPO"]) / "src"))
    try:
        from self_audit_maskfree.export import comparison_completeness, validate_freeze
    except ImportError as exc:
        problems.append(f"cannot import the export module to validate the freeze: {exc}")
    else:
        try:
            validate_freeze(freeze_path, require_complete=True)
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            problems.append(f"freeze validation failed: {exc}")
        manifest = json.loads(freeze_path.read_text(encoding="utf-8"))
        primary_manifest = manifest
        completeness = manifest.get("completeness") or comparison_completeness(
            manifest.get("compared_methods", []))
        if not completeness.get("complete"):
            problems.append(
                "freeze does not enumerate every compared method; missing: "
                f"{completeness.get('missing')}")

deployment_freeze_path = run_dir / "exports" / "deployment" / "freeze_manifest.json"
if deployment_freeze_path.is_file():
    try:
        from self_audit_maskfree.export import validate_freeze as validate_deployment_freeze
        validate_deployment_freeze(deployment_freeze_path, require_complete=True)
    except Exception as exc:  # noqa: BLE001 - deployment is a required final artifact
        problems.append(f"deployment freeze validation failed: {exc}")

verification_path = run_dir / "reports" / "verification.json"
if not verification_path.is_file():
    problems.append("missing required image-only verification report: reports/verification.json")
else:
    try:
        verification_payload = json.loads(verification_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - malformed final artifact is a failure
        problems.append(f"verification report is unreadable: {exc}")
    else:
        rows = verification_payload.get("rows")
        if not isinstance(rows, list) or not rows:
            problems.append("verification report contains no rows")
        if verification_payload.get("status") in ("failed", "error"):
            problems.append(
                f"verification report status={verification_payload.get('status')!r}")
        if isinstance(rows, list):
            expected_units = finalization.get("units") if isinstance(finalization, dict) else None
            if not isinstance(expected_units, int) or isinstance(expected_units, bool):
                problems.append(f"finalization.units={expected_units!r} is not an integer")
            elif len(rows) != expected_units:
                problems.append(
                    f"verification row count={len(rows)} disagrees with finalization.units={expected_units}"
                )
            row_ids = [str(row.get("unit_id")) for row in rows
                       if isinstance(row, dict) and row.get("unit_id") is not None]
            if len(row_ids) != len(set(row_ids)):
                problems.append("verification report contains duplicate unit_id rows")
            if len(row_ids) != len(rows):
                problems.append("verification report has a row without unit_id")
            if isinstance(primary_manifest, dict):
                primary_completeness = primary_manifest.get("completeness") or {}
                required_methods = {
                    str(method) for method in primary_completeness.get("required", [])
                }
                method_units = primary_manifest.get("method_units") or {}
                expected_unit_ids = {
                    str(unit_id)
                    for unit_ids in method_units.values()
                    for unit_id in (unit_ids if isinstance(unit_ids, list) else [])
                }
                if expected_unit_ids and set(row_ids) != expected_unit_ids:
                    problems.append(
                        "verification unit set disagrees with the frozen per-unit set: "
                        f"missing={sorted(expected_unit_ids - set(row_ids))}, "
                        f"extra={sorted(set(row_ids) - expected_unit_ids)}"
                    )
                for row in rows:
                    methods = row.get("methods") if isinstance(row, dict) else None
                    if not isinstance(methods, dict):
                        problems.append(
                            f"verification row {row.get('unit_id') if isinstance(row, dict) else None!r} "
                            "has no per-unit methods map"
                        )
                        continue
                    missing_methods = sorted(required_methods - {str(name) for name in methods})
                    if missing_methods:
                        problems.append(
                            f"verification row {row.get('unit_id')!r} is missing compared methods: "
                            f"{missing_methods}"
                        )
        print(f"verification_status={verification_payload.get('status', 'rows_present')!r}")

failure_path = run_dir / "reports" / "failure_report.json"
if failure_path.is_file():
    try:
        failure_payload = json.loads(failure_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - malformed failure is still a failure
        problems.append(f"failure_report.json is unreadable: {exc}")
    else:
        problems.append(
            "run has a recorded software failure: "
            f"{failure_payload.get('stage')}: {failure_payload.get('error')}")

print(f"run_dir={run_dir}")
print(f"run_id={report.get('run_id')!r} epochs_completed={epochs} last_completed_epoch={last}")
print(f"device={report.get('device')!r} amp={report.get('amp_enabled')!r}")
print(f"physical_batch={report.get('physical_batch')!r} "
      f"accumulation_steps={report.get('accumulation_steps')!r} "
      f"effective_batch={report.get('effective_batch')!r}")
print(f"global_optimizer_steps={report.get('global_optimizer_steps')!r} "
      f"component_steps={report.get('component_steps')!r}")
print(f"scientific_outcome={report.get('scientific_outcome')!r}")

if problems:
    print("COMPLETION CHECK: FAILED")
    for problem in problems:
        print(f"  - {problem}")
    raise SystemExit(1)
print("COMPLETION CHECK: PASSED - runtime completion only.")
print("The scientific outcome is the separate scientific_outcome field printed above; "
      "read it before calling this run a success.")
PY
}

if [ -n "$VERIFY_RUN" ]; then
  verify_run_dir "$VERIFY_RUN"
  exit 0
fi

# --------------------------------------------------------------------------- #
# stage 1: resolve
# --------------------------------------------------------------------------- #
[ -n "$DATASET" ] || { usage >&2; die "no dataset given"; }
case "$DATASET" in
  acdc|mnms) ;;
  *) die "dataset must be acdc or mnms, got '$DATASET'" ;;
esac

RUN_FULL="${RUN_FULL:-0}"
ALLOW_CPU="${ALLOW_CPU:-0}"
WORKSPACE="${WORKSPACE:-$DEFAULT_WORKSPACE}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-150}"
BATCH_SIZE="${BATCH_SIZE:-8}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"
EPOCH_VALIDATION="${EPOCH_VALIDATION:-1}"
EPOCH_REFERENCE_CONFIG="${EPOCH_REFERENCE_CONFIG:-}"
MASKFREE_FALLBACK="${MASKFREE_FALLBACK:-}"
MASKFREE_GPU_UUID="${MASKFREE_GPU_UUID:-}"
MASKFREE_GPU_OVERRIDE="${MASKFREE_GPU_OVERRIDE:-}"
MASKFREE_ALLOW_GPU_NAME="${MASKFREE_ALLOW_GPU_NAME:-}"
DATA_ROOT="${DATA_ROOT:-}"
REQUESTED_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

case "$EPOCH_VALIDATION" in
  0|1) ;;
  *) die "EPOCH_VALIDATION must be 0 or 1, got '$EPOCH_VALIDATION'" ;;
esac

TEMPLATE_CONFIG="configs/maskfree_${DATASET}_150.yaml"
[ -f "$TEMPLATE_CONFIG" ] || die "missing template config: $TEMPLATE_CONFIG"

case "$WORKSPACE" in
  /*) ;;
  *) die "WORKSPACE must be an absolute path, got '$WORKSPACE'" ;;
esac

# Explicit pre-run batch fallback. Resolved here, before anything allocates.
CONTRASTIVE_NOTE="none: requested physical batch ${BATCH_SIZE}, accumulation ${ACCUM_STEPS}"
case "$MASKFREE_FALLBACK" in
  "")
    ;;
  4x2)
    BATCH_SIZE=4; ACCUM_STEPS=2
    CONTRASTIVE_NOTE="microbatch4 x accumulation2 for effective batch 8; optimizer microbatch grouping differs and W4 contrastive negatives remain within-image"
    ;;
  2x4)
    BATCH_SIZE=2; ACCUM_STEPS=4
    CONTRASTIVE_NOTE="microbatch2 x accumulation4 for effective batch 8; optimizer microbatch grouping differs and W4 contrastive negatives remain within-image"
    ;;
  *)
    die "MASKFREE_FALLBACK must be empty, 4x2 or 2x4, got '$MASKFREE_FALLBACK'" ;;
esac
EFFECTIVE_BATCH=$(( BATCH_SIZE * ACCUM_STEPS ))

RUN_ID="${RUN_ID:-maskfree150_${DATASET}_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
case "$RUN_ID" in
  */*|.|..) die "RUN_ID must be a single path component, got '$RUN_ID'" ;;
esac

RUN_DIR="$WORKSPACE/runs/maskfree150/$DATASET/$RUN_ID"
CONFIG_DIR="$WORKSPACE/run_configs/maskfree150/$RUN_ID"
RESOLVED_CONFIG="$CONFIG_DIR/${DATASET}.yaml"
LOG_DIR="$WORKSPACE/logs/maskfree150/$RUN_ID"

case "$RUN_DIR" in
  "$WORKSPACE"/runs/maskfree150/*) ;;
  *) die "refusing: run directory '$RUN_DIR' escapes the workspace run root" ;;
esac

# --------------------------------------------------------------------------- #
# GPU identity gate
# --------------------------------------------------------------------------- #
DEVICE="cuda"
GPU_REPORT="not probed"
GPU_TABLE=""
GPU_INDEX=""
GPU_UUID=""
GPU_NAME=""
GPU_SELECTOR_SOURCE="none"
GPU_OVERRIDE_RECORD="${MASKFREE_GPU_OVERRIDE}"
if [ -z "$GPU_OVERRIDE_RECORD" ] && [ -n "$MASKFREE_ALLOW_GPU_NAME" ]; then
  GPU_OVERRIDE_RECORD="product_name_substring=${MASKFREE_ALLOW_GPU_NAME}"
fi
if [ "$ALLOW_CPU" = "1" ]; then
  DEVICE="cpu"
  GPU_REPORT="skipped: ALLOW_CPU=1 (a CPU run is a software check, not the contracted experiment)"
else
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    GPU_REPORT="nvidia-smi not found on this host"
    if [ "$RUN_FULL" = "1" ]; then
      die "no nvidia-smi: cannot verify GPU identity. Set ALLOW_CPU=1 for a CPU software check, or run on the GPU host."
    fi
  else
    # Clear the caller's CUDA visibility mask while querying NVML so a numeric
    # selector always refers to the host's physical index. Python is started
    # only after the selected physical GPU is replaced with its UUID below.
    GPU_TABLE="$(env -u CUDA_VISIBLE_DEVICES nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader 2>/dev/null || true)"
    if [ -z "$GPU_TABLE" ]; then
      GPU_REPORT="nvidia-smi returned no GPUs"
      if [ "$RUN_FULL" = "1" ]; then die "nvidia-smi reported no GPUs"; fi
    else
      GPU_SELECTOR="$MASKFREE_GPU_UUID"
      if [ -n "$GPU_SELECTOR" ]; then
        GPU_SELECTOR_SOURCE="explicit_uuid"
      else
        GPU_SELECTOR="$REQUESTED_CUDA_VISIBLE_DEVICES"
        if [ -n "$GPU_SELECTOR" ]; then GPU_SELECTOR_SOURCE="cuda_visible_devices"; fi
      fi
      if [ -z "$GPU_SELECTOR" ]; then
        GPU_REPORT="visible GPUs:
$GPU_TABLE
MASKFREE_GPU_UUID and CUDA_VISIBLE_DEVICES are unset"
        if [ "$RUN_FULL" = "1" ]; then
          die "no physical GPU selector: set MASKFREE_GPU_UUID or a single CUDA_VISIBLE_DEVICES index/UUID; this script will not choose a device for you."
        fi
      else
        case "$GPU_SELECTOR" in
          *,*|*" "*) die "GPU selector '$GPU_SELECTOR' is ambiguous; use one physical index or UUID" ;;
        esac
        MATCH="$(printf '%s\n' "$GPU_TABLE" | awk -F', *' -v selector="$GPU_SELECTOR" \
          '$1 == selector || $2 == selector { print; found=1 } END { if (!found) exit 1 }' || true)"
        [ -n "$MATCH" ] || die "GPU selector '$GPU_SELECTOR' matches no physical GPU reported by nvidia-smi:
$GPU_TABLE"
        GPU_INDEX="$(printf '%s\n' "$MATCH" | awk -F', *' '{print $1}')"
        GPU_UUID="$(printf '%s\n' "$MATCH" | awk -F', *' '{print $2}')"
        GPU_NAME="$(printf '%s\n' "$MATCH" | awk -F', *' '{print $3}')"
        # If both selectors were supplied, they must resolve to the same
        # physical GPU. This prevents an inherited numeric mask from being
        # silently ignored when an explicit UUID is added.
        if [ -n "$MASKFREE_GPU_UUID" ] && [ -n "$REQUESTED_CUDA_VISIBLE_DEVICES" ]; then
          REQUESTED_MATCH="$(printf '%s\n' "$GPU_TABLE" | awk -F', *' -v selector="$REQUESTED_CUDA_VISIBLE_DEVICES" \
            '$1 == selector || $2 == selector { print; found=1 } END { if (!found) exit 1 }' || true)"
          [ -n "$REQUESTED_MATCH" ] || die "CUDA_VISIBLE_DEVICES='$REQUESTED_CUDA_VISIBLE_DEVICES' matches no physical GPU"
          REQUESTED_UUID="$(printf '%s\n' "$REQUESTED_MATCH" | awk -F', *' '{print $2}')"
          [ "$REQUESTED_UUID" = "$GPU_UUID" ] || die "MASKFREE_GPU_UUID='$MASKFREE_GPU_UUID' disagrees with CUDA_VISIBLE_DEVICES='$REQUESTED_CUDA_VISIBLE_DEVICES'"
        fi
        if ! printf '%s' "$GPU_NAME" | grep -qF "RTX 4070" && [ -z "$GPU_OVERRIDE_RECORD" ]; then
          die "selected GPU '$GPU_NAME' is not an RTX 4070; set MASKFREE_GPU_OVERRIDE to record an explicit rented-GPU identity"
        fi
        if printf '%s' "$GPU_NAME" | grep -qF "$UNSUPPORTED_GPU_PATTERNS"; then
          if [ -z "$MASKFREE_ALLOW_GPU_NAME" ] && [ -z "$MASKFREE_GPU_OVERRIDE" ]; then
            die "selected GPU '$GPU_NAME' matches the unsupported list ('$UNSUPPORTED_GPU_PATTERNS'). Set MASKFREE_ALLOW_GPU_NAME or MASKFREE_GPU_OVERRIDE to override deliberately."
          fi
          if [ -n "$MASKFREE_ALLOW_GPU_NAME" ] && \
             ! printf '%s' "$GPU_NAME" | grep -qF "$MASKFREE_ALLOW_GPU_NAME" && \
             [ -z "$MASKFREE_GPU_OVERRIDE" ]; then
            die "MASKFREE_ALLOW_GPU_NAME='$MASKFREE_ALLOW_GPU_NAME' does not identify selected GPU '$GPU_NAME'"
          fi
        fi
        GPU_REPORT="selected index=$GPU_INDEX name=$GPU_NAME uuid=$GPU_UUID selector=$GPU_SELECTOR override=${GPU_OVERRIDE_RECORD:-none}"
        export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
        export CUDA_VISIBLE_DEVICES="$GPU_UUID"
      fi
    fi
  fi
fi

# CUDA must see this setting before torch is imported. It is harmless for a
# CPU software check and is recorded in the launcher environment.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

INVENTORY_CMD="$PYTHON scripts/prepare_maskfree_data.py --dataset $DATASET --root <config data_root> --output $RUN_DIR/manifests"
PREFLIGHT_CMD="$PYTHON scripts/train_maskfree.py --config $RESOLVED_CONFIG --preflight"
TRAIN_CMD="$PYTHON scripts/train_maskfree.py --config $RESOLVED_CONFIG"

cat <<PLAN
[maskfree] ---------------------------------------------------------------
[maskfree] dataset            : $DATASET
[maskfree] repo root          : $ROOT
[maskfree] workspace          : $WORKSPACE
[maskfree] run id             : $RUN_ID
[maskfree] run dir            : $RUN_DIR
[maskfree] template config    : $TEMPLATE_CONFIG
[maskfree] resolved config    : $RESOLVED_CONFIG (written read-only, 0444)
[maskfree] data root override : ${DATA_ROOT:-<template default>}
[maskfree] total epochs       : $TOTAL_EPOCHS
[maskfree] physical batch     : $BATCH_SIZE
[maskfree] accumulation       : $ACCUM_STEPS
[maskfree] effective batch    : $EFFECTIVE_BATCH
[maskfree] batch fallback     : $CONTRASTIVE_NOTE
[maskfree] epoch validation   : enabled=$EPOCH_VALIDATION reference_config=${EPOCH_REFERENCE_CONFIG:-<none>} (report-only frozen students on dev; separate evaluator; no checkpoint selection)
[maskfree] device             : $DEVICE
[maskfree] audit override     : ${AUDIT_DEVICE:-<template default>}
[maskfree] timing / logging   : ${TIMING_MODE:-<template default>} / ${LOGGING_MODE:-<template default>}
[maskfree] CPU candidates     : ${CANDIDATE_WORKERS:-<template default>} workers x ${CANDIDATE_WORKER_THREADS:-<template default>} threads
[maskfree] cache / prefetch   : ${DATA_CACHE_BYTES:-<template default>} bytes / ${PREFETCH_BATCHES:-<template default>} batches, ${PREFETCH_MAX_BYTES:-<template default>} bytes
[maskfree] gpu gate           : $GPU_REPORT
[maskfree] interpreter        : $PYTHON
[maskfree] ---------------------------------------------------------------
[maskfree] stage 2 inventory  : $INVENTORY_CMD
[maskfree] stage 3 preflight  : $PREFLIGHT_CMD
[maskfree] stage 4 train      : $TRAIN_CMD
[maskfree] stage 5 verify     : scripts/run_maskfree_full.sh --verify-run $RUN_DIR
[maskfree] ---------------------------------------------------------------
PLAN

if [ "$RUN_FULL" != "1" ]; then
  log "RUN_FULL is not 1: printed the resolved plan and exited without executing."
  log "Nothing was written, no GPU was used, no data was read."
  log "Set RUN_FULL=1 to execute. Full execution status for this project is NOT STARTED."
  exit 0
fi

# --------------------------------------------------------------------------- #
# execution
# --------------------------------------------------------------------------- #
if [ -e "$RUN_DIR" ]; then
  die "run directory already exists: $RUN_DIR (a run id is immutable; pick a new RUN_ID)"
fi
if [ -e "$RESOLVED_CONFIG" ]; then
  die "resolved config already exists: $RESOLVED_CONFIG (refusing to overwrite a run's identity)"
fi

mkdir -p "$CONFIG_DIR" "$LOG_DIR" "$RUN_DIR/reports"

# Persist the physical-device decision before inventory or training. The
# original visibility request and the resolved UUID are both retained so a
# later audit can distinguish CUDA's logical index from the host's GPU index.
MASKFREE_GPU_IDENTITY="$RUN_DIR/reports/gpu_identity.json" \
MASKFREE_GPU_TABLE="$GPU_TABLE" \
MASKFREE_GPU_INDEX="$GPU_INDEX" \
MASKFREE_GPU_UUID_VALUE="$GPU_UUID" \
MASKFREE_GPU_NAME="$GPU_NAME" \
MASKFREE_GPU_OVERRIDE_RECORD="$GPU_OVERRIDE_RECORD" \
MASKFREE_GPU_SELECTOR_SOURCE="$GPU_SELECTOR_SOURCE" \
MASKFREE_REQUESTED_CVD="$REQUESTED_CUDA_VISIBLE_DEVICES" \
MASKFREE_RESOLVED_CVD="${CUDA_VISIBLE_DEVICES:-}" \
MASKFREE_DEVICE="$DEVICE" \
MASKFREE_BATCH="$BATCH_SIZE" \
MASKFREE_ACCUM="$ACCUM_STEPS" \
MASKFREE_EFFECTIVE_BATCH="$EFFECTIVE_BATCH" \
MASKFREE_RUN_ID="$RUN_ID" \
"$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

requested = os.environ.get("MASKFREE_REQUESTED_CVD", "")
uuid = os.environ.get("MASKFREE_GPU_UUID_VALUE", "")
override = os.environ.get("MASKFREE_GPU_OVERRIDE_RECORD") or None
selection = os.environ.get("MASKFREE_GPU_SELECTOR_SOURCE", "none")
payload = {
    "schema": "maskfree_gpu_identity.v1",
    "run_id": os.environ["MASKFREE_RUN_ID"],
    "device": os.environ["MASKFREE_DEVICE"],
    "requested_cuda_visible_devices": requested or None,
    "resolved_cuda_visible_devices": os.environ.get("MASKFREE_RESOLVED_CVD") or None,
    "selection": selection,
    "resolved_physical_index": os.environ.get("MASKFREE_GPU_INDEX") or None,
    "resolved_uuid": uuid or None,
    "resolved_name": os.environ.get("MASKFREE_GPU_NAME") or None,
    "expected_name": "RTX 4070",
    "override_record": override,
    "observed_gpu_inventory": [
        line for line in os.environ.get("MASKFREE_GPU_TABLE", "").splitlines() if line.strip()
    ],
    "physical_batch": int(os.environ["MASKFREE_BATCH"]),
    "accumulation_steps": int(os.environ["MASKFREE_ACCUM"]),
    "effective_batch": int(os.environ["MASKFREE_EFFECTIVE_BATCH"]),
    "identity_status": (
        "resolved" if uuid else
        "cpu_software_check" if os.environ["MASKFREE_DEVICE"] == "cpu" else
        "unresolved_plan"
    ),
}
target = Path(os.environ["MASKFREE_GPU_IDENTITY"])
target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

# ---------------------------------------------------------------------------
# stage 1b: the resolved config
#
# Written through W5's own MaskfreeConfig loader and its frozen dataclass, so
# the only fields that can exist are the ones its schema declares and every
# operator override is re-validated by MaskfreeConfig.__post_init__ before any
# data is touched. The file is then made read-only: it is this run's identity,
# and the trainer's resume guard compares against it.
#
# output_dir is the WORKSPACE ROOT. The trainer appends
# runs/maskfree150/<dataset>/<run_id>, which is exactly $RUN_DIR above.
# ---------------------------------------------------------------------------
log "stage 1b: writing the resolved config through W5's MaskfreeConfig schema"
MASKFREE_TEMPLATE="$TEMPLATE_CONFIG" \
MASKFREE_RESOLVED="$RESOLVED_CONFIG" \
MASKFREE_RUN_ID="$RUN_ID" \
MASKFREE_WORKSPACE="$WORKSPACE" \
MASKFREE_DATA_ROOT="$DATA_ROOT" \
MASKFREE_TOTAL_EPOCHS="$TOTAL_EPOCHS" \
MASKFREE_BATCH="$BATCH_SIZE" \
MASKFREE_ACCUM="$ACCUM_STEPS" \
MASKFREE_DEVICE="$DEVICE" \
MASKFREE_ALLOW_CPU="$ALLOW_CPU" \
MASKFREE_EPOCH_VALIDATION="$EPOCH_VALIDATION" \
MASKFREE_EPOCH_REFERENCE_CONFIG="$EPOCH_REFERENCE_CONFIG" \
MASKFREE_LAUNCH_ENV="$LOG_DIR/launch_env.sh" \
MASKFREE_REPO="$ROOT" \
"$PYTHON" - <<'PY'
import dataclasses
import os
import shlex
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(os.environ["MASKFREE_REPO"]) / "src"))
from self_audit_maskfree.config import RUNTIME_FIELDS, load_config

config = load_config(os.environ["MASKFREE_TEMPLATE"])
overrides = {
    "run_id": os.environ["MASKFREE_RUN_ID"],
    "output_dir": os.environ["MASKFREE_WORKSPACE"],
    "total_epochs": int(os.environ["MASKFREE_TOTAL_EPOCHS"]),
    "batch_size": int(os.environ["MASKFREE_BATCH"]),
    "accumulation_steps": int(os.environ["MASKFREE_ACCUM"]),
    "device": os.environ["MASKFREE_DEVICE"],
    "allow_cpu": os.environ["MASKFREE_ALLOW_CPU"] == "1",
    "epoch_validation": os.environ["MASKFREE_EPOCH_VALIDATION"] == "1",
}
for field in ("audit_device",) + RUNTIME_FIELDS:
    value = os.environ.get(field.upper())
    if value is not None:
        overrides[field] = int(value) if isinstance(getattr(config, field), int) else value
data_root = os.environ.get("MASKFREE_DATA_ROOT") or ""
if data_root:
    overrides["data_root"] = data_root
epoch_reference_config = os.environ.get("MASKFREE_EPOCH_REFERENCE_CONFIG") or ""
if epoch_reference_config:
    # This is a control-plane value for the isolated child evaluator. The
    # training process deliberately does not open or validate that path.
    overrides["epoch_reference_config"] = epoch_reference_config

# dataclasses.replace re-runs MaskfreeConfig.__post_init__, so an override that
# violates the schema fails here, not deep inside a 150-epoch run.
resolved = dataclasses.replace(config, **overrides)

target = Path(os.environ["MASKFREE_RESOLVED"])
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(yaml.safe_dump(resolved.to_dict(), sort_keys=True), encoding="utf-8")
target.chmod(0o444)

# Hand the resolved values the later stages need back to the shell, quoted.
env_path = Path(os.environ["MASKFREE_LAUNCH_ENV"])
env_path.parent.mkdir(parents=True, exist_ok=True)
env_path.write_text(
    "\n".join(
        f"{key}={shlex.quote(str(value))}"
        for key, value in (
            ("RESOLVED_DATA_ROOT", resolved.data_root),
            ("RESOLVED_PROTOCOL", resolved.protocol),
            ("RESOLVED_SEED", resolved.seed),
            ("RESOLVED_DEPTH_AXIS", resolved.depth_axis),
            ("RESOLVED_IMAGE_SIZE", resolved.image_size),
            ("RESOLVED_EPOCH_VALIDATION", resolved.epoch_validation),
            ("RESOLVED_EPOCH_REFERENCE_CONFIG", resolved.epoch_reference_config or ""),
        )
    )
    + "\n",
    encoding="utf-8",
)
print(f"[maskfree] resolved config written read-only: {target}")
PY

# shellcheck source=/dev/null
. "$LOG_DIR/launch_env.sh"

log "stage 2: image-only inventory (read-only; never opens a mask tree)"
"$PYTHON" scripts/prepare_maskfree_data.py \
  --dataset "$DATASET" \
  --root "$RESOLVED_DATA_ROOT" \
  --output "$RUN_DIR/manifests" \
  --seed "$RESOLVED_SEED" \
  --protocol "$RESOLVED_PROTOCOL" \
  --depth-axis "$RESOLVED_DEPTH_AXIS" \
  --image-size "$RESOLVED_IMAGE_SIZE" 2>&1 \
  | tee "$LOG_DIR/inventory_${DATASET}.log"

log "stage 3: bounded preflight and batch gate (real fit/score/backward/checkpoint/export)"
"$PYTHON" scripts/train_maskfree.py --config "$RESOLVED_CONFIG" --preflight 2>&1 \
  | tee "$LOG_DIR/preflight_${DATASET}.log"

GATE_RECEIPT="$RUN_DIR/reports/gate_receipt.json"
[ -f "$GATE_RECEIPT" ] || die "preflight produced no $GATE_RECEIPT; the full run is gated on that receipt"
MASKFREE_GATE="$GATE_RECEIPT" MASKFREE_BATCH="$BATCH_SIZE" MASKFREE_ACCUM="$ACCUM_STEPS" \
"$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

receipt = json.loads(Path(os.environ["MASKFREE_GATE"]).read_text(encoding="utf-8"))
if receipt.get("status") != "pass":
    raise SystemExit(
        f"[maskfree] ERROR: batch/GPU gate did not pass: status={receipt.get('status')!r} "
        f"reason={receipt.get('reason')!r}. Resolve the fallback explicitly with "
        "MASKFREE_FALLBACK before the run; there is no mid-run retry."
    )
requested = (int(os.environ["MASKFREE_BATCH"]), int(os.environ["MASKFREE_ACCUM"]))
observed = (receipt.get("physical_batch"), receipt.get("accumulation_steps"))
if observed != requested:
    raise SystemExit(
        f"[maskfree] ERROR: the gate ran at physical/accumulation {observed}, the run was "
        f"requested at {requested}. A gate receipt only licenses the shape it measured."
    )
print(f"[maskfree] gate receipt: status=pass device={receipt.get('device')!r} "
      f"physical_batch={observed[0]} accumulation_steps={observed[1]} "
      f"effective_batch={receipt.get('effective_batch')}")
PY

log "stage 4: the single ${TOTAL_EPOCHS}-epoch global timeline, both student arms"
"$PYTHON" scripts/train_maskfree.py --config "$RESOLVED_CONFIG" 2>&1 \
  | tee "$LOG_DIR/train_${DATASET}.log"

log "stage 5: completion check"
verify_run_dir "$RUN_DIR"

log "run complete: $RUN_DIR"
printf '%s\n' "$RUN_DIR" > "$LOG_DIR/last_run_dir.txt"
