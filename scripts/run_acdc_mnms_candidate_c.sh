#!/usr/bin/env bash
# Sequential native Candidate C training: ACDC from epoch 0, then M&Ms from epoch 0.
# The two runs are independent; M&Ms never loads the ACDC checkpoint.
#
# CURRICULUM selects which schedule profile the generated run configs are built
# from.  It is a closed enum and an unrecognised value is a hard error.
#
#   joint_from_start (DEFAULT)
#       configs/self_audit_joint_from_start{,_mnms}.yaml
#       One interval [0, 130): trainable=all, objective=retained_final_annotation,
#       rollout=threshold_gate.  Annotator and Auditor are both trained and the
#       accept/reject gate is consulted from the first optimizer step.
#
#   staged
#       configs/self_audit_full{,_mnms}.yaml
#       The canonical 3-interval curriculum (annotation_bootstrap 0-99,
#       auditor_training 100-119, joint_self_audit 120-129).  Its original
#       native behaviour is retained unchanged, including
#       predicted_history_exposure=True at weight 0.1.
#
# Both profiles are run as fresh trainings from initialization.  Neither passes
# a resume argument, and neither may be used to continue a checkpoint produced
# under the other profile: the schedule and config signature differ and the
# lineage guards that reject such a continuation must not be bypassed.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p runs logs run_configs

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
# Native batch size, unchanged from the existing override.  It is applied to
# every interval of the generated run configs and does not alter the checked-in
# profile YAML files.  It has not been measured under the joint-from-start
# profile, where the gated joint rollout runs from epoch 0; lower it at
# invocation if the run runs out of memory.
export BATCH_SIZE="${BATCH_SIZE:-8}"

CURRICULUM="${CURRICULUM:-joint_from_start}"
case "$CURRICULUM" in
  joint_from_start|staged) ;;
  *)
    echo "Error: CURRICULUM='${CURRICULUM}' is not supported." >&2
    echo "Valid values: joint_from_start (default) | staged" >&2
    exit 2
    ;;
esac

if [[ "$CURRICULUM" == "staged" ]]; then
  ACDC_SOURCE_CONFIG="configs/self_audit_full.yaml"
  MNMS_SOURCE_CONFIG="configs/self_audit_full_mnms.yaml"
else
  ACDC_SOURCE_CONFIG="configs/self_audit_joint_from_start.yaml"
  MNMS_SOURCE_CONFIG="configs/self_audit_joint_from_start_mnms.yaml"
fi

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
ACDC_RUN="${ACDC_RUN:-acdc_candidate_c_${CURRICULUM}_4070_${STAMP}}"
MNMS_RUN="${MNMS_RUN:-mnms_candidate_c_${CURRICULUM}_4070_${STAMP}}"

# Each invocation writes its generated run configs into its OWN directory.
#
# This matters because the M&Ms leg starts hours after generation: the ACDC leg
# runs to completion first.  With a shared filename, a second invocation started
# in the meantime would rewrite the config the first invocation's pending M&Ms
# leg is about to read, and that leg would silently train under the second
# invocation's BATCH_SIZE.  A timestamp alone does not fix it (two invocations
# can share a second), so the directory is always created with mktemp, which
# fails rather than reuse an existing path.  There is deliberately no way to
# point this at an existing directory: an override would reintroduce exactly the
# sharing this prevents.  The directory is kept after the run: the generated
# configs are the provenance record of what was actually executed.
RUN_CONFIG_DIR="$(mktemp -d "run_configs/${CURRICULUM}_${STAMP}_XXXXXX")"
ACDC_CONFIG="${RUN_CONFIG_DIR}/acdc_candidate_c_${CURRICULUM}.yaml"
MNMS_CONFIG="${RUN_CONFIG_DIR}/mnms_candidate_c_${CURRICULUM}.yaml"

echo "=============================================================================="
echo "Self-Audit sequential Candidate C native training"
echo "Curriculum     : ${CURRICULUM}"
echo "ACDC profile   : ${ACDC_SOURCE_CONFIG}"
echo "M&Ms profile   : ${MNMS_SOURCE_CONFIG}"
echo "GPU visibility : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Batch size     : ${BATCH_SIZE} (run-local override, applied to every interval)"
echo "ACDC run       : ${ACDC_RUN}"
echo "M&Ms run       : ${MNMS_RUN}"
echo "Run configs    : ${RUN_CONFIG_DIR} (unique per invocation, kept as provenance)"
echo "=============================================================================="

# Generate run-specific configs from the checked-in profile configs so this
# script is self-contained and the checked-in YAML files remain unchanged.
# The resolved schedule is printed here, from the generated configs themselves.
CURRICULUM="$CURRICULUM" \
ACDC_SOURCE_CONFIG="$ACDC_SOURCE_CONFIG" \
MNMS_SOURCE_CONFIG="$MNMS_SOURCE_CONFIG" \
ACDC_CONFIG="$ACDC_CONFIG" \
MNMS_CONFIG="$MNMS_CONFIG" \
python - <<'PY'
from pathlib import Path
import os
import sys

import yaml

SRC = Path.cwd() / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Imported at module scope on purpose: a preflight that cannot validate must
# fail here, not print a warning and let an unvalidated config reach training.
from self_audit.training.unified_config import load_unified_config

batch_size = int(os.environ.get("BATCH_SIZE", "8"))
if batch_size <= 0:
    raise ValueError(f"BATCH_SIZE must be > 0, got {batch_size}")

curriculum = os.environ["CURRICULUM"]
if curriculum not in {"joint_from_start", "staged"}:
    raise ValueError(f"Unsupported CURRICULUM {curriculum!r}")

# The staged curriculum keeps its original native behaviour: the auxiliary
# predicted-history rollout is switched ON at weight 0.1, which is the only way
# its frozen bootstrap and auditor intervals see predicted history at all.
#
# The joint-from-start profile leaves the flag as its config declares it (OFF).
# The trainer's retained_final_annotation branch does not consult the flag at
# all -- the gated joint rollout is its own forward and always consumes accepted
# predicted auditor feedback -- so the flag is inert for this profile either
# way, and OFF simply records that no auxiliary pass is requested.  It is not a
# feedback switch.
force_exposure = curriculum == "staged"

pairs = [
    (
        Path(os.environ["ACDC_SOURCE_CONFIG"]),
        Path(os.environ["ACDC_CONFIG"]),
        f"self_audit_acdc_candidate_c_{curriculum}",
    ),
    (
        Path(os.environ["MNMS_SOURCE_CONFIG"]),
        Path(os.environ["MNMS_CONFIG"]),
        f"self_audit_mnms_candidate_c_{curriculum}",
    ),
]

for src, dst, experiment_name in pairs:
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
    cfg["experiment"]["name"] = experiment_name
    cfg["model"]["window_mode"] = "candidate_c"
    if force_exposure:
        cfg["training"]["rollout"]["predicted_history_exposure"] = True
        cfg["training"]["rollout"]["predicted_history_weight"] = 0.1

    # Requested native batch size. Keep it run-local so the checked-in profile
    # configs remain untouched; BATCH_SIZE can be overridden at invocation.
    for interval in cfg["training"]["schedule"]["intervals"]:
        interval["batch_size"] = batch_size

    dst.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create: the file is claimed and written in one step, so there is
    # no window between "does it exist?" and "write it" for another invocation
    # to slip through.  An existing path here means the per-invocation directory
    # was bypassed, not that a rerun is harmless, and the OS raises
    # FileExistsError before any byte is written.
    try:
        with open(dst, "x", encoding="utf-8") as handle:
            handle.write(yaml.safe_dump(cfg, sort_keys=False))
    except FileExistsError as exc:
        raise FileExistsError(
            f"Refusing to overwrite an existing generated run config: {dst}. "
            "Each invocation must generate into its own fresh directory."
        ) from exc
    # Provenance: the executed config is read-only once written.
    dst.chmod(0o444)

    # Validate the generated YAML through the real strict loader so a broken
    # profile fails now rather than after the dataset has been built.
    load_unified_config(dst)

    intervals = cfg["training"]["schedule"]["intervals"]
    schedule = [
        f"{it['start_epoch']}-{it['end_epoch'] - 1}:{it['batch_size']}"
        for it in intervals
    ]
    print(
        f"[config] {dst}: strict schema validation OK "
        f"window_mode={cfg['model']['window_mode']} "
        f"predicted_history_exposure={cfg['training']['rollout']['predicted_history_exposure']} "
        f"batch_schedule={','.join(schedule)}"
    )
    for it in intervals:
        print(
            f"[schedule] {dst}: epochs [{it['start_epoch']}, {it['end_epoch']}) "
            f"name={it['name']} trainable={it['trainable']} "
            f"objective={it['objective']} rollout={it['rollout']} "
            f"encoder_lr={it['encoder_lr']} annotation_lr={it['annotation_lr']} "
            f"auditor_lr={it['auditor_lr']}"
        )
    print(
        f"[schedule] {dst}: total_epochs={cfg['training']['schedule']['total_epochs']} "
        f"best_selection_min_epoch={cfg['checkpoint']['best_selection_min_epoch']} "
        f"warmup_epochs={cfg['training']['lr_curve']['warmup_epochs']} "
        f"(optimizer LR ramp only; no curriculum warmup interval)"
    )
    if curriculum == "joint_from_start":
        print(
            f"[schedule] {dst}: auditor feedback and accept/reject are live from "
            f"epoch 0; predicted_history_exposure=False means only that no "
            f"auxiliary rollout is requested, and the joint objective ignores "
            f"that flag anyway"
        )
PY

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
  2>&1 | tee "logs/${ACDC_RUN}.log"

printf '>>> ACDC COMPLETE: %s\n' "$ACDC_RUN"

# -----------------------------------------------------------------------------
# 2) Native M&Ms — a new independent full 130-epoch run from initialization.
#    No resume and no ACDC checkpoint are supplied here by design.
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
  2>&1 | tee "logs/${MNMS_RUN}.log"

printf '>>> M&Ms COMPLETE: %s\n' "$MNMS_RUN"

echo
echo "=============================================================================="
echo "ALL NATIVE RUNS COMPLETE"
echo "Curriculum : $CURRICULUM"
echo "ACDC       : runs/$ACDC_RUN"
echo "M&Ms       : runs/$MNMS_RUN"
echo "Configs    : $RUN_CONFIG_DIR"
echo "=============================================================================="
