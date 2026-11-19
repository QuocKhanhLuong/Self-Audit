#!/usr/bin/env bash
# One-container A4-SA pipeline for Modal: prepare check -> S1 -> S2 -> S3-SA -> A4-SA -> final evaluation.
# Re-running this script resumes: finished stages are skipped (DONE markers), the unfinished stage resumes
# from its last checkpoint.  Budget caps come from BUDGET.md as env vars: MAX_HOURS, CAP_HOURS_{S1,S2,S3,A4,FINAL}.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=/root/mount_data/runs/cbkd-sa-a4-20261119
PREP=/root/mount_data/prepared/ACDC/cbkd-sa-a4-20261119
CFG=$RUN/A4_SA_CONFIG.json
mkdir -p "$RUN"
test -f "$PREP/train.json" && test -f "$PREP/val.json" && test -f "$PREP/test.json" || { echo "prepared data missing: $PREP" >&2; exit 2; }
python scripts/cbkd_sa_make_config.py --prepared "$PREP" --out "$CFG"
python scripts/cbkd_sa_pipeline.py --config "$CFG" --out "$RUN" --device cuda
