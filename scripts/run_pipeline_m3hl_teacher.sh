#!/usr/bin/env bash
# M3HL-teacher campaign. Usage: SELF_AUDIT_ROOT=... [MAX_HOURS=..] [CAP_HOURS_<STEP>=..] scripts/run_pipeline_m3hl_teacher.sh [--steps "M3WAIT ..."] [--device cuda]
# T-M3HL training (official recipe) is run separately in <run>/T-M3HL; see reports/m3hl-teacher-20261121/RUNNING.md.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${SELF_AUDIT_ROOT:?set SELF_AUDIT_ROOT}"
exec python scripts/m3hl_pipeline.py "$@"
