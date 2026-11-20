#!/usr/bin/env bash
# W2S-Match on the SSL4MIS split. Usage: SELF_AUDIT_ROOT=... [MAX_HOURS=..] [CAP_HOURS_<STEP>=..] scripts/run_pipeline_w2s_ssl4mis.sh [--steps "W WFINAL SAT ..."] [--device cuda]
# GPU pinning on the 4070 server: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 (set by the caller).
set -euo pipefail
cd "$(dirname "$0")/.."
: "${SELF_AUDIT_ROOT:?set SELF_AUDIT_ROOT}"
exec python scripts/w2s_ssl4mis_pipeline.py "$@"
