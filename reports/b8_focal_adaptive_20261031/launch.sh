#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
exec /workspace/cptu-runtime-20261030/bin/python /workspace/Self-Audit-b8-focal-runner-20261031/scripts/run_b8_focal_adaptive.py --plan /workspace/b8-focal-adaptive-setup-20261031/LOCKED_FOCAL_PLAN.json --plan-sha256 e6724c68a2844ef93683d919dc82f316f23beaa72ca5d897379220e00cec159c > /workspace/ema_full_B8_focal_adaptive_20261031T030334Z_dbcf83c/runner.log 2>&1
