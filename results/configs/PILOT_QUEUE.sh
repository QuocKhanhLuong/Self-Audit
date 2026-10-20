#!/bin/bash
set -euo pipefail
exec > >(tee -a /workspace/self-audit-v2/logs/pilots.log) 2>&1
cd /root/Self-Audit-acdc-v2-run2
manifest=/workspace/self-audit-v2/cache/L8_S42/train.json
printf 'Waiting for verified L8_S42 preparation...\n'
while [ ! -f /workspace/self-audit-v2/cache/L8_S42/PREPARATION.json ]; do
 if [ -f /workspace/self-audit-v2/logs/DATA_VERIFICATION_REVIEWED.json ]; then
  python -c 'import json; assert json.load(open("/workspace/self-audit-v2/logs/DATA_VERIFICATION_REVIEWED.json"))["status"] == "PASS"'
 fi
 sleep 15
done
data_sha=$(sha256sum "$manifest" | cut -d ' ' -f1)
for scope in head decoder; do
 python train.py --config "/workspace/self-audit-v2/${scope}-pilot.json" --model-config /workspace/self-audit-v2/BACKBONE.json --data "$manifest" --data-sha256 "$data_sha" --out "/workspace/self-audit-v2/runs/pilot_B16_${scope}_32" --device cuda
 done
printf 'BOTH REAL PATIENT PILOTS COMPLETED\n'
