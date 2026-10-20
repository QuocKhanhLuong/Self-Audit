#!/bin/bash
set -euo pipefail
exec >> /workspace/self-audit-v2/vram24/logs/test-inference.log 2>&1
cd /root/Self-Audit-vram24
printf 'Waiting for the final priority-arm frozen VAL evaluation; TEST cannot affect training.\n'
while [ ! -f /workspace/self-audit-v2/vram24/evaluation/L8_S42_joint_no_guard_C256_B52_10000/METRICS.json ]; do sleep 30; done
for method in supervised supervised_extra ema_full ema_block joint_guard independent_guard sign joint_no_class joint_no_guard; do
 run="/workspace/self-audit-v2/vram24/runs/L8_S42_${method}_C256_B52_10000"
 sha=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["final_manifest_sha256"])' "$run/REPORT.json")
 python scripts/infer_acdc_test.py --checkpoint "$run/final" --checkpoint-sha256 "$sha" --root /workspace/self-audit-v2/data/raw/ACDC/testing --out "/workspace/self-audit-v2/vram24/test-inference/L8_S42_${method}_C256_B52_10000" --batch 16
done
