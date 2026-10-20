#!/bin/bash
set -euo pipefail
exec > >(tee -a /workspace/self-audit-v2/vram24/logs/pilot.log) 2>&1
cd /root/Self-Audit-vram24
manifest=/workspace/self-audit-v2/cache/L8_S42/train.json
sha=$(sha256sum "$manifest" | cut -d ' ' -f1)
python train.py --config /workspace/self-audit-v2/vram24/pilot_C256_B52_72.json --model-config /workspace/self-audit-v2/BACKBONE.json --data "$manifest" --data-sha256 "$sha" --out /workspace/self-audit-v2/vram24/runs/pilot_C256_B52_72 --device cuda
