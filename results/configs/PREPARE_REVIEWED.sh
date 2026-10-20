#!/bin/bash
set -euo pipefail
exec >> /workspace/self-audit-v2/logs/data-preparation-reviewed.log 2>&1
cd /root/Self-Audit-acdc-v2-run2
while kill -0 26164 2>/dev/null; do sleep 15; done
python - <<'CHECK'
from pathlib import Path
lines=Path('/workspace/self-audit-v2/logs/rclone-check.txt').read_text().splitlines()
assert len(lines)==1053 and all(x.startswith('= ') for x in lines), 'remote checksum comparison incomplete or mismatched'
print('REMOTE CHECK: all 1053 files match')
CHECK
python scripts/verify_acdc_download.py --root /workspace/self-audit-v2/data/raw/ACDC --split splits/acdc_reviewed_locked_train80_val18.json --out /workspace/self-audit-v2/logs/DATA_VERIFICATION_REVIEWED.json
for labels in 8 4 16; do
  for subset_seed in 42 123 2026; do
    python - <<'PY'
from pathlib import Path
import shutil
raw=Path('/workspace/self-audit-v2/data/raw/ACDC/training')
# Conservative bound: native cache float32 may exceed source integer sizes.
required=1024**3 if Path('/workspace/self-audit-v2/cache/L8_S42/PREPARATION.json').exists() else 4*sum(p.stat().st_size for p in raw.rglob('*4d.nii*'))+1024**3
free=shutil.disk_usage('/workspace/self-audit-v2').free
if free < required:
    raise SystemExit(f'Insufficient free disk for another cache: free={free}, required_bound={required}; preserving all completed caches')
PY
    extra=()
    if [ "$labels" = 8 ] && [ "$subset_seed" = 42 ]; then extra=(--evaluation); else extra=(--image-cache /workspace/self-audit-v2/cache/L8_S42); fi
    python prepare_acdc.py --root /workspace/self-audit-v2/data/raw/ACDC/training --split splits/acdc_reviewed_locked_train80_val18.json --labels "$labels" --seed "$subset_seed" --out "/workspace/self-audit-v2/cache/L${labels}_S${subset_seed}" "${extra[@]}"
  done
done
printf 'DATA PREPARATION COMPLETED; UTC: '
date -u --iso-8601=seconds
