#!/bin/bash
set -euo pipefail
exec > >(tee -a /workspace/self-audit-v2/vram24/logs/priority-training.log) 2>&1
cd /root/Self-Audit-vram24
pilot=/workspace/self-audit-v2/vram24/runs/pilot_C256_B52_72
printf 'Waiting for fixed-crop batch52 full-coverage pilot.\n'
while [ ! -f "$pilot/REPORT.json" ]; do
 if [ -f "$pilot/FAILURE.json" ]; then printf 'Pilot failed; no full run launched.\n';exit 1;fi
 sleep 5
done
python - <<'CHECK'
import json
from pathlib import Path
from self_audit.model import digest
p=Path('/workspace/self-audit-v2/vram24/runs/pilot_C256_B52_72')
r=json.loads((p/'REPORT.json').read_text());m=json.loads((p/'final/MANIFEST.json').read_text())
assert r['status']=='COMPLETED' and not r['synthetic'] and r['steps']==72
assert digest(p/'final/MANIFEST.json')==r['final_manifest_sha256']
assert not r['label_reads_outside_budget'] and set(r['seen_unlabeled'])==set(m['unlabeled_ids'])
rows=[json.loads(s) for s in (p/'steps.jsonl').read_text().splitlines()]
assert all(s['gradient_norm']>0 and s['supervised']['patient']!=s['audit']['patient'] for s in rows)
assert all(s['batch_size']==52 and s['crop_size']==256 for s in rows)
print('C256/B52 pilot PASS; all 72 U patients visited; fresh full-run initialization follows.')
CHECK
manifest=/workspace/self-audit-v2/cache/L8_S42/train.json
dev=/workspace/self-audit-v2/cache/L8_S42/development.json
data_sha=$(sha256sum "$manifest" | cut -d ' ' -f1)
dev_sha=$(sha256sum "$dev" | cut -d ' ' -f1)
for method in supervised supervised_extra ema_full ema_block joint_guard independent_guard sign joint_no_class joint_no_guard; do
 run="/workspace/self-audit-v2/vram24/runs/L8_S42_${method}_C256_B52_10000"
 printf 'Starting crop256 batch52 %s at ' "$method";date -u --iso-8601=seconds
 python train.py --config "results/configs/L8_S42_${method}.json" --model-config /workspace/self-audit-v2/BACKBONE.json --data "$manifest" --data-sha256 "$data_sha" --out "$run" --device cuda --allow-full-run
 checkpoint_sha=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["final_manifest_sha256"])' "$run/REPORT.json")
 evaluation="/workspace/self-audit-v2/vram24/evaluation/L8_S42_${method}_C256_B52_10000"
 python - <<'DISK'
import json,shutil
from pathlib import Path
import numpy as np
p=Path('/workspace/self-audit-v2/cache/L8_S42/development.json');o=json.loads(p.read_text())
need=sum(4*np.load(p.parent/r['image'],mmap_mode='r',allow_pickle=False).nbytes for r in o['patients'])+2*1024**3
assert shutil.disk_usage('/workspace').free>need,'Insufficient disk; preserve artifacts and stop queue'
DISK
 python evaluate.py --checkpoint "$run/final" --checkpoint-sha256 "$checkpoint_sha" --data "$dev" --data-sha256 "$dev_sha" --out "$evaluation" --device cuda
 python scripts/render_acdc_results.py --evaluation "$evaluation" --data "$dev" --out "results/figures/L8_S42_${method}_C256_B52_10000" --train-run "$run"
done
printf 'All nine priority C256/B52 arms finished; other budgets/seeds remain pending.\n'
