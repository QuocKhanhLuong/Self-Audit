#!/bin/bash
set -euo pipefail
exec > >(tee -a /workspace/self-audit-v2/logs/priority-training.log) 2>&1
cd /root/Self-Audit-acdc-v2-run2
printf 'Waiting for both 32-step real patient pilots.\n'
while [ ! -f /workspace/self-audit-v2/runs/pilot_B16_decoder_32/REPORT.json ]; do
 for scope in head decoder; do
  if [ -f "/workspace/self-audit-v2/runs/pilot_B16_${scope}_32/FAILURE.json" ]; then
   printf 'Pilot failure: stopping full-run queue.\n'; exit 1
  fi
 done
 sleep 15
done
python - <<'CHECK'
from pathlib import Path
import json
for scope in ['head','decoder']:
 p=Path('/workspace/self-audit-v2/runs')/f'pilot_B16_{scope}_32'
 r=json.loads((p/'REPORT.json').read_text())
 assert r['status']=='COMPLETED' and not r['synthetic'] and r['steps']==32
 assert not r['label_reads_outside_budget']
 rows=[json.loads(s) for s in (p/'steps.jsonl').read_text().splitlines()]
 assert all(row['gradient_norm']>0 and row['supervised']['patient']!=row['audit']['patient'] for row in rows)
 assert (p/'final/weights.pt').is_file()
print('Both pilots completed and mechanical receipts checked; no quality claim.')
CHECK
manifest=/workspace/self-audit-v2/cache/L8_S42/train.json
dev=/workspace/self-audit-v2/cache/L8_S42/development.json
data_sha=$(sha256sum "$manifest" | cut -d ' ' -f1)
dev_sha=$(sha256sum "$dev" | cut -d ' ' -f1)
for method in supervised supervised_extra ema_full ema_block joint_guard independent_guard sign joint_no_class joint_no_guard; do
 run="/workspace/self-audit-v2/runs/L8_S42_${method}_B16_10000"
 printf 'Starting %s at ' "$method"; date -u --iso-8601=seconds
 python train.py --config "results/configs/L8_S42_${method}.json" --model-config /workspace/self-audit-v2/BACKBONE.json --data "$manifest" --data-sha256 "$data_sha" --out "$run" --device cuda --allow-full-run
 checkpoint_sha=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["final_manifest_sha256"])' "$run/REPORT.json")
 evaluation="/workspace/self-audit-v2/evaluation/L8_S42_${method}_B16_10000"
 python - <<'DISK'
import json,shutil
from pathlib import Path
import numpy as np
p=Path('/workspace/self-audit-v2/cache/L8_S42/development.json');o=json.loads(p.read_text())
need=sum(4*np.load(p.parent/r['image'],mmap_mode='r',allow_pickle=False).nbytes for r in o['patients'])+2*1024**3
assert shutil.disk_usage('/workspace').free>need, 'Not enough disk for frozen probabilities: stop queue, preserve artifacts'
DISK
 python evaluate.py --checkpoint "$run/final" --checkpoint-sha256 "$checkpoint_sha" --data "$dev" --data-sha256 "$dev_sha" --out "$evaluation" --device cuda
 python scripts/render_acdc_results.py --evaluation "$evaluation" --data "$dev" --out "results/figures/L8_S42_${method}_B16_10000" --train-run "$run"
 printf 'Completed frozen evaluation for %s at ' "$method"; date -u --iso-8601=seconds
done
printf 'L8 subset-seed42 priority arm queue completed. Other budgets/subset seeds remain pending.\n'
