import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
root=Path('/dev/shm/b52_eval_test50')
print('B52 raw-student historical TEST50 evaluator | 50 cines | Ctrl-C exits viewer only',flush=True)
while True:
    now=datetime.now(timezone.utc).strftime('%H:%M:%S UTC')
    if (root/'METRICS.json').exists():
        m=json.loads((root/'METRICS.json').read_text())
        scores={k:round(v['dice'],4) for k,v in m['macro_patient'].items()}
        print(f'{now} COMPLETE | {scores} | metrics={root / "METRICS.json"}',flush=True)
        break
    count=len(list((root/'predictions').glob('patient*.npy'))) if (root/'predictions').exists() else 0
    size=subprocess.check_output(['du','-sh',str(root)],text=True).split()[0] if root.exists() else '0'
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
    print(f'{now} cine files {count:2d}/50 | frozen-prob storage {size} | {gpu}',flush=True)
    time.sleep(5)
