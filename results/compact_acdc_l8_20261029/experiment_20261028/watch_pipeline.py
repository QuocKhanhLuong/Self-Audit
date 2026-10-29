import json
import time
from datetime import datetime, timezone
from pathlib import Path

root = Path('/workspace/compact-acdc-l8/20261028')
eval_root = Path('/dev/shm/b8_eval_test50')
print('ACDC L8 | B8 raw-student TEST50 -> frozen KD gate | Ctrl-C exits viewer only', flush=True)
while True:
    now = datetime.now(timezone.utc).strftime('%H:%M:%S UTC')
    metrics = eval_root / 'METRICS.json'
    if metrics.exists():
        m = json.loads(metrics.read_text())
        classes = {k: v['dice'] for k, v in m['macro_patient'].items()}
        macro4 = sum(classes.values()) / 4
        fg = sum(classes[k] for k in ('RV', 'MYO', 'LV')) / 3
        print(f'{now} B8 TEST50 COMPLETE | Macro4={macro4:.6f} FG={fg:.6f} | {metrics}', flush=True)
        break
    pred_dir = eval_root / 'predictions'
    n = len(list(pred_dir.glob('*.npy'))) if pred_dir.exists() else 0
    err = root / 'b8_recovery/eval.stderr.log'
    if err.exists() and err.stat().st_size:
        print(f'{now} B8 EVAL ERROR | {err}', flush=True)
        break
    print(f'{now} B8 TEST50 predictions {n}/50 | output {eval_root}', flush=True)
    time.sleep(5)
