import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
root=Path('/workspace/compact-acdc-l8/20261028/b8_recovery/full')
log=root/'steps.jsonl'
print('B8 EMA full recovery | exact historical config | target 10000 steps | Ctrl-C exits viewer only',flush=True)
while True:
    now=datetime.now(timezone.utc).strftime('%H:%M:%S UTC')
    if (root/'REPORT.json').exists():
        r=json.loads((root/'REPORT.json').read_text())
        print(f'{now} COMPLETE | steps={r.get("steps")} | report={root / "REPORT.json"}',flush=True)
        break
    if log.exists():
        count=int(subprocess.check_output(['wc','-l',str(log)],text=True).split()[0])
        with log.open('rb') as f:
            f.seek(max(0,f.seek(0,2)-65536));lines=f.read().splitlines()
        try: row=json.loads(lines[-1]) if lines else {}
        except json.JSONDecodeError: row={}
        loss=row.get('supervised_loss'); loss_text=f'{loss:.5f}' if isinstance(loss,(int,float)) else '...'
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
        print(f'{now} step {count:5d}/10000 | loss {loss_text} | {gpu}',flush=True)
    else: print(f'{now} waiting for train preflight',flush=True)
    time.sleep(5)
