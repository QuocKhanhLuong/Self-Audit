import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

arm = sys.argv[1]
root = Path('/workspace/compact-acdc-l8/20261028') / f'{arm}_full'
log = root / 'TRAIN_LOG.jsonl'
print(f'ACDC L8 | {arm} 10,000 steps | Ctrl-C exits viewer only', flush=True)
while True:
    now = datetime.now(timezone.utc).strftime('%H:%M:%S UTC')
    report = root / 'REPORT.json'
    if report.exists():
        data = json.loads(report.read_text())
        print(f'{now} COMPLETE | steps={data.get("steps")} status={data.get("status")} | {report}', flush=True)
        break
    n = 0
    row = {}
    if log.exists():
        n = int(subprocess.check_output(['wc', '-l', str(log)], text=True).split()[0])
        with log.open('rb') as f:
            f.seek(max(0, f.seek(0, 2) - 65536))
            lines = f.read().splitlines()
        try:
            row = json.loads(lines[-1]) if lines else {}
        except json.JSONDecodeError:
            row = {}
    err = root.parent / f'{arm}_full.stderr.log'
    if err.exists() and err.stat().st_size:
        print(f'{now} STDERR | {err}', flush=True)
    try:
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader'], text=True).strip()
    except Exception:
        gpu = 'GPU telemetry unavailable'
    print(f'{now} step {n:5d}/10000 | status={row.get("status", "preflight")} | loss={row.get("supervised_loss")} | {gpu}', flush=True)
    time.sleep(5)
