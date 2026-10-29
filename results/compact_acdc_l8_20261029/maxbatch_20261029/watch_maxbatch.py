import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


root = Path('/workspace/compact-acdc-l8/20261029-maxbatch')
run = root / 'wide_kd_batch152_full'
log = run / 'TRAIN_LOG.jsonl'
print('ACDC L8 | compact wide plain KD | batch 152 | 10,000 steps', flush=True)
while True:
    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    report = run / 'REPORT.json'
    if report.exists():
        data = json.loads(report.read_text())
        print(f'{now} COMPLETE | {data.get("steps")}/10000 | {data.get("status")} | {report}', flush=True)
        break
    row = {}
    if log.exists():
        with log.open('rb') as f:
            f.seek(max(0, f.seek(0, 2) - 65536))
            lines = f.read().splitlines()
        if lines:
            try:
                row = json.loads(lines[-1])
            except json.JSONDecodeError:
                pass
    err = root / 'wide_kd_batch152_full.stderr.log'
    err_bytes = err.stat().st_size if err.exists() else 0
    try:
        gpu = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader'],
            text=True,
        ).strip()
    except Exception:
        gpu = 'GPU telemetry unavailable'
    step = row.get('step', -1) + 1
    print(f'{now} step {step:5d}/10000 | {row.get("status", "preflight")} | loss={row.get("supervised_loss")} | stderr={err_bytes}B | {gpu}', flush=True)
    time.sleep(5)
