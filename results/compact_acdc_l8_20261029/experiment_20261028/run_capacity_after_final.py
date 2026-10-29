import json
import subprocess
import sys
import time
from pathlib import Path


root = Path('/workspace/compact-acdc-l8/20261028')
result = root / 'FINAL_RESULT.json'
blocker = root / 'FINALIZE_BLOCKER.json'
while not result.exists():
    if blocker.exists():
        raise SystemExit('Finalization blocked; skip batch200 capacity profile')
    time.sleep(10)

final = json.loads(result.read_text())
if not final.get('benchmark_path') or not Path(final['benchmark_path']).is_file():
    raise SystemExit('Benchmark receipt missing; skip capacity profile')

out = root / 'batch200_capacity_profile'
if out.exists():
    raise SystemExit('Capacity profile output exists; refusing overwrite')
cmd = [
    sys.executable, 'train.py',
    '--config', str(root / 'batch200_capacity_config.json'),
    '--data', '/workspace/Self-Audit/data/prepared/ACDC_L8_S42/train.json',
    '--data-sha256', 'e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4',
    '--out', str(out), '--device', 'cuda', '--allow-full-run',
]
with (root / 'batch200_capacity.stdout.log').open('w') as stdout, (root / 'batch200_capacity.stderr.log').open('w') as stderr:
    completed = subprocess.run(cmd, cwd='/workspace/Self-Audit-compact', stdout=stdout, stderr=stderr)
raise SystemExit(completed.returncode)
