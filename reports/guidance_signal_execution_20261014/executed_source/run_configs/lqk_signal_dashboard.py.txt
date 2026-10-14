#!/usr/bin/env python3
"""Read-only display of the fixed-step guidance signal experiment."""
import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
JOB = Path((ROOT/'runs/LATEST_GUIDANCE_SIGNAL.txt').read_text().strip())
def read(path):
    return json.loads(path.read_text()) if path.exists() else {}
manifest = read(JOB/'LAUNCH_MANIFEST.json')
state = read(JOB/'STATUS.json')
print(datetime.datetime.now().astimezone().strftime('%d/%m %H:%M:%S %Z'))
print('GUIDANCE SIGNAL Q/Z | RTX 4070 | commit ' + manifest.get('git_commit', '')[:7])
print('Status:', state.get('status', 'STARTING'), '| stage:', state.get('stage', '-'))
print('Budget: 64 optimizer steps per arm; 4 selected TRAIN patients; seed 42')
print('Completed steps:', state.get('completed_optimizer_steps', {'Q': 0, 'Z': 0}))
if manifest.get('signals'):
    signals = read(Path(manifest['signals'])/'FROZEN.json')
    print('Reused prepared patients:', ', '.join(signals.get('patients', [])))
else:
    print('Prepared patients:', ', '.join(state.get('prepared_patients', [])) or 'none yet')
if state.get('status') == 'WAITING_FOR_GPU':
    print('GPU free:', state.get('gpu_free_mib'), '/ required', manifest['gpu_min_free_mib'], 'MiB')
    print('Unrelated GPU job is preserved; automatic start when enough VRAM is free.')
if len(sys.argv) > 1 and sys.argv[1] == 'losses':
    for arm in ('Q', 'Z'):
        trace = read(JOB/'pilot_Q_Z_64'/arm/'TRACE.json')
        print('\nARM', arm, '(trace published when arm completes)')
        for row in trace.get('steps', [])[-8:]:
            sem = row['semantic'][arm]
            print(f"{row['optimizer_step']:2d}/64 feat={row['feature_loss']:.5f} CE={sem['ce']:.5f} H={sem['entropy']:.5f} KL={sem['kl']:.5f}")
else:
    print('W&B:', state.get('wandb_url', 'initializing'))
    print('Train-arm Dice/IoU: not evaluated by the current pilot protocol')
    print('Stagewise VAL metrics refer to the original frozen checkpoint.')
    print('Log:', JOB/'launcher.log')
if state.get('error'):
    print('ERROR:', state['error'])
