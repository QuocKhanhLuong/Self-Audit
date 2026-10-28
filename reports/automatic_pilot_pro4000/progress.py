from pathlib import Path
import json,subprocess,datetime
ROOT=Path('/workspace/Self-Audit');R=ROOT/'reports/automatic_pilot_pro4000';base=ROOT/'runs/pro4000';arms=['scalar_brier','joint_brier','scalar_dice','joint_dice','frozen','support_only','plain_head_u','extra_supervised','smaller_lr','full_ema']
def status(regime,arm):
 p=base/regime/arm
 if (p/'REPORT.json').exists():
  x=json.loads((p/'REPORT.json').read_text());return f"DONE {x['steps']:5d} {x['phase_cost_seconds']:6.1f}s"
 if (p/'FAILURE.json').exists():return 'FAILED - inspect FAILURE.json'
 if (p/'steps.jsonl').exists():
  with (p/'steps.jsonl').open('rb') as f:f.seek(max(0,(p/'steps.jsonl').stat().st_size-100000));lines=f.read().splitlines()
  if lines:
   x=json.loads(lines[-1]);return f"RUN {x['step']:6d} {x['step_seconds']:6.2f}s/it"
 if p.exists():return 'INITIALIZING'
 return 'PENDING'
print(datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),' | B128 fresh trajectories; full_ema microbatch 40')
print(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu,power.draw','--format=csv,noheader'],text=True).strip())
print('RAM cgroup GiB:',round(int(Path('/sys/fs/cgroup/memory.current').read_text())/2**30,1),'/120.7')
print(f"{'Arm':19} {'Equal update (128)':27} Equal time (600s)")
for a in arms:print(f'{a:19} {status("equal_update_B128_M40",a):27} {status("equal_time_B128_M40",a)}')
for regime in ['equal_update_B128_M40','equal_time_B128_M40']:
 p=R/f'evaluation_progress_{regime}.json'
 if p.exists():x=json.loads(p.read_text());print('Evaluation',regime,len(x['completed_arms']),'arms completed')
print('Raw logs: reports/automatic_pilot_pro4000/optimized128-full-training-live.log')
