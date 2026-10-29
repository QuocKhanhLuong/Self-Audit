import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

root = Path('/workspace/compact-acdc-l8/20261028')
scalar = root / 'scalar_guard_full'
joint = root / 'joint_guard_full'
plain_report = json.loads((root / 'plain_kd_full/REPORT.json').read_text())
while not (scalar / 'REPORT.json').exists():
    time.sleep(10)

report = json.loads((scalar / 'REPORT.json').read_text())
checks = {
    'status': report.get('status') == 'COMPLETED',
    'steps': report.get('steps') == 10000,
    'initialization': report.get('initialization', {}).get('initial_state_sha256') == plain_report.get('initialization', {}).get('initial_state_sha256'),
    'teacher_cache': report.get('teacher_cache_manifest_sha256') == plain_report.get('teacher_cache_manifest_sha256'),
    'source_digest': report.get('source_digest') == plain_report.get('source_digest'),
    'label_budget': report.get('label_budget') == 8,
    'checkpoint_manifest': (scalar / 'final/MANIFEST.json').is_file() and hashlib.sha256((scalar / 'final/MANIFEST.json').read_bytes()).hexdigest() == report.get('final_manifest_sha256'),
}
receipt = {'schema': 'self_audit.compact_handoff.v1', 'from': 'scalar_guard', 'to': 'joint_guard', 'checks': checks, 'status': 'PASS' if all(checks.values()) else 'BLOCKED'}
(root / 'SCALAR_TO_JOINT_HANDOFF.json').write_text(json.dumps(receipt, indent=2) + '\n')
if not all(checks.values()):
    raise SystemExit('Scalar-to-joint handoff blocked: ' + str(checks))
if joint.exists():
    raise SystemExit('Joint output already exists; refusing overwrite')

subprocess.run([
    'tmux', 'respawn-window', '-k', '-t', 'ssh_tmux:compact-l8-log',
    'python /workspace/compact-acdc-l8/20261028/watch_compact_arm.py joint_guard',
], check=True)
cmd = [
    sys.executable, 'train.py', '--config', str(root / 'kd_configs/joint_guard.json'),
    '--data', '/workspace/Self-Audit/data/prepared/ACDC_L8_S42/train.json',
    '--data-sha256', 'e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4',
    '--out', str(joint), '--device', 'cuda', '--allow-full-run',
]
with (root / 'joint_guard_full.stdout.log').open('w') as stdout, (root / 'joint_guard_full.stderr.log').open('w') as stderr:
    result = subprocess.run(cmd, cwd='/workspace/Self-Audit-compact', stdout=stdout, stderr=stderr)
raise SystemExit(result.returncode)
