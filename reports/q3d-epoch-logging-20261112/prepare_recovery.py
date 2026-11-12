"""Stop only the declared Q3D job at its next durable root checkpoint."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/root/Self-Audit')
BASE = Path('/root/mount_data/runs/q3d-resenc-pixdim-20261112')
PLAN = BASE / 'campaign-v2/PLAN.json'
STATE = PLAN.parent / 'RUN_STATE.json'
AUDIT = BASE / 'epoch-amendment'
JOB = 'shared.acdc_a_k8_seed42.teacher.t'
sys.path.insert(0, str(ROOT))
from scripts.q3d_campaign import validate_campaign_plan
from self_audit.candidate_q3d.config import load_config
from self_audit.candidate_q3d import engine


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if head != sys.argv[1] or sha(PLAN) != sys.argv[2]:
        raise RuntimeError('source/plan identity changed before recovery')
    plan = validate_campaign_plan(PLAN)
    job = next(j for j in plan['stage_jobs'] if j['id'] == JOB)
    output = Path(job['output'])
    checkpoint = output / 'latest.pt'
    sidecar = output / 'latest.pt.sha256'
    if AUDIT.exists():
        raise RuntimeError('amendment audit already exists; no implicit retry')
    before = sidecar.read_text()
    print('Waiting for next durable root checkpoint before stopping the declared job.', flush=True)
    deadline = time.monotonic() + 1200
    while sidecar.read_text() == before:
        if time.monotonic() > deadline:
            raise TimeoutError('no new checkpoint; live job was not stopped')
        state = json.loads(STATE.read_text())
        running = [name for name, row in state['jobs'].items() if row['status'] == 'RUNNING']
        if running != [JOB]:
            raise RuntimeError('active stage changed; live jobs were not stopped')
        time.sleep(0.25)
    # Supervisor is configured stopasgroup/killasgroup; verify no descendant remains.
    pid = int(subprocess.check_output(['supervisorctl', 'pid', 'q3d-campaign'], text=True).strip())
    pairs = [tuple(map(int, line.split())) for line in subprocess.check_output(
        ['ps', '-eo', 'pid=,ppid='], text=True).splitlines()]
    owned = {pid}
    while True:
        found = {child for child, parent in pairs if parent in owned}
        if found.issubset(owned):
            break
        owned |= found
    subprocess.run(['supervisorctl', 'stop', 'q3d-campaign'], check=True)
    live = {int(line.split()[0]): line.split()[1] for line in subprocess.check_output(
        ['ps', '-eo', 'pid=,stat='], text=True).splitlines()}
    if any(p in live and not live[p].startswith('Z') for p in owned):
        raise RuntimeError('a declared writer is still alive; no files modified')
    AUDIT.mkdir()
    shutil.copy2(STATE, AUDIT / 'RUN_STATE.before.json')
    shutil.copy2(output / 'metrics.jsonl', AUDIT / 'metrics.before.jsonl')
    for prefix, path in [('root', checkpoint), ('stage_T', output / 'stages/t/latest.pt')]:
        if path.exists():
            os.link(path, AUDIT / (prefix + '.pt'))
            shutil.copy2(path.with_suffix('.pt.sha256'), AUDIT / (prefix + '.pt.sha256'))
    config = load_config(job['config_path'], real_run=True)
    data = engine._load_data(config, None)
    identity = engine._data_identity_without_labels(data)
    saved = engine._load_checkpoint(checkpoint, config_sha256=engine._object_sha256(config),
                                    data_identity=identity)
    cursor = saved['stage_cursor']
    if (saved['stage'] != 'T' or type(cursor) is not int or cursor % 400 or
            not 0 < cursor < 8000 or saved['completed_stages']):
        raise RuntimeError('root checkpoint is not the expected incomplete T boundary')
    raw = (AUDIT / 'metrics.before.jsonl').read_bytes()
    lines = raw.splitlines(keepends=True)
    rows = []
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
    expected = saved['metrics']
    if rows[:len(expected)] != expected:
        raise RuntimeError('metrics are not an exact checkpoint prefix; no rewind performed')
    suffix = b''.join(lines[len(expected):])
    (AUDIT / 'metrics.replayed_suffix.jsonl').write_bytes(suffix)
    restored = ''.join(json.dumps(row, sort_keys=True) + '\n' for row in expected)
    temp = output / 'metrics.epoch-recovery.tmp'
    temp.write_text(restored)
    temp.replace(output / 'metrics.jsonl')
    receipt = {
        'status': 'PREPARED_STOPPED', 'old_code_commit': head, 'plan_sha256': sha(PLAN),
        'job_id': JOB, 'checkpoint_path': str(checkpoint), 'checkpoint_sha256': sha(checkpoint),
        'config_sha256': saved['config_sha256'], 'data_identity': identity,
        'resume_update': cursor, 'old_last_logged_update': max(r.get('update', 0) for r in rows),
        'replayed_metric_records': len(rows) - len(expected),
        'metric_archive_sha256': sha(AUDIT / 'metrics.before.jsonl'),
        'command': json.loads(STATE.read_text())['jobs'][JOB]['command'],
    }
    (AUDIT / 'PREPARED.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
