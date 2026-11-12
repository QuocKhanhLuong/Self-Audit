"""Explicit one-time recovery for the approved observational-metrics amendment."""
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path('/root/Self-Audit')
BASE = Path('/root/mount_data/runs/q3d-resenc-pixdim-20261112')
PLAN = BASE / 'campaign-v2/PLAN.json'
STATE = PLAN.parent / 'RUN_STATE.json'
AUDIT = BASE / 'epoch-amendment'
sys.path.insert(0, str(ROOT))
from scripts import q3d_campaign as campaign
from self_audit.candidate_q3d.campaign_runtime import _verified_output, _receipt_is_synthetic
from self_audit.candidate_q3d.config import load_config
from self_audit.candidate_q3d import engine


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def main():
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if head != sys.argv[1] or sha(PLAN) != sys.argv[2]:
        raise RuntimeError('amended source/plan SHA mismatch')
    if (AUDIT / 'RESUME_STARTED.json').exists():
        raise RuntimeError('explicit recovery already started; no automatic retry')
    prepared = json.loads((AUDIT / 'PREPARED.json').read_text())
    if prepared['status'] != 'PREPARED_STOPPED' or prepared['plan_sha256'] != sha(PLAN):
        raise RuntimeError('missing prepared recovery identity')
    if sha(prepared['checkpoint_path']) != prepared['checkpoint_sha256']:
        raise RuntimeError('prepared root checkpoint changed')
    plan = campaign.validate_campaign_plan(PLAN)
    state = json.loads(STATE.read_text())
    before = json.loads((AUDIT / 'RUN_STATE.before.json').read_text())
    if state != before:
        raise RuntimeError('campaign state changed after stopping')
    job = next(j for j in plan['stage_jobs'] if j['id'] == prepared['job_id'])
    record = state['jobs'][job['id']]
    if job['stage'] != 'T' or record['status'] != 'RUNNING' or record['command'] != prepared['command']:
        raise RuntimeError('recovery no longer targets the declared T command')
    experiment = dict(next(e for e in plan['experiments'] if e['id'] == job['experiment_id']))
    experiment['output'] = job['output']
    expected = campaign._job_command(experiment, config_path=Path(job['config_path']),
        train_ref=job['train_manifest'], dev_ref=job['development_manifest'],
        device='cuda', synthetic=False, stage='T', stage_artifacts={})
    if expected != record['command']:
        raise RuntimeError('stored T command differs from the immutable plan')
    root_latest = Path(job['output']) / 'latest.pt'
    if Path(prepared['checkpoint_path']) != root_latest:
        raise RuntimeError('prepared checkpoint is not the exact job root')
    config = load_config(job['config_path'], real_run=True)
    identity = engine._data_identity_without_labels(engine._load_data(config, None))
    config_sha = engine._object_sha256(config)
    if config_sha != prepared['config_sha256'] or identity != prepared['data_identity']:
        raise RuntimeError('live config/cohort identity changed since preparation')
    engine._load_checkpoint(root_latest, config_sha256=config_sha, data_identity=identity)
    write(AUDIT / 'RESUME_STARTED.json', {**prepared, 'new_code_commit': head, 'status': 'RESUMING'})
    write(BASE / 'QUEUE_STATUS.json', {'status': 'RUNNING', 'setting': 'v1', 'k': 8,
        'through_stage': 'P', 'current_stage': 'T', 'code_commit': head,
        'plan_sha256': sha(PLAN), 'resume_update': prepared['resume_update'],
        'observational_metrics_amendment': True})
    message = 'EPOCH_AMENDMENT_T_RESUME from update=' + str(prepared['resume_update']) + ' code=' + head
    print(message, flush=True)
    with (BASE / 'launch.log').open('a') as handle:
        handle.write(message + '\n')
    result = subprocess.run(record['command'], cwd=ROOT, capture_output=True, text=True)
    (AUDIT / 'resume.stdout.txt').write_text(result.stdout)
    (AUDIT / 'resume.stderr.txt').write_text(result.stderr)
    if result.returncode:
        raise RuntimeError('resumed T failed; queue remains stopped; inspect resume.stderr.txt')
    receipt = json.loads(result.stdout)
    if (receipt.get('status') != 'COMPLETED' or _receipt_is_synthetic(receipt)
            or receipt.get('evidence_status') != 'TRAIN_EXECUTION'
            or receipt.get('runtime_synthetic') is not False
            or receipt.get('active_stages') != ['T'] or receipt.get('completed_stages') != ['T']
            or receipt.get('config_sha256') != prepared['config_sha256']
            or receipt.get('data_identity') != prepared['data_identity']
            or receipt.get('completed_updates') != 8000):
        raise RuntimeError('resumed T returned an incompatible completion receipt')
    if Path(receipt['checkpoint']) != root_latest:
        raise RuntimeError('completion receipt does not reference the exact job root')
    if sha(root_latest) != receipt['checkpoint_sha256']:
        raise RuntimeError('terminal checkpoint hash mismatch')
    final = engine._load_checkpoint(root_latest, config_sha256=config_sha, data_identity=identity)
    if (final.get('schema') != engine.STATE_SCHEMA or final.get('stage') is not None
            or final.get('stage_cursor') != 0 or final.get('completed_stages') != ['T']
            or final.get('completed_updates_total') != 8000
            or final.get('synthetic_gate_override_requested') is not False
            or final.get('synthetic_gate_override_used', False) is not False):
        raise RuntimeError('terminal root checkpoint is not a complete real T state')
    artifact = _verified_output(job)
    record.update({'status': 'COMPLETED', 'exit_code': 0, 'receipt': copy.deepcopy(receipt),
                   'stdout_tail': result.stdout[-4000:], 'artifact': artifact,
                   'observational_metrics_amendment': {'from_commit': prepared['old_code_commit'],
                       'to_commit': head, 'resume_update': prepared['resume_update'],
                       'checkpoint_sha256': prepared['checkpoint_sha256']}})
    write(STATE, state)
    write(AUDIT / 'RESUME_COMPLETED.json', {'status': 'COMPLETED', 'receipt': receipt, 'artifact': artifact})
    spec = importlib.util.spec_from_file_location('ordered_queue', BASE / 'run_ordered.py')
    queue = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(queue)
    queue.log('EPOCH_AMENDMENT_T_COMPLETED: continuing original queue with shared T artifact')
    queue.main()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        write(AUDIT / 'FAILED.json', {'status': 'FAILED', 'error': str(exc), 'traceback': traceback.format_exc()})
        write(BASE / 'QUEUE_STATUS.json', {'status': 'FAILED', 'error': str(exc),
                                          'recovery_audit': str(AUDIT)})
        raise
