"""Software-only recovery fixtures: no patient data, GPU or server execution."""
import copy
import json
import os
from pathlib import Path
import subprocess

import pytest

from scripts import q3d_resume_queue as queue

REASON = 'User-authorized full-pipeline exploratory continuation 2026-11-12'


@pytest.fixture
def recovery(tmp_path, monkeypatch):
    base = tmp_path / 'run'
    base.mkdir()
    q = queue.ResumeQueue(base, 'source', 'plan', REASON, process_reader=lambda: '1 /sbin/init\n')
    output = base / 'q'
    output.mkdir()
    config = {'runtime': {'synthetic': False}, 'recipe': 'unchanged'}
    config_path = base / 'config.json'
    queue.atomic_json(config_path, config)
    job = {'id': 'shared.q', 'stage': 'Q', 'method': 'candidate_q3d', 'output': str(output),
           'config_path': str(config_path), 'experiment_id': 'v1k8', 'dependencies': [],
           'stage_artifact_inputs': {}, 'train_manifest': {}, 'development_manifest': {}}
    experiment = {'id': 'v1k8', 'setting': 'v1', 'k': 8, 'arm': 'v1',
                  'output': str(output), 'stage_job_ids': [job['id']]}
    q.plan = {'stage_jobs': [job], 'experiments': [experiment]}
    q.jobs = {job['id']: job}
    command = queue.campaign._job_command(experiment, config_path=config_path,
                  train_ref={}, dev_ref={}, device='cuda', synthetic=False, stage='Q', stage_artifacts={})
    state = {'schema': queue.campaign.STATE_SCHEMA, 'plan_sha256': 'plan',
             'jobs': {job['id']: {'status': 'FAILED', 'command': command, 'error': 'quality gate failed',
                                 'exit_code': None, 'input_manifest_sha256': {}}},
             'experiments': {'v1k8': {'status': 'NOT_RUN'}}}
    queue.atomic_json(q.state_path, state)
    checkpoint = {'schema': queue.engine.STATE_SCHEMA, 'config_sha256': queue.engine._object_sha256(config),
                  'data_identity': {'cohort': 'fixture'}, 'stage': 'Q', 'stage_cursor': 1000,
                  'completed_stages': ['T'], 'completed_updates_total': 1000,
                  'metrics': [{'update': 1000, 'gate': {'passed': False}}],
                  'gate': {'passed': False, 'status': 'NO_GO'},
                  'synthetic_gate_override_requested': False}
    queue.engine._atomic_torch_save(checkpoint, output / 'latest.pt')
    (output / 'metrics.jsonl').write_text(json.dumps(checkpoint['metrics'][0]) + '\n')
    monkeypatch.setattr(q, 'assert_pins', lambda: None)
    monkeypatch.setattr(queue, 'load_config', lambda *a, **k: config)
    monkeypatch.setattr(queue.engine, '_load_data', lambda *a: object())
    monkeypatch.setattr(queue.engine, '_data_identity_without_labels', lambda *a: {'cohort': 'fixture'})
    calls = []

    def finish(command, **kwargs):
        calls.append(list(command))
        latest = queue.engine._load_checkpoint(output / 'latest.pt',
                    config_sha256=checkpoint['config_sha256'], data_identity=checkpoint['data_identity'])
        latest.update(stage=None, stage_cursor=0, completed_stages=['T', job['stage']],
                      exploratory=True, exploratory_quality_waiver=q.policy)
        digest = queue.engine._atomic_torch_save(latest, output / 'latest.pt')
        receipt = {'status': 'COMPLETED', 'evidence_status': 'TRAIN_EXECUTION', 'runtime_synthetic': False,
                   'active_stages': [job['stage']], 'completed_stages': latest['completed_stages'],
                   'completed_updates': latest['completed_updates_total'],
                   'config_sha256': latest['config_sha256'], 'data_identity': latest['data_identity'],
                   'checkpoint': str(output / 'latest.pt'), 'checkpoint_sha256': digest,
                   'tokenizer_gate': latest['gate'], 'exploratory': True,
                   'exploratory_quality_waiver': q.policy}
        return subprocess.CompletedProcess(command, 0, json.dumps(receipt), '')

    q.child_runner = finish
    return q, job, checkpoint, calls


def test_exact_stored_command_resumes_and_archives_gate_failure(recovery):
    q, job, checkpoint, calls = recovery
    before = q.state()
    latest = Path(job['output']) / 'latest.pt'
    original_inode, original_sha = latest.stat().st_ino, queue.sha(latest)
    with latest.with_name('metrics.jsonl').open('a') as stream:
        stream.write('{"uncommitted": true}\npartial')
    q.recover(job)
    assert calls == [before['jobs'][job['id']]['command'] + [queue.WAIVER, REASON]]
    after = q.state()
    record = after['jobs'][job['id']]
    assert record['status'] == 'COMPLETED'
    assert record['receipt']['tokenizer_gate']['passed'] is False
    assert record['receipt']['tokenizer_gate']['status'] == 'NO_GO'
    assert record['exploratory'] is True
    assert after['experiments'] == before['experiments']
    audit = Path(record['recovery_history'][0]['audit_dir'])
    assert (audit / 'latest.pt').stat().st_ino == original_inode
    assert queue.sha(audit / 'latest.pt') == original_sha
    assert queue.read(audit / 'RUN_STATE.json') == before
    assert (audit / 'metrics.uncommitted.jsonl').read_text() == '{"uncommitted": true}\npartial'
    assert latest.with_name('metrics.jsonl').read_text() == json.dumps(checkpoint['metrics'][0]) + '\n'
    assert len(list((q.base / 'exploratory/attempts').glob('*/stdout.txt'))) == 1


@pytest.mark.parametrize('stage', ['T', 'Q', 'P', 'J'])
def test_every_stage_recovers_without_editing_config(recovery, stage):
    q, job, checkpoint, calls = recovery
    config_before = Path(job['config_path']).read_bytes()
    job['stage'] = stage
    state = q.state()
    command = state['jobs'][job['id']]['command']
    command[command.index('--q3d-stage') + 1] = stage
    queue.atomic_json(q.state_path, state)
    q.recover(job)
    assert calls[0][calls[0].index('--q3d-stage') + 1] == stage
    assert Path(job['config_path']).read_bytes() == config_before


@pytest.mark.parametrize('damage', ['hash', 'sidecar', 'schema', 'config', 'data', 'missing'])
def test_bad_checkpoint_blocks_before_child(recovery, damage):
    q, job, checkpoint, calls = recovery
    latest = Path(job['output']) / 'latest.pt'
    if damage == 'hash':
        latest.write_bytes(latest.read_bytes() + b'tamper')
    elif damage == 'sidecar':
        latest.with_suffix('.pt.sha256').unlink()
    elif damage == 'missing':
        latest.unlink()
    else:
        checkpoint[{'schema': 'schema', 'config': 'config_sha256', 'data': 'data_identity'}[damage]] = 'wrong'
        queue.engine._atomic_torch_save(checkpoint, latest)
    with pytest.raises((ValueError, FileNotFoundError)):
        q.recover(job)
    assert calls == []
    assert q.state()['jobs'][job['id']]['status'] == 'FAILED'


def test_divergent_metrics_block_recovery(recovery):
    q, job, _, calls = recovery
    (Path(job['output']) / 'metrics.jsonl').write_text('{"different": 1}\n')
    with pytest.raises(ValueError, match='metrics prefix'):
        q.recover(job)
    assert calls == []


def test_stored_running_valid_receipt_is_reused_without_replay(recovery):
    q, job, _, calls = recovery
    record = q.state()['jobs'][job['id']]
    result = q.runner(record['command'] + [queue.WAIVER, REASON])
    assert result.returncode == 0
    state = q.state()
    state['jobs'][job['id']]['status'] = 'RUNNING'
    queue.atomic_json(q.state_path, state)
    calls.clear()
    q.recover(job)
    assert calls == []
    record = q.state()['jobs'][job['id']]
    assert record['status'] == 'COMPLETED'
    assert record['recovery_history'][0]['reused_receipt'].endswith('stdout.txt')


def test_completed_job_and_experiment_skip(recovery, monkeypatch):
    q, job, _, calls = recovery
    q.recover(job)
    calls.clear()
    state = q.state()
    state['experiments']['v1k8']['status'] = 'COMPLETED'
    queue.atomic_json(q.state_path, state)
    monkeypatch.setattr(queue.campaign, 'execute_campaign', lambda *a, **kw: pytest.fail('completed experiment relaunched'))
    q.recover(job)
    q.train('v1', 8)
    assert calls == []


def test_stored_command_mutation_blocks(recovery):
    q, job, _, calls = recovery
    state = q.state()
    state['jobs'][job['id']]['command'].append('--synthetic')
    queue.atomic_json(q.state_path, state)
    with pytest.raises(ValueError, match='immutable planned command'):
        q.recover(job)
    assert calls == []


@pytest.mark.parametrize('field,value', [('runtime_synthetic', True), ('checkpoint_sha256', 'bad'),
                                       ('active_stages', ['J']), ('exploratory', False)])
def test_invalid_completion_cannot_mark_job_completed(recovery, field, value):
    q, job, _, calls = recovery
    finish = q.child_runner
    def bad_receipt(command, **kwargs):
        result = finish(command, **kwargs)
        receipt = json.loads(result.stdout)
        receipt[field] = value
        result.stdout = json.dumps(receipt)
        return result
    q.child_runner = bad_receipt
    with pytest.raises(ValueError):
        q.recover(job)
    assert q.state()['jobs'][job['id']]['status'] == 'FAILED'


def test_artifact_verified_before_record_completion(recovery, monkeypatch):
    q, job, _, calls = recovery
    job['artifact_output'] = {'manifest_path': '/fake/artifact.json'}
    def fail(*a, **kw):
        raise ValueError('artifact SHA mismatch')
    monkeypatch.setattr(queue.runtime, '_verified_output', fail)
    with pytest.raises(ValueError, match='artifact SHA mismatch'):
        q.recover(job)
    assert q.state()['jobs'][job['id']]['status'] == 'FAILED'


def test_order_and_explicit_waiver(tmp_path, monkeypatch):
    q = queue.ResumeQueue(tmp_path, 'source', 'plan', REASON, process_reader=lambda: '1 /sbin/init\n')
    plan = {'selection': {'synthetic': False}, 'stage_jobs': [],
            'experiments': [{'id': f'{s}{k}', 'setting': s, 'k': k, 'stage_job_ids': []} for s, k in queue.ORDER]}
    queue.atomic_json(tmp_path / 'profile_args.json', ['python', 'profile.py', '--report', '/old/T.json'])
    monkeypatch.setattr(q, 'assert_pins', lambda: None)
    monkeypatch.setattr(queue.campaign, 'validate_campaign_plan', lambda *a: plan)
    monkeypatch.setattr(q, 'validate_profile', lambda *a: {})
    calls = []
    monkeypatch.setattr(q, 'train', lambda s, k, through='J': calls.append((s, k, through)))
    monkeypatch.setattr(q, 'profile_j', lambda e: calls.append(('profile', 8, 'J')))
    q.run()
    assert calls == [('v1', 8, 'P'), ('profile', 8, 'J')] + [(s, k, 'J') for s, k in queue.ORDER]
    status = queue.read(tmp_path / 'QUEUE_STATUS.json')
    assert status['status'] == 'COMPLETED' and status['exploratory'] is True
    assert status['exploratory_quality_waiver']['reason'] == REASON


def test_campaign_gets_waiver_and_recovery_precedes_execution(recovery, monkeypatch):
    q, job, _, calls = recovery
    events = []
    original_recover = q.recover
    def recover(j):
        original_recover(j)
        events.append('recovered')
    def execute(*a, **kwargs):
        assert q.state()['jobs'][job['id']]['status'] == 'COMPLETED'
        assert kwargs['exploratory_quality_waiver'] == REASON
        assert kwargs['through_stage'] == 'J'
        events.append('executed')
        return {'status': 'COMPLETED'}
    monkeypatch.setattr(q, 'recover', recover)
    monkeypatch.setattr(queue.campaign, 'execute_campaign', execute)
    q.train('v1', 8)
    assert events == ['recovered', 'executed']


def test_process_absence_fails_closed(tmp_path):
    with pytest.raises(RuntimeError, match='still present'):
        queue.assert_children_absent(tmp_path, process_reader=lambda: f'12 python train.py --out {tmp_path}/q\n')
    with pytest.raises(RuntimeError, match='positively establish'):
        queue.assert_children_absent(tmp_path, process_reader=lambda: '')
    def failed_ps():
        raise subprocess.CalledProcessError(1, 'ps')
    with pytest.raises(subprocess.CalledProcessError):
        queue.assert_children_absent(tmp_path, process_reader=failed_ps)


def test_lock_refuses_second_writer_and_keeps_inode(tmp_path):
    with queue.queue_lock(tmp_path):
        inode = (tmp_path / 'exploratory-queue.lock').stat().st_ino
        with pytest.raises(RuntimeError, match='holds the lock'):
            with queue.queue_lock(tmp_path):
                pytest.fail('second writer acquired lock')
    with queue.queue_lock(tmp_path):
        assert (tmp_path / 'exploratory-queue.lock').stat().st_ino == inode


def test_tail_starts_at_current_offset_and_waits_for_complete_line(recovery):
    q, job, _, _ = recovery
    path = Path(job['output']) / 'metrics.jsonl'
    q.offsets[path] = path.stat().st_size
    with path.open('a') as stream:
        stream.write('{"RV": 0.8')
    q.poll_metrics()
    assert not (q.base / 'launch.log').exists()
    with path.open('a') as stream:
        stream.write(', "MYO": 0.5, "LV": 0.9}\n')
    q.poll_metrics()
    q.poll_metrics()
    log = (q.base / 'launch.log').read_text()
    assert log.count('METRIC') == 1 and '"MYO": 0.5' in log
    assert '"update": 1000' not in log


def test_reason_must_match_existing_waiver():
    with pytest.raises(ValueError, match='same explicit waiver'):
        queue.with_waiver(['python', queue.WAIVER, 'old'], REASON)
    with pytest.raises(ValueError, match='same explicit waiver'):
        queue.with_waiver(['python', queue.WAIVER], REASON)


def test_source_and_plan_pins_checked(tmp_path, monkeypatch):
    q = queue.ResumeQueue(tmp_path, 'expected', 'digest', REASON)
    monkeypatch.setattr(queue.subprocess, 'check_output', lambda *a, **kw: 'other\n')
    with pytest.raises(ValueError, match='HEAD or immutable plan'):
        q.assert_pins()


@pytest.mark.parametrize('script', ['run_ordered.py', 'resume_queue.py', 'q3d_resume_queue.py', 'q3d_resume_epoch_queue.py', 'q3d_campaign.py'])
def test_legacy_queue_writer_without_base_blocks(tmp_path, script):
    with pytest.raises(RuntimeError, match='queue process still present'):
        queue.assert_children_absent(tmp_path, process_reader=lambda: f'789 python /root/Self-Audit/{script} sha plan\n')
    queue.assert_children_absent(tmp_path, process_reader=lambda: f'{os.getpid()} python {script}\n1 /sbin/init\n')


def test_completed_joint_checkpoint_revalidated_before_skip(recovery):
    q, job, _, calls = recovery
    job['stage'] = 'J'
    state = q.state()
    command = state['jobs'][job['id']]['command']
    command[command.index('--q3d-stage') + 1] = 'J'
    queue.atomic_json(q.state_path, state)
    q.recover(job)
    state = q.state()
    state['experiments']['v1k8']['status'] = 'COMPLETED'
    queue.atomic_json(q.state_path, state)
    latest = Path(job['output']) / 'latest.pt'
    latest.write_bytes(latest.read_bytes() + b'tamper')
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        q.train('v1', 8)
    assert len(calls) == 1


def test_completed_artifact_pin_revalidated(recovery, monkeypatch):
    q, job, _, _ = recovery
    state = q.state()
    state['experiments']['v1k8']['status'] = 'COMPLETED'
    state['jobs'][job['id']].update(status='COMPLETED', artifact={'sha256': 'pinned'})
    queue.atomic_json(q.state_path, state)
    job['artifact_output'] = {'manifest_path': '/fake/manifest.json'}
    def changed(j, *, pinned_sha):
        assert pinned_sha == 'pinned'
        raise ValueError('shared artifact changed after production')
    monkeypatch.setattr(queue.runtime, '_verified_output', changed)
    with pytest.raises(ValueError, match='changed after production'):
        q.train('v1', 8)


def test_failed_profile_receipt_archived_and_retry_imports_pins(recovery, monkeypatch):
    q, job, _, _ = recovery
    experiment = q.plan['experiments'][0]
    experiment.update(config_path=job['config_path'], config_sha256=queue.sha(job['config_path']),
                      stage_artifact_producers={'T': 't', 'Q': 'q', 'P': 'p'})
    state = q.state()
    artifacts = {}
    for stage in 'TQP':
        jid = stage.lower()
        q.jobs[jid] = {'id': jid, 'stage': stage}
        state['jobs'][jid] = {'status': 'COMPLETED', 'artifact': {'sha256': jid * 64}}
        artifacts[jid] = {'path': str(q.base / (stage + '.json')), 'sha256': jid * 64}
    queue.atomic_json(q.state_path, state)
    monkeypatch.setattr(queue.runtime, '_verified_output', lambda j, **kw: artifacts[j['id']])
    target = q.base / 'exploratory/GPU_PROFILE_J.json'
    failed = {'status': 'FAILED', 'exploratory': True, 'exploratory_quality_waiver': q.policy}
    queue.atomic_json(target, failed)
    old_bytes = target.read_bytes()
    queue.atomic_json(q.base / 'profile_args.json', ['python', 'scripts/q3d_profile_gpu.py',
                     '--stage', 'T', '--config', job['config_path'], '--report', '/old/T.json'])
    calls = []
    def profile(command, **kwargs):
        calls.append(command)
        assert command[command.index('--report') + 1] == str(target)
        assert command[-2:] == [queue.WAIVER, REASON]
        assert all(stage + '=' + artifacts[stage.lower()]['sha256'] in command for stage in 'TQP')
        queue.atomic_json(target, {'status': 'COMPLETED', 'evidence_status': 'GPU_PROFILE',
                    'execution_mode': 'REAL_CUDA_PROFILE', 'config_sha256': experiment['config_sha256'],
                    'stage_results': {'J': {'measured_updates': 20}}, 'exploratory': True,
                    'exploratory_quality_waiver': q.policy,
                    'stage_artifact_inputs': {s: {'manifest_path': artifacts[s.lower()]['path'],
                       'manifest_sha256': artifacts[s.lower()]['sha256']} for s in 'TQP'}})
        return subprocess.CompletedProcess(command, 0, '{}', '')
    q.child_runner = profile
    q.profile_j(experiment)
    assert len(calls) == 1
    history = list((q.base / 'exploratory/profile-history').glob('*/GPU_PROFILE_J.json'))
    assert len(history) == 1 and history[0].read_bytes() == old_bytes
    q.profile_j(experiment)
    assert len(calls) == 1


def test_plan_hash_mismatch_blocks(tmp_path, monkeypatch):
    q = queue.ResumeQueue(tmp_path, 'source', 'wrong', REASON)
    queue.atomic_json(q.plan_path, {})
    monkeypatch.setattr(queue.subprocess, 'check_output', lambda *a, **kw: 'source\n')
    with pytest.raises(ValueError, match='immutable plan SHA mismatch'):
        q.assert_pins()
