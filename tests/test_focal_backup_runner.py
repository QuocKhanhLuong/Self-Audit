import importlib.util
import io
import json
import copy
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('b8_runner', ROOT / 'scripts/run_b8_focal_adaptive.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_upload_exit_zero_does_not_certify_remote_bytes(tmp_path, monkeypatch):
    payload = tmp_path / 'weights.pt'
    payload.write_bytes(b'actual weights')
    backup = runner.DriveBackup('KhanhDrive:new', 'parent', tmp_path / 'log', attempts=1)
    monkeypatch.setattr(runner, 'command', lambda *args: None)
    monkeypatch.setattr(backup, 'readback', lambda name: (14, 'wrong'))
    with pytest.raises(RuntimeError, match='UPLOAD_PENDING'):
        backup.upload_verified(payload, 'weights.pt')
    assert payload.read_bytes() == b'actual weights'


def test_verified_backup_records_actual_id_size_and_sha(tmp_path, monkeypatch):
    payload = tmp_path / 'weights.pt'
    payload.write_bytes(b'actual weights')
    backup = runner.DriveBackup('KhanhDrive:new', 'parent', tmp_path / 'log', attempts=1)
    monkeypatch.setattr(runner, 'command', lambda *args: None)
    monkeypatch.setattr(backup, 'readback', lambda name: (payload.stat().st_size, runner.sha256(payload)))
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *a, **kw: json.dumps({'Size': payload.stat().st_size, 'ID': 'observed-id'}))
    result = backup.upload_verified(payload, 'weights.pt')
    assert result['drive_id'] == 'observed-id'
    assert result['sha256'] == runner.sha256(payload)
    assert result['readback'] == 'SHA256_MATCH'


def test_readback_stream_checks_process_exit(tmp_path, monkeypatch):
    class FailedStream:
        stdout = io.BytesIO(b'partial payload')
        def wait(self):
            return 1
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda *a, **kw: FailedStream())
    backup = runner.DriveBackup('KhanhDrive:new', 'parent', tmp_path / 'log', attempts=1)
    with pytest.raises(RuntimeError, match='readback failed'):
        backup.readback('weights.pt')


def test_write_probe_needs_bytes_but_not_metadata_query(tmp_path, monkeypatch):
    payload = tmp_path / 'probe.json'; payload.write_bytes(b'probe')
    backup = runner.DriveBackup('KhanhDrive:new', 'parent', tmp_path / 'log', attempts=1)
    monkeypatch.setattr(runner, 'command', lambda *a: None)
    monkeypatch.setattr(backup, 'readback', lambda n: (5, runner.sha256(payload)))
    def forbidden(*a, **kw):
        raise AssertionError('probe does not need a metadata request')
    monkeypatch.setattr(runner.subprocess, 'check_output', forbidden)
    proof = backup.upload_verified(payload, 'probe.json', collect_id=False)
    assert proof['readback'] == 'SHA256_MATCH' and proof['drive_id'] is None


def test_flat_upload_attempts_once_then_only_verifies(tmp_path, monkeypatch):
    payload = tmp_path / 'weights.pt'; payload.write_bytes(b'actual weights')
    backup = runner.DriveBackup('KhanhDrive:', 'fresh-parent', tmp_path / 'log', attempts=3, flat_new_folder=True)
    uploads = []
    def upload(argv, log):
        assert argv[:2] == ['rclone', 'copy'] and '--no-check-dest' in argv
        staged = Path(argv[2]) / 'raw_student__raw_student.pt'
        assert staged.read_bytes() == payload.read_bytes()
        uploads.append(argv)
    monkeypatch.setattr(runner, 'command', upload)
    monkeypatch.setattr(runner.time, 'sleep', lambda n: None)
    def quota(name):
        raise RuntimeError('readback metadata quota')
    monkeypatch.setattr(backup, 'readback', quota)
    for _ in range(2):
        with pytest.raises(RuntimeError, match='UPLOAD_PENDING'):
            backup.upload_verified(payload, 'raw_student/raw_student.pt')
    assert len(uploads) == 1
    intent = json.loads((tmp_path / 'UPLOAD_PROGRESS.json').read_text())['files']['raw_student/raw_student.pt']
    assert intent['upload_state'] == 'UPLOAD_COMMAND_SUCCEEDED'
    assert intent['sha256'] == runner.sha256(payload)


def test_uncertain_flat_upload_never_blindly_creates_duplicate(tmp_path, monkeypatch):
    payload = tmp_path / 'weights.pt'; payload.write_bytes(b'actual weights')
    backup = runner.DriveBackup('KhanhDrive:', 'fresh-parent', tmp_path / 'log', attempts=1, flat_new_folder=True)
    calls = []
    def crash(*args):
        calls.append(args)
        raise RuntimeError('connection died after possible server commit')
    monkeypatch.setattr(runner, 'command', crash)
    monkeypatch.setattr(backup, 'readback', lambda n: (_ for _ in ()).throw(RuntimeError('quota')))
    for _ in range(2):
        with pytest.raises(RuntimeError, match='UPLOAD_PENDING'):
            backup.upload_verified(payload, 'full_checkpoint/weights.pt')
    assert len(calls) == 1
    with pytest.raises(ValueError, match='different bytes'):
        backup.upload_new_flat_file(payload, 'full_checkpoint/weights.pt', 100, 'wrong')
    with pytest.raises(ValueError, match='unsafe'):
        backup.remote_name('ambiguous__path/weights.pt')


def test_plan_hash_rejection_precedes_run_creation(tmp_path):
    plan = tmp_path / 'plan.json'
    plan.write_text('{}')
    with pytest.raises(ValueError, match='plan hash mismatch'):
        runner.run(plan, '0' * 64)
    assert list(tmp_path.iterdir()) == [plan]


def test_atomic_json_never_replaces_on_invalid_data(tmp_path):
    path = tmp_path / 'STATUS.json'
    runner.atomic_json(path, {'status': 'READY'})
    before = path.read_bytes()
    with pytest.raises(ValueError):
        runner.atomic_json(path, {'invalid': float('nan')})
    assert path.read_bytes() == before


def test_metadata_failure_preserves_readback_evidence_without_reupload(tmp_path, monkeypatch):
    payload = tmp_path / 'weights.pt'; payload.write_bytes(b'actual weights')
    backup = runner.DriveBackup('KhanhDrive:new', 'parent', tmp_path / 'log', attempts=3)
    uploads = []
    monkeypatch.setattr(runner, 'command', lambda *a: uploads.append(a))
    monkeypatch.setattr(backup, 'readback', lambda n: (payload.stat().st_size, runner.sha256(payload)))
    monkeypatch.setattr(runner.time, 'sleep', lambda n: None)
    def quota(*a, **kw):
        raise subprocess.CalledProcessError(1, a[0])
    monkeypatch.setattr(runner.subprocess, 'check_output', quota)
    with pytest.raises(RuntimeError, match='UPLOAD_PENDING'):
        backup.upload_verified(payload, 'weights.pt')
    assert len(uploads) == 1
    proof = json.loads((tmp_path / 'UPLOAD_PROGRESS.json').read_text())['files']['weights.pt']
    assert proof['sha256'] == runner.sha256(payload)
    assert proof['readback'] == 'SHA256_MATCH'
    assert proof['drive_id'] is None


@pytest.mark.parametrize('mutation', [None, 'batch_size', 'steps', 'seed', 'pseudo_threshold', 'focal_gamma', 'scheduler'])
def test_variant_recipe_only_allows_two_declared_changes(mutation):
    actual = json.loads((ROOT / 'configs/b8_focal_adaptive.json').read_text())
    original = copy.deepcopy(actual)
    original.pop('supervised_loss'); original.pop('lr_scheduler')
    if mutation in ('batch_size', 'steps', 'seed', 'pseudo_threshold'):
        actual[mutation] += 1
    elif mutation == 'focal_gamma':
        actual['supervised_loss']['gamma'] = 3
    elif mutation == 'scheduler':
        actual['lr_scheduler']['factor'] = .1
    if mutation:
        with pytest.raises(ValueError):
            runner.validate_recipe(original, actual)
    else:
        runner.validate_recipe(original, actual)


def test_failed_weight_upload_blocks_evaluation_and_preserves_run(tmp_path, monkeypatch):
    root = tmp_path / 'run'
    final = root / 'training/final'
    final.mkdir(parents=True)
    (final / 'weights.pt').write_bytes(b'full training payload')
    (final / 'MANIFEST.json').write_text(json.dumps({'step': 10000, 'payload_sha256': runner.sha256(final / 'weights.pt')}))
    (root / 'training/REPORT.json').write_text(json.dumps({'status': 'COMPLETED', 'steps': 10000}))
    config = tmp_path / 'config.json'
    config.write_text('{}')
    data = tmp_path / 'data.json'
    data.write_text('{}')
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps({'run_root': str(root), 'drive_remote': 'KhanhDrive:new',
                               'drive_parent_id': 'parent', 'source': str(tmp_path),
                               'python': 'python', 'config': str(config), 'data': str(data),
                               'backup_files': []}))
    monkeypatch.setattr(runner, 'validate_plan', lambda value: None)
    commands = []
    def export_only(argv, *args):
        commands.append(argv)
        assert '--export-source' in argv
        (root / 'raw_student').mkdir(exist_ok=True)
        (root / 'raw_student/raw_student.pt').write_bytes(b'student payload')
        (root / 'raw_student/RAW_STUDENT.json').write_text('{}')
    monkeypatch.setattr(runner, 'command', export_only)
    attempted = []
    def upload(self, path, name, **kwargs):
        attempted.append(name)
        if name == 'full_checkpoint/weights.pt':
            raise RuntimeError('UPLOAD_PENDING: disconnected')
        return {'readback': 'SHA256_MATCH'}
    monkeypatch.setattr(runner.DriveBackup, 'upload_verified', upload)
    for _ in range(2):
        with pytest.raises(RuntimeError, match='UPLOAD_PENDING'):
            runner.run(plan, runner.sha256(plan), resume=True)
        assert (final / 'weights.pt').read_bytes() == b'full training payload'
        assert json.loads((root / 'STATUS.json').read_text())['status'] == 'UPLOAD_PENDING'
    assert len(commands) == 2
    assert attempted.count('raw_student/raw_student.pt') == 2
    assert (root / 'PARTIAL_WEIGHT_BACKUP.json').exists()
    assert not (root / 'TRAIN_STARTED.json').exists()
