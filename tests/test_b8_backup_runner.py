import importlib.util
import io
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('b8_runner', ROOT / 'scripts/run_b8_reproduction.py')
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
