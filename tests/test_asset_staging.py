import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from self_audit import asset_staging as s


def test_credentials_are_temporary_private_and_removed_on_failure():
    name = None
    with pytest.raises(RuntimeError):
        with s.rclone_config({"RCLONE_CONFIG_TEXT": "[drive]\ntype = drive\ntoken = private-token\n",
                              "RCLONE_REMOTE_NAME": "drive:"}) as (name, remote):
            assert remote == "drive"
            assert Path(name).parent == Path('/tmp')
            assert os.stat(name).st_mode & 0o777 == 0o600
            assert Path(name).read_text().endswith("private-token\n")
            raise RuntimeError("test interruption")
    assert name and not Path(name).exists()


@pytest.mark.parametrize('value', ['', '.', '/assets/x', '../x', 'drive:x', 'x\\y', 'x\ny'])
def test_reject_path_escape(value):
    with pytest.raises(ValueError):
        s.relative_path(value)


def test_wrong_remote_section_fails_without_secret_echo():
    with pytest.raises(ValueError) as exc:
        with s.rclone_config({'RCLONE_CONFIG_TEXT': '[real]\ntoken = VERY_PRIVATE',
                              'RCLONE_REMOTE_NAME': 'wrong'}):
            pass
    assert 'VERY_PRIVATE' not in str(exc.value)


def test_rclone_error_redacted_and_secret_not_in_child_env(monkeypatch):
    monkeypatch.setenv('RCLONE_CONFIG_TEXT', 'private-config')
    def run(argv, **kw):
        assert 'RCLONE_CONFIG_TEXT' not in kw['env']
        assert argv[:3] == ['rclone', '--config', '/tmp/private.conf']
        return subprocess.CompletedProcess(argv, 1, b'', b'Bearer secret-value: invalid_grant')
    monkeypatch.setattr(s.subprocess, 'run', run)
    with pytest.raises(RuntimeError) as exc:
        s.Rclone('/tmp/private.conf', 'drive').listing()
    assert 'OAUTH_INVALID_GRANT' in str(exc.value)
    assert 'secret-value' not in str(exc.value)


def test_whole_drive_recursive_scan_forbidden():
    with pytest.raises(ValueError):
        s.Rclone('unused', 'drive').listing('', recursive=True)


class FakeDrive:
    def __init__(self, data):
        self.data = data
        self.calls = []
    def listing(self, path):
        return [{'Size': len(self.data), 'IsDir': False}]
    def copy_file(self, source, destination):
        self.calls.append(source)
        Path(destination).write_bytes(self.data)
    def source(self, path):
        return 'observed:' + path


def test_metadata_hashes_exact_bytes_and_preserves_conflicts(tmp_path):
    data = b'{ "original_bytes": true }\n'
    drive = FakeDrive(data)
    receipt = s.initial_receipt()
    entry = {'source': 'observed/manifest.json', 'target': 'manifest.json',
             'sha256': hashlib.sha256(data).hexdigest()}
    s.copy_metadata(tmp_path, drive, [entry], receipt)
    assert (tmp_path/'manifest.json').read_bytes() == data
    assert not s.verify_files(tmp_path, receipt)
    (tmp_path/'manifest.json').write_bytes(b'preserve')
    with pytest.raises(ValueError):
        s.copy_metadata(tmp_path, drive, [entry], receipt)
    assert (tmp_path/'manifest.json').read_bytes() == b'preserve'
    assert s.verify_files(tmp_path, receipt) == ['manifest.json']


def test_wrong_metadata_hash_is_not_committed(tmp_path):
    receipt = s.initial_receipt()
    with pytest.raises(ValueError):
        s.copy_metadata(tmp_path, FakeDrive(b'wrong'), [{'source': 'x.json', 'target': 'x.json',
                          'sha256': 'a'*64}], receipt)
    assert not (tmp_path/'x.json').exists()
    assert not list(tmp_path.glob('*.partial'))
    assert not receipt['files']


def test_payload_requires_exact_role_manifest_and_hash(tmp_path):
    drive = FakeDrive(b'weights')
    receipt = s.initial_receipt()
    with pytest.raises(ValueError):
        s.copy_payload(tmp_path, drive, {'asset': 'student_weights', 'sha256': 'b'*64}, receipt)
    with pytest.raises(ValueError):
        s.copy_payload(tmp_path, drive, {'asset': 'student_weights',
                                        'sha256': s.PINS['student_weights']}, receipt)
    assert not drive.calls


def test_payload_full_byte_hash_and_atomic_failure(tmp_path, monkeypatch):
    weights = b'fixture weights, not a real checkpoint'
    expected = hashlib.sha256(weights).hexdigest()
    meta = json.dumps({'payload_sha256': expected}).encode()
    (tmp_path/'MANIFEST.json').write_bytes(meta)
    monkeypatch.setitem(s.PINS, 'student_weights', expected)
    monkeypatch.setitem(s.PINS, 'student_manifest', hashlib.sha256(meta).hexdigest())
    receipt = s.initial_receipt()
    receipt['assets']['student_manifest'] = {'path': 'MANIFEST.json'}
    request = {'asset': 'student_weights', 'sha256': expected,
               'source': 'known/weights.pt', 'target': 'export/weights.pt'}
    with pytest.raises(ValueError, match='SHA256'):
        s.copy_payload(tmp_path, FakeDrive(b'wrong bytes'), request, receipt)
    assert not (tmp_path/'export/weights.pt').exists()
    assert not list(tmp_path.rglob('*.partial'))
    s.copy_payload(tmp_path, FakeDrive(weights), request, receipt)
    assert (tmp_path/'export/weights.pt').read_bytes() == weights
    assert not s.verify_files(tmp_path, receipt)


def test_git_metadata_preserves_manifest_bytes(tmp_path, monkeypatch):
    import io
    raw = b'{ "payload_sha256": "fixture" }\n'
    monkeypatch.setitem(s.PINS, 'teacher_manifest', hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(s.urllib.request, 'urlopen', lambda *a, **kw: io.BytesIO(raw))
    r = s.initial_receipt()
    s.stage_git_metadata(tmp_path, [{'asset': 'teacher_manifest', 'source': 'known/MANIFEST.json',
                                    'target': 'teacher/MANIFEST.json'}], r)
    assert (tmp_path/'teacher/MANIFEST.json').read_bytes() == raw
    assert not s.verify_files(tmp_path, r)


def test_runtime_closure_checks_all_native_source_bytes(tmp_path):
    (tmp_path/'self_audit').mkdir()
    (tmp_path/'self_audit/a.py').write_text('a = 1\n')
    (tmp_path/'train.py').write_text('pass\n')
    expected = hashlib.sha256(('a.py:' + s.sha256(tmp_path/'self_audit/a.py') + '\ntrain.py:' +
                              s.sha256(tmp_path/'train.py')+'\n').encode()).hexdigest()
    assert s.source_digest(tmp_path) == expected
    (tmp_path/'self_audit/extra.py').write_text('pass\n')
    assert s.source_digest(tmp_path) != expected


def test_receipt_never_ready_from_manifests_alone():
    r = s.initial_receipt()
    r['assets'] = {k: {'sha256': v} for k,v in s.PINS.items()}
    assert s.refresh_status(r)['status'] == 'BLOCKED'
    assert r['missing'] == ['manifest_referenced_ACDC_TRAIN_VAL18_prepared_files']


def test_modal_script_gpu_secret_scope_and_persistence():
    import ast
    tree = ast.parse(Path('scripts/modal_stage_assets.py').read_text())
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name in ('stage_assets', 'verify_assets'):
        decorator = functions[name].decorator_list[0]
        kwargs = {kw.arg: kw.value for kw in decorator.keywords}
        assert 'gpu' not in kwargs
        assert ('secrets' in kwargs) == (name == 'stage_assets')
    assert ast.literal_eval({kw.arg: kw.value for kw in functions['verify_assets'].decorator_list[0].keywords}['single_use_containers'])
    assert 'MODAL_TASK_ID' in ast.unparse(functions['verify_assets'])
    assert 'volume.commit()' in ast.unparse(functions['stage_assets'])
