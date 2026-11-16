import hashlib
import zipfile
import io
import subprocess
from unittest.mock import patch

import pytest

from scripts.pg_download_dataset import safe_relative, validate_inventory, verify_file, direct_file, range_download


def record(path):
    data = path.read_bytes()
    return {'Path': path.name, 'Size': len(data), 'IsDir': False,
            'Hashes': {'md5': hashlib.md5(data).hexdigest(), 'sha256': hashlib.sha256(data).hexdigest()}}


def test_source_hash_detects_corruption(tmp_path):
    p = tmp_path / 'a.nii'
    p.write_bytes(b'original')
    source = record(p)
    validate_inventory([source])
    assert verify_file(p, source)['sha256'] == source['Hashes']['sha256']
    p.write_bytes(b'changed!')
    with pytest.raises(ValueError, match='MD5 mismatch'):
        verify_file(p, source)


def test_archive_and_path_guard(tmp_path):
    for name in ['../escape', '/absolute', 'a/../../b', 'a\\b']:
        with pytest.raises(ValueError):
            safe_relative(name)
    archive = tmp_path / 'bad.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('../escape', 'payload')
    with pytest.raises(ValueError, match='unsafe source path'):
        verify_file(archive, record(archive))


def test_inventory_rejects_missing_hash_and_duplicates():
    row = {'Path': 'a', 'Size': 1, 'IsDir': False, 'Hashes': {'md5': 'a' * 32}}
    with pytest.raises(ValueError, match='duplicate'):
        validate_inventory([row, row])
    with pytest.raises(ValueError, match='missing source MD5'):
        validate_inventory([{**row, 'Hashes': {}}])


def test_direct_transport_seals_only_verified_bytes(tmp_path):
    source = tmp_path / 'source'
    source.write_bytes(b'patient-volume')
    row = {**record(source), 'Path': 'patient/image.nii', 'ID': 'valid-id'}
    response = io.BytesIO(source.read_bytes())
    response.headers = {'Content-Type': 'application/octet-stream'}
    with patch('urllib.request.urlopen', return_value=response):
        direct_file(tmp_path / 'raw', row, attempts=1)
    output = tmp_path / 'raw/patient/image.nii'
    assert output.read_bytes() == source.read_bytes()
    with patch('urllib.request.urlopen', side_effect=AssertionError('must reuse verified data')):
        direct_file(tmp_path / 'raw', row)


def test_direct_transport_never_promotes_html_or_wrong_content(tmp_path):
    row = {'Path': 'patient/image.nii', 'Size': 4, 'ID': 'valid-id', 'Hashes': {'md5': hashlib.md5(b'good').hexdigest()}}
    for content, mime in ((b'html', 'text/html'), (b'fail', 'application/octet-stream')):
        response = io.BytesIO(content)
        response.headers = {'Content-Type': mime}
        with patch('urllib.request.urlopen', return_value=response), pytest.raises(ValueError):
            direct_file(tmp_path, row, attempts=1)
        assert not (tmp_path / row['Path']).exists()


def test_interrupted_rclone_bytes_preserved_before_verified_replacement(tmp_path):
    p = tmp_path / 'image.nii'
    p.write_bytes(b'good')
    row = {**record(p), 'ID': 'valid-id'}
    p.write_bytes(b'bad!')
    response = io.BytesIO(b'good')
    response.headers = {'Content-Type': 'application/octet-stream'}
    with patch('urllib.request.urlopen', return_value=response):
        direct_file(tmp_path, row, attempts=1)
    assert p.read_bytes() == b'good'
    saved, = tmp_path.glob('image.nii.interrupted-*')
    assert saved.read_bytes() == b'bad!'


def test_range_download_rejects_server_ignoring_range(tmp_path):
    def response(argv, **kwargs):
        from pathlib import Path
        Path(argv[argv.index('--output') + 1]).write_bytes(b'good')
        return subprocess.CompletedProcess(argv, 0, stdout='200')
    with patch('subprocess.run', side_effect=response), pytest.raises(ValueError, match='range response'):
        range_download('https://example.test/file?id=x', tmp_path/'partial', 4)
