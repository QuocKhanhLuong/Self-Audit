"""Download a provider-materialized original CINE and audit without mutation."""
import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def main():
    payload = json.loads(sys.stdin.read())
    pid, name, subset = payload['pid'], payload['title'], payload['subset']
    assert name == pid + '_4d.nii' and subset in ('training', 'testing')
    root = Path('/root/mount_data/audits/acdc-drive-original-20261112')
    path = root / subset / pid / name
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt = root / 'receipts' / (pid + '.json')
    receipt.parent.mkdir(parents=True, exist_ok=True)
    if receipt.exists():
        print(json.dumps({'pid': pid, 'status': 'ALREADY_AUDITED'}))
        return
    if not path.exists():
        partial = path.with_suffix('.nii.part')
        with urllib.request.urlopen(payload['download_url'], timeout=120) as src, partial.open('wb') as dst:
            for block in iter(lambda: src.read(8 * 1024 * 1024), b''):
                dst.write(block)
        assert partial.stat().st_size == int(payload['size']), 'Drive metadata size mismatch'
        partial.rename(path)
    assert path.stat().st_size == int(payload['size'])
    im = nib.load(str(path))
    array = np.asanyarray(im.dataobj)
    sha = digest(path)
    existing = Path('/root/mount_data/datasets/ACDC/raw') / subset / pid / name
    existing_sha = digest(existing) if existing.exists() else None
    pixdim = np.asarray(im.header.get_zooms()[:3], dtype=np.float64)
    lengths = np.linalg.norm(im.affine[:3, :3], axis=0)
    expected_bytes = int(im.dataobj.offset) + int(np.prod(im.shape)) * im.get_data_dtype().itemsize
    finite = bool(np.isfinite(array).all())
    row = {
        'patient_id': pid, 'subset': subset, 'file_id': payload['id'], 'source_url': payload['url'],
        'downloaded_path': str(path), 'downloaded_at_utc': datetime.now(timezone.utc).isoformat(),
        'bytes': path.stat().st_size, 'drive_metadata_bytes': int(payload['size']),
        'expected_nifti_bytes': expected_bytes, 'file_size_exact': path.stat().st_size == expected_bytes,
        'sha256': sha, 'existing_path': str(existing), 'existing_sha256': existing_sha,
        'identical_to_server': sha == existing_sha, 'shape_xyzt': list(im.shape),
        'dtype': str(im.get_data_dtype()), 'all_voxels_finite': finite,
        'pixdim_xyz_native': pixdim.tolist(), 'selected_affine_xyz': im.affine.tolist(),
        'selected_affine_axis_lengths': lengths.tolist(),
        'sform_code': int(im.header['sform_code']), 'qform_code': int(im.header['qform_code']),
        'spatial_unit': im.header.get_xyzt_units()[0],
        'affine_pixdim_disagree': not bool(np.allclose(lengths, pixdim, rtol=1e-5, atol=1e-6)),
        'pixdim_valid_nonunit': bool(np.isfinite(pixdim).all() and (pixdim > 0).all() and not np.allclose(pixdim, 1)),
        'evidence_status': 'REAL_DATA_INTEGRITY_AUDIT_NO_TRAINING',
        'mask_access': 'NOT_READ',
    }
    row['integrity_status'] = 'PASS' if finite and row['file_size_exact'] else 'FAIL'
    receipt.write_text(json.dumps(row, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: row[k] for k in ('patient_id', 'integrity_status', 'identical_to_server', 'affine_pixdim_disagree')}))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # Do not print temporary provider download credentials from exception URLs.
        print(json.dumps({'status': 'FAILED', 'exception_type': type(exc).__name__}))
        sys.exit(1)
