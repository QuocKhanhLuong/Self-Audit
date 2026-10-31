#!/usr/bin/env python3
"""Restore only a pinned evaluation cohort from raw ACDC, checking every hash."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import nibabel as nib


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_checked(path, array, expected):
    path = Path(path)
    if path.exists():
        if digest(path) != expected:
            raise ValueError(f'existing evaluation asset differs: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        np.save(stream, array, allow_pickle=False)
    if digest(temporary) != expected:
        raise ValueError(f'restored evaluation asset differs: {path}')
    os.replace(temporary, path)


def restore(manifest, expected, raw_root, out):
    manifest, raw_root, out = Path(manifest), Path(raw_root), Path(out)
    if digest(manifest) != expected:
        raise ValueError('evaluation manifest mismatch')
    m = json.loads(manifest.read_text())
    if m['role'] not in {'development', 'external_frozen'} or m['synthetic']:
        raise ValueError('real evaluation cohort required')
    files = []
    for record in m['patients']:
        patient = record['id']
        if not patient.startswith('patient') or not patient[7:].isdigit():
            raise ValueError('invalid patient identity')
        candidates = list((raw_root / patient).glob(patient + '_4d.nii*'))
        if len(candidates) != 1:
            raise ValueError('ambiguous/missing raw cine')
        raw_path = candidates[0]
        if digest(raw_path) != record['geometry_source_image']['sha256']:
            raise ValueError('raw cine hash mismatch')
        raw_image = nib.load(raw_path)
        raw = np.asarray(raw_image.dataobj)
        if raw.ndim != 4 or not np.isfinite(raw).all():
            raise ValueError('invalid cine')
        lo, hi = np.percentile(raw, [.5, 99.5])
        x = np.clip((raw-lo)/max(float(hi-lo), 1e-6), 0, 1).transpose(3, 2, 1, 0).astype(np.float32)
        target = out / Path(record['image']).name
        save_checked(target, x, record['image_sha256'])
        files.append({'path': str(target), 'sha256': digest(target)})
        for frame, mr in record['masks'].items():
            t = int(frame)
            candidates = list((raw_root / patient).glob(f'{patient}_frame{t+1:02d}_gt.nii*'))
            if len(candidates) != 1:
                raise ValueError('ambiguous/missing reference')
            mask_path = candidates[0]
            mask = nib.load(mask_path)
            image = nib.load(str(mask_path).replace('_gt.nii', '.nii'))
            if mask.shape != image.shape or not np.allclose(mask.affine, image.affine):
                raise ValueError('mask/frame geometry mismatch')
            if not np.array_equal(np.asarray(image.dataobj), raw[..., t]):
                raise ValueError('frame/cine voxel mismatch')
            y = np.asarray(mask.dataobj)
            if not np.isin(y, [0, 1, 2, 3, 255]).all():
                raise ValueError('unknown classes')
            target = out / Path(mr['path']).name
            save_checked(target, y.transpose(2, 1, 0).astype(np.uint8), mr['sha256'])
            files.append({'path': str(target), 'sha256': digest(target)})
        print(patient, 'verified', flush=True)
    receipt = {'status': 'VERIFIED', 'manifest_sha256': expected, 'patients': len(m['patients']),
               'files': files, 'scope': 'evaluation asset preparation only; no model selection or training'}
    (out / 'RESTORATION.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'manifest-sha256', 'raw-root', 'out'):
        parser.add_argument('--' + name, required=True)
    a = parser.parse_args()
    restore(a.manifest, a.manifest_sha256, a.raw_root, a.out)
