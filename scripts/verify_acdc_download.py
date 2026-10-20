#!/usr/bin/env python3
"""Inventory raw ACDC without opening any reference mask array.

Run only after rclone copy and checksum check succeed. This verifies local image
geometry/content, not transport completeness. Never alters the locked split.
"""
import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify(root, split_path):
    split = json.loads(split_path.read_text())
    train, val = split['train_patients'], split['val_patients']
    errors, records, hashes = [], [], {}
    if len(set(train + val)) != len(train + val):
        errors.append('locked split contains overlapping or duplicate IDs')
    expected = {f'patient{i:03d}' for i in range(1, 101)}
    actual = {p.name for p in (root / 'training').glob('patient*') if p.is_dir()}
    if actual != expected or set(train + val) != expected:
        errors.append('training directory or locked split does not cover exactly patient001..100')
    for pid in sorted(actual):
        folder = root / 'training' / pid
        cine_paths = list(folder.glob(f'{pid}_4d.nii*'))
        if len(cine_paths) != 1 or not (folder / 'Info.cfg').is_file():
            errors.append(f'{pid}: missing unique cine or Info.cfg')
            continue
        path = cine_paths[0]
        try:
            image = nib.load(path)
            a = np.asarray(image.dataobj)
            if a.ndim != 4 or not np.isfinite(a).all() or not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-12:
                raise ValueError('invalid finite 4D image/affine')
            digest = sha(path)
            if digest in hashes:
                errors.append(f'duplicate cine content: {hashes[digest]}, {pid}; split review required')
            hashes[digest] = pid
            # Image frames only: no discovery or opening of U reference arrays.
            frames = list(folder.glob(f'{pid}_frame*.nii*'))
            frames = [p for p in frames if '_gt' not in p.name]
            if len(frames) != 2:
                errors.append(f'{pid}: expected two ED/ES image frames')
            for fp in frames:
                frame = nib.load(fp)
                t = int(fp.name.split('_frame')[1].split('.')[0]) - 1
                if frame.shape != a.shape[:3] or not 0 <= t < a.shape[3] or not np.array_equal(np.asarray(frame.dataobj), a[..., t]):
                    errors.append(f'{pid}: image frame/cine exact voxel bridge failed')
            records.append({'patient': pid, 'role': 'TRAIN' if pid in train else 'VAL', 'cine_sha256': digest, 'shape_xyzt': list(image.shape), 'affine': image.affine.tolist(), 'spacing_header': [float(x) for x in image.header.get_zooms()], 'geometry_status': 'image_header_only_not_HD95_certification'})
        except Exception as exc:
            errors.append(f'{pid}: {type(exc).__name__}: {exc}')
    test_ids = sorted(p.name for p in (root / 'testing').glob('patient*') if p.is_dir())
    if set(test_ids) != {f'patient{i:03d}' for i in range(101, 151)}:
        errors.append('official testing directory does not cover exactly patient101..150')
    return {'status': 'PASS' if not errors else 'FAIL', 'root': str(root), 'split_sha256': sha(split_path), 'train_expected': len(train), 'val_expected': len(val), 'training_folders_observed': len(actual), 'training_cines_verified': len(records), 'official_test_folders_observed': len(test_ids), 'mask_arrays_opened': 0, 'transport_integrity': 'requires separate successful rclone check', 'patients': records, 'errors': errors}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--split', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        p.error('output must be new; preserve previous verification receipts')
    report = verify(a.root.resolve(), a.split.resolve())
    a.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'patients'}, indent=2))
    raise SystemExit(0 if report['status'] == 'PASS' else 1)
