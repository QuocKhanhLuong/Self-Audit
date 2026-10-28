"""Prepare image-disjoint M&Ms TEST135 for frozen, offline evaluation only.

Native float32 cine arrays live in a private tmpfs scratch directory because the
22.3 GiB derived cache would exhaust the shared persistent workspace. The raw
archives and NIfTI source files remain unchanged and the recipe is repeatable.
"""
import csv
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path('/workspace/Self-Audit')
sys.path.insert(0, str(ROOT))
from self_audit.model import digest
from self_audit.data import write_json

R = ROOT / 'reports/automatic_pilot_pro4000'
raw = ROOT / 'data/MnM'
scratch = Path('/dev/shm/self_audit_mnm_test135_20261028')
assert not scratch.exists(), 'derived TEST scratch already exists; preserve it'
scratch.mkdir()
quarantined = set(json.loads((R / 'cross_dataset_voxel_duplicates.json').read_text())['quarantined_mnm_ids'])
split = json.loads((R / 'mnm_patient_splits.json').read_text())
original_test = set(split['original_test'])
assert len(original_test) == 136 and len(original_test & quarantined) == 1
metadata_file = next(raw.rglob('*.csv'))
metadata = {row['External code']: row for row in csv.DictReader(metadata_file.open())}
images = sorted(p for p in raw.rglob('*_sa.nii.gz') if 'Testing' in p.parts)
assert len(images) == 136
look = np.arange(256, dtype=np.uint8)
look[1] = 3
look[3] = 1
records = []
mask_count = 0
unknown_temporal_spacing = 0
for i, image_path in enumerate(images, 1):
    patient = image_path.name.split('_sa')[0]
    assert patient in original_test
    if patient in quarantined:
        continue
    ni = nib.load(image_path)
    image = np.asarray(ni.dataobj)
    assert image.ndim == 4 and np.isfinite(image).all()
    assert all(float(v) > 0 for v in ni.header.get_zooms()[:3])
    if float(ni.header.get_zooms()[3]) == 0:
        unknown_temporal_spacing += 1
    lo, hi = np.percentile(image, [.5, 99.5])
    normalized = np.clip((image - lo) / max(float(hi - lo), 1e-6), 0, 1).transpose(3, 2, 1, 0).astype(np.float32)
    image_file = scratch / f'MnM_{patient}.npy'
    np.save(image_file, normalized)
    mask_path = image_path.with_name(image_path.name.replace('_sa.nii', '_sa_gt.nii'))
    mn = nib.load(mask_path)
    assert mn.shape == ni.shape and np.allclose(mn.affine, ni.affine)
    mask = np.asarray(mn.dataobj)
    assert float(np.max(np.abs(mask - np.rint(mask)))) < 1e-4
    labels = np.rint(mask).astype(np.uint8)
    assert np.isin(labels, [0, 1, 2, 3, 255]).all()
    frames = sorted({int(metadata[patient]['ED']), int(metadata[patient]['ES'])})
    assert len(frames) == 2 and 0 <= frames[0] < image.shape[3] and 0 <= frames[1] < image.shape[3]
    record = {'id': 'MnM_' + patient, 'image': str(image_file), 'image_sha256': digest(image_file), 'masks': {}}
    for t in frames:
        out_mask = scratch / f'MnM_{patient}_t{t:03d}_mask.npy'
        np.save(out_mask, look[labels[..., t]].transpose(2, 1, 0))
        record['masks'][str(t)] = {'path': str(out_mask), 'sha256': digest(out_mask)}
        mask_count += 1
    records.append(record)
    fs = os.statvfs(scratch)
    free_gib = fs.f_bavail * fs.f_frsize / 2**30
    if free_gib < 4:
        raise RuntimeError(f'tmpfs safety threshold reached after {patient}: {free_gib:.1f} GiB free')
    if i % 10 == 0:
        print('PREPARED_TEST', len(records), 'of 135', 'tmpfs_free_GiB', round(free_gib, 1), flush=True)

assert len(records) == 135 and mask_count == 270 and unknown_temporal_spacing == 135
assert len({x['id'] for x in records}) == 135
manifest_path = R / 'MnM_TEST135_manifest.json'
assert not manifest_path.exists()
write_json(manifest_path, {
    'schema': 'self_audit.data.v2', 'role': 'external_frozen', 'synthetic': False,
    'scope': 'Drive M&Ms original TEST136 minus one cross-role exact-image duplicate; 135 patients. External frozen evaluation only; GT not independently challenge certified.',
    'patients': records,
})
receipt = {
    'schema': 'self_audit.pro4000.mnm_test_preparation.v1', 'status': 'PASS',
    'original_test_patients': 136, 'active_test_patients': 135,
    'quarantined_test_ids': sorted(original_test & quarantined),
    'source_nifti_images': 136, 'prepared_images': 135,
    'prepared_annotated_frames': mask_count,
    'unknown_temporal_spacing_images': unknown_temporal_spacing,
    'original_files_modified': False,
    'normalization': 'per-cine image-only .5/99.5 percentiles, native TZHW float32, no resize',
    'label_mapping': {'0': 0, '1': 3, '2': 2, '3': 1, '255': 255},
    'source_split_csv_sha256': digest(metadata_file),
    'manifest': str(manifest_path), 'manifest_sha256': digest(manifest_path),
    'scratch': str(scratch), 'scratch_persistence': 'tmpfs only; regenerate with this script after instance restart',
    'training_access': False,
}
(R / 'mnm_test_preparation.json').write_text(json.dumps(receipt, indent=2) + '\n')
print('MNM_TEST135_PREPARED', json.dumps(receipt), flush=True)
