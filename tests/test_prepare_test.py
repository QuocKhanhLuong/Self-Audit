"""TEST cache boundaries; manufactured arrays establish no patient efficacy."""
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from scripts.prepare_acdc_test import prepare


@pytest.fixture
def inputs(tmp_path):
    raw_test, raw_train = tmp_path / 'testing', tmp_path / 'training'
    affine = np.eye(4)
    source = np.arange(8*8*2*2, dtype=np.float32).reshape(8, 8, 2, 2)
    for index in [1] + list(range(101, 151)):
        pid = f'patient{index:03d}'
        folder = (raw_train if index == 1 else raw_test) / pid
        folder.mkdir(parents=True)
        raw = source + index * 300
        nib.save(nib.Nifti1Image(raw, affine), folder / f'{pid}_4d.nii')
        if index == 1:
            # This deliberately invalid annotation must never be opened.
            (folder / f'{pid}_frame01_gt.nii').write_bytes(b'forbidden TRAIN annotation')
            continue
        for frame in (1, 2):
            nib.save(nib.Nifti1Image(raw[..., frame-1], affine), folder / f'{pid}_frame{frame:02d}.nii')
            mask = np.zeros((8, 8, 2), dtype=np.uint8); mask[2:5, 2:5] = 3
            nib.save(nib.Nifti1Image(mask, affine), folder / f'{pid}_frame{frame:02d}_gt.nii')
    split = tmp_path / 'split.json'
    split.write_text(json.dumps(dict(train=['patient001'], val=[], test=[f'patient{i:03d}' for i in range(101, 151)])))
    protocol = tmp_path / 'protocol.json'
    protocol.write_text(json.dumps(dict(checkpoints={a: {} for a in ('baseline', 'fixed', 'meta')})))
    return raw_test, raw_train, split, protocol, tmp_path / 'cache'


def test_external_test_cache_does_not_read_train_masks(inputs):
    receipt = prepare(*inputs)
    assert receipt['patients'] == 50 and receipt['reference_frames'] == 100
    assert receipt['train_u_val_mask_reads'] == 0
    assert receipt['raw_train_val_cines_compared'] == 1
    manifest = json.loads((inputs[-1] / 'test.json').read_text())
    assert manifest['role'] == 'external_frozen'
    assert all(len(p['masks']) == 2 for p in manifest['patients'])


def test_external_test_rejects_duplicate_train_cine(inputs):
    raw_test, raw_train, *_ = inputs
    src = raw_train / 'patient001/patient001_4d.nii'
    (raw_test / 'patient101/patient101_4d.nii').write_bytes(src.read_bytes())
    with pytest.raises(ValueError, match='duplicate cine content'):
        prepare(*inputs)


def test_external_test_rejects_broken_reference_bridge(inputs):
    path = inputs[0] / 'patient101/patient101_frame01.nii'
    image = nib.load(path)
    changed = np.asarray(image.dataobj).copy() + 1
    nib.save(nib.Nifti1Image(changed, image.affine), path)
    with pytest.raises(ValueError, match='voxel bridge mismatch'):
        prepare(*inputs)


def test_external_test_reports_previously_quarantined_match(inputs):
    raw_test, raw_train, split, *_ = inputs
    excluded = raw_train / 'patient055'; excluded.mkdir()
    (excluded / 'patient055_4d.nii').write_bytes((raw_test / 'patient118/patient118_4d.nii').read_bytes())
    metadata = json.loads(split.read_text()); metadata['excluded_patients'] = ['patient055']
    split.write_text(json.dumps(metadata))
    receipt = prepare(*inputs)
    assert receipt['patients'] == 50
    assert receipt['exact_cine_overlap_with_active_train_val'] == 0
    assert receipt['quarantined_raw_cine_matches'] == [dict(test_id='patient118', excluded_raw_ids=['patient055'])]
