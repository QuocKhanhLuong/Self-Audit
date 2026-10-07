"""Empty-prediction consumer contracts; synthetic fixtures are not quality evidence."""
import json

import numpy as np
import pytest

from pseudolabel_v3_fixtures import seal_test_teacher
from self_audit_pseudolabel.data_v3 import is_reference_path, read_patient_splits
from self_audit_pseudolabel.freeze import export_pseudo_npz, sha256_file, verify_frozen
from self_audit_pseudolabel.pipeline_v3 import discover, inventory
from test_pseudolabel_v3_checkup import evaluator
from test_pseudolabel_v3_regressions import _nifti_fixture


@pytest.mark.parametrize('unknown', [False, True], ids=['background', 'unknown'])
@pytest.mark.parametrize('empty_reference', [False, True], ids=['foreground-reference', 'empty-reference'])
def test_empty_predictions_evaluate_without_bypassing_freeze(tmp_path, monkeypatch, unknown, empty_reference):
    import nibabel as nib

    data, split = _nifti_fixture(tmp_path)
    if empty_reference:
        # Build independent synthetic references before preparing the teacher
        # artifact. Their contents never determine any exported prediction.
        for path in data.rglob('*_gt.nii.gz'):
            image = nib.load(path)
            empty = nib.Nifti1Image(np.zeros(image.shape, np.uint8), image.affine, image.header)
            nib.save(empty, path)

    run = tmp_path / 'freeze'
    run.mkdir()
    entries = []
    load = nib.load
    image_reads = []

    def image_only_load(path, *args, **kwargs):
        assert not is_reference_path(path), 'teacher fixture preparation opened a reference'
        image_reads.append(str(path))
        return load(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(nib, 'load', image_only_load)
        splits = read_patient_splits(split)
        records = inventory(discover('acdc', data), splits)
        for record in records:
            x, y, z_count, t_count = record['shape']
            label = np.full((y, x), 255 if unknown else 0, np.uint8)
            valid = np.full((y, x), 0 if unknown else 1, np.uint8)
            soft = np.zeros((4, y, x), np.float32)
            soft[0] = 1
            for t in range(t_count):
                for z in range(z_count):
                    metadata = {'patient_id': record['patient_id'], 't': t, 'z': z}
                    path = f"{record['patient_id']}_{t}_{z}.npz"
                    digest = export_pseudo_npz(
                        run / path, pseudo_label=label, valid=valid,
                        soft_label=soft, metadata=metadata,
                    )
                    entries.append({**metadata, 'path': path, 'sha256': digest, 'split': record['split']})
        config = {
            'dataset': 'acdc', 'image_records': records, 'export_records': records,
            'split_patients': splits, 'producer_patient_ids': splits['train'],
        }
        seal_test_teacher(run, entries, config)
        payload = verify_frozen(run / 'FROZEN.json', expected_role='teacher_freeze')

    assert image_reads
    assert payload['config']['synthetic_fixture'] is True
    assert all(entry['valid_foreground'] == 0 for entry in payload['entries'])
    frozen_digest = sha256_file(run / 'FROZEN.json')
    out = tmp_path / 'scores.json'
    result = evaluator()(['--run', str(run), '--references', str(data), '--out', str(out)])

    # Exercise persisted JSON as well as the in-process return: both-empty
    # classes are null, never NaN, while missed reference anatomy scores zero.
    saved = json.loads(out.read_text())
    assert saved == result
    assert saved['manifest_id'] == payload['manifest_id']
    assert saved['patients'] == 1 and saved['scored_slices'] == 4
    assert saved['patient_rows'][0]['patient'] == 'patient002'
    assert saved['empty_class_policy'] == 'exclude_both_empty'
    assert saved['UNKNOWN_policy'] == 'missed_reference_anatomy'
    expected_dice = None if empty_reference else 0.0
    metric_rows = [saved, *saved['patient_rows'], *saved['frame_rows'],
                   saved['phase_macro'], *saved['phase_metrics'].values()]
    for row in metric_rows:
        assert row['lv'] == expected_dice
        assert row['foreground_mean'] == expected_dice
        assert row['rv'] is None and row['myo'] is None
        assert row['known_fraction'] == (0.0 if unknown else 1.0)
    assert len(saved['reference_rows']) == 2
    assert {row['phase'] for row in saved['reference_rows']} == {'ED', 'ES'}
    assert all(row['sha256'] == sha256_file(row['path']) for row in saved['reference_rows'])
    assert sha256_file(run / 'FROZEN.json') == frozen_digest
    verify_frozen(run / 'FROZEN.json', expected_role='teacher_freeze')
