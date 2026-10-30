"""Exercise the actual subprocess runner, segmented resume, export and native scoring."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from scripts.run_transfer_utility_pilot import (
    ROOT, _run_logged, execute_arm, evaluate_screening_decision, read_json, write_json,
)
from self_audit.compact_artifacts import digest
from self_audit.compact_data import synthetic_compact_fixture


def test_actual_segmented_runner(tmp_path):
    train, train_sha = synthetic_compact_fixture(tmp_path / 'train', size=16,
        n_labeled=1, n_unlabeled=2, num_frames=2, num_slices=2)
    cache = tmp_path / 'cache'
    _run_logged([sys.executable, str(ROOT / 'scripts/build_compact_teacher_cache.py'),
        '--data-manifest', str(train), '--data-sha256', train_sha,
        '--output-dir', str(cache), '--synthetic-teacher', '--device', 'cpu'], tmp_path, 'cache', 60)
    cache_file = cache / 'CACHE_MANIFEST.json'
    cfg = read_json(ROOT / 'configs/transfer_utility/scalar_guard_cross_patient_utility.json')
    cfg.update(steps=3, warmup_steps=1, batch_size=1, labeled_batch_size=1,
        unlabeled_batch_size=1, probe_batch_size=1, crop_size=16, max_trials=8,
        pseudo_threshold=0.0, save_every=1)
    cfg['model']['widths'] = [4, 8]
    ids = [r['id'] for r in read_json(train)['labeled']]
    cfg['data'] = dict(manifest_sha256=train_sha, budget_k=len(ids), labeled_ids=ids)
    cfg['teacher'] = {'lineage': {'budget_k': len(ids), 'exact_labeled_ids': ids}}
    config = tmp_path / 'config.json'
    write_json(config, cfg)
    rng = np.random.default_rng(514)
    cine = rng.random((2, 2, 16, 16), dtype=np.float32)
    dev = tmp_path / 'dev'
    dev.mkdir()
    image, label = dev / 'dev_image.npy', dev / 'dev_label.npy'
    np.save(image, cine)
    np.save(label, (cine[0] * 4).astype(np.uint8))
    val = dev / 'val.json'
    write_json(val, {'schema': 'self_audit.native_manifest.v1', 'role': 'development',
        'synthetic': True, 'patients': [{'id': 'synthetic_dev', 'domain': 'manufactured',
        'image': image.name, 'image_sha256': digest(image), 'axes': ['T','Z','Y','X'],
        'affine': np.eye(4).tolist(), 'masks': {'0': {'path': label.name, 'sha256': digest(label)}}}]})
    kwargs = dict(data_manifest=train, data_manifest_sha256=train_sha, val_manifest=val,
        val_manifest_sha256=digest(val), device='cpu', synthetic=True,
        teacher_cache=cache_file, teacher_cache_sha256=digest(cache_file), watchdog_seconds=120)
    result = execute_arm('scalar_guard_cross_patient_utility', config, tmp_path / 'run',
                         stop_after_steps=[1, 2, 3], **kwargs)
    assert result['status'] == 'COMPLETED', result
    assert result['steps'] == 3
    assert len(result['segments']) == len(result['evaluations']) == 3
    assert len(result['initial_state_sha256']) == len(result['data_schedule_sha256']) == 64
    assert all(e['coverage']['complete'] and e['synthetic'] for e in result['evaluations'].values())
    assert read_json(Path(result['final_checkpoint']) / 'MANIFEST.json')['step'] == 3
    assert [json.loads(x)['step'] for x in (tmp_path/'run/TRAIN_LOG.jsonl').read_text().splitlines()] == [0,1,2]
    with pytest.raises(FileExistsError):
        execute_arm('scalar_guard_cross_patient_utility', config, tmp_path / 'run', **kwargs)
    bad = execute_arm('scalar_guard_cross_patient_utility', config, tmp_path / 'bad',
                      **{**kwargs, 'val_manifest_sha256': '0'*64})
    assert bad['status'] == 'INCOMPLETE'
    assert bad['evaluations'] == {}


def test_missing_provenance_cannot_screen_go():
    names = ('plain_kd','scalar_guard_original','scalar_guard_larger_tie','scalar_guard_cross_patient_utility')
    arms = {n: {'status':'COMPLETED','evaluations':{'step_002000':{
        'synthetic':False,'class_first_macro4':.9 if 'cross_patient' in n else .8,
        'patients':{},'coverage':{'complete':True},'status':'COMPLETED'}}} for n in names}
    assert evaluate_screening_decision(arms)['verdict'] == 'BLOCKED'
    for arm in arms.values():
        arm['evaluations']['step_002000']['patients'] = {f'p{i}':{} for i in range(18)}
    assert evaluate_screening_decision(arms)['verdict'] == 'BLOCKED'
