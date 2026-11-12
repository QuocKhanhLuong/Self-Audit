"""CLI cache interoperability with the trainer's snapshot and source identities."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from scripts.q3d_physio_cache import build_cache, _load_post_t_teacher
from self_audit.candidate_a3d.data import synthetic_fixture
from self_audit.candidate_q3d.config import canonical_config, validate_config
from self_audit.candidate_q3d.engine import (
    _atomic_torch_save, _build_teacher, _build_tokenizer, _data_identity_without_labels,
    _load_data, _object_sha256, _state_sha256,
)
from self_audit.candidate_q3d.physiology import load_physiology_cache

torch.set_num_threads(1)


def _snapshot(tmp_path: Path, arm='c1'):
    fixture = synthetic_fixture(tmp_path / 'fixture', n_labeled=2, n_unlabeled=2,
                                n_evaluation=1, shape_tzyx=(2, 4, 8, 9))
    config = canonical_config(arm, k=2)
    config['runtime'].update(synthetic=True, device='cpu')
    for name, path_key, sha_key in [('train', 'train', 'train_sha256'),
                                    ('development', 'evaluation', 'evaluation_sha256')]:
        config['dataset']['manifests'][name] = {
            'path': str(fixture[path_key]), 'sha256': fixture[sha_key], 'status': 'READY',
        }
    config = validate_config(config)
    data = _load_data(config, None)
    teacher, _ = _build_teacher(config, synthetic=True)
    teacher.eval().requires_grad_(False)
    model_sha = _state_sha256(teacher)
    # This is an explicit manufactured safe payload for loader testing; it
    # makes no claim that the randomly initialized teacher was fine-tuned.
    payload = {
        'schema': 'self_audit.candidate_q3d.resume.v3',
        'config_sha256': _object_sha256(config),
        'data_identity': _data_identity_without_labels(data),
        'completed_stages': ['T'], 'teacher_state': teacher.state_dict(),
        'teacher_best_sha256': model_sha,
    }
    checkpoint = tmp_path / 'synthetic_snapshot.pt'
    file_sha = _atomic_torch_save(payload, checkpoint)
    return config, data, teacher, payload, checkpoint, file_sha


def test_post_t_cli_cache_uses_engine_model_and_source_identity(tmp_path):
    config, data, teacher, _, checkpoint, file_sha = _snapshot(tmp_path)
    target = tmp_path / 'cache.json'
    summary = build_cache(config, config_path=tmp_path/'config.json',
                          teacher_factory=None, output_path=target,
                          teacher_checkpoint=str(checkpoint), teacher_sha256=file_sha)
    assert summary['teacher_checkpoint_file_sha256'] == file_sha
    assert summary['teacher_sha256'] == _state_sha256(teacher)
    assert summary['teacher_sha256'] != file_sha
    assert summary['synthetic_fixture'] is True
    result = load_physiology_cache(target, teacher_sha256=_state_sha256(teacher),
                                  codebook_sha256=None, protocol='A', config=config,
                                  source_identity=data.cache_source_identity())
    assert {r['id'] for r in result['payload']['patients']} == set(data.unlabeled_patients)
    assert data.student_data.label_reads == set()


@pytest.mark.parametrize('change,match', [
    ('stage', 'completed stage T'), ('config', 'config identity'),
    ('cohort', 'protocol/cohort'), ('state', 'VAL-selected'),
])
def test_post_t_loader_rejects_wrong_stage_config_cohort_or_snapshot(tmp_path, change, match):
    config, _, _, payload, checkpoint, _ = _snapshot(tmp_path)
    payload = copy.deepcopy(payload)
    if change == 'stage':
        payload['completed_stages'] = []
    elif change == 'config':
        payload['config_sha256'] = '0' * 64
    elif change == 'cohort':
        payload['data_identity']['protocol'] = 'B'
    else:
        payload['teacher_best_sha256'] = '0' * 64
    checksum = _atomic_torch_save(payload, checkpoint)
    with pytest.raises(ValueError, match=match):
        _load_post_t_teacher(config, checkpoint, checksum, torch.device('cpu'))


def test_post_t_loader_requires_digest_sidecar_before_safe_load(tmp_path):
    config, _, _, _, checkpoint, checksum = _snapshot(tmp_path)
    checkpoint.with_suffix('.pt.sha256').write_text('')
    with pytest.raises(ValueError, match='sidecar'):
        _load_post_t_teacher(config, checkpoint, checksum, torch.device('cpu'))


def test_cache_derives_codebook_digest_from_frozen_q_snapshot(tmp_path):
    config, data, teacher, payload, checkpoint, checksum = _snapshot(tmp_path, 'v1')
    with pytest.raises(ValueError, match='completed Q'):
        _load_post_t_teacher(config, checkpoint, checksum, torch.device('cpu'))
    tokenizer = _build_tokenizer(config, teacher, torch.device('cpu'))
    # Manufactured frozen artifact for loader verification, never a passing
    # tokenizer training gate or evidence of patient-quality code assignments.
    tokenizer.C0.copy_(torch.nn.functional.normalize(torch.randn_like(tokenizer.C0), dim=1))
    tokenizer.is_initialized.fill_(True)
    tokenizer.is_frozen.fill_(True)
    tokenizer.reconstructor = None
    tokenizer.anatomy_head = None
    payload['completed_stages'] = ['T', 'Q']
    payload['tokenizer_state'] = tokenizer.state_dict()
    checksum = _atomic_torch_save(payload, checkpoint)
    kwargs = dict(config_path=tmp_path/'config.json', teacher_factory=None,
                  output_path=tmp_path/'vq_cache.json', teacher_checkpoint=str(checkpoint),
                  teacher_sha256=checksum)
    result = build_cache(config, **kwargs)
    assert result['codebook_sha256'] == tokenizer.codebook_sha256
    load_physiology_cache(kwargs['output_path'], teacher_sha256=_state_sha256(teacher),
                         codebook_sha256=tokenizer.codebook_sha256, protocol='A',
                         config=config, source_identity=data.cache_source_identity())
    with pytest.raises(ValueError, match='differs from the frozen'):
        build_cache(config, codebook_sha256='0'*64, **kwargs)


def test_standalone_cache_rejects_joint_snapshot_and_online_epoch(tmp_path):
    config, _, _, payload, checkpoint, _ = _snapshot(tmp_path)
    payload['completed_stages'].append('J')
    checksum = _atomic_torch_save(payload, checkpoint)
    with pytest.raises(ValueError, match='before joint'):
        _load_post_t_teacher(config, checkpoint, checksum, torch.device('cpu'))
    with pytest.raises(ValueError, match='O1 refreshes'):
        build_cache(config, config_path=tmp_path/'config.json', teacher_factory=None,
                    output_path=tmp_path/'cache.json', teacher_checkpoint=str(checkpoint),
                    teacher_sha256=checksum, refreshed_epoch=10)
    checkpoint.write_bytes(b'tampered')
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        _load_post_t_teacher(config, checkpoint, checksum, torch.device('cpu'))
