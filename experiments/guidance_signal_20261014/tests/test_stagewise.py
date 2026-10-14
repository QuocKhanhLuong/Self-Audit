"""Offline CPU contracts; no real model, data, teacher or weight download."""
from __future__ import annotations
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments.guidance_signal_20261014 import runtime, stagewise as sw, stagewise_table as table


@pytest.fixture(scope='module')
def rt():
    return runtime.load_runtime()


def fake_arrays():
    cfg = dict(min_probability=.7, min_margin=.2, min_reliability=.05)
    # Includes UNKNOWN after gating, disallowed high-confidence RV, and ties.
    p = np.array([[[[[.8, .1, .05, .05], [.05, .85, .05, .05]],
                     [[.25, .25, .25, .25], [.05, .05, .1, .8]]]]], np.float32)
    rel = np.array([[[[.1, .1], [0, .1]]]], np.float32)
    raw, labels, _ = sw.gated(p, cfg, reliability=rel)
    bank = dict(r0=p.copy(), q=p.copy(), reliability=rel, labels=labels)
    native = np.repeat(np.repeat(p, 2, axis=2), 3, axis=3)
    ck = dict(config=cfg, allowed_classes=[True, False, True, True], status='PARTIAL_SEMANTIC_TRAINED_NOT_QUALITY_CERTIFIED')
    return bank, p, native, ck


def test_nearest_mapping_preserves_unknown_and_named_labels():
    labels = np.array([[[[255, 1], [2, 3]]]], np.uint8)
    mapped = sw.native_labels(labels, (3, 5))
    assert mapped.shape == (1, 1, 3, 5)
    assert mapped[0, 0].tolist() == [[255, 255, 255, 1, 1], [255, 255, 255, 1, 1], [2, 2, 2, 3, 3]]
    assert set(np.unique(mapped)) == {1, 2, 3, 255}


def test_raw_argmax_separate_from_original_permission_gating():
    bank, common, native, ck = fake_arrays()
    outputs, rows, projected = sw.stage_arrays(bank, common, native, ck)
    assert set(outputs) == set(table.STAGES)
    assert (outputs['native_raw_argmax'] == 1).sum() == 6
    assert (outputs['native_gated'] == 1).sum() == 0
    assert (outputs['q_gated'] == 1).sum() == 6  # original bank has no student permission gate
    assert np.all(outputs['native_gated'][outputs['native_raw_argmax'] == 1] == 255)
    assert np.allclose(projected.sum(-1), 1)
    rv = next(r for r in rows if r['stage'] == 'native' and r['class_name'] == 'RV')
    assert rv['raw_argmax_pixels'] == 6 and rv['permission_pass_pixels'] == 0
    assert ck['allowed_classes'] == [True, False, True, True]


def test_q_gate_reconstruction_fails_closed():
    bank, common, native, ck = fake_arrays()
    bank['labels'][0, 0, 0, 0] = 255
    with pytest.raises(ValueError, match='Q gate reconstruction'):
        sw.stage_arrays(bank, common, native, ck)


def test_representation_only_keeps_raw_diagnostics_but_abstains():
    bank, common, native, ck = fake_arrays(); ck['status'] = 'REPRESENTATION_ONLY'
    outputs, _, _ = sw.stage_arrays(bank, common, native, ck)
    assert np.all(outputs['native_gated'] == 255)
    assert np.any(outputs['native_raw_argmax'] != 255)
    assert np.all(outputs['student_common_gated'] == 255)


def test_fidelity_before_thresholds_and_undefined_empty_strata():
    bank, common, _, ck = fake_arrays()
    rows = sw.fidelity_scalars('synthetic', bank['q'], common, bank['reliability'], ck['allowed_classes'])
    rv = next(r for r in rows if r['class_name'] == 'RV')
    myo = next(r for r in rows if r['class_name'] == 'MYO')
    assert rv['pixels'] == 1 and rv['argmax_agreement'] == 1
    assert rv['effective_weight_mass'] == 0 and rv['weighted_kl'] is None
    assert myo['pixels'] == 0 and myo['kl_q_to_student'] is None


def test_fake_student_matches_pinned_native_prediction(rt):
    class FakeModel(torch.nn.Module):
        def forward(self, x):
            logits = torch.stack([x[:, 0], -x[:, 0], x[:, 1], x[:, 2]], 1)
            return {'logits': logits}
    cine = np.random.default_rng(42).normal(size=(2, 1, 8, 8)).astype(np.float32)
    ck = dict(config=dict(grid=3, min_probability=.7, min_margin=.2),
              allowed_classes=[True, False, True, True], status='SEMANTIC_TRAINED_NOT_QUALITY_CERTIFIED')
    common, native = sw._student_probabilities(rt, FakeModel(), ck, cine, 'cpu', 1)
    _, labels, _ = sw.gated(native, ck['config'], allowed=ck['allowed_classes'])
    assert common.shape == (2, 1, 3, 3, 4)
    assert np.array_equal(labels, rt.student.predict_cine(FakeModel(), ck, cine, 'cpu', 1))


def frozen_fixture(tmp_path, rt):
    folder = tmp_path / 'generated'; folder.mkdir()
    patients = sw._cohort(rt)
    recipe, _ = sw._published(rt, 'run_evidence/run/RECIPE.json')
    meta = {p: dict(shape_xyzt=[2, 2, 1, 2], affine=np.eye(4).tolist(),
                   val_coded_uncoded='UNKNOWN') for p in patients}
    for stage in table.STAGES:
        sub = folder / stage; sub.mkdir()
        for pid in patients:
            np.savez_compressed(sub / (pid + '.npz'), labels=np.full((2, 1, 2, 2), 255, np.uint8))
        rt.io.freeze(sub, dict(role='student_predictions', diagnostic_stage=stage, split='val', patients=patients,
            images=meta, student_status='SEMANTIC_TRAINED_NOT_QUALITY_CERTIFIED',
            checkpoint_sha256=sw.CHECKPOINT_SHA256, locked_splits=recipe['splits'], train_patients=recipe['splits']['train'], gt_read=False))
    rt.io.write_json(folder / 'GENERATION.json', dict(private_arrays=False))
    rt.io.freeze(folder, dict(role='stagewise_complete', split='val', patients=patients, stages=list(table.STAGES),
        stage_manifest_sha256={s: rt.io.digest(folder / s / 'FROZEN.json') for s in table.STAGES},
        source_sha256=rt.source_sha256, source_commit=rt.snapshot_sha, experiment_source_sha256=sw.experiment_hashes(rt.io), gt_read=False))
    return folder


def test_complete_freeze_and_missing_patient_rejected(tmp_path, rt):
    folder = frozen_fixture(tmp_path, rt)
    assert len(sw.verify_complete(folder, rt)['patients']) == 18
    (folder / 'q_raw' / (sw._cohort(rt)[0] + '.npz')).unlink()
    with pytest.raises(ValueError, match='inventory'):
        sw.verify_complete(folder, rt)


def test_score_no_reference_access_before_full_freeze(tmp_path, rt, monkeypatch):
    folder = frozen_fixture(tmp_path, rt)
    (folder / 'native_raw_argmax' / (sw._cohort(rt)[-1] + '.npz')).write_bytes(b'tampered')
    reads = []
    fake_eval = SimpleNamespace(resolve_references=lambda *a, **k: reads.append('REFERENCE_READ'),
                                evaluate=lambda *a, **k: reads.append('REFERENCE_SCORE'))
    fake_rt = SimpleNamespace(io=rt.io, source_sha256=rt.source_sha256, snapshot_sha=rt.snapshot_sha, evaluate=fake_eval)
    monkeypatch.setattr(sw.runtime, 'load_runtime', lambda: fake_rt)
    with pytest.raises(ValueError, match='hash mismatch'):
        sw.score(generated=folder, out=tmp_path / 'score', reference_root=tmp_path / 'sentinel')
    assert reads == []
    assert not (tmp_path / 'score').exists()


def test_score_rejects_wrong_original_frames_before_reference_bytes(tmp_path, rt, monkeypatch):
    folder = frozen_fixture(tmp_path, rt)
    fake_eval = SimpleNamespace(resolve_references=lambda *a, **k: [dict(patient_id='patient004', frame_index=1, path='/must/not/read')])
    fake_rt = SimpleNamespace(io=rt.io, source_sha256=rt.source_sha256, snapshot_sha=rt.snapshot_sha, evaluate=fake_eval)
    monkeypatch.setattr(sw.runtime, 'load_runtime', lambda: fake_rt)
    with pytest.raises(ValueError, match='original locked VAL36'):
        sw.score(generated=folder, out=tmp_path / 'score', reference_root=tmp_path / 'sentinel')
    assert not (tmp_path / 'score').exists()


def test_test_split_and_reference_arguments_to_generate_rejected():
    with pytest.raises(ValueError, match='TEST excluded'):
        sw.score(generated='absent', out='absent', split='test')
    with pytest.raises(ValueError, match='TEST excluded'):
        sw.generate(run='x', root='x', out='x', expected_checkpoint_sha256='x', teacher_checkpoint='x', teacher_config='x', split='test')
    with pytest.raises(SystemExit) as exc:
        sw.main(['GENERATE', '--run', 'x', '--root', 'x', '--out', 'x', '--expected-checkpoint-sha256', 'x',
                 '--teacher-checkpoint', 'x', '--teacher-config', 'x', '--reference-root', '/must/not/read'])
    assert exc.value.code == 2


def test_saved_table_counts_nulls_strata_and_failures(tmp_path):
    reductions = table.EVIDENCE.parent / 'guidance_research_continuation_20261014/PROVENANCE_METRICS_checks.json'
    before = table.digest(reductions)
    rows, summary = table.build_tables(reductions=reductions)
    assert len(rows) == 2880 and sum(r['availability'] == 'AVAILABLE' for r in rows) == 288
    assert summary['patient_count'] == 18 and summary['labeled_frames'] == 36
    rv = summary['native_gated']['new']['overall']['classes']['RV']
    assert rv['zero_prediction_patients'] == 18 and rv['missing_prediction_frames'] == 36
    assert rv['patient_macro']['precision'] is None and rv['patient_macro']['recall'] == 0
    assert summary['unavailable']['raw_RV_argmax'] == 'UNKNOWN'
    assert summary['native_gated']['new']['strata']['aligned']['patient_count'] == 11
    assert summary['native_gated']['new']['strata']['verified_recovery']['patient_count'] == 7
    for row in rows:
        assert row['val_coded_uncoded'] == 'UNKNOWN'
        assert row['old_new_error_regions_status'] == 'UNKNOWN'
        if row['availability'] == 'NOT_MEASURED':
            assert 'tp' not in row and 'precision' not in row
        else:
            assert row['correct_prediction_coverage'] == row['recall']
    table.write_tables(tmp_path, reductions=reductions)
    with (tmp_path / 'STAGEWISE_DIAGNOSTICS.csv').open() as f:
        exported = list(csv.DictReader(f))
    missing = next(r for r in exported if r['stage'] == 'r0_raw')
    assert missing['tp'] == missing['precision'] == ''
    with pytest.raises(FileExistsError):
        table.write_tables(tmp_path, reductions=reductions)
    assert table.digest(reductions) == before


def test_unknown_is_false_negative_in_original_evaluator(rt):
    labels = np.full((2, 2, 1), 255, np.uint8)
    target = np.ones_like(labels)
    score = rt.evaluate.score_volume(labels, target)
    assert score['classes']['RV']['fn'] == 4 and score['classes']['RV']['dice'] == 0
    assert score['coverage'] == 0
    with pytest.raises(ValueError, match='native XYZ'):
        rt.evaluate.score_volume(labels, np.ones((3, 2, 1), np.uint8))


def test_full_generate_with_explicit_synthetic_backend_freezes_all18(tmp_path, rt, monkeypatch):
    """Exercise GENERATE sequencing with tiny fake inputs, never a teacher fallback."""
    import sys
    recipe, _ = sw._published(rt, 'run_evidence/run/RECIPE.json')
    recipe = {**recipe, 'config': rt.core.GuidanceConfig(grid=8, feature_dim=4).dictionary(), 'orientation_policy': 'cardinal_v1'}
    # dataclass tuple is serialized as a list in original JSON/checkpoints.
    recipe['config'] = json.loads(json.dumps(recipe['config']))
    patients = recipe['splits']['val']; train = recipe['splits']['train']
    root = tmp_path / 'images'; root.mkdir()
    records = {}
    for pid in patients:
        path = root / (pid + '.dat'); path.write_bytes(b'synthetic-image-only'); records[pid] = path
    cine = np.random.default_rng(9).normal(size=(2, 1, 8, 8)).astype(np.float32)
    def reader(path):
        return cine.copy(), dict(image_sha256=rt.io.digest(path), shape_xyzt=[8, 8, 1, 2],
            affine=np.eye(4).tolist(), normalization={'fixture': 'synthetic'}, patient_left_axis='+x')
    old = {'images': {pid: reader(path)[1] for pid, path in records.items()}}
    teacher_provenance = dict(frozen=True, fixture='SYNTHETIC_ONLY')
    run = tmp_path / 'original'; (run / 'bank').mkdir(parents=True)
    (run / 'bank/FROZEN.json').write_text('{}')
    ck = dict(config=recipe['config'], allowed_classes=[True] * 4, class_patients=[len(train)] * 4,
        status='SEMANTIC_TRAINED_NOT_QUALITY_CERTIFIED', locked_splits=recipe['splits'],
        train_patients=train, teacher=teacher_provenance, bank_manifest_sha256=rt.io.digest(run / 'bank/FROZEN.json'))
    bank_manifest = dict(patients=train, teacher=teacher_provenance,
                         patient_reports={pid: {'class_pixels': [1] * 4} for pid in train})
    class FakeTeacher:
        def __init__(self, **kwargs):
            assert kwargs['offline'] is True
            self.provenance = teacher_provenance
        def extract(self, image, grid, feature_dim, tile_batch):
            return np.ones((*image.shape[:2], grid, grid, feature_dim), np.float16)
    class FakeModel(torch.nn.Module):
        def forward(self, x):
            return {'logits': torch.stack([x[:, 0], -x[:, 0], x[:, 1], x[:, 2]], 1)}
    def make_bank(image, features, axis, config, **kwargs):
        p = np.zeros((*image.shape, 4), np.float32); p[..., 0] = .8; p[..., 1:] = .2 / 3
        rel = np.full(image.shape, .1, np.float32)
        _, labels, _ = sw.gated(p, recipe['config'], reliability=rel)
        return dict(r0=p, q=p.copy(), reliability=rel, labels=labels), {'status': 'SYNTHETIC_ONLY'}
    fake_rt = SimpleNamespace(io=rt.io, core=SimpleNamespace(GuidanceConfig=rt.core.GuidanceConfig, make_bank=make_bank),
        teacher=SimpleNamespace(FrozenCineMA=FakeTeacher),
        student=SimpleNamespace(load_student=lambda *a: (FakeModel(), ck), contexts=rt.student.contexts, predict_cine=rt.student.predict_cine),
        source_sha256=rt.source_sha256, snapshot_sha=rt.snapshot_sha)
    monkeypatch.setattr(rt.io, 'read_cine', reader)
    monkeypatch.setattr(rt.io, 'discover', lambda *a: records)
    monkeypatch.setattr(sw.runtime, 'load_runtime', lambda: fake_rt)
    monkeypatch.setattr(sw, '_validate_original', lambda *a: (run, recipe, bank_manifest, {'teacher': teacher_provenance}, old, {}))
    monkeypatch.setitem(sys.modules, 'nibabel', SimpleNamespace(load=lambda *a: SimpleNamespace(header={'qform_code': 0, 'sform_code': 0})))
    result = sw.generate(run=run, root=root, out=tmp_path / 'generated', expected_checkpoint_sha256=sw.CHECKPOINT_SHA256,
                         teacher_checkpoint=tmp_path / 'FAKE_NOT_OPENED', teacher_config=tmp_path / 'FAKE_NOT_OPENED', device='cpu')
    assert result['patients'] == 18 and result['stages'] == 10
    frozen = sw.verify_complete(result['output'], fake_rt)
    assert frozen['gt_read'] is False
    out = Path(result['output'])
    assert not (out / 'PRIVATE_ARRAYS_DO_NOT_PUBLISH').exists()
    assert len(list(out.glob('*/*.npz'))) == 18 * 10
    with (out / 'FIDELITY_SCALARS.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 18 * 4 and {r['val_coded_uncoded'] for r in rows} == {'uncoded'}
    assert json.loads((out / 'GENERATION.json').read_text())['train_permissions_updated'] is False


def test_true_kl_disjoint_support_is_infinite_not_clipped_proxy():
    q = np.array([[[[[1., 0., 0., 0.]]]]], np.float32)
    p = np.array([[[[[0., 1., 0., 0.]]]]], np.float32)
    rows = sw.fidelity_scalars('fake', q, p, np.ones(q.shape[:-1]), [True] * 4)
    assert rows[0]['kl_status'] == 'INFINITE_SUPPORT_MISMATCH'
    assert rows[0]['infinite_kl_pixels'] == 1 and rows[0]['kl_q_to_student'] is None
    assert rows[0]['weighted_kl_status'] == 'INFINITE_SUPPORT_MISMATCH'
    # Finite logits can underflow probabilities: exact log_softmax retains finite KL.
    logp = np.array([[[[[-1000., 0., -1000., -1000.]]]]])
    rows = sw.fidelity_scalars('fake', q, p, np.ones(q.shape[:-1]), [True] * 4, logp)
    assert rows[0]['kl_status'] == 'FINITE' and rows[0]['kl_q_to_student'] == 1000.


@pytest.mark.parametrize('marker', ['FROZEN.json', 'PREDICTIONS_LOCKED.json'])
def test_output_guard_rejects_locked_ancestors(tmp_path, marker):
    frozen = tmp_path / 'frozen'; frozen.mkdir(); (frozen / marker).write_text('{}')
    with pytest.raises(ValueError, match='frozen or locked'):
        sw._fresh(frozen / 'child' / 'new')
    with pytest.raises(ValueError, match='frozen or locked'):
        table.write_tables(frozen / 'child')


def test_output_guard_rejects_input_overlap(tmp_path):
    source = tmp_path / 'original'; source.mkdir()
    with pytest.raises(ValueError, match='overlap'):
        sw._fresh(source / 'new', inputs=[source])
    with pytest.raises(ValueError, match='overlap'):
        table.write_tables(table.EVIDENCE / 'must_not_write')


def test_table_rejects_changed_count_bytes_even_without_reductions(tmp_path):
    import shutil
    evidence = tmp_path / 'evidence'; evidence.mkdir()
    for relative in ('MANIFEST.json', 'run_evidence/run/RECIPE.json', 'baseline_forensics/val_counts/DIAGNOSTICS.json',
                     'run_evidence/post_training/val_counts/DIAGNOSTICS.json'):
        dest = evidence / relative; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(table.EVIDENCE / relative, dest)
    path = evidence / 'baseline_forensics/val_counts/DIAGNOSTICS.json'
    data = json.loads(path.read_text()); data['patient_frame_counts'][0]['tp'] += 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='scalar input hash mismatch'):
        table.build_tables(evidence)
