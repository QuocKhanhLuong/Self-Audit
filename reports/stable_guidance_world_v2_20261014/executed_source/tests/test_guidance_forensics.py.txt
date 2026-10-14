"""Manufactured regression tests, not evidence of cardiac accuracy."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from self_audit_guidance.core import GuidanceConfig, anatomical_anchors
from self_audit_guidance.experimental_geometry import world_left_geometry, header_geometry
from self_audit_guidance.experimental_diagnostics import (
    semantic_terms, scalar_terms, epoch_terms, shared_gradient_norms,
    resolution_control, gate_probabilities, bank_statistics, replay_active_schedule)
from self_audit_guidance.student import GuidanceStudent, guidance_loss, train_student, load_student
from self_audit_guidance.io import read_cine, write_json, freeze, digest

spec = importlib.util.spec_from_file_location('forensics_cli', Path(__file__).parents[1]/'scripts/diagnose_stable_guidance.py')
cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize('angle', [0, 30, 45, 60, 90])
@pytest.mark.parametrize('flip,swap', [(1, False), (-1, False), (1, True), (-1, True)])
def test_world_centroids_equal_affine_with_anisotropic_pixel_centers(angle, flip, swap):
    theta = np.deg2rad(angle)
    a = np.eye(4)
    a[:2, :2] = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) @ np.diag([2*flip, 3])
    if swap: a[:, [0, 1]] = a[:, [1, 0]]
    a[:3, 3] = [20, -40, 10]
    g = world_left_geometry(a, (120, 280), (24, 40), sform_code=1, qform_code=0)
    assert g['valid']
    transform = np.asarray(g['bank_xy_to_native_xy_pixel_centers'])
    c1, c2 = np.array([2., 3.]), np.array([7., 12.])  # bank y,x
    def position(yx):
        xy = transform @ [yx[1], yx[0], 1]
        return -(a @ [*xy[:2], 2, 1])[0]
    assert np.dot(g['left_vector_yx'], c2-c1) == pytest.approx(position(c2)-position(c1))
    assert (transform @ [0, 0, 1])[:2] == pytest.approx([3, 2])


@pytest.mark.parametrize('kind', ['nan', 'singular', 'fallback', 'out_of_plane', 'bad_bottom_row'])
def test_uninformative_or_invalid_geometry_abstains(kind):
    a = np.eye(4); sc = 1
    if kind == 'nan': a[0, 0] = np.nan
    if kind == 'singular': a[0, 0] = 0
    if kind == 'fallback': sc = 0
    if kind == 'out_of_plane': a[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    if kind == 'bad_bottom_row': a[3, 0] = 1
    g = world_left_geometry(a, (32, 32), (8, 8), sform_code=sc, qform_code=0)
    assert not g['valid'] and g['left_vector_yx'] == [0, 0]


def test_real_nifti_header_authority_and_cardinal_45_gap(tmp_path):
    import nibabel as nib
    a = np.eye(4); theta = np.pi/4
    a[:2, :2] = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    n = nib.Nifti1Image(np.ones((20, 16, 2, 2), np.float32), a)
    n.set_sform(a, 1); n.set_qform(a, 1)
    path = tmp_path/'cine.nii'; nib.save(n, path)
    cine, meta = read_cine(path)
    assert meta['patient_left_axis'] == ''
    g = header_geometry(meta, cine.shape[-2:], (8, 8))
    assert g['valid'] and g['selected_form'] == 'sform' and g['sform_qform_agree']
    assert g['left_vector_yx'] == pytest.approx([2/np.sqrt(2), -2.5/np.sqrt(2)])
    n.set_sform(np.eye(4), 1); nib.save(n, path)
    with pytest.raises(ValueError, match='changed'): header_geometry(meta, (16, 20), (8, 8))


def test_45_degree_readout_repair_and_invalid_geometry_still_abstains(monkeypatch):
    import self_audit_guidance.core as core
    y, x = np.mgrid[:40, :40]; radius = np.hypot(x-18, y-20)
    labels = np.zeros((40, 40), int)
    labels[radius < 10] = 1; labels[radius < 6] = 2
    labels[np.hypot(x-28, y-20) < 3] = 3
    image = np.array([-.5, .2, 1.5, 1.5], np.float32)[labels][None, None]
    feat = np.eye(4, dtype=np.float32)[labels][None, None]
    monkeypatch.setattr(core, '_cluster', lambda *args: labels[None, None])
    cfg = GuidanceConfig(grid=40, feature_dim=4, clusters=(4,), min_region_pixels=3)
    baseline = anatomical_anchors(image, feat, '', cfg)
    repaired = anatomical_anchors(image, feat, '', cfg, left_vector_yx=(1., -1.))
    invalid = anatomical_anchors(image, feat, '+x', cfg, left_vector_yx=(0., 0.))
    assert baseline[2]['rv_hypotheses'] == 0
    assert baseline[2]['wall_cavity_hypotheses'] > 0
    assert repaired[2]['rv_hypotheses'] > 0 and (repaired[0][..., 1] > .25).any()
    assert invalid[2]['rv_hypotheses'] == 0  # never falls back to supplied cardinal side
    old = anatomical_anchors(image, feat, '-x', cfg)
    aligned = anatomical_anchors(image, feat, '-x', cfg, left_vector_yx=(0., -2.))
    assert np.array_equal(old[0], aligned[0]) and np.array_equal(old[1], aligned[1])


@pytest.mark.parametrize('enabled', [True, False])
def test_ce_entropy_kl_match_actual_loss_and_gradients_unchanged(enabled):
    torch.manual_seed(8); model = GuidanceStudent(4, 8)
    x = torch.randn(2, 3, 12, 20); out = model(x)
    q = torch.rand(2, 4, 6, 7); q[:, 3] += 2; q /= q.sum(1, keepdim=True)
    q[0, :, 0, 0] = torch.tensor([1., 0, 0, 0])
    w = torch.rand(2, 6, 7); w[0, 0, 0] = 0
    fl, sl, active = guidance_loss(out, torch.randn(2, 4, 6, 7), q, w, enabled)
    terms = semantic_terms(out['logits'], q, w, enabled)
    assert terms['active'] == active
    assert float(terms['ce'].detach()) == pytest.approx(float(sl.detach()), abs=1e-6)
    assert float(terms['ce'].detach()) == pytest.approx(float((terms['entropy']+terms['kl']).detach()), abs=1e-6)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    norms = shared_gradient_norms(model, fl, sl)
    assert norms['feature_norm'] > 0
    assert (norms['semantic_norm'] > 0) == enabled
    assert all(p.grad is None for p in model.parameters())
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())


def test_inactive_batches_not_omitted_from_original_epoch_mean():
    q = torch.tensor([.1, .1, .1, .7]).reshape(1, 4, 1, 1)
    active = scalar_terms(semantic_terms(q.log(), q, torch.ones(1, 1, 1)))
    inactive = scalar_terms(semantic_terms(q.log(), q, torch.zeros(1, 1, 1)))
    result = epoch_terms([active, inactive])
    assert result['active_fraction'] == .5
    assert result['ce'] == pytest.approx(active['entropy']/2)
    assert abs(result['kl']) < 1e-6


def test_coarse_nullspace_changes_native_but_not_fixed_projection():
    result = cli.probes()['nullspace']
    assert result['native_argmax_changed_pixels'] == 2
    assert result['coarse_max_difference'] < 1e-6
    q = torch.tensor([.1, .1, .1, .7]).reshape(1, 4, 1, 1)
    x = q.log().expand(1, 4, 2, 2).clone(); y = x.clone()
    y[:, 1] += torch.tensor([[20., -20.], [-20., 20.]])
    assert torch.allclose(resolution_control(x, (1, 1))[1], resolution_control(y, (1, 1))[1])


def test_unknown_training_mass_and_class_permissions_are_different():
    q = np.broadcast_to([.1, .1, .2, .6], (1, 1, 2, 2, 4)).copy()
    data = dict(q=q, r0=q, reliability=np.ones((1, 1, 2, 2))*.2,
                labels=np.full((1, 1, 2, 2), 255))
    result = bank_statistics(data, [True]*4, GuidanceConfig().dictionary())
    assert result['classes'][3]['soft_weight_on_unknown'] == 4
    assert result['classes'][3]['hard_accepted'] == 0
    assert result['classes'][1]['target_mass'] > 0  # all soft classes on a weighted pixel
    blocked = bank_statistics(data, [True, True, True, False], GuidanceConfig().dictionary())
    assert all(c['target_mass'] == 0 for c in blocked['classes'])
    p = torch.tensor([.02, .02, .02, .94]).reshape(1, 4, 1, 1)
    labels, _ = gate_probabilities(p, GuidanceConfig().dictionary(), [True]*4, True)
    assert labels.item() == 255


def test_metric_reduction_preserves_nulls_patient_weight_and_strata():
    def row(patient, frame, tp, fn, recovery=False):
        counts = {c: dict(tp=0, fp=0, fn=0) for c in cli.CLASSES}
        counts['LV'] = dict(tp=tp, fp=0, fn=fn)
        return dict(patient_id=patient, frame_index=frame,
                    metrics=dict(classes=counts, coverage=1. if tp else 0.),
                    geometry_verification={'policy':'test'} if recovery else None)
    rows = [row('a', 0, 4, 0), row('a', 1, 4, 0), row('b', 0, 0, 4, True)]
    result, flat = cli.counts_summary(rows)
    assert result['foreground']['dice'] == .5  # not 2/3 pooled or frame average
    assert result['foreground']['precision'] == 1
    assert result['defined_patients']['LV']['precision'] == 1
    assert result['classes']['RV']['dice'] is None
    assert flat[-1]['stratum'] == 'verified_recovery'
    with pytest.raises(ValueError, match='duplicate'): cli.counts_summary(rows+[rows[0]])


def test_missing_artifacts_are_blocked_and_frozen_output_refused(tmp_path):
    result = cli.audit_run(tmp_path/'absent', 1, 1, cli.EXPECTED_CHECKPOINT)
    assert result['status'] == 'BLOCKED' and 'MISSING RECIPE.json' in result['blockers']
    frozen = tmp_path/'frozen'; frozen.mkdir(); freeze(frozen, {'role':'test'})
    with pytest.raises(SystemExit): cli.main(['probes', '--out', str(frozen/'new')])
    assert not (frozen/'new').exists()


def test_opt_in_logging_does_not_change_training_or_frozen_bank(tmp_path):
    bank = tmp_path/'bank'; bank.mkdir()
    rng = np.random.default_rng(3)
    q = np.zeros((2, 1, 8, 8, 4), np.float32); q[:] = [.1, .1, .1, .7]
    pids = ['p1', 'p2']; cfg = GuidanceConfig(grid=8, feature_dim=4).dictionary()
    reports = {p: dict(class_pixels=[0, 0, 0, 128]) for p in pids}
    for pid in pids:
        np.savez_compressed(bank/(pid+'.npz'), image=rng.normal(size=(2, 1, 12, 12)).astype(np.float32),
            features=rng.normal(size=(2, 1, 8, 8, 4)).astype(np.float16), q=q, r0=q,
            reliability=np.full((2, 1, 8, 8), .1, np.float32), labels=np.full((2, 1, 8, 8), 3, np.uint8))
    freeze(bank, dict(role='guidance_bank', split='train', patients=pids,
        locked_splits=dict(train=pids, val=['v'], test=['t']), config=cfg,
        patient_reports=reports, teacher={'frozen':True, 'provider':'manufactured'},
        image_records={p: dict(raw_image=str(tmp_path/(p+'_absent.nii')), image_sha256=p,
                               shape_xyzt=[12, 12, 1, 2], affine=np.eye(4).tolist(),
                               patient_left_axis='-x') for p in pids}))
    before = digest(bank/'FROZEN.json')
    a = train_student(bank, tmp_path/'baseline', epochs=1, batch_size=2, device='cpu', width=8, threads=1)
    b = train_student(bank, tmp_path/'logged', epochs=1, batch_size=2, device='cpu', width=8, threads=1, diagnostics=True)
    assert all(torch.equal(a['model'][k], b['model'][k]) for k in a['model'])
    assert before == digest(bank/'FROZEN.json')
    import json
    row = json.loads((tmp_path/'logged/training.json').read_text())[0]
    d = row['semantic_diagnostics']
    assert d['ce'] == pytest.approx(row['semantic_loss'], abs=1e-6)
    assert d['ce'] == pytest.approx(d['entropy']+d['kl'], abs=1e-6)
    activity = {p: np.ones((2, 1), bool) for p in pids}
    schedule = replay_active_schedule(activity, pids, 42, 1, 2)
    assert sum(r['active_batches'] for r in schedule) == b['semantic_updates']
    # Exercise the full read-only checkpoint diagnostic on manufactured artifacts.
    import shutil
    from self_audit_guidance.io import verify
    shutil.copytree(tmp_path/'logged', tmp_path/'student')
    m = verify(bank, 'guidance_bank')
    recipe = dict(config=m['config'], batch_size=2, seed=42, epochs=1)
    write_json(tmp_path/'RECIPE.json', recipe)
    write_json(tmp_path/'EFFECTIVE_SPLIT.json', m['locked_splits'])
    write_json(tmp_path/'PREFLIGHT_test.json', dict(status='PASS_EXECUTION_ONLY_QUALITY_UNVERIFIED',
        checks=dict(native_cine_inventory=dict(splits=m['locked_splits'], images=m['image_records']))))
    locks = {}
    for split in ('val', 'test'):
        folder = tmp_path/f'predictions_{split}'; folder.mkdir()
        freeze(folder, dict(role='student_predictions', split=split))
        locks[split] = digest(folder/'FROZEN.json')
    write_json(tmp_path/'PREDICTIONS_LOCKED.json', dict(predictions=locks))
    result = cli.audit_run(tmp_path, 2, 1, digest(tmp_path/'student/last.pt'))
    assert result['status'] == 'BOUNDED_TRAIN_FIDELITY_AUDIT'
    assert result['fixed_bank_activation_schedule'][0]['active_batches'] == b['semantic_updates']
    assert result['patients']['p1']['geometry']['status'] == 'BLOCKED'
    write_json(tmp_path/'diagnostic_result.json', result)  # all values JSON serializable
    assert before == digest(bank/'FROZEN.json')


def test_opt_in_prepare_records_geometry_without_default_change_or_reference_reads(tmp_path):
    import nibabel as nib
    from self_audit_guidance.pipeline import prepare
    from self_audit_guidance.io import verify
    class ManufacturedTeacher:
        provenance = {'frozen':True, 'provider':'manufactured_no_accuracy_claim'}
        def extract(self, cine, grid, feature_dim, tile_batch):
            return np.ones((*cine.shape[:2], grid, grid, feature_dim), np.float32)
    root = tmp_path/'images'; pid = 'p1'; (root/pid).mkdir(parents=True)
    a = np.eye(4); a[:2, :2] = [[2**-.5, -2**-.5], [2**-.5, 2**-.5]]
    nii = nib.Nifti1Image(np.random.default_rng(7).normal(size=(20, 16, 1, 2)).astype(np.float32), a)
    nii.set_sform(a, 1); nib.save(nii, root/pid/(pid+'_4d.nii'))
    # Invalid reference-like files prove discovery does not open them.
    (root/pid/(pid+'_gt.nii')).write_text('must never be opened')
    (root/pid/'Info.cfg').write_text('must never be opened')
    split = tmp_path/'split.json'; write_json(split, dict(train=[pid], val=['v'], test=['t']))
    cfg = tmp_path/'cfg.json'; write_json(cfg, GuidanceConfig(grid=8, feature_dim=4, clusters=(2,)).dictionary())
    prepare(root, 'acdc', split, tmp_path/'world_bank', cfg, patients=0, device='cpu',
            teacher=ManufacturedTeacher(), orientation_policy='world_left_v2')
    m = verify(tmp_path/'world_bank', 'guidance_bank')
    assert m['orientation_policy'] == 'world_left_v2'
    assert m['image_records'][pid]['world_left_geometry']['valid']
    assert m['image_records'][pid]['patient_left_axis'] == ''  # original metadata retained
    prepare(root, 'acdc', split, tmp_path/'legacy_bank', cfg, patients=0, device='cpu', teacher=ManufacturedTeacher())
    old = verify(tmp_path/'legacy_bank', 'guidance_bank')
    assert 'orientation_policy' not in old and 'world_left_geometry' not in old['image_records'][pid]
