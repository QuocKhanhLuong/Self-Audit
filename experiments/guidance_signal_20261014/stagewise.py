"""Frozen, bounded VAL-only stage diagnostics using the private executed source.

GENERATE has no reference arguments. SCORE verifies the complete all-patient,
all-stage freeze before reference resolution. All arrays stay on the execution
host. Only scalar CSV/JSON summaries are suitable for publication. No training,
permission update, threshold sweep, teacher download, or TEST operation exists.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from . import runtime
from .stagewise_table import CLASSES, STAGES, count_metrics, guard_output, summarize

CHECKPOINT_SHA256 = '90bc16bb9f89963b452355376673e3ba9aa24b38870e4595237d75e402ce873e'
SCHEMA = 'self_audit.stagewise_diagnostics.v1'
R0_DEFINITION = ('Original core.make_bank R0: fixed CineMA features plus anatomy-prior votes, '
                 'not a pretrained semantic head or trained stage; raw=np.argmax (lowest class wins ties); '
                 'diagnostic R0 gate uses original probability/margin thresholds and the same Q-derived '
                 'propagated reliability as make_bank, without student permissions.')
MAPPING = 'nearest_floor_yx: native[y,x]=coarse[floor(y*Gh/H),floor(x*Gw/W)]; T/Z unchanged; UNKNOWN=255 preserved'
PROJECTION = 'student common-grid softmax probabilities bilinear to native H/W, align_corners=False, then original student gates'


def _json(io, path):
    return io.read_json(Path(path))


def _fresh(out, inputs=()):
    return guard_output(out, inputs=inputs)


def _published(rt, relative):
    """Verify published text bytes against the hash-pinned publication manifest."""
    base = runtime.SNAPSHOT_ROOT
    manifest = _json(rt.io, base / 'MANIFEST.json')
    entries = {r['published_path']: r for r in manifest['files']}
    entry = entries[relative]
    path = base / relative
    if rt.io.digest(path) != entry['published_sha256']:
        raise ValueError('published evidence changed: ' + relative)
    return _json(rt.io, path), entry['original_sha256']


def _cohort(rt):
    recipe, _ = _published(rt, 'run_evidence/run/RECIPE.json')
    ids = recipe['splits']['val']
    if len(ids) != 18 or len(set(ids)) != 18:
        raise ValueError('locked VAL18 missing')
    return ids


def experiment_hashes(io):
    base = Path(__file__).parent
    return {name: io.digest(base / name) for name in ('stagewise.py', 'stagewise_table.py', 'runtime.py')}


def native_labels(labels, native_hw):
    """Predetermined nearest-neighbor label mapping; never interpolates labels/GT."""
    labels = np.asarray(labels)
    if labels.ndim != 4 or not np.isin(labels, [0, 1, 2, 3, 255]).all():
        raise ValueError('TZYX named/UNKNOWN labels required')
    h, w = native_hw
    if min(h, w) < 1:
        raise ValueError('invalid native grid')
    yy = np.arange(h) * labels.shape[-2] // h
    xx = np.arange(w) * labels.shape[-1] // w
    return labels[:, :, yy[:, None], xx[None, :]].astype(np.uint8)


def gated(probability, cfg, *, reliability=None, allowed=None, enabled=True):
    p = np.asarray(probability)
    if p.ndim != 5 or p.shape[-1] != 4 or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(-1), 1, atol=2e-5, rtol=0):
        raise ValueError('finite TZYXC simplex probabilities required')
    top = np.sort(p, axis=-1)
    raw = p.argmax(-1).astype(np.uint8)
    masks = dict(probability=top[..., -1] >= cfg['min_probability'],
                 margin=top[..., -1] - top[..., -2] >= cfg['min_margin'])
    if reliability is not None:
        if reliability.shape != raw.shape or not np.isfinite(reliability).all():
            raise ValueError('reliability grid mismatch')
        masks['reliability'] = reliability >= cfg['min_reliability']
    if allowed is not None:
        if len(allowed) != 4 or any(type(x) is not bool for x in allowed):
            raise ValueError('original four class permissions required')
        masks['permission'] = np.asarray(allowed, dtype=bool)[raw]
    valid = np.full(raw.shape, bool(enabled))
    for mask in masks.values():
        valid &= mask
    return raw, np.where(valid, raw, 255).astype(np.uint8), masks


def gate_scalars(pid, stage, raw, selected, masks):
    rows = []
    for c, name in enumerate(CLASSES):
        domain = raw == c
        n = int(domain.sum())
        row = dict(patient_id=pid, stage=stage, class_name=name, raw_argmax_pixels=n,
                   accepted_pixels=int(((selected == c) & domain).sum()),
                   rejected_pixels=int(((selected == 255) & domain).sum()),
                   raw_argmax_fraction=n / raw.size,
                   selection_rate=int((selected == c).sum()) / n if n else None)
        row.update({k + '_pass_pixels': int((v & domain).sum()) for k, v in masks.items()})
        rows.append(row)
    return rows


def fidelity_scalars(pid, q, student, reliability, allowed, student_log_prob=None):
    """Class strata are Q argmax BEFORE confidence/permission thresholds."""
    if q.shape != student.shape or q.shape[:-1] != reliability.shape:
        raise ValueError('fidelity requires aligned common-grid probabilities')
    if not np.isfinite(q).all() or not np.isfinite(student).all() or (q < 0).any() or (student < 0).any():
        raise ValueError('invalid probability support')
    log_q = np.zeros_like(q, dtype=np.float64)
    positive = q > 0
    log_q[positive] = np.log(q[positive])
    if student_log_prob is None:
        log_p = np.full_like(student, -np.inf, dtype=np.float64)
        log_p[student > 0] = np.log(student[student > 0])
    else:
        log_p = np.asarray(student_log_prob, dtype=np.float64)
        if log_p.shape != q.shape or np.isnan(log_p).any() or np.isposinf(log_p).any():
            raise ValueError('invalid common-grid log probabilities')
    terms = np.zeros_like(log_q)
    terms[positive] = q[positive] * (log_q[positive] - log_p[positive])
    kl = terms.sum(-1)
    target, prediction = q.argmax(-1), student.argmax(-1)
    effective = reliability * np.asarray(allowed)[target]
    rows = []
    for c, name in enumerate(CLASSES):
        m = target == c; n = int(m.sum()); weight = float(effective[m].sum(dtype=np.float64))
        infinite = int(np.isposinf(kl[m]).sum())
        weighted_domain = m & (effective > 0)
        weighted_infinite = int(np.isposinf(kl[weighted_domain]).sum())
        rows.append(dict(patient_id=pid, class_name=name, stratum='Q_RAW_ARGMAX_BEFORE_THRESHOLDS',
                         pixels=n, kl_q_to_student=float(kl[m].mean()) if n and not infinite else None,
                         kl_status='INFINITE_SUPPORT_MISMATCH' if infinite else ('FINITE' if n else 'EMPTY_STRATUM'),
                         infinite_kl_pixels=infinite, kl_definition='sum Q*(log Q-log P); 0 log(0/P)=0; no clipping',
                         effective_weight_mass=weight,
                         weighted_kl=float((kl[weighted_domain] * effective[weighted_domain]).sum(dtype=np.float64) / weight) if weight > 0 and not weighted_infinite else None,
                         weighted_kl_status='INFINITE_SUPPORT_MISMATCH' if weighted_infinite else ('FINITE' if weight > 0 else 'ZERO_WEIGHT'),
                         argmax_agreement=float((target[m] == prediction[m]).mean()) if n else None,
                         original_permission=bool(allowed[c])))
    return rows


def _student_probabilities(rt, model, ck, cine, device, batch_size, return_log_prob=False):
    import torch
    from torch.nn import functional as F
    grid = ck['config']['grid']; t, z, h, w = cine.shape
    common = np.empty((t, z, grid, grid, 4), np.float32)
    native = np.empty((t, z, h, w, 4), np.float32)
    common_logp = np.empty_like(common)
    ids = list(np.ndindex((t, z)))
    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            current = ids[start:start + batch_size]
            x = torch.from_numpy(rt.student.contexts(cine, current)).to(device)
            logits = model(x)['logits'].float()
            pn = logits.softmax(1).permute(0, 2, 3, 1).cpu().numpy()
            # Exactly the training readout: interpolate logits, then softmax.
            coarse_logits = F.interpolate(logits, (grid, grid), mode='bilinear', align_corners=False)
            pc = coarse_logits.softmax(1).permute(0, 2, 3, 1).cpu().numpy()
            logpc = coarse_logits.log_softmax(1).permute(0, 2, 3, 1).cpu().numpy()
            for j, (ti, zi) in enumerate(current):
                native[ti, zi], common[ti, zi], common_logp[ti, zi] = pn[j], pc[j], logpc[j]
    return (common, native, common_logp) if return_log_prob else (common, native)


def stage_arrays(bank, common, native, ck):
    """Pure tensor stage construction, also used by synthetic local preflight."""
    import torch
    from torch.nn import functional as F
    cfg = ck['config']; allowed = ck['allowed_classes']
    enabled = ck['status'] != 'REPRESENTATION_ONLY'
    h, w = native.shape[-3:-1]
    source = dict(r0=bank['r0'], q=bank['q'], student_common=common, native=native)
    flat = torch.from_numpy(common).flatten(0, 1).permute(0, 3, 1, 2)
    projection = F.interpolate(flat, (h, w), mode='bilinear', align_corners=False).permute(0, 2, 3, 1).reshape(*native.shape).numpy()
    source['coarse_projection'] = projection
    outputs, gate_rows = {}, []
    for name, prob in source.items():
        teacher_stage = name in ('r0', 'q')
        raw, selected, masks = gated(prob, cfg,
            reliability=bank['reliability'] if teacher_stage else None,
            allowed=None if teacher_stage else allowed, enabled=True if teacher_stage else enabled)
        if name == 'q' and not np.array_equal(selected, bank['labels']):
            raise ValueError('Q gate reconstruction differs from original make_bank')
        raw_name = 'native_raw_argmax' if name == 'native' else name + '_raw'
        outputs[raw_name] = native_labels(raw, (h, w))
        outputs[name + '_gated'] = native_labels(selected, (h, w))
        gate_rows.extend(gate_scalars('', name, raw, selected, masks))
    if set(outputs) != set(STAGES):
        raise AssertionError('incomplete stage implementation')
    return outputs, gate_rows, projection


def _validate_original(rt, run, expected_checkpoint_sha256, teacher_checkpoint, teacher_config):
    io = rt.io; run = Path(run).resolve()
    expected_recipe, recipe_sha = _published(rt, 'run_evidence/run/RECIPE.json')
    locks = {}
    for rel, published in [('RECIPE.json', 'run_evidence/run/RECIPE.json'),
                           ('bank/FROZEN.json', 'run_evidence/run/bank/FROZEN.json'),
                           ('student/FROZEN.json', 'run_evidence/run/student/FROZEN.json'),
                           ('predictions_val/FROZEN.json', 'run_evidence/run/predictions_val/FROZEN.json')]:
        _, original_sha = _published(rt, published)
        path = run / rel
        if io.digest(path) != original_sha:
            raise ValueError('original run identity mismatch: ' + rel)
        locks[str(path)] = original_sha
    recipe = _json(io, run / 'RECIPE.json')
    bank = io.verify(run / 'bank', 'guidance_bank')
    student = io.verify(run / 'student', 'student')
    preds = _json(io, run / 'predictions_val/FROZEN.json')
    if expected_checkpoint_sha256 != CHECKPOINT_SHA256 or io.digest(run / 'student/last.pt') != expected_checkpoint_sha256:
        raise ValueError('original checkpoint SHA mismatch')
    if student['files']['last.pt'] != expected_checkpoint_sha256:
        raise ValueError('student manifest checkpoint mismatch')
    if student['bank_manifest_sha256'] != io.digest(run / 'bank/FROZEN.json'):
        raise ValueError('student TRAIN-bank identity mismatch')
    if recipe['config'] != expected_recipe['config'] or bank['config'] != recipe['config']:
        raise ValueError('original recipe thresholds/config changed')
    if recipe['source_sha256'] != rt.source_sha256 or bank['source_sha256'] != rt.source_sha256 or student['source_sha256'] != rt.source_sha256:
        raise ValueError('original run did not use the pinned executed source')
    if recipe['splits'] != expected_recipe['splits'] or bank['locked_splits'] != recipe['splits']:
        raise ValueError('original split mismatch')
    if bank['split'] != 'train' or bank['patients'] != recipe['splits']['train']:
        raise ValueError('full original TRAIN bank required')
    if preds['patients'] != recipe['splits']['val'] or preds['checkpoint_sha256'] != expected_checkpoint_sha256:
        raise ValueError('original VAL prediction identity mismatch')
    if recipe['dataset'] != 'acdc' or recipe.get('orientation_policy', 'cardinal_v1') != bank.get('orientation_policy', 'cardinal_v1'):
        raise ValueError('original dataset/geometry policy mismatch')
    for path, expected in [(teacher_checkpoint, recipe['expected_teacher_sha256']), (teacher_config, recipe['teacher_config_sha256'])]:
        if not Path(path).is_file() or io.digest(path) != expected:
            raise ValueError('local original teacher file hash mismatch; downloads are forbidden')
        locks[str(Path(path).resolve())] = expected
    locks[str(run / 'student/last.pt')] = expected_checkpoint_sha256
    return run, recipe, bank, student, preds, locks


def _write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _check_locks(io, locks):
    for path, expected in locks.items():
        if io.digest(path) != expected:
            raise ValueError('locked input changed: ' + path)


def generate(*, run, root, out, expected_checkpoint_sha256, teacher_checkpoint,
             teacher_config, device='cuda', split='val', private_arrays=False):
    if split != 'val':
        raise ValueError('only locked VAL18 is permitted; TEST excluded')
    out = _fresh(out, inputs=(run, root, teacher_checkpoint, teacher_config)); rt = runtime.load_runtime(); io = rt.io
    run, recipe, bank_manifest, student_manifest, old_predictions, locks = _validate_original(
        rt, run, expected_checkpoint_sha256, teacher_checkpoint, teacher_config)
    if run in out.parents or Path(root).resolve() in out.parents:
        raise ValueError('generation output must be separate from original run and image roots')
    experiment_source = experiment_hashes(io)
    locks.update({str(Path(__file__).parent / name): sha for name, sha in experiment_source.items()})
    patients = _cohort(rt)
    model, ck = rt.student.load_student(run / 'student', device)
    if ck['locked_splits'] != recipe['splits'] or ck['train_patients'] != bank_manifest['patients'] or ck['config'] != recipe['config']:
        raise ValueError('checkpoint recipe/cohort mismatch')
    class_patients = [sum(bank_manifest['patient_reports'][p]['class_pixels'][c] > 0 for p in bank_manifest['patients']) for c in range(4)]
    if ck['class_patients'] != class_patients or ck['allowed_classes'] != [n >= 2 for n in class_patients]:
        raise ValueError('original TRAIN-derived permissions inconsistent; no updates allowed')
    if ck['bank_manifest_sha256'] != io.digest(run / 'bank/FROZEN.json') or ck['teacher'] != bank_manifest['teacher']:
        raise ValueError('checkpoint source provenance mismatch')
    teacher = rt.teacher.FrozenCineMA(device=device, checkpoint=teacher_checkpoint, config=teacher_config, offline=True)
    if teacher.provenance != bank_manifest['teacher'] or teacher.provenance != student_manifest['teacher']:
        raise ValueError('teacher provenance differs from original run')
    records = io.discover(root, recipe['dataset'], patients)
    out.mkdir(parents=True, exist_ok=False)
    for stage in STAGES:
        (out / stage).mkdir()
    if private_arrays:
        (out / 'PRIVATE_ARRAYS_DO_NOT_PUBLISH').mkdir()
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(recipe['threads'])
    cfg = rt.core.GuidanceConfig(**recipe['config'])
    inventory, reports, gates, fidelity = {}, {}, [], []
    for pid in patients:
        cine, meta = io.read_cine(records[pid])
        original_meta = old_predictions['images'][pid]
        for key in ('image_sha256', 'shape_xyzt', 'affine', 'normalization'):
            if meta[key] != original_meta[key]:
                raise ValueError('original VAL image/preprocessing mismatch: ' + pid + ':' + key)
        # Read actual VAL image header codes; never borrow TRAIN patient strata.
        import nibabel as nib
        header = nib.load(str(records[pid])).header
        meta['qform_code'] = int(header['qform_code']); meta['sform_code'] = int(header['sform_code'])
        meta['val_coded_uncoded'] = 'coded' if meta['qform_code'] or meta['sform_code'] else 'uncoded'
        if cine.shape[0] * cine.shape[1] * cfg.grid ** 2 > cfg.max_nodes:
            raise ValueError('original graph node bound exceeded')
        features = teacher.extract(cine, grid=cfg.grid, feature_dim=cfg.feature_dim, tile_batch=recipe['tile_batch'])
        if features.shape != (*cine.shape[:2], cfg.grid, cfg.grid, cfg.feature_dim) or not np.isfinite(features).all():
            raise ValueError('original teacher feature reconstruction failed')
        small = F.interpolate(torch.from_numpy(cine.copy()).flatten(0, 1)[:, None],
                              (cfg.grid, cfg.grid), mode='bilinear', align_corners=False)[:, 0].reshape(*cine.shape[:2], cfg.grid, cfg.grid).numpy()
        geometry = {}
        if recipe.get('orientation_policy', 'cardinal_v1') == 'world_left_v2':
            meta['world_left_geometry'] = rt.experimental_geometry.header_geometry(meta, cine.shape[-2:], small.shape[-2:])
            geometry['left_vector_yx'] = meta['world_left_geometry']['left_vector_yx']
        elif recipe.get('orientation_policy', 'cardinal_v1') != 'cardinal_v1':
            raise ValueError('unknown original geometry policy')
        bank, report = rt.core.make_bank(small, features, meta['patient_left_axis'], cfg, **geometry)
        common, native, common_logp = _student_probabilities(rt, model, ck, cine, device, recipe['batch_size'], return_log_prob=True)
        outputs, gate_rows, projection = stage_arrays(bank, common, native, ck)
        # Independent check against the exact archived native prediction implementation.
        exact = rt.student.predict_cine(model, ck, cine, device, recipe['batch_size'])
        if not np.array_equal(outputs['native_gated'], exact):
            raise ValueError('native gated reconstruction differs from pinned predict_cine')
        for stage, labels in outputs.items():
            np.savez_compressed(out / stage / (pid + '.npz'), labels=labels)
        for row in gate_rows:
            row.update(patient_id=pid, val_coded_uncoded=meta['val_coded_uncoded'])
        gates.extend(gate_rows)
        fidelity.extend({**r, 'val_coded_uncoded': meta['val_coded_uncoded']} for r in fidelity_scalars(pid, bank['q'], common, bank['reliability'], ck['allowed_classes'], common_logp))
        if private_arrays:
            np.savez_compressed(out / 'PRIVATE_ARRAYS_DO_NOT_PUBLISH' / (pid + '.npz'),
                                r0=bank['r0'], q=bank['q'], reliability=bank['reliability'],
                                student_common=common, native=native, coarse_projection=projection)
        inventory[pid] = meta; reports[pid] = report
        locks[str(Path(records[pid]).resolve())] = meta['image_sha256']
        print('[STAGEWISE_GENERATE]', pid, 'complete', flush=True)
    _check_locks(io, locks)
    for stage in STAGES:
        io.freeze(out / stage, dict(role='student_predictions', diagnostic_stage=stage, split='val',
            patients=patients, images=inventory, student_status=ck['status'], locked_splits=ck['locked_splits'],
            train_patients=ck['train_patients'], checkpoint_sha256=expected_checkpoint_sha256,
            unknown=255, labels_axis_order='TZYX', quality_evaluated=False, gt_read=False))
    if private_arrays:
        io.freeze(out / 'PRIVATE_ARRAYS_DO_NOT_PUBLISH', dict(role='private_stage_arrays', patients=patients))
    _write_csv(out / 'GATE_SCALARS.csv', gates); _write_csv(out / 'FIDELITY_SCALARS.csv', fidelity)
    io.write_json(out / 'GENERATION.json', dict(schema=SCHEMA, status='GENERATED_NOT_SCORED', split='val',
        patients=patients, source_sha256=rt.source_sha256, source_commit=rt.snapshot_sha,
        experiment_source_sha256=experiment_source,
        checkpoint_sha256=expected_checkpoint_sha256, input_locks=locks, config=ck['config'],
        allowed_classes=ck['allowed_classes'], train_permissions_updated=False,
        r0_definition=R0_DEFINITION, coarse_native_mapping=MAPPING, projection_control=PROJECTION,
        geometry_policy=recipe.get('orientation_policy', 'cardinal_v1'), teacher=teacher.provenance,
        patient_specific_teacher_inference='DIAGNOSTIC_ONLY_NO_TRAINING_OR_PERMISSION_FEEDBACK',
        scalar_scope='all cine frames; scoring selects original reference frame indices later',
        patient_reports=reports, gt_read=False, test_excluded=True, private_arrays=private_arrays))
    subdirs = [*STAGES] + (['PRIVATE_ARRAYS_DO_NOT_PUBLISH'] if private_arrays else [])
    io.freeze(out, dict(role='stagewise_complete', split='val', patients=patients, stages=list(STAGES),
        stage_manifest_sha256={s: io.digest(out / s / 'FROZEN.json') for s in subdirs},
        source_sha256=rt.source_sha256, source_commit=rt.snapshot_sha,
        experiment_source_sha256=experiment_source, gt_read=False))
    verify_complete(out, rt)
    return dict(status='GENERATED_NOT_SCORED', patients=18, stages=len(STAGES), output=str(out))


def verify_complete(folder, rt=None):
    """No reference argument or operation; verify every payload before SCORE."""
    rt = rt or runtime.load_runtime(); io = rt.io; folder = Path(folder)
    if folder.is_symlink() or (folder / 'FROZEN.json').is_symlink():
        raise ValueError('symlink generation identity refused')
    m = io.verify(folder, 'stagewise_complete')
    if m.get('split') != 'val' or m.get('patients') != _cohort(rt) or m.get('stages') != list(STAGES):
        raise ValueError('complete locked VAL18 all-stage freeze required')
    if (m.get('source_sha256') != rt.source_sha256 or m.get('source_commit') != rt.snapshot_sha
            or m.get('experiment_source_sha256') != experiment_hashes(io) or m.get('gt_read') is not False):
        raise ValueError('frozen source/phase identity mismatch')
    manifests = m.get('stage_manifest_sha256', {})
    generation = _json(io, folder / 'GENERATION.json')
    expected_dirs = set(STAGES) | ({'PRIVATE_ARRAYS_DO_NOT_PUBLISH'} if generation.get('private_arrays') else set())
    if set(manifests) != expected_dirs or {p.name for p in folder.iterdir() if p.is_dir()} != expected_dirs:
        raise ValueError('stage directory inventory differs from complete freeze')
    original_recipe, _ = _published(rt, 'run_evidence/run/RECIPE.json')
    reference_inventory = None
    for name in sorted(expected_dirs):
        sub = folder / name
        if sub.is_symlink() or io.digest(sub / 'FROZEN.json') != manifests[name]:
            raise ValueError('stage manifest changed: ' + name)
        sm = io.verify(sub, 'private_stage_arrays' if name.startswith('PRIVATE_') else 'student_predictions')
        if sm.get('patients') != m['patients'] or set(sm['files']) != {p + '.npz' for p in m['patients']}:
            raise ValueError('incomplete per-patient stage payloads')
        if name.startswith('PRIVATE_'):
            continue
        if sm.get('split') != 'val' or sm.get('diagnostic_stage') != name or sm.get('gt_read') is not False:
            raise ValueError('invalid frozen prediction stage')
        if sm.get('checkpoint_sha256') != CHECKPOINT_SHA256 or sm.get('locked_splits') != original_recipe['splits'] or sm.get('train_patients') != original_recipe['splits']['train']:
            raise ValueError('checkpoint or cohort leakage in stage freeze')
        if set(sm['images']) != set(m['patients']):
            raise ValueError('missing native image metadata')
        if reference_inventory is None:
            reference_inventory = sm['images']
        elif sm['images'] != reference_inventory:
            raise ValueError('stage image/native geometry identity mismatch')
        for pid in m['patients']:
            with np.load(sub / (pid + '.npz'), allow_pickle=False) as data:
                if set(data.files) != {'labels'}:
                    raise ValueError('unexpected stage payload schema')
                labels = data['labels']
                if labels.shape != tuple(reversed(sm['images'][pid]['shape_xyzt'])) or not np.isin(labels, [0, 1, 2, 3, 255]).all():
                    raise ValueError('invalid native stage labels')
    return m


def score(*, generated, out, reference_root=None, reference_manifest=None,
          allow_verified_acdc_frame_grid=False, split='val'):
    if split != 'val':
        raise ValueError('TEST excluded')
    rt = runtime.load_runtime(); io = rt.io
    complete = verify_complete(generated, rt)  # MUST precede reference resolution/read.
    out = _fresh(out, inputs=[generated] + [p for p in (reference_root, reference_manifest) if p is not None])
    frozen_sha = io.digest(Path(generated) / 'FROZEN.json')
    if Path(generated).resolve() in out.parents:
        raise ValueError('SCORE output must not mutate the generation directory')
    # Freeze verification above completes before any reference-manifest/root access.
    expected, _ = _published(rt, 'run_evidence/run/evaluation_val/METRICS.json')
    expected_refs = {(f['patient_id'], f['frame_index']): f['reference_sha256'] for f in expected['frames']}
    pm = _json(io, Path(generated) / 'native_gated/FROZEN.json')
    entries = rt.evaluate.resolve_references(pm, reference_root, reference_manifest)
    if {(e['patient_id'], e['frame_index']) for e in entries} != set(expected_refs):
        raise ValueError('reference frames differ from original locked VAL36 evaluation')
    for entry in entries:
        if io.digest(entry['path']) != expected_refs[(entry['patient_id'], entry['frame_index'])]:
            raise ValueError('reference bytes differ from original locked VAL evaluation')
    out.mkdir(parents=True, exist_ok=False)
    rows, summaries, reference_identity = [], {}, None
    for stage in STAGES:
        # Reuse original evaluator including exact frame/cine identity recovery,
        # zero-based frame checks, no GT resize, no label permutation, and UNKNOWN FN.
        report = rt.evaluate.evaluate(Path(generated) / stage, out / stage,
            reference_root=reference_root, reference_manifest=reference_manifest,
            allow_verified_acdc_frame_grid=allow_verified_acdc_frame_grid)
        frames = report['frames']
        if len(frames) != 36 or {r['patient_id'] for r in frames} != set(complete['patients']) or any(sum(r['patient_id'] == pid for r in frames) != 2 for pid in complete['patients']):
            raise ValueError('SCORE requires original 18 patients / 36 labeled frames')
        identity = {(r['patient_id'], r['frame_index']): r['reference_sha256'] for r in frames}
        if reference_identity is not None and identity != reference_identity:
            raise ValueError('reference frame/array identity changed across stages')
        reference_identity = identity
        pm = _json(io, Path(generated) / stage / 'FROZEN.json')
        for frame in frames:
            stratum = 'verified_recovery' if frame['geometry_verification'] is not None else 'aligned'
            for name, value in frame['metrics']['classes'].items():
                rows.append(dict(stage=stage, availability='AVAILABLE', patient_id=frame['patient_id'],
                    frame_index=frame['frame_index'], class_name=name, geometry_stratum=stratum,
                    val_coded_uncoded=pm['images'][frame['patient_id']].get('val_coded_uncoded', 'UNKNOWN'),
                    **count_metrics(value['tp'], value['fp'], value['fn']),
                    frame_coverage=frame['metrics']['coverage'], hd95_mm=value['hd95_mm'],
                    surface_status=value['surface_status'], old_new_error_regions_status='UNKNOWN',
                    joint_correctness_status='UNKNOWN'))
        stage_rows = [r for r in rows if r['stage'] == stage]
        summaries[stage] = dict(overall=report['summary'], count_summary=summarize(stage_rows),
            count_strata={s: summarize([r for r in stage_rows if r['geometry_stratum'] == s])
                          for s in sorted({r['geometry_stratum'] for r in stage_rows})}, strata={s: rt.evaluate.aggregate_scores([
            f for f in frames if ('verified_recovery' if f['geometry_verification'] is not None else 'aligned') == s])
            for s in ('aligned', 'verified_recovery')})
    verify_complete(generated, rt)
    if io.digest(Path(generated) / 'FROZEN.json') != frozen_sha:
        raise ValueError('complete generation freeze changed during scoring')
    # Keep recovery/aligned strata on scalar-only host exports; mixed patients stay explicit.
    geometry_by_patient = {}
    for row in rows:
        geometry_by_patient.setdefault(row['patient_id'], set()).add(row['geometry_stratum'])
    for filename in ('GATE_SCALARS.csv', 'FIDELITY_SCALARS.csv'):
        with (Path(generated) / filename).open(newline='') as f:
            source_rows = list(csv.DictReader(f))
        for row in source_rows:
            strata = geometry_by_patient[row['patient_id']]
            row['evaluation_geometry_stratum'] = next(iter(strata)) if len(strata) == 1 else 'mixed'
            row['scalar_scope'] = 'all_cine_frames_patient_class_summary_not_only_reference_frames'
        _write_csv(out / filename, source_rows)
    _write_csv(out / 'STAGEWISE_DIAGNOSTICS.csv', rows)
    io.write_json(out / 'STAGEWISE_AVAILABILITY.json', dict(schema=SCHEMA,
        status='SCORED_FROZEN_VAL_DIAGNOSTICS', generation_manifest_sha256=frozen_sha,
        patient_count=18, labeled_frames=36, summaries=summaries,
        limitations=['No paired baseline arrays: old/new error regions and joint correctness remain UNKNOWN.',
                     'Diagnostics do not authorize threshold, class permission, or model selection changes.'],
        test_excluded=True, references_used_for_selection=False))
    io.freeze(out, dict(role='stagewise_scored', split='val', generation_manifest_sha256=frozen_sha,
        stage_manifest_sha256={s: io.digest(out / s / 'FROZEN.json') for s in STAGES}))
    return dict(status='SCORED_FROZEN_VAL_DIAGNOSTICS', rows=len(rows), output=str(out))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='phase', required=True)
    a = sub.add_parser('GENERATE')
    for key in ('run', 'root', 'out', 'expected-checkpoint-sha256', 'teacher-checkpoint', 'teacher-config'):
        a.add_argument('--' + key, required=True)
    a.add_argument('--device', default='cuda'); a.add_argument('--split', choices=['val'], default='val')
    a.add_argument('--private-arrays', action='store_true')
    a = sub.add_parser('SCORE'); a.add_argument('--generated', required=True); a.add_argument('--out', required=True)
    g = a.add_mutually_exclusive_group(required=True)
    g.add_argument('--reference-root'); g.add_argument('--reference-manifest')
    a.add_argument('--allow-verified-acdc-frame-grid', action='store_true')
    a.add_argument('--split', choices=['val'], default='val')
    args = vars(p.parse_args(argv)); phase = args.pop('phase')
    result = generate(**args) if phase == 'GENERATE' else score(**args)
    print(json.dumps(result, sort_keys=True)); return 0


if __name__ == '__main__':
    raise SystemExit(main())
