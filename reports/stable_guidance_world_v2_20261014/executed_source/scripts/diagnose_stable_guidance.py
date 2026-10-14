#!/usr/bin/env python3
"""Isolated forensics. No training, threshold fitting or reference-mask reads."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from self_audit_guidance.io import digest, read_json, write_json, verify, load_bank_patient
from self_audit_guidance.experimental_geometry import world_left_geometry, header_geometry
from self_audit_guidance.experimental_diagnostics import (
    semantic_terms, scalar_terms, shared_gradient_norms, bank_statistics,
    resolution_control, gate_probabilities, replay_active_schedule)

EXPECTED_CHECKPOINT = '516e569c9141e2d8edb00b598df5cf12c8a9a4741cf1ae0cca8cbf23d6d03353'
CLASSES = ('BG', 'RV', 'MYO', 'LV')


def mean(values):
    present = [x for x in values if x is not None]
    return sum(present)/len(present) if present else None


def counts_summary(frames):
    """Independent reduction from counts; never copy cached metric values."""
    patients, flat = {}, []
    seen = set()
    for row in frames:
        key = row['patient_id'], row['frame_index']
        if key in seen:
            raise ValueError('duplicate patient/frame count record')
        seen.add(key)
        patient = patients.setdefault(key[0], {c: [] for c in CLASSES})
        for c in CLASSES:
            counts = row['metrics']['classes'][c]
            tp, fp, fn = [counts[k] for k in ('tp', 'fp', 'fn')]
            if any(type(v) is not int or v < 0 for v in (tp, fp, fn)):
                raise ValueError('invalid confusion counts')
            values = dict(dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                          iou=tp/(tp+fp+fn) if tp+fp+fn else None,
                          precision=tp/(tp+fp) if tp+fp else None,
                          recall=tp/(tp+fn) if tp+fn else None)
            patient[c].append(values)
            flat.append(dict(patient_id=key[0], frame_index=key[1], class_name=c,
                             tp=tp, fp=fp, fn=fn, **values,
                             stratum='verified_recovery' if row.get('geometry_verification') else 'aligned'))
    per_patient = {p: {c: {m: mean([r[m] for r in rows])
                            for m in ('dice', 'iou', 'precision', 'recall')}
                       for c, rows in groups.items()} for p, groups in patients.items()}
    classes = {c: {m: mean([p[c][m] for p in per_patient.values()])
                   for m in ('dice', 'iou', 'precision', 'recall')} for c in CLASSES}
    coverage = mean([mean([r['metrics']['coverage'] for r in frames if r['patient_id'] == p])
                     for p in patients])
    return dict(patient_count=len(patients), frames=len(frames), classes=classes,
                foreground={m: mean([classes[c][m] for c in CLASSES[1:]])
                            for m in ('dice', 'iou', 'precision', 'recall')},
                coverage=coverage, per_patient=per_patient,
                defined_patients={c: {m: sum(p[c][m] is not None for p in per_patient.values())
                                      for m in ('dice', 'iou', 'precision', 'recall')} for c in CLASSES}), flat


def metrics_audit(folder):
    manifest = verify(folder, 'independent_evaluation')
    report = read_json(Path(folder)/'METRICS.json')
    if report['split'] != 'val' or manifest['split'] != 'val':
        raise ValueError('this development diagnostic permits exposed VAL only, never TEST')
    if manifest['prediction_manifest_sha256'] != report['prediction_manifest_sha256']:
        raise ValueError('evaluation prediction binding mismatch')
    overall, flat = counts_summary(report['frames'])
    strata = {}
    for label, recovered in [('aligned', False), ('verified_recovery', True)]:
        rows = [r for r in report['frames'] if bool(r.get('geometry_verification')) == recovered]
        strata[label] = counts_summary(rows)[0]
    expected = report['summary']['macro_foreground_dice']
    if not np.isclose(overall['foreground']['dice'], expected, atol=1e-12, rtol=0):
        raise ValueError('counts disagree with published evaluator Dice')
    return dict(status='COUNTS_RECOMPUTED_GEOMETRY_RECEIPTS_NOT_REEXECUTED', overall=overall,
                strata=strata, patient_frame_counts=flat,
                metrics_sha256=digest(Path(folder)/'METRICS.json'),
                reference_informed_development=True)


def probes():
    q = torch.tensor([.1, .1, .1, .7]).reshape(1, 4, 1, 1)
    logits = q.log().expand(1, 4, 2, 2).clone()
    altered = logits.clone()
    altered[:, 1] += torch.tensor([[20., -20.], [-20., 20.]])
    a = scalar_terms(semantic_terms(logits, q, torch.ones(1, 1, 1)))
    b = scalar_terms(semantic_terms(altered, q, torch.ones(1, 1, 1)))
    rotations = []
    for angle in (0, 30, 45, 60, 90):
        theta = np.deg2rad(angle)
        aff = np.eye(4)
        aff[:2, :2] = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        rotations.append(dict(angle_degrees=angle,
            legacy_cardinal_available=bool(np.max(np.abs(aff[0, :2])) >= .9),
            world_geometry=world_left_geometry(aff, (192, 256), (48, 48), sform_code=1, qform_code=0)))
    assert abs(a['ce']-b['ce']) < 1e-6
    return dict(status='MANUFACTURED_MATHEMATICAL_CHECKS_NOT_PATIENT_EVIDENCE',
                orientation=rotations, nullspace=dict(baseline=a, altered=b,
                    native_argmax_changed_pixels=int((logits.argmax(1) != altered.argmax(1)).sum()),
                    native_pixels=4, logit_perturbation=20.,
                    coarse_max_difference=float((F.interpolate(logits, (1, 1), mode='bilinear', align_corners=False)-
                                                F.interpolate(altered, (1, 1), mode='bilinear', align_corners=False)).abs().max())))


def audit_run(run, max_patients, max_batches, expected_sha):
    """Bounded checkpoint replay; no new bank, teacher downloads or optimizer."""
    from self_audit_guidance.student import load_student, contexts, guidance_loss
    run = Path(run).resolve()
    if (run/'run/RECIPE.json').exists():
        run = run/'run'
    result = dict(run_path=str(run), status='BLOCKED', blockers=[], max_patients=max_patients,
                  max_batches_per_patient=max_batches, reference_masks_read=False,
                  official_test_touched=False, model_updates=0)
    required = ['RECIPE.json', 'EFFECTIVE_SPLIT.json', 'PREDICTIONS_LOCKED.json',
                'bank/FROZEN.json', 'student/FROZEN.json', 'student/last.pt',
                'student/training.json', 'student/REPORT.json',
                'predictions_val/FROZEN.json', 'predictions_test/FROZEN.json']
    result['inventory'] = {name: digest(run/name) if (run/name).is_file() else None for name in required}
    result['preflight_paths'] = [p.name for p in sorted(run.glob('PREFLIGHT*.json'))]
    for name, value in result['inventory'].items():
        if value is None: result['blockers'].append('MISSING '+name)
    if not result['preflight_paths']: result['blockers'].append('MISSING PREFLIGHT')
    if result['blockers']: return result
    recipe = read_json(run/'RECIPE.json')
    result['resolved_recipe'] = recipe
    if digest(run/'student/last.pt') != expected_sha:
        raise ValueError('checkpoint hash does not match the requested run')
    bank = verify(run/'bank', 'guidance_bank')
    student = verify(run/'student', 'student')
    result['preflights'] = []
    matched_preflight = False
    for name in result['preflight_paths']:
        pf = read_json(run/name)
        inventory = pf.get('checks', {}).get('native_cine_inventory', {})
        matches = (pf.get('status') == 'PASS_EXECUTION_ONLY_QUALITY_UNVERIFIED'
                   and inventory.get('splits') == bank['locked_splits']
                   and all(inventory.get('images', {}).get(pid, {}).get('image_sha256') ==
                           bank['image_records'][pid]['image_sha256'] for pid in bank['patients']))
        matched_preflight |= matches
        result['preflights'].append(dict(file=name, sha256=digest(run/name), status=pf.get('status'),
                                         train_inventory_matches_bank=matches))
    if not matched_preflight:
        raise ValueError('no passing preflight binds the effective split and TRAIN image hashes')
    if student['bank_manifest_sha256'] != digest(run/'bank/FROZEN.json'):
        raise ValueError('student bound to a different bank')
    from self_audit_guidance.io import read_splits
    if read_splits(run/'EFFECTIVE_SPLIT.json') != bank['locked_splits']:
        raise ValueError('effective split disagrees with bank')
    model, ck = load_student(run/'student', 'cpu')
    if ck['locked_splits'] != bank['locked_splits'] or ck['train_patients'] != bank['patients']:
        raise ValueError('checkpoint split disagrees with bank')
    if ck['config'] != bank['config'] or recipe['config'] != bank['config']:
        raise ValueError('recipe/bank/student config mismatch')
    counts = [sum(bank['patient_reports'][p]['class_pixels'][c]>0 for p in bank['patients']) for c in range(4)]
    if ck['class_patients'] != counts or ck['allowed_classes'] != [n>=2 for n in counts]:
        raise ValueError('checkpoint permissions disagree with bank hard-label patient counts')
    if ck['bank_manifest_sha256'] != digest(run/'bank/FROZEN.json'):
        raise ValueError('checkpoint bound to a different bank')
    result.update(checkpoint_sha256=expected_sha, resolved_config=bank['config'],
                  semantic_updates_recorded=ck['semantic_updates'], allowed_classes=ck['allowed_classes'],
                  historical_active_fraction='UNAVAILABLE_UNLESS_LOGGED_NOT_INFERRED_FROM_CE',
                  training_history=read_json(run/'student/training.json'),
                  student_report=read_json(run/'student/REPORT.json'), patients={})
    # Verify frozen prediction identities without opening official TEST arrays.
    locks = read_json(run/'PREDICTIONS_LOCKED.json')
    for split in ('val', 'test'):
        if digest(run/f'predictions_{split}/FROZEN.json') != locks['predictions'][split]:
            raise ValueError('prediction pair lock mismatch')
    result['prediction_payload_verification'] = 'VAL/TEST manifest hashes only; arrays not read'
    result['source_receipts'] = dict(recipe=recipe.get('source_sha256'),
                                     bank=bank.get('source_sha256'), student=student.get('source_sha256'))
    frame_activity = {}
    for pid in bank['patients'][:max_patients]:
        data = load_bank_patient(run/'bank', pid)
        stats = bank_statistics(data, ck['allowed_classes'], bank['config'])
        winning = data['q'].argmax(-1)
        frame_activity[pid] = ((winning != 0) & (data['reliability'] > 0)
                              & np.asarray(ck['allowed_classes'])[winning]).any(axis=(-2, -1))
        meta = bank['image_records'][pid]
        try: stats['geometry'] = header_geometry(meta, data['image'].shape[-2:], data['q'].shape[-3:-1])
        except (OSError, ValueError) as exc: stats['geometry'] = dict(status='BLOCKED', reason=str(exc))
        stats['legacy_patient_left_axis'] = meta['patient_left_axis']
        stats['recorded_bank_report'] = bank['patient_reports'][pid]
        if [c['hard_accepted'] for c in stats['classes']] != bank['patient_reports'][pid]['class_pixels']:
            raise ValueError('bank hard counts disagree with frozen permissions source')
        stats['batches'] = []
        ids = list(np.ndindex(data['image'].shape[:2]))
        batch_size = recipe['batch_size']
        starts = list(range(0, len(ids), batch_size))
        eligible = [s for s in starts if any(frame_activity[pid][i] for i in ids[s:s+batch_size])]
        # No loss or reference is consulted. Fix the first eligible block for
        # meaningful CE/H/KL; retain an inactive block if no active one exists.
        if eligible:
            first = starts.index(eligible[0]); starts = starts[first:]+starts[:first]
        stats['batch_selection'] = 'first_eligible_block_then_cyclic_lexicographic_TZ_not_population_sampling'
        for start in starts[:max_batches]:
            indices = ids[start:start+batch_size]
            ti, zi = np.asarray(indices).T
            x = torch.from_numpy(contexts(data['image'], indices))
            target = torch.from_numpy(data['q'][ti, zi]).permute(0, 3, 1, 2)
            teacher = torch.from_numpy(data['features'][ti, zi].astype(np.float32)).permute(0, 3, 1, 2)
            weight = torch.from_numpy(data['reliability'][ti, zi])*torch.tensor(ck['allowed_classes'])[target.argmax(1)]
            output = model(x)
            fl, sl, active = guidance_loss(output, teacher, target, weight, any(ck['allowed_classes'][1:]))
            terms = scalar_terms(semantic_terms(output['logits'], target, weight, any(ck['allowed_classes'][1:])))
            if not np.isclose(terms['ce'], float(sl.detach()), atol=1e-6, rtol=0):
                raise ValueError('diagnostic CE does not match the actual loss')
            gradients = shared_gradient_norms(model, fl, sl)
            with torch.no_grad():
                native = output['logits'].softmax(1)
                coarse, projected = resolution_control(output['logits'], target.shape[-2:])
                gates = {}; maps = {}
                for name, p in [('native', native), ('coarse', coarse), ('projected_control', projected)]:
                    maps[name], gates[name] = gate_probabilities(p, ck['config'], ck['allowed_classes'], ck['status']=='REPRESENTATION_ONLY')
                    gates[name]['argmax_counts'] = [int((p.argmax(1) == c).sum()) for c in range(4)]
                fidelity = dict(coarse_argmax_agreement=float((coarse.argmax(1)==target.argmax(1)).float().mean()),
                    native_projected_argmax_disagreement=float((native.argmax(1)!=projected.argmax(1)).float().mean()),
                    native_projected_gated_disagreement=float((maps['native']!=maps['projected_control']).float().mean()))
            stats['batches'].append(dict(indices_tz=[list(v) for v in indices], semantic=terms,
                shared_encoder_gradients=gradients, gates=gates, fidelity_not_accuracy=fidelity))
        result['patients'][pid] = stats
    result.update(status='BOUNDED_TRAIN_FIDELITY_AUDIT', patients_inspected=len(result['patients']),
                  all_train_patients_inspected=len(result['patients'])==len(bank['patients']),
                  original_epoch_replay=False, actual_historical_gradients='NOT_RECONSTRUCTIBLE_FROM_FINAL_CHECKPOINT')
    if result['all_train_patients_inspected']:
        schedule = replay_active_schedule(frame_activity, bank['patients'], recipe['seed'],
                                          recipe['epochs'], recipe['batch_size'])
        if sum(r['active_batches'] for r in schedule) != ck['semantic_updates']:
            raise ValueError('fixed-bank batch replay disagrees with recorded semantic updates')
        result['fixed_bank_activation_schedule'] = schedule
        result['historical_active_fraction'] = 'REPLAYED_FROM_FIXED_BANK_AND_SEED_COUNTER_MATCHED_NOT_GRADIENT_HISTORY'
    # Recheck immutability after forward/autograd.
    verify(run/'bank', 'guidance_bank'); verify(run/'student', 'student')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['probes', 'audit', 'metrics'])
    parser.add_argument('--run', type=Path)
    parser.add_argument('--evaluation', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-patients', type=int, default=1)
    parser.add_argument('--max-batches', type=int, default=1)
    parser.add_argument('--expected-checkpoint-sha256', default=EXPECTED_CHECKPOINT)
    args = parser.parse_args(argv)
    if min(args.max_patients, args.max_batches) < 1: parser.error('positive bounded budgets required')
    if args.mode=='audit' and args.run is None: parser.error('--run required')
    if args.mode=='metrics' and args.evaluation is None: parser.error('--evaluation required')
    out = args.out.resolve()
    for source in (args.run, args.evaluation):
        if source and out.is_relative_to(source.resolve()):
            parser.error('diagnostics must be outside original run/frozen folders')
    if any((parent/'FROZEN.json').exists() for parent in [out, *out.parents]):
        parser.error('refusing output inside a frozen artifact')
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    started = time.perf_counter()
    try:
        if args.mode == 'probes': result = probes()
        elif args.mode == 'metrics': result = metrics_audit(args.evaluation)
        else: result = audit_run(args.run, args.max_patients, args.max_batches, args.expected_checkpoint_sha256)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        result = dict(status='BLOCKED', error_type=type(exc).__name__, reason=str(exc))
    result.update(elapsed_seconds=time.perf_counter()-started, training_run=False)
    write_json(out/'DIAGNOSTICS.json', result)
    rows = result.get('patient_frame_counts')
    if rows:
        with (out/'DIAGNOSTICS.csv').open('x') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(result['status'])
    return 2 if result['status']=='BLOCKED' else 0


if __name__ == '__main__':
    raise SystemExit(main())
