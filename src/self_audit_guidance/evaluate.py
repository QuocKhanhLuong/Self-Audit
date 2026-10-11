"""Independent reference evaluator. Import/call only AFTER predictions are frozen.

This module is never imported by the teacher, bank builder or student trainer.
References cannot change the checkpoint, class mapping or abstention thresholds.
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from .io import digest, read_json, write_json, verify, freeze

CLASS_NAMES = ('BG', 'RV', 'MYO', 'LV')


def score_volume(pred, reference, affine=None, spatial_unit='unknown'):
    """Score the WHOLE labeled volume. UNKNOWN contributes foreground FN.

    Classes absent in BOTH prediction and reference are not assigned Dice=1;
    they are marked absent and excluded from the corresponding macro mean.
    HD95 is null with an explicit reason when either surface is empty.
    """
    pred, reference = np.asarray(pred), np.asarray(reference)
    if pred.shape != reference.shape or pred.ndim != 3:
        raise ValueError('predictions and references must share a native XYZ grid')
    if not np.isin(pred, [0, 1, 2, 3, 255]).all() or not np.isin(reference, [0, 1, 2, 3]).all():
        raise ValueError('invalid reference/prediction classes; no remapping is inferred')
    scale = {'mm': 1., 'meter': 1000., 'micron': .001}.get(spatial_unit)
    rows = {}
    for c in range(4):
        a, b = pred == c, reference == c
        tp, fp, fn = int((a & b).sum()), int((a & ~b).sum()), int((~a & b).sum())
        denom = 2 * tp + fp + fn
        row = dict(tp=tp, fp=fp, fn=fn,
                   dice=2 * tp / denom if denom else None,
                   iou=tp / (tp + fp + fn) if tp + fp + fn else None,
                   predicted_pixels=int(a.sum()), reference_pixels=int(b.sum()),
                   hd95_mm=None, surface_status='NOT_REQUESTED_BG' if c == 0 else 'BOTH_EMPTY')
        if c and a.any() and b.any():
            if affine is None or scale is None:
                row['surface_status'] = 'SPATIAL_UNIT_OR_AFFINE_UNAVAILABLE'
            else:
                affine = np.asarray(affine, dtype=float)
                if affine.shape != (4,4) or not np.isfinite(affine).all():
                    raise ValueError('invalid surface affine')
                def surface(mask):
                    xyz = np.argwhere(mask & ~ndi.binary_erosion(mask, border_value=0))
                    return (xyz @ affine[:3,:3].T + affine[:3,3]) * scale
                sa, sb = surface(a), surface(b)
                # Pooled bidirectional 95th percentile, explicitly not max of two percentiles.
                distances = np.r_[cKDTree(sb).query(sa, workers=1)[0],
                                  cKDTree(sa).query(sb, workers=1)[0]]
                row.update(hd95_mm=float(np.percentile(distances, 95)), surface_status='MEASURED')
        elif c and (a.any() or b.any()):
            row['surface_status'] = 'MISSING_PREDICTION' if not a.any() else 'FALSE_POSITIVE_ONLY'
        rows[CLASS_NAMES[c]] = row
    valid = pred != 255
    return dict(classes=rows, pixels=int(pred.size), coverage=float(valid.mean()),
                unknown_pixels=int((~valid).sum()), accuracy_all_pixels=float((pred == reference).mean()),
                # This secondary metric is NEVER substituted for all-pixel Dice.
                accuracy_accepted_pixels=float((pred[valid] == reference[valid]).mean()) if valid.any() else None)


def _mean(values):
    present = [float(x) for x in values if x is not None]
    return sum(present) / len(present) if present else None


def aggregate_scores(rows):
    patients = {}
    for row in rows:
        patients.setdefault(row['patient_id'], []).append(row)
    per_patient = {}
    for pid, scans in patients.items():
        per_patient[pid] = dict(
            n_labeled_frames=len(scans),
            coverage=_mean([s['metrics']['coverage'] for s in scans]),
            dice={c: _mean([s['metrics']['classes'][c]['dice'] for s in scans]) for c in CLASS_NAMES},
            hd95_mm={c: _mean([s['metrics']['classes'][c]['hd95_mm'] for s in scans]) for c in CLASS_NAMES[1:]})
    dice = {c: _mean([r['dice'][c] for r in per_patient.values()]) for c in CLASS_NAMES}
    return dict(patient_count=len(per_patient), labeled_frames=len(rows), patient_macro_dice=dice,
                macro_foreground_dice=_mean([dice[c] for c in CLASS_NAMES[1:]]),
                coverage_patient_macro=_mean([r['coverage'] for r in per_patient.values()]),
                per_patient=per_patient, both_empty_class_policy='excluded_not_one',
                hd95_empty_policy='null_and_report_failure_count',
                hd95_definition='95th_percentile_of_pooled_bidirectional_native_surface_distances')


def resolve_references(pred_manifest, reference_root=None, reference_manifest=None):
    patients = set(pred_manifest['patients'])
    if (reference_root is None) == (reference_manifest is None):
        raise ValueError('provide exactly one of reference_root (ACDC) or reference_manifest')
    entries = []
    if reference_manifest is not None:
        source = Path(reference_manifest).resolve()
        raw = read_json(source)
        if raw.get('schema') != 'self_audit.references.v1':
            raise ValueError('explicit references require schema self_audit.references.v1')
        for row in raw.get('entries', []):
            if row.get('patient_id') not in patients:
                continue  # shared val/test manifest; never read excluded paths
            if set(row) != {'patient_id', 'frame_index', 'path'} or type(row['frame_index']) is not int:
                raise ValueError('reference entries need patient_id, zero-based frame_index and path')
            p = Path(row['path'])
            entries.append({**row, 'path': str((source.parent / p).resolve() if not p.is_absolute() else p.resolve())})
    else:
        root = Path(reference_root).resolve()
        for pid in sorted(patients):
            folder = root / pid
            seen = []
            for p in folder.glob(f'{pid}_frame*_gt.nii*'):
                m = re.fullmatch(re.escape(pid) + r'_frame(\d+)_gt\.nii(?:\.gz)?', p.name)
                if m:
                    seen.append(dict(patient_id=pid, frame_index=int(m[1])-1, path=str(p.resolve())))
            if len(seen) != 2:
                raise ValueError(f'{pid}: require exactly two ACDC annotated frames, found {len(seen)}; '
                                 'use an explicit manifest for other reference layouts')
            entries += seen
    keys = [(e['patient_id'], e['frame_index']) for e in entries]
    if len(set(keys)) != len(keys) or len({e['path'] for e in entries}) != len(entries):
        raise ValueError('duplicate reference frame or reference file')
    if {e['patient_id'] for e in entries} != patients:
        raise ValueError('references must cover every prediction patient; no silently smaller cohort')
    for e in entries:
        shape = pred_manifest['images'][e['patient_id']]['shape_xyzt']
        if not 0 <= e['frame_index'] < shape[3] or not Path(e['path']).is_file():
            raise ValueError(f'invalid reference frame/path: {e}')
    return sorted(entries, key=lambda e: (e['patient_id'], e['frame_index']))


def evaluate(predictions, out, reference_root=None, reference_manifest=None):
    """References first become visible here, after checking the frozen artifact."""
    folder = Path(predictions); m = verify(folder, 'student_predictions')
    if m.get('split') not in ('val','test'):
        raise ValueError('only held-out frozen predictions can be evaluated')
    if set(m['patients']) & set(m.get('train_patients', [])):
        raise ValueError('training patient in held-out evaluator')
    frozen_sha = digest(folder / 'FROZEN.json')
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    entries = resolve_references(m, reference_root, reference_manifest)
    import nibabel as nib
    rows = []
    for pid in sorted(m['patients']):
        with np.load(folder / (pid+'.npz'), allow_pickle=False) as data:
            if set(data.files) != {'labels'}:
                raise ValueError('unexpected prediction payload')
            labels = data['labels']
        meta = m['images'][pid]
        if labels.shape != tuple(reversed(meta['shape_xyzt'])):
            raise ValueError('native prediction shape does not match frozen image')
        for e in [r for r in entries if r['patient_id'] == pid]:
            before = digest(e['path']); ref = nib.load(e['path'])
            if ref.shape != tuple(meta['shape_xyzt'][:3]) or not np.allclose(ref.affine, meta['affine'], rtol=0, atol=1e-5):
                raise ValueError(f'{pid}: reference/native geometry mismatch; no resizing is permitted')
            values = np.asarray(ref.dataobj)
            if not np.isfinite(values).all() or not np.isin(values,[0,1,2,3]).all():
                raise ValueError('invalid reference labels')
            if digest(e['path']) != before:
                raise ValueError('reference changed during evaluation')
            prediction = labels[e['frame_index']].transpose(2,1,0)
            metrics = score_volume(prediction, values, np.asarray(meta['affine']), meta.get('spatial_unit','unknown'))
            rows.append({**e, 'reference_sha256': before, 'metrics': metrics})
    verify(folder,'student_predictions')
    if digest(folder/'FROZEN.json') != frozen_sha:
        raise ValueError('predictions changed during scoring')
    report = dict(schema='self_audit.guidance_evaluation.v1', split=m['split'],
                  status='SCORED_FROZEN_PREDICTIONS_NOT_CLINICALLY_VALIDATED',
                  student_status=m['student_status'], prediction_manifest_sha256=frozen_sha,
                  selection_feedback=False, unknown_is_not_background=True,
                  summary=aggregate_scores(rows), frames=rows)
    out.mkdir(parents=True,exist_ok=False); write_json(out/'METRICS.json',report)
    freeze(out,dict(role='independent_evaluation', split=m['split'],
                    prediction_manifest_sha256=frozen_sha, reference_used_for_selection=False))
    print('[EVALUATION]',m['split'],report['summary'],flush=True)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',required=True); p.add_argument('--out',required=True)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--reference-root'); g.add_argument('--reference-manifest')
    evaluate(**vars(p.parse_args(argv)))
    return 0
