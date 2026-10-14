"""Publish scalar stage availability from already published VAL counts only.

No images, model arrays, references, inference, or provenance-file rewrites.
Run: python -m experiments.guidance_signal_20261014.stagewise_table --out NEW_DIR
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
from .runtime import MANIFEST_SHA256

SOURCE_COMMIT = 'b937e3b746608725e43d3bc3cef8861666d152dc'
EVIDENCE = Path(__file__).resolve().parents[2] / 'reports/stable_guidance_world_v2_20261014'
STAGES = ('r0_raw', 'r0_gated', 'q_raw', 'q_gated', 'student_common_raw',
          'student_common_gated', 'native_raw_argmax', 'native_gated',
          'coarse_projection_raw', 'coarse_projection_gated')
CLASSES = ('BG', 'RV', 'MYO', 'LV')
FIELDS = ('run', 'stage', 'availability', 'patient_id', 'frame_index', 'class_name',
          'geometry_stratum', 'val_coded_uncoded', 'tp', 'fp', 'fn',
          'predicted_pixels', 'reference_pixels', 'dice', 'iou', 'precision',
          'precision_status', 'recall', 'correct_prediction_coverage',
          'frame_coverage', 'raw_argmax_pixels', 'raw_argmax_status',
          'joint_correctness_status', 'old_new_error_regions_status')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def guard_output(path, inputs=(), *, existing_directory=False):
    path = Path(path).resolve()
    if path.exists() and not (existing_directory and path.is_dir()):
        raise FileExistsError(path)
    for parent in (path, *path.parents):
        if any((parent / marker).exists() for marker in ('FROZEN.json', 'PREDICTIONS_LOCKED.json')):
            raise ValueError('output is inside a frozen or locked artifact')
    for source in inputs:
        source = Path(source).resolve()
        if path == source or source in path.parents or path in source.parents:
            raise ValueError('output/input path overlap')
    return path


def mean(values):
    defined = [v for v in values if v is not None]
    return sum(defined) / len(defined) if defined else None


def count_metrics(tp, fp, fn):
    if any(type(v) is not int or v < 0 for v in (tp, fp, fn)):
        raise ValueError('counts must be nonnegative integers')
    return dict(tp=tp, fp=fp, fn=fn, predicted_pixels=tp + fp,
                reference_pixels=tp + fn,
                dice=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                iou=tp / (tp + fp + fn) if tp + fp + fn else None,
                precision=tp / (tp + fp) if tp + fp else None,
                precision_status='DEFINED' if tp + fp else 'UNDEFINED_NO_PREDICTIONS',
                recall=tp / (tp + fn) if tp + fn else None,
                correct_prediction_coverage=tp / (tp + fn) if tp + fn else None)


def validate_counts(rows, patients):
    keys = [(r['patient_id'], r['frame_index'], r['class_name']) for r in rows]
    if len(keys) != len(set(keys)) or len(rows) != 36 * 4:
        raise ValueError('require unique 36-frame x 4-class counts')
    if {r['patient_id'] for r in rows} != set(patients) or len(patients) != 18:
        raise ValueError('require locked VAL18 cohort')
    for pid in patients:
        frames = {r['frame_index'] for r in rows if r['patient_id'] == pid}
        if len(frames) != 2:
            raise ValueError('require exactly two labeled frames per patient')
        for frame in frames:
            subset = [r for r in rows if (r['patient_id'], r['frame_index']) == (pid, frame)]
            if {r['class_name'] for r in subset} != set(CLASSES):
                raise ValueError('missing named class')
            for row in subset:
                count_metrics(row['tp'], row['fp'], row['fn'])


def summarize(rows):
    patients = sorted({r['patient_id'] for r in rows})
    result = {}
    for c in CLASSES:
        selected = [r for r in rows if r['class_name'] == c]
        patient_metrics = {}
        for pid in patients:
            rs = [r for r in selected if r['patient_id'] == pid]
            patient_metrics[pid] = {k: mean([count_metrics(r['tp'], r['fp'], r['fn'])[k] for r in rs])
                                    for k in ('dice', 'iou', 'precision', 'recall', 'correct_prediction_coverage')}
        result[c] = dict(
            patient_macro={k: mean([v[k] for v in patient_metrics.values()]) for k in next(iter(patient_metrics.values()))},
            precision_defined_patients=sum(v['precision'] is not None for v in patient_metrics.values()),
            zero_prediction_patients=sum(sum(r['tp'] + r['fp'] for r in selected if r['patient_id'] == pid) == 0 for pid in patients),
            patients_with_any_missing_prediction_frame=sum(any(r['tp'] + r['fp'] == 0 and r['tp'] + r['fn'] > 0 for r in selected if r['patient_id'] == pid) for pid in patients),
            zero_true_positive_patients=sum(sum(r['tp'] for r in selected if r['patient_id'] == pid) == 0 for pid in patients),
            missing_prediction_frames=sum(r['tp'] + r['fp'] == 0 and r['tp'] + r['fn'] > 0 for r in selected),
            totals={k: sum(r[k] for r in selected) for k in ('tp', 'fp', 'fn')})
    return dict(patient_count=len(patients), labeled_frames=len(rows) // 4, classes=result)


def build_tables(evidence=EVIDENCE, reductions=None):
    evidence = Path(evidence)
    manifest = evidence / 'MANIFEST.json'
    if manifest.is_symlink() or digest(manifest) != MANIFEST_SHA256:
        raise ValueError('published input manifest hash mismatch')
    entries = {e['published_path']: e for e in json.loads(manifest.read_text())['files']}
    def published(relative):
        path = evidence / relative
        if path.is_symlink() or digest(path) != entries[relative]['published_sha256']:
            raise ValueError('published scalar input hash mismatch: ' + relative)
        return json.loads(path.read_text())
    recipe = published('run_evidence/run/RECIPE.json')
    patients = recipe['splits']['val']
    paths = {'baseline': evidence / 'baseline_forensics/val_counts/DIAGNOSTICS.json',
             'new': evidence / 'run_evidence/post_training/val_counts/DIAGNOSTICS.json'}
    sources = {k: published(str(p.relative_to(evidence)))['patient_frame_counts'] for k, p in paths.items()}
    reduction = json.loads(Path(reductions).read_text()) if reductions else None
    for label, rows in sources.items():
        validate_counts(rows, patients)
        if reduction:
            if reduction['evidence_commit'] != SOURCE_COMMIT or reduction['status'] != 'PASS':
                raise ValueError('invalid independent reduction evidence')
            if reduction['recomputed'][label]['overall']['frame_metrics'] != rows:
                raise ValueError('published count / independent reduction mismatch')
    indexed = [{(r['patient_id'], r['frame_index'], r['class_name']): r for r in sources[k]} for k in paths]
    if indexed[0].keys() != indexed[1].keys():
        raise ValueError('old/new frame identity mismatch')
    if any(a['tp'] + a['fn'] != indexed[1][k]['tp'] + indexed[1][k]['fn'] for k, a in indexed[0].items()):
        raise ValueError('old/new reference totals mismatch')
    table = []
    summaries = {}
    for label, rows in sources.items():
        summaries[label] = {'overall': summarize(rows), 'strata': {s: summarize([r for r in rows if r['stratum'] == s]) for s in sorted({r['stratum'] for r in rows})}}
        for stage in STAGES:
            for row in rows:
                available = stage == 'native_gated'
                item = dict(run=label, stage=stage, availability='AVAILABLE' if available else 'NOT_MEASURED',
                            patient_id=row['patient_id'], frame_index=row['frame_index'], class_name=row['class_name'],
                            geometry_stratum=row['stratum'], val_coded_uncoded='UNKNOWN',
                            raw_argmax_status='UNKNOWN', joint_correctness_status='UNKNOWN',
                            old_new_error_regions_status='UNKNOWN')
                if available:
                    item.update(count_metrics(row['tp'], row['fp'], row['fn']))
                    frame = [r for r in rows if (r['patient_id'], r['frame_index']) == (row['patient_id'], row['frame_index'])]
                    item['frame_coverage'] = sum(r['tp'] + r['fp'] for r in frame) / sum(r['tp'] + r['fn'] for r in frame)
                table.append(item)
    availability = dict(
        schema='self_audit.stagewise_available_counts.v1', source_commit=SOURCE_COMMIT,
        publication_manifest_sha256=MANIFEST_SHA256,
        status='SAVED_NATIVE_GATED_COUNTS_AVAILABLE_OTHER_STAGES_NOT_MEASURED',
        patients=patients, patient_count=18, labeled_frames=36, ontology=dict(enumerate(CLASSES)), unknown=255,
        stages={s: 'AVAILABLE' if s == 'native_gated' else 'NOT_MEASURED' for s in STAGES},
        sources={k: dict(path=str(p.relative_to(evidence)), sha256=digest(p)) for k, p in paths.items()},
        independent_reductions_sha256=digest(reductions) if reductions else None,
        native_gated=summaries,
        definitions=dict(precision='TP/(TP+FP); null/blank when no predictions',
                         correct_prediction_coverage='TP/(TP+FN)=TP/GT; identical to named-class recall',
                         frame_coverage='sum_c(TP_c+FP_c)/sum_c(TP_c+FN_c); all named classes, UNKNOWN excluded from numerator',
                         patient_macro='mean over defined frame metrics within patient, then mean over defined patients',
                         zero_prediction_patients='zero predicted class pixels across both frames',
                         missing_prediction_frames='GT positive and predicted class count zero'),
        unavailable=dict(raw_RV_argmax='UNKNOWN', old_new_error_regions='UNKNOWN', joint_correctness='UNKNOWN',
                         val_coded_uncoded='UNKNOWN: no actual VAL qform/sform code metadata; TRAIN strata are not substituted',
                         teacher_student_fidelity='NOT_MEASURED: requires aligned probability arrays'),
        limitations=['Saved counts cannot locate pixels or establish matched joint correctness.',
                     'Equal reference totals do not prove identical reference arrays.',
                     'Native gated zero RV predictions do not establish zero raw RV argmax.',
                     'No model execution, reference read, or TEST scoring was performed.'])
    return table, availability


def write_tables(out, evidence=EVIDENCE, reductions=None):
    out = guard_output(out, inputs=(evidence,), existing_directory=True)
    targets = [out / 'STAGEWISE_DIAGNOSTICS.csv', out / 'STAGEWISE_AVAILABILITY.json']
    for target in targets:
        guard_output(target, inputs=([reductions] if reductions else []))
    if any(p.exists() for p in targets):
        raise FileExistsError('stagewise outputs must be fresh; refusing overwrite')
    rows, availability = build_tables(evidence, reductions)
    out.mkdir(parents=True, exist_ok=True)
    with targets[0].open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    with targets[1].open('x') as f:
        json.dump(availability, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n')
    return dict(rows=len(rows), available_rows=sum(r['availability'] == 'AVAILABLE' for r in rows),
                patient_count=18, labeled_frames=36, outputs=[str(p) for p in targets])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True); p.add_argument('--evidence', type=Path, default=EVIDENCE)
    p.add_argument('--reductions', type=Path)
    a = p.parse_args(argv)
    print(json.dumps(write_tables(a.out, a.evidence, a.reductions), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
