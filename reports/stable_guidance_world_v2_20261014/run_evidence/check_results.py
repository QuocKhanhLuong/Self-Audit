"""Wait for this training job, verify frozen counts, and compare the baseline."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

job = Path(__file__).resolve().parent
plan = json.loads((job / 'LAUNCH_MANIFEST.json').read_text())
destination = job / 'post_training'
destination.mkdir(exist_ok=True)
state_path = destination / 'STATUS.json'


def save(payload):
    temporary = state_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    temporary.replace(state_path)


save({'status': 'WAITING_FOR_TRAINING_AND_EVALUATION', 'reference_masks_read': False})
try:
    while not (job / 'EXIT_CODE').exists():
        time.sleep(15)
    pipeline = json.loads((job / 'run/RUN_STATUS.json').read_text())
    if not pipeline['status'].startswith('COMPLETED_VAL_SCORED'):
        raise RuntimeError('Pipeline did not complete evaluation: '+pipeline['status'])
    evaluation = job / 'run/evaluation_val'
    subprocess.run([plan['python'], str(job / 'source/scripts/diagnose_stable_guidance.py'),
                    'metrics', '--evaluation', str(evaluation), '--out', str(destination / 'val_counts')], check=True)
    sys.path.insert(0, str(job / 'source/scripts'))
    from publish_guidance_metrics import evaluation_scalars
    current_report = json.loads((evaluation / 'METRICS.json').read_text())
    baseline_report = json.loads((Path(plan['original_job']) / 'run/evaluation_val_verified_grid/METRICS.json').read_text())
    identity = lambda report: sorted((r['patient_id'], r['frame_index']) for r in report['frames'])
    if identity(current_report) != identity(baseline_report):
        raise ValueError('Baseline and current evaluated cohorts differ')
    current = evaluation_scalars(current_report)
    baseline = evaluation_scalars(baseline_report)
    metrics = ('dice', 'iou', 'precision', 'recall')
    with (destination / 'MAIN_METRICS.csv').open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['class', *[m+'_percent' for m in metrics]])
        for name in ('RV', 'MYO', 'LV', 'foreground_macro'):
            prefix = 'val/' if name == 'foreground_macro' else 'val/'+name+'/'
            writer.writerow([name, *[None if current[prefix+m] is None else 100*current[prefix+m] for m in metrics]])
    comparisons = []
    for metric in (*metrics, 'coverage'):
        old, new = baseline['val/'+metric], current['val/'+metric]
        comparisons.append({'metric': metric, 'baseline_percent': None if old is None else old*100,
                            'new_percent': None if new is None else new*100,
                            'delta_percentage_points': None if old is None or new is None else (new-old)*100})
    with (destination / 'COMPARISON.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    summary = {'status': 'COUNTS_VERIFIED_AND_BASELINE_COMPARED', 'metrics': current,
               'comparison': comparisons, 'baseline_job': plan['original_job'],
               'same_evaluated_patient_frames': True, 'reference_masks_read': False,
               'auroc_status': 'UNAVAILABLE_HARD_LABEL_EXPORTS',
               'official_test_status': 'EXPORTED_NOT_SCORED_UNLABELED',
               'checkpoint_selection': 'fixed_last_epoch_no_reference_feedback',
               'metric_aggregation': 'frame_then_patient_then_foreground_class_macro',
               'undefined_metrics': 'null; excluded from corresponding macro',
               'geometry_receipts_reexecuted_by_this_check': False}
    (destination / 'RESULTS.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    save(summary)
    print(json.dumps(summary), flush=True)
except Exception as exc:
    save({'status': 'BLOCKED', 'error': type(exc).__name__+': '+str(exc)})
    raise
