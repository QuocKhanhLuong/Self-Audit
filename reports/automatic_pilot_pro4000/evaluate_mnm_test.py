"""Frozen TEST135 generalization of the five declared/diagnostic ACDC arms."""
import json
import sys
from pathlib import Path

ROOT = Path('/workspace/Self-Audit')
sys.path.insert(0, str(ROOT))
from self_audit.model import digest
from self_audit.automatic_evaluation import evaluate_automatic
from scripts.run_automatic_pilot import collate_evaluations

R = ROOT / 'reports/automatic_pilot_pro4000'
RUNS = ROOT / 'runs/pro4000'
tag = 'equal_time_B128_M40_MnM_TEST135'
manifest = R / 'MnM_TEST135_manifest.json'
prep = json.loads((R / 'mnm_test_preparation.json').read_text())
assert prep['status'] == 'PASS' and prep['manifest_sha256'] == digest(manifest)
query = json.loads(manifest.read_text())
assert query['role'] == 'external_frozen' and len(query['patients']) == 135
matrix_path = RUNS / 'equal_time_B128_M40/summary.json'
matrix = json.loads(matrix_path.read_text())
arms = ('scalar_dice', 'joint_dice', 'smaller_lr', 'extra_supervised', 'full_ema')
evaluations = {}
results = []
for arm in arms:
    item = matrix['arms'][arm]
    checkpoint = Path(item['output_directory']) / 'final'
    out = RUNS / f'evaluation_{tag}' / arm
    print('MNM_TEST135_EVALUATION_START', arm, flush=True)
    metrics = evaluate_automatic(
        checkpoint=checkpoint,
        checkpoint_sha=item['receipt']['final_manifest_sha256'],
        data_manifest=manifest,
        data_sha=digest(manifest),
        out=out,
        device='cuda',
        batch_size=4,
        threads=4,
        compress_predictions=True,
    )
    assert metrics['total_patients_scored'] == 135
    assert metrics['prediction_coverage'] == metrics['scoring_coverage'] == 1.0
    evaluations[arm] = {'directory': str(out), 'metrics_sha256': digest(out / 'METRICS.json')}
    results.append({'arm': arm, 'metrics': metrics, 'training_report': item['receipt']})
    (R / f'evaluation_progress_{tag}.json').write_text(json.dumps({
        'completed_arms': list(evaluations), 'total_arms': len(arms)
    }, indent=2) + '\n')
    print('MNM_TEST135_EVALUATION_COMPLETE', arm, metrics['foreground_patient_macro'], flush=True)

index = {
    'schema': 'self_audit.automatic_evaluation_index.v1',
    'matrix_report': {'path': str(matrix_path), 'sha256': digest(matrix_path)},
    'evaluations': evaluations,
}
index_path = R / f'evaluation_index_{tag}.json'
index_path.write_text(json.dumps(index, indent=2) + '\n')
decision = collate_evaluations(index_path, digest(index_path))
(R / f'decision_{tag}.json').write_text(json.dumps(decision, indent=2, allow_nan=False) + '\n')
(R / f'results_{tag}.json').write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')
print('MNM_TEST135_EVALUATION_PHASE_COMPLETE', decision['decision_gate'], flush=True)
