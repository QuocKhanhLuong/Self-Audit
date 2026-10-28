"""Supplemental, independently scored M&Ms VAL32 generalization control."""
import json
import sys
from pathlib import Path

ROOT = Path('/workspace/Self-Audit')
sys.path.insert(0, str(ROOT))
from self_audit.model import digest
from self_audit.automatic_evaluation import evaluate_automatic

R = ROOT / 'reports/automatic_pilot_pro4000'
matrix_path = ROOT / 'runs/pro4000/equal_time_B128_M40/summary.json'
matrix = json.loads(matrix_path.read_text())
item = matrix['arms']['full_ema']
query = Path(json.loads((R / 'mnm_loading_smoke.json').read_text())['generalization_manifest'])
out = ROOT / 'runs/pro4000/evaluation_equal_time_B128_M40_MnM_verified_VAL/full_ema'
metrics = evaluate_automatic(
    checkpoint=Path(item['output_directory']) / 'final',
    checkpoint_sha=item['receipt']['final_manifest_sha256'],
    data_manifest=query,
    data_sha=digest(query),
    out=out,
    device='cuda',
    batch_size=4,
    threads=4,
    compress_predictions=True,
)
receipt = {
    'schema': 'self_audit.pro4000.supplementary_generalization.v1',
    'arm': 'full_ema',
    'cohort': 'MnM_verified_VAL32',
    'training_dataset': 'ACDC_L8_S42 only',
    'query_manifest': str(query),
    'query_manifest_sha256': digest(query),
    'checkpoint': str(Path(item['output_directory']) / 'final'),
    'checkpoint_manifest_sha256': item['receipt']['final_manifest_sha256'],
    'output_directory': str(out),
    'metrics_sha256': digest(out / 'METRICS.json'),
    'foreground_patient_macro': metrics['foreground_patient_macro'],
    'macro_patient': metrics['macro_patient'],
    'prediction_coverage': metrics['prediction_coverage'],
    'scoring_coverage': metrics['scoring_coverage'],
    'total_patients_scored': metrics['total_patients_scored'],
    'quality_status': 'MEASURED',
}
(R / 'mnm_full_ema_supplemental.json').write_text(json.dumps(receipt, indent=2) + '\n')
(R / 'results_equal_time_B128_M40_MnM_full_ema.json').write_text(json.dumps([
    {'arm': 'full_ema', 'metrics': metrics, 'training_report': item['receipt']}
], indent=2, allow_nan=False) + '\n')
print('MNM_FULL_EMA_EVALUATION_COMPLETE', json.dumps(receipt), flush=True)
