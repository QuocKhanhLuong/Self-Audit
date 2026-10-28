"""Independent file-level audit of finished checkpoints and native exports."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/workspace/Self-Audit')
R = ROOT / 'reports/automatic_pilot_pro4000'
RUNS = ROOT / 'runs/pro4000'

def load(path):
    return json.loads(Path(path).read_text())

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

matrix_checks = []
for regime in ('equal_update_B128_M40', 'equal_time_B128_M40'):
    matrix = load(R / f'{regime}.json')
    for arm, item in matrix['arms'].items():
        checkpoint = Path(item['output_directory']) / 'final'
        manifest = checkpoint / 'MANIFEST.json'
        meta = load(manifest)
        assert digest(manifest) == item['receipt']['final_manifest_sha256']
        assert digest(checkpoint / 'weights.pt') == meta['payload_sha256']
        assert meta['training_manifest_sha256'] == matrix['training_manifest']['sha256']
        assert meta['source_sha256'] == item['receipt']['source_sha256']
        assert meta['step'] == item['receipt']['steps']
        matrix_checks.append({'regime': regime, 'arm': arm,
                              'manifest_sha256': digest(manifest),
                              'payload_sha256': meta['payload_sha256']})

export_checks = []
for tag, expected_patients in (
        ('equal_update_B128_M40', 50),
        ('equal_time_B128_M40', 50),
        ('equal_time_B128_M40_MnM_verified_VAL', 32)):
    index = load(R / f'evaluation_index_{tag}.json')
    for arm, item in index['evaluations'].items():
        out = Path(item['directory'])
        assert digest(out / 'METRICS.json') == item['metrics_sha256']
        frozen = load(out / 'PREDICTIONS_FROZEN.json')
        assert digest(out / 'RECIPE.json') == frozen['recipe_sha256']
        assert frozen['total_patients_exported'] == expected_patients
        assert len(frozen['files']) == expected_patients and frozen['coverage'] == 1.0
        for record in frozen['files']:
            assert digest(out / 'predictions' / record['file']) == record['sha256']
        export_checks.append({'tag': tag, 'arm': arm, 'patients': len(frozen['files']),
                              'metrics_sha256': item['metrics_sha256'],
                              'frozen_receipt_sha256': digest(out / 'PREDICTIONS_FROZEN.json')})

supplement = load(R / 'mnm_full_ema_supplemental.json')
out = Path(supplement['output_directory'])
frozen = load(out / 'PREDICTIONS_FROZEN.json')
assert digest(out / 'METRICS.json') == supplement['metrics_sha256']
assert frozen['checkpoint_sha256'] == supplement['checkpoint_manifest_sha256']
assert len(frozen['files']) == 32 and frozen['coverage'] == 1.0
for record in frozen['files']:
    assert digest(out / 'predictions' / record['file']) == record['sha256']
export_checks.append({'tag': 'equal_time_B128_M40_MnM_verified_VAL_supplemental',
                      'arm': 'full_ema', 'patients': 32,
                      'metrics_sha256': supplement['metrics_sha256'],
                      'frozen_receipt_sha256': digest(out / 'PREDICTIONS_FROZEN.json')})

report = {'schema': 'self_audit.pro4000.output_audit.v1', 'status': 'PASS',
          'checkpoint_count': len(matrix_checks), 'evaluation_count': len(export_checks),
          'native_prediction_file_count': sum(x['patients'] for x in export_checks),
          'checkpoints': matrix_checks, 'evaluations': export_checks}
(R / 'OUTPUT_AUDIT.json').write_text(json.dumps(report, indent=2) + '\n')
print('OUTPUT_AUDIT_PASS', report['checkpoint_count'], report['evaluation_count'],
      report['native_prediction_file_count'])
