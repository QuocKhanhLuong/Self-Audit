import hashlib
import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path


root = Path('/workspace/compact-acdc-l8/20261029-maxbatch')
prior = Path('/workspace/compact-acdc-l8/20261028')
source = Path('/workspace/Self-Audit-compact')
bench_source = Path('/workspace/Self-Audit-benchmark')
run_dir = root / 'wide_kd_batch152_full'
train_manifest = Path('/workspace/Self-Audit/data/prepared/ACDC_L8_S42/train.json')
train_sha = 'e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4'
val_manifest = prior / 'val18_native.json'
val_sha = 'cac4426393ca11959f56df2876869e825443916cdec02c6d95ec8da3fac0ca50'
test_manifest = prior / 'test50_native.json'
test_sha = '68251014b877c25c4e182bbd05c570c24b0788e483435d68cb41ca350bc45242'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(tag, args, cwd):
    with (root / f'{tag}.stdout.log').open('w') as stdout, (root / f'{tag}.stderr.log').open('w') as stderr:
        result = subprocess.run(args, cwd=cwd, stdout=stdout, stderr=stderr)
    if result.returncode:
        raise RuntimeError(f'{tag} exit {result.returncode}; see {root / f"{tag}.stderr.log"}')


def evaluate(tag, export_dir, export_sha, manifest, manifest_sha, out):
    command(tag, [sys.executable, 'scripts/evaluate_compact_student.py',
        '--export', str(export_dir), '--export-sha256', export_sha,
        '--manifest', str(manifest), '--manifest-sha256', manifest_sha,
        '--train-manifest', str(train_manifest), '--train-manifest-sha256', train_sha,
        '--out', str(out), '--device', 'cuda', '--batch-size', '8', '--normalization', 'prepared'], source)
    metrics = json.loads((out / 'METRICS.json').read_text())
    if metrics['evaluation_manifest_sha256'] != manifest_sha or metrics['export_manifest_sha256'] != export_sha:
        raise RuntimeError(f'{tag} lineage mismatch')
    return metrics


def main():
    while not (run_dir / 'REPORT.json').exists():
        check = subprocess.run(['tmux', 'display-message', '-p', '-t', 'ssh_tmux:maxbatch-full', '#{pane_dead}'],
            text=True, capture_output=True)
        if check.returncode or check.stdout.strip() == '1':
            raise RuntimeError('Training window exited without REPORT.json; see wide_kd_batch152_full.stderr.log')
        time.sleep(30)

    report = json.loads((run_dir / 'REPORT.json').read_text())
    prior_report = json.loads((prior / 'wide_plain_kd_full/REPORT.json').read_text())
    config = root / 'wide_kd_batch152_full.json'
    cache = root / 'CACHE_MANIFEST.json'
    checks = {
        'completed_10000': report.get('status') == 'COMPLETED' and report.get('steps') == 10000,
        'batch_152': report.get('batch_sizes') == {'labeled': 152, 'unlabeled': 152},
        'same_source': report.get('source_digest') == prior_report.get('source_digest'),
        'l8_train_manifest': report.get('train_manifest_sha256') == train_sha and report.get('label_budget') == 8,
        'teacher_cache_manifest': report.get('teacher_cache_manifest_sha256') == sha(cache),
        'manifest_sha': sha(run_dir / 'final/MANIFEST.json') == report.get('final_manifest_sha256'),
        'weights_sha': sha(run_dir / 'final/weights.pt') == report.get('final_weights_sha256'),
        'config_exists': config.is_file(),
    }
    if not all(checks.values()):
        raise RuntimeError(f'Max batch final checkpoint verification failed: {checks}')

    export = root / 'compact_student_batch152_export'
    command('export_batch152', [sys.executable, 'scripts/export_compact_student.py',
        '--checkpoint', str(run_dir / 'final'), '--checkpoint-sha256', report['final_manifest_sha256'],
        '--out', str(export)], source)
    export_sha = sha(export / 'MANIFEST.json')
    meta = json.loads((export / 'MANIFEST.json').read_text())
    if meta.get('teacher_included') is not False or meta.get('external_teacher_required') is not False:
        raise RuntimeError('Export is not standalone')

    val_out = Path('/workspace/maxbatch152-val18-20261029')
    test_out = Path('/workspace/maxbatch152-test50-20261029')
    val = evaluate('evaluate_batch152_val18', export, export_sha, val_manifest, val_sha, val_out)
    test = evaluate('evaluate_batch152_test50', export, export_sha, test_manifest, test_sha, test_out)

    bench_path = root / 'BATCH152_QUALITY_SPEED_MEMORY.json'
    command('benchmark_batch152', [sys.executable, 'scripts/benchmark_compact_student.py',
        '--compact-export', str(export), '--compact-export-sha256', export_sha,
        '--manifest', str(test_manifest), '--manifest-sha256', test_sha,
        '--baseline-checkpoint', str(prior / 'b8_recovery/full/final'),
        '--baseline-checkpoint-sha256', 'ae36d026ce26ca188627e6ea4699695e9eb99501312ce05d5324b81e44d3ddaa',
        '--baseline-teacher-identity', str(prior / 'b8_recovery/TEACHER_IDENTITY.json'),
        '--baseline-teacher-identity-sha256', '1cc82863264331c98d1ffd48fa5e2da61cdc6dcce5bcc1a55aa94762bfe4526b',
        '--historical-source', '/workspace/Self-Audit-history-0f54455b',
        '--out', str(bench_path), '--device', 'cuda', '--batch-size', '8', '--warmup', '2', '--repeats', '5', '--normalization', 'prepared'], bench_source)
    bench = json.loads(bench_path.read_text())
    if bench['compact_student']['status'] != 'COMPLETED' or bench['baseline_student']['status'] != 'COMPLETED':
        raise RuntimeError('Benchmark incomplete')

    prior_val = json.loads(Path('/workspace/wide-plain-kd-val18-20261029/METRICS.json').read_text())
    better_val = (val['class_first_macro4'], val['class_first_foreground_macro']) > (
        prior_val['class_first_macro4'], prior_val['class_first_foreground_macro'])
    selected = export if better_val else prior / 'compact_student_selected'
    selected_sha = export_sha if better_val else '197dd4367877c51bc21b91d172f25442549ef9a3d0e422d28118bb07308067de'
    selected_export = root / 'compact_student_selected_by_val'
    shutil.copytree(selected, selected_export)
    if sha(selected_export / 'MANIFEST.json') != selected_sha:
        raise RuntimeError('Selected export hash mismatch')

    result = {
        'schema': 'self_audit.compact_max_batch.v1',
        'status': 'COMPLETE',
        'new_factor': 'batch 152; 19x the matched batch-8 compact arms; same L8 IDs and fixed raw B8 teacher',
        'checks': checks,
        'config_path': str(config),
        'config_sha256': sha(config),
        'checkpoint_manifest_sha256': report['final_manifest_sha256'],
        'checkpoint_weights_sha256': report['final_weights_sha256'],
        'export_path': str(export),
        'export_manifest_sha256': export_sha,
        'export_weights_sha256': sha(export / 'weights.pt'),
        'val_macro4': val['class_first_macro4'],
        'val_foreground': val['class_first_foreground_macro'],
        'test50_macro4': test['class_first_macro4'],
        'test50_foreground': test['class_first_foreground_macro'],
        'test50_class_dice': {k: v['dice'] for k, v in test['macro_patient'].items()},
        'test50_exposed_development': True,
        'decision': 'GO' if test['class_first_macro4'] >= 0.87 else 'NO_GO',
        'selected_by_val': 'batch152' if better_val else 'batch8',
        'selected_export_path': str(selected_export),
        'selected_export_manifest_sha256': selected_sha,
        'train_elapsed_seconds': report['elapsed_seconds'],
        'train_peak_vram_allocated_bytes': report['profile']['cuda_peak_allocated_bytes'],
        'train_peak_vram_reserved_bytes': report['profile']['cuda_peak_reserved_bytes'],
        'val_metrics_path': str(val_out / 'METRICS.json'),
        'test_metrics_path': str(test_out / 'METRICS.json'),
        'benchmark_path': str(bench_path),
        'benchmark_sha256': sha(bench_path),
    }
    (root / 'MAXBATCH_RESULT.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        (root / 'MAXBATCH_BLOCKER.json').write_text(json.dumps({'error': str(exc), 'traceback': traceback.format_exc()}, indent=2) + '\n')
        raise
