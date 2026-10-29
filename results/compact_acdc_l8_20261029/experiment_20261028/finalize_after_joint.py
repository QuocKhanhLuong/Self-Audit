import hashlib
import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path('/workspace/compact-acdc-l8/20261028')
TRAIN_SOURCE = Path('/workspace/Self-Audit-compact')
BENCH_SOURCE = Path('/workspace/Self-Audit-benchmark')
TRAIN_MANIFEST = Path('/workspace/Self-Audit/data/prepared/ACDC_L8_S42/train.json')
TRAIN_SHA = 'e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4'
VAL_MANIFEST = ROOT / 'val18_native.json'
VAL_SHA = 'cac4426393ca11959f56df2876869e825443916cdec02c6d95ec8da3fac0ca50'
TEST_MANIFEST = ROOT / 'test50_native.json'
TEST_SHA = '68251014b877c25c4e182bbd05c570c24b0788e483435d68cb41ca350bc45242'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(tag, cmd, cwd):
    with (ROOT / f'{tag}.stdout.log').open('w') as stdout, (ROOT / f'{tag}.stderr.log').open('w') as stderr:
        result = subprocess.run(cmd, cwd=cwd, stdout=stdout, stderr=stderr)
    if result.returncode:
        raise RuntimeError(f'{tag} failed with exit {result.returncode}; see {ROOT / f"{tag}.stderr.log"}')


def check_report(arm, plain):
    folder = ROOT / f'{arm}_full'
    report = json.loads((folder / 'REPORT.json').read_text())
    checks = {
        'complete': report.get('status') == 'COMPLETED' and report.get('steps') == 10000,
        'same_initialization': report.get('initialization', {}).get('initial_state_sha256') == plain['initialization']['initial_state_sha256'],
        'same_teacher_cache': report.get('teacher_cache_manifest_sha256') == plain['teacher_cache_manifest_sha256'],
        'same_source': report.get('source_digest') == plain['source_digest'],
        'label_budget': report.get('label_budget') == 8,
        'manifest_sha': sha(folder / 'final/MANIFEST.json') == report.get('final_manifest_sha256'),
        'weights_sha': sha(folder / 'final/weights.pt') == report.get('final_weights_sha256'),
    }
    if not all(checks.values()):
        raise RuntimeError(f'{arm} final verification failed: {checks}')
    return report


def export(arm, checkpoint_sha):
    out = ROOT / f'{arm}_export'
    run(f'export_{arm}', [sys.executable, 'scripts/export_compact_student.py',
        '--checkpoint', str(ROOT / f'{arm}_full/final'),
        '--checkpoint-sha256', checkpoint_sha, '--out', str(out)], TRAIN_SOURCE)
    return out, sha(out / 'MANIFEST.json')


def evaluate_val(arm, export_dir, export_sha):
    out = Path(f'/workspace/{arm.replace("_", "-")}-val18-20261028')
    run(f'eval_{arm}_val18', [sys.executable, 'scripts/evaluate_compact_student.py',
        '--export', str(export_dir), '--export-sha256', export_sha,
        '--manifest', str(VAL_MANIFEST), '--manifest-sha256', VAL_SHA,
        '--train-manifest', str(TRAIN_MANIFEST), '--train-manifest-sha256', TRAIN_SHA,
        '--out', str(out), '--device', 'cuda', '--batch-size', '8', '--normalization', 'prepared'], TRAIN_SOURCE)
    metrics = json.loads((out / 'METRICS.json').read_text())
    if metrics.get('evaluation_manifest_sha256') != VAL_SHA or metrics.get('export_manifest_sha256') != export_sha:
        raise RuntimeError(f'{arm} VAL metric lineage mismatch')
    return out, metrics


def main():
    while not (ROOT / 'joint_guard_full/REPORT.json').exists():
        time.sleep(10)
    plain = json.loads((ROOT / 'plain_kd_full/REPORT.json').read_text())
    check_report('scalar_guard', plain)
    check_report('joint_guard', plain)

    arm_exports = {
        'supervised': (ROOT / 'supervised_final_export', '26e7d500c8514778704a063dcad8310a00db3919cada937956d35b7bee3cba5e'),
        'plain_kd': (ROOT / 'plain_kd_export', 'ce2fd65a1d1ae5708aae1b7187ae7eb0b21d8ed1d93612cfaf38aec346964753'),
    }
    for arm in ('scalar_guard', 'joint_guard'):
        report = json.loads((ROOT / f'{arm}_full/REPORT.json').read_text())
        arm_exports[arm] = export(arm, report['final_manifest_sha256'])

    val_results = {
        'supervised': json.loads(Path('/workspace/compact-eval-val18-final/METRICS.json').read_text()),
        'plain_kd': json.loads(Path('/workspace/plain-kd-val18-20261028/METRICS.json').read_text()),
    }
    for arm in ('scalar_guard', 'joint_guard'):
        _, val_results[arm] = evaluate_val(arm, *arm_exports[arm])
    for arm, metrics in val_results.items():
        if metrics.get('evaluation_manifest_sha256') != VAL_SHA or metrics.get('export_manifest_sha256') != arm_exports[arm][1]:
            raise RuntimeError(f'{arm} VAL receipt mismatch')

    rank = sorted(val_results, key=lambda a: (val_results[a]['class_first_macro4'], val_results[a]['class_first_foreground_macro']), reverse=True)
    chosen = rank[0]
    selection = {
        'schema': 'self_audit.compact_final_selection.v1',
        'rule': 'highest VAL18 class-first Macro4; foreground tie break; TEST50 never used for compact selection',
        'ranked_arms': [{'method': a, 'macro4': val_results[a]['class_first_macro4'], 'foreground': val_results[a]['class_first_foreground_macro']} for a in rank],
        'selected_method': chosen,
        'selected_export_manifest_sha256': arm_exports[chosen][1],
        'val_manifest_sha256': VAL_SHA,
    }
    (ROOT / 'FINAL_SELECTION.json').write_text(json.dumps(selection, indent=2) + '\n')
    final_export = ROOT / 'compact_student_final'
    shutil.copytree(arm_exports[chosen][0], final_export)
    if sha(final_export / 'MANIFEST.json') != arm_exports[chosen][1]:
        raise RuntimeError('final standalone export manifest changed during copy')

    test_out = Path('/workspace/compact-final-test50-20261028')
    run('eval_final_test50', [sys.executable, 'scripts/evaluate_compact_student.py',
        '--export', str(final_export), '--export-sha256', arm_exports[chosen][1],
        '--manifest', str(TEST_MANIFEST), '--manifest-sha256', TEST_SHA,
        '--train-manifest', str(TRAIN_MANIFEST), '--train-manifest-sha256', TRAIN_SHA,
        '--out', str(test_out), '--device', 'cuda', '--batch-size', '8', '--normalization', 'prepared'], TRAIN_SOURCE)
    test_metrics = json.loads((test_out / 'METRICS.json').read_text())

    bench_out = ROOT / 'QUALITY_SPEED_MEMORY.json'
    run('benchmark_final', [sys.executable, 'scripts/benchmark_compact_student.py',
        '--compact-export', str(final_export), '--compact-export-sha256', arm_exports[chosen][1],
        '--manifest', str(TEST_MANIFEST), '--manifest-sha256', TEST_SHA,
        '--baseline-checkpoint', str(ROOT / 'b8_recovery/full/final'),
        '--baseline-checkpoint-sha256', 'ae36d026ce26ca188627e6ea4699695e9eb99501312ce05d5324b81e44d3ddaa',
        '--baseline-teacher-identity', str(ROOT / 'b8_recovery/TEACHER_IDENTITY.json'),
        '--baseline-teacher-identity-sha256', '1cc82863264331c98d1ffd48fa5e2da61cdc6dcce5bcc1a55aa94762bfe4526b',
        '--historical-source', '/workspace/Self-Audit-history-0f54455b',
        '--out', str(bench_out), '--device', 'cuda', '--batch-size', '8', '--warmup', '2', '--repeats', '5', '--normalization', 'prepared'], BENCH_SOURCE)
    bench = json.loads(bench_out.read_text())
    if bench.get('compact_student', {}).get('status') != 'COMPLETED' or bench.get('baseline_student', {}).get('status') != 'COMPLETED':
        raise RuntimeError('final benchmark did not complete both models: ' + str(bench.get('baseline_student')))
    class_dice = {k: v['dice'] for k, v in test_metrics['macro_patient'].items()}
    quality = {
        'schema': 'self_audit.compact_final_result.v1',
        'decision': 'GO' if test_metrics['class_first_macro4'] >= 0.87 else 'NO_GO',
        'selected_method': chosen,
        'test50_macro4': test_metrics['class_first_macro4'],
        'test50_foreground': test_metrics['class_first_foreground_macro'],
        'test50_class_dice': class_dice,
        'test50_exposed_development': True,
        'export_path': str(final_export),
        'export_manifest_sha256': arm_exports[chosen][1],
        'benchmark_path': str(bench_out),
        'benchmark_sha256': sha(bench_out),
        'test_metrics_path': str(test_out / 'METRICS.json'),
        'test_metrics_sha256': sha(test_out / 'METRICS.json'),
        'train_elapsed_seconds': {a: json.loads((ROOT / f'{a}_full/REPORT.json').read_text())['elapsed_seconds'] for a in ('supervised', 'plain_kd', 'scalar_guard', 'joint_guard')},
        'teacher_train_elapsed_seconds': 2440.043427208904,
        'teacher_cache_elapsed_seconds': 88.75977764395066,
    }
    (ROOT / 'FINAL_RESULT.json').write_text(json.dumps(quality, indent=2) + '\n')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        (ROOT / 'FINALIZE_BLOCKER.json').write_text(json.dumps({'error': str(exc), 'traceback': traceback.format_exc()}, indent=2) + '\n')
        raise
