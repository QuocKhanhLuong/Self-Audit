import hashlib
import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path


root = Path('/workspace/compact-acdc-l8/20261028')
train_source = Path('/workspace/Self-Audit-compact')
bench_source = Path('/workspace/Self-Audit-benchmark')
wide = root / 'wide_plain_kd_full'
plain = root / 'plain_kd_full'
train_manifest = Path('/workspace/Self-Audit/data/prepared/ACDC_L8_S42/train.json')
train_sha = 'e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4'
val_manifest = root / 'val18_native.json'
val_sha = 'cac4426393ca11959f56df2876869e825443916cdec02c6d95ec8da3fac0ca50'
test_manifest = root / 'test50_native.json'
test_sha = '68251014b877c25c4e182bbd05c570c24b0788e483435d68cb41ca350bc45242'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(tag, cmd, cwd):
    with (root / f'{tag}.stdout.log').open('w') as stdout, (root / f'{tag}.stderr.log').open('w') as stderr:
        result = subprocess.run(cmd, cwd=cwd, stdout=stdout, stderr=stderr)
    if result.returncode:
        raise RuntimeError(f'{tag} exit {result.returncode}; see {root / f"{tag}.stderr.log"}')


def main():
    while not (wide / 'REPORT.json').exists():
        time.sleep(10)
    wr = json.loads((wide / 'REPORT.json').read_text())
    pr = json.loads((plain / 'REPORT.json').read_text())
    checks = {
        'complete': wr.get('status') == 'COMPLETED' and wr.get('steps') == 10000,
        'fixed_teacher': wr.get('teacher_cache_manifest_sha256') == pr.get('teacher_cache_manifest_sha256'),
        'same_source': wr.get('source_digest') == pr.get('source_digest'),
        'same_l8': wr.get('label_budget') == pr.get('label_budget') == 8 and wr.get('train_manifest_sha256') == pr.get('train_manifest_sha256') == train_sha,
        'manifest_sha': sha(wide / 'final/MANIFEST.json') == wr.get('final_manifest_sha256'),
        'weights_sha': sha(wide / 'final/weights.pt') == wr.get('final_weights_sha256'),
    }
    schedules_compared = 0
    schedule_mismatches = 0
    with (wide / 'TRAIN_LOG.jsonl').open() as wlog, (plain / 'TRAIN_LOG.jsonl').open() as plog:
        for wrow, prow in zip(wlog, plog):
            w = json.loads(wrow)
            p = json.loads(prow)
            schedules_compared += 1
            if w.get('step') != p.get('step') or w.get('data_schedule_sha256') != p.get('data_schedule_sha256'):
                schedule_mismatches += 1
    checks['matched_l_u_schedule'] = schedules_compared == 10000 and schedule_mismatches == 0
    if not all(checks.values()):
        raise RuntimeError(f'Wide capacity trial lineage failed: {checks}; compared={schedules_compared}, mismatches={schedule_mismatches}')

    export = root / 'wide_plain_kd_export'
    run('export_wide_plain_kd', [sys.executable, 'scripts/export_compact_student.py',
        '--checkpoint', str(wide / 'final'), '--checkpoint-sha256', wr['final_manifest_sha256'], '--out', str(export)], train_source)
    export_sha = sha(export / 'MANIFEST.json')
    val_out = Path('/workspace/wide-plain-kd-val18-20261029')
    run('eval_wide_plain_kd_val18', [sys.executable, 'scripts/evaluate_compact_student.py',
        '--export', str(export), '--export-sha256', export_sha,
        '--manifest', str(val_manifest), '--manifest-sha256', val_sha,
        '--train-manifest', str(train_manifest), '--train-manifest-sha256', train_sha,
        '--out', str(val_out), '--device', 'cuda', '--batch-size', '8', '--normalization', 'prepared'], train_source)
    wm = json.loads((val_out / 'METRICS.json').read_text())
    pm = json.loads(Path('/workspace/plain-kd-val18-20261028/METRICS.json').read_text())
    if wm.get('evaluation_manifest_sha256') != val_sha or wm.get('export_manifest_sha256') != export_sha:
        raise RuntimeError('Wide VAL18 lineage mismatch')
    if pm.get('evaluation_manifest_sha256') != val_sha:
        raise RuntimeError('Small plain KD VAL18 lineage mismatch')
    selected = 'wide_plain_kd' if (wm['class_first_macro4'], wm['class_first_foreground_macro']) > (pm['class_first_macro4'], pm['class_first_foreground_macro']) else 'plain_kd'

    bench_path = root / 'WIDE_QUALITY_SPEED_MEMORY.json'
    run('benchmark_wide_plain_kd', [sys.executable, 'scripts/benchmark_compact_student.py',
        '--compact-export', str(export), '--compact-export-sha256', export_sha,
        '--manifest', str(test_manifest), '--manifest-sha256', test_sha,
        '--baseline-checkpoint', str(root / 'b8_recovery/full/final'),
        '--baseline-checkpoint-sha256', 'ae36d026ce26ca188627e6ea4699695e9eb99501312ce05d5324b81e44d3ddaa',
        '--baseline-teacher-identity', str(root / 'b8_recovery/TEACHER_IDENTITY.json'),
        '--baseline-teacher-identity-sha256', '1cc82863264331c98d1ffd48fa5e2da61cdc6dcce5bcc1a55aa94762bfe4526b',
        '--historical-source', '/workspace/Self-Audit-history-0f54455b',
        '--out', str(bench_path), '--device', 'cuda', '--batch-size', '8', '--warmup', '2', '--repeats', '5', '--normalization', 'prepared'], bench_source)
    bench = json.loads(bench_path.read_text())
    if bench['compact_student']['status'] != 'COMPLETED' or bench['baseline_student']['status'] != 'COMPLETED':
        raise RuntimeError('Wide benchmark incomplete')

    chosen_export = export if selected == 'wide_plain_kd' else root / 'compact_student_final'
    chosen_sha = export_sha if selected == 'wide_plain_kd' else 'ce2fd65a1d1ae5708aae1b7187ae7eb0b21d8ed1d93612cfaf38aec346964753'
    selected_export = root / 'compact_student_selected'
    shutil.copytree(chosen_export, selected_export)
    if sha(selected_export / 'MANIFEST.json') != chosen_sha:
        raise RuntimeError('Selected artifact hash changed on copy')

    if selected == 'wide_plain_kd':
        test_out = Path('/workspace/wide-plain-kd-test50-20261029')
        run('eval_wide_plain_kd_test50', [sys.executable, 'scripts/evaluate_compact_student.py',
            '--export', str(selected_export), '--export-sha256', chosen_sha,
            '--manifest', str(test_manifest), '--manifest-sha256', test_sha,
            '--train-manifest', str(train_manifest), '--train-manifest-sha256', train_sha,
            '--out', str(test_out), '--device', 'cuda', '--batch-size', '8', '--normalization', 'prepared'], train_source)
    else:
        test_out = Path('/workspace/compact-final-test50-20261028')
    tm = json.loads((test_out / 'METRICS.json').read_text())
    result = {
        'schema': 'self_audit.compact_capacity_selection.v1',
        'selection_rule': 'higher VAL18 class-first Macro4; foreground tie break; TEST50 excluded',
        'checks': checks,
        'schedule_steps_compared': schedules_compared,
        'schedule_mismatches': schedule_mismatches,
        'small_plain_val_macro4': pm['class_first_macro4'],
        'wide_plain_val_macro4': wm['class_first_macro4'],
        'small_plain_val_foreground': pm['class_first_foreground_macro'],
        'wide_plain_val_foreground': wm['class_first_foreground_macro'],
        'selected_method': selected,
        'selected_export_path': str(selected_export),
        'selected_export_manifest_sha256': chosen_sha,
        'selected_test50_macro4': tm['class_first_macro4'],
        'selected_test50_foreground': tm['class_first_foreground_macro'],
        'selected_test50_class_dice': {k: v['dice'] for k, v in tm['macro_patient'].items()},
        'decision': 'GO' if tm['class_first_macro4'] >= 0.87 else 'NO_GO',
        'wide_train_elapsed_seconds': wr['elapsed_seconds'],
        'wide_peak_train_vram_allocated_bytes': wr['profile']['cuda_peak_allocated_bytes'],
        'wide_peak_train_vram_reserved_bytes': wr['profile']['cuda_peak_reserved_bytes'],
        'wide_export_manifest_sha256': export_sha,
        'wide_benchmark_path': str(bench_path),
        'wide_benchmark_sha256': sha(bench_path),
        'wide_val_metrics_path': str(val_out / 'METRICS.json'),
        'wide_val_metrics_sha256': sha(val_out / 'METRICS.json'),
        'selected_test_metrics_path': str(test_out / 'METRICS.json'),
        'selected_test_metrics_sha256': sha(test_out / 'METRICS.json'),
    }
    (root / 'WIDE_CAPACITY_RESULT.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        (root / 'WIDE_FINALIZE_BLOCKER.json').write_text(json.dumps({'error': str(exc), 'traceback': traceback.format_exc()}, indent=2) + '\n')
        raise
