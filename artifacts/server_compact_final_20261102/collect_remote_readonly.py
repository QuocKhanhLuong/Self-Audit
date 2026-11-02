"""Read-only server verification and allowlisted tar stream; never runs evaluation.

Run through SSH with Python on stdin. Stdout contains a tar archive, not text.
Only metadata, source, logs and the three selected student exports are transferred.
MRI, reference masks and prediction arrays stay on the server.
"""
from pathlib import Path
import datetime
import hashlib
import io
import json
import math
import sys
import tarfile

BASE = Path('/workspace/selfaudit-20261101/final_eval_after_git_20261102')
RUNTIME = Path('/workspace/Self-Audit-kd-local-20261102')
files = {}
verified = {}


def sha(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def check(path, expected):
    actual = sha(path)
    assert actual == expected, str(path)
    verified[str(path)] = actual
    return actual


def read(path):
    return json.loads(Path(path).read_text())


def add(path, relative):
    path = Path(path)
    assert path.is_file(), str(path)
    assert path.suffix in ('.json', '.jsonl', '.log', '.py') or relative in (
        'S0/best.pt', 'S1/best.pt', 'S2/best.pt',
        'S0/best_export/weights.pt', 'S1/best_export/weights.pt', 'S2/best_export/weights.pt')
    assert relative not in files
    files[relative] = path


def evaluation(folder, metrics_sha, frozen_sha, export_sha, payload_sha, manifest_sha, train_sha, count, payloads):
    folder = Path(folder)
    check(folder / 'METRICS.json', metrics_sha)
    check(folder / 'PREDICTIONS_FROZEN.json', frozen_sha)
    m = read(folder / 'METRICS.json')
    f = read(folder / 'PREDICTIONS_FROZEN.json')
    r = read(folder / 'RECIPE.json')
    assert m['status'] == 'COMPLETED'
    assert m['synthetic'] is False and f['synthetic'] is False and r['synthetic'] is False
    assert m['frozen_predictions_sha256'] == frozen_sha
    assert m['coverage']['complete'] is True and f['coverage']['complete'] is True
    assert len(m['patients']) == len(f['predictions']) == count
    assert sorted(m['patients']) == sorted(x['id'] for x in f['predictions'])
    for doc in (m, f, r):
        assert doc['export_manifest_sha256'] == export_sha
        assert doc['export_payload_sha256'] == payload_sha
        assert doc['evaluation_manifest_sha256'] == manifest_sha
    assert m['train_manifest_sha256'] == r['train_manifest_sha256'] == train_sha
    assert r['batch_size'] == 8 and r['normalization'] == 'prepared'
    assert m['recipe_sha256'] == f['recipe_sha256'] == sha(folder / 'RECIPE.json')
    for c in ('BG', 'RV', 'MYO', 'LV'):
        values = [v[c] for v in m['patients'].values() if v[c] is not None]
        assert len(values) == m['macro_patient'][c]['present_patients'] == count
        assert math.isclose(sum(values) / count, m['macro_patient'][c]['dice'], abs_tol=1e-12)
    assert math.isclose(sum(m['macro_patient'][c]['dice'] for c in ('BG','RV','MYO','LV'))/4, m['class_first_macro4'], abs_tol=1e-12)
    assert math.isclose(sum(m['macro_patient'][c]['dice'] for c in ('RV','MYO','LV'))/3, m['class_first_foreground_macro'], abs_tol=1e-12)
    if payloads:
        for item in f['predictions']:
            pred = folder / 'predictions' / item['prediction_file']
            assert pred.resolve().is_relative_to((folder / 'predictions').resolve())
            check(pred, item['prediction_sha256'])
    return m


def main():
    status = read(BASE / 'STATUS.json')
    plan = read(BASE / 'PLAN.json')
    launch = read(BASE / 'LAUNCH.json')
    assert status['status'] == 'COMPLETED' and set(status['results']) == {'S0','S1','S2'}
    assert status['training'] == 'NOT_RUN_EVALUATION_ONLY' and plan['new_training_steps'] == 0
    check(BASE / 'PLAN.json', launch['plan_sha256'])
    check(BASE / 'complete_evaluation.py', plan['script_sha256'])
    for path, expected in plan['old_receipt_pins'].items():
        check(path, expected)
    for path, expected in plan['runtime_files'].items():
        check(RUNTIME / path, expected)
        add(RUNTIME / path, 'runtime_source/' + path)
    for name in ('PLAN.json','LAUNCH.json','STATUS.json','live.log','complete_evaluation.py'):
        add(BASE / name, name)
    summary = {}
    for arm in ('S0','S1','S2'):
        root = BASE / arm
        result = read(root / 'RESULT.json')
        best = read(root / 'BEST.json')
        lock = read(root / 'RUN_LOCK.json')
        selection = read(root / 'SELECTION_LOCK_BEFORE_TEST.json')
        assert result == status['results'][arm] and result['status'] == 'COMPLETED'
        assert result['drive_backup'] == 'UPLOAD_PENDING'
        assert selection['selection_complete'] is True and selection['test_used_for_selection'] is False
        assert selection['best'] == best and selection['validation_checkpoint_count'] == 10
        check(root / 'RUN_LOCK.json', plan['arms'][arm]['lock_sha256'])
        assert lock['protocol']['eval_batch_size'] == 8 and lock['protocol']['steps'] == 10000
        config_sha = hashlib.sha256(json.dumps(lock['config'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        assert config_sha == plan['arms'][arm]['final']['config_sha256']
        for role in ('train','val','test'):
            ref = lock['inputs'][role]
            check(ref['path'],ref['sha256'])
            add(ref['path'], arm + '/input_manifests/' + role + '.json')
        for path in sorted(root.rglob('*')):
            if path.is_file() and path.suffix in ('.json','.jsonl','.log'):
                add(path, str(path.relative_to(BASE)))
        weight_sha = check(root / 'best.pt',result['best_pt_sha256'])
        assert weight_sha == best['best_pt_sha256']
        check(root / 'best_export/weights.pt',weight_sha)
        check(root / 'best_export/MANIFEST.json',best['export_manifest_sha256'])
        export = read(root / 'best_export/MANIFEST.json')
        assert export['payload_sha256'] == weight_sha and export['synthetic'] is False
        for name in ('best.pt','best_export/weights.pt'):
            add(root / name,arm + '/' + name)
        for label,cp,manifest_sha,payload_sha in (
            ('selected_checkpoint',best['checkpoint_path'],best['checkpoint_manifest_sha256'],best['checkpoint_payload_sha256']),
            ('final_checkpoint',plan['arms'][arm]['final']['checkpoint_path'],plan['arms'][arm]['final']['manifest_sha256'],plan['arms'][arm]['final']['payload_sha256'])):
            check(Path(cp)/'MANIFEST.json',manifest_sha)
            check(Path(cp)/'weights.pt',payload_sha)
            add(Path(cp)/'MANIFEST.json',arm+'/'+label+'/MANIFEST.json')
        entries = [json.loads(x) for x in (root/'VALIDATION_LOG.jsonl').read_text().splitlines() if x.strip()]
        assert sorted(e['step'] for e in entries) == list(range(1000,10001,1000))
        winner = sorted(entries,key=lambda e:(-e['macro4'],e['step']))[0]
        assert winner['step'] == best['step'] == result['best_step']
        assert winner['metrics_sha256'] == best['val_metrics_sha256']
        val_metrics = None
        for row in entries:
            m = evaluation(row['validation_dir'],row['metrics_sha256'],row['predictions_frozen_sha256'],row['export_manifest_sha256'],read(Path(row['export_path'])/'MANIFEST.json')['payload_sha256'],lock['inputs']['val']['sha256'],lock['inputs']['train']['sha256'],18,row['step']==best['step'])
            assert m['class_first_macro4'] == row['macro4'] and m['class_first_foreground_macro'] == row['foreground_macro']
            for name in ('METRICS.json','PREDICTIONS_FROZEN.json','RECIPE.json'):
                add(Path(row['validation_dir'])/name,arm+'/validation_history/step_'+str(row['step']).zfill(6)+'/'+name)
            if row['step'] == best['step']:
                assert row['predictions_frozen_sha256'] == best['predictions_frozen_sha256'] == sha(best['predictions_frozen_path'])
                val_metrics = m
        assert val_metrics['class_first_macro4'] == result['val_macro4'] == best['val_macro4']
        assert val_metrics['class_first_foreground_macro'] == result['val_foreground'] == best['foreground_macro']
        tm = evaluation(root/'test50',result['test_metrics_sha256'],result['test_frozen_sha256'],best['export_manifest_sha256'],weight_sha,lock['inputs']['test']['sha256'],lock['inputs']['train']['sha256'],50,True)
        assert tm['class_first_macro4'] == result['test_macro4']
        assert tm['class_first_foreground_macro'] == result['test_foreground']
        assert (root/'SELECTION_LOCK_BEFORE_TEST.json').stat().st_mtime_ns <= (root/'test50/RECIPE.json').stat().st_mtime_ns <= (root/'test50/PREDICTIONS_FROZEN.json').stat().st_mtime_ns <= (root/'test50/METRICS.json').stat().st_mtime_ns
        summary[arm] = {'best_step':best['step'],'best_pt_sha256':weight_sha,'config_sha256':config_sha,'validation_checkpoints':10,'selected_val_predictions_hashed':18,'test_predictions_hashed':50,'val_macro4':result['val_macro4'],'test_macro4':result['test_macro4']}
    receipt = {'status':'PASS','verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Read-only verification of actual server files; no evaluation or training launched','old_receipts_unchanged':True,'drive_backup':'UPLOAD_PENDING','prediction_arrays_transferred':False,'images_or_masks_transferred':False,'arms':summary,'verified_server_files':verified}
    inventory = {name:{'server_path':str(path),'sha256':sha(path),'bytes':path.stat().st_size} for name,path in files.items()}
    with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as tar:
        for name,path in files.items():
            tar.add(path,arcname=name,recursive=False)
        for name,data in [('REMOTE_VERIFICATION.json',receipt),('TRANSFER_MANIFEST.json',inventory)]:
            payload=(json.dumps(data,indent=2,sort_keys=True)+'\n').encode()
            info=tarfile.TarInfo(name); info.size=len(payload)
            tar.addfile(info,io.BytesIO(payload))


if __name__ == '__main__':
    main()
