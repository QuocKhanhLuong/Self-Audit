import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

OUT = Path('/workspace/cptu-launch-20261031')
PLAN = Path('/workspace/b8-repro-setup-20261031/LOCKED_B8_PLAN.json')
EXPECTED_PLAN = '239c6023c14d058fa667b62ae53fca09a523d2dd739b496a09b1431fff166125'

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def write(p, data):
    p = Path(p)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    os.replace(tmp, p)

def main():
    OUT.mkdir(exist_ok=True)
    with (OUT/'EVAL.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert sha(PLAN) == EXPECTED_PLAN, 'plan hash changed'
        p = json.loads(PLAN.read_text())
        root = Path(p['run_root'])
        final = root/'training/final'
        report = json.loads((root/'training/REPORT.json').read_text())
        manifest = json.loads((final/'MANIFEST.json').read_text())
        assert report['status'] == 'COMPLETED' and report['steps'] == manifest['step'] == 10000
        assert manifest['config']['method'] == 'ema_full'
        assert sha(final/'weights.pt') == manifest['payload_sha256'] == 'ae89c3378fe02f30bf5e64c8977bfc260e126a4c7b9218510eb6dcdfc99cb8e1'
        assert sha(final/'MANIFEST.json') == 'a743b23eaf07a1a0ad957399547199c7ddd877654f0faa090b3faa61801520f1'
        assert sha(root/'raw_student/raw_student.pt') == '100e811c1cd1b02db99d8d0e78b4c26ab5a1ba96454ff1a039dd2c25c5279ef2'
        assert sha(p['evaluator']) == '0dc928c24dfa6ba860a8805088b29391dba03051e78555a9d5243c87938756ff'
        assert subprocess.check_output(['git','-C',p['source'],'rev-parse','HEAD'],text=True).strip() == p['source_commit']
        assert not subprocess.check_output(['git','-C',p['source'],'status','--porcelain','--untracked-files=no'],text=True).strip()
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(), 'GPU occupied'
        write(OUT/'BACKUP_DEFERRED_AUTHORIZATION.json', {
            'user_instruction': 'train đi', 'context':'Explicit request to start four arms after being told Drive backup blocks the pipeline.',
            'backup_status':'UPLOAD_PENDING', 'policy':'Proceed with local hash-verified final raw B8; do not claim Drive backup complete.',
            'teacher_plan_sha256':EXPECTED_PLAN, 'time':time.time(),
            'quality_gate_unchanged':'final raw-student historical TEST50 Macro4 >= 0.87; VAL also reported; no tuning',
            'training_budget_unchanged':{'steps_per_arm':2000,'warmup_steps':1000,'seed':42},
        })
        summaries = {}
        commands = []
        for domain in ('val18','test50'):
            item = p['evaluation'][domain]
            assert sha(item['path']) == item['sha256']
            destination = root/(domain+'_evaluation')
            assert not destination.exists(), 'Existing evaluation preserved, inspect before resume'
            cmd = [p['python'],p['evaluator'],'--checkpoint',str(final),'--checkpoint-sha256',sha(final/'MANIFEST.json'),
                   '--data',item['path'],'--data-sha256',item['sha256'],'--out',str(destination),
                   '--device','cuda','--batch-size','1','--threads','2']
            write(OUT/'EVAL_STATUS.json',{'status':'RUNNING','stage':domain,'time':time.time(),'command':cmd})
            start = time.monotonic()
            with (OUT/(domain+'.log')).open('x') as log:
                result = subprocess.run(cmd,cwd=p['source'],stdout=log,stderr=subprocess.STDOUT,timeout=7200)
            commands.append({'domain':domain,'argv':cmd,'exit_code':result.returncode,'elapsed_seconds':time.monotonic()-start})
            write(OUT/'EVAL_COMMANDS.json', commands)
            result.check_returncode()
            m = json.loads((destination/'METRICS.json').read_text())
            classes = {k:m['macro_patient'][k]['dice'] for k in ('BG','RV','MYO','LV')}
            assert len(m['patients']) == (18 if domain == 'val18' else 50)
            summaries[domain] = {'class_dice':classes,'Macro4':sum(classes.values())/4,
                'foreground':sum(classes[k] for k in ('RV','MYO','LV'))/3,'patients':len(m['patients']),
                'metrics_path':str(destination/'METRICS.json'),'metrics_sha256':sha(destination/'METRICS.json')}
            write(OUT/'TEACHER_METRICS.json',summaries)
            print(json.dumps({'stage':domain,'metrics':summaries[domain]}),flush=True)
        write(OUT/'EVAL_STATUS.json',{'status':'COMPLETED','time':time.time(),'metrics':summaries,
            'quality_gate':'GO' if summaries['test50']['Macro4'] >= .87 else 'NO_GO','backup_status':'UPLOAD_PENDING'})

if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        OUT.mkdir(exist_ok=True)
        write(OUT/'EVAL_STATUS.json',{'status':'BLOCKED','time':time.time(),'error':repr(exc)})
        raise
