import hashlib
import json
from pathlib import Path
import subprocess
import time
import psutil

out=Path('/workspace/cptu-launch-20261031')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    p=Path(p); tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(d,indent=2)+'\n'); tmp.replace(p)

names=['plain_kd','scalar_guard_original','scalar_guard_larger_tie','scalar_guard_cross_patient_utility']
arms={}
for name in names:
    root=out/'pilot/runs'/name
    segments=[]
    for log in sorted(root.glob('segments/*/TRAIN_LOG.jsonl')):
        lines=log.read_text().splitlines()
        last=json.loads(lines[-1]) if lines else {}
        segments.append({'log':str(log),'rows':len(lines),'last_zero_based_step':last.get('step'),
                         'last_status':last.get('status')})
    arms[name]={'segments':segments,'observed_steps':max([s['last_zero_based_step']+1 for s in segments if s['last_zero_based_step'] is not None]+[0])}
launch=json.loads((out/'LAUNCH_COMMAND.json').read_text())
process=json.loads((out/'RUNNER_PROCESS.json').read_text())
try:
    root=psutil.Process(process['pid'])
    processes=[{'pid':p.pid,'argv':p.cmdline()} for p in [root]+root.children(recursive=True)]
except psutil.Error:
    processes=[]
result={'status':'RUNNING' if processes else 'CHECK_RUNNER_EXIT','observed_unix':time.time(),
        'source_commit':launch['source_commit'],'plan_sha256':launch['plan_sha256'],
        'teacher_quality_gate':'NO_GO','teacher_test50_macro4':0.8682269380656147,
        'exploratory_authorized':True,'backup_status':'UPLOAD_PENDING',
        'required_arms':arms,'processes':processes,
        'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True).strip(),
        'server_checks':'949 passed, 2 warnings; check_repo PASS',
        'cuda_replay':json.loads((out/'cuda_verification/VERIFICATION.json').read_text())['status'],
        'profile':json.loads((out/'pilot/profile/PROFILE_RECEIPT.json').read_text())['status'],
        'tmux':launch['tmux'],'run_root':str(out),
        'teacher_receipt_sha256':sha(out/'CPTU_TEACHER_RECEIPT.json'),
        'cache_manifest_sha256':sha(out/'teacher_cache/CACHE_MANIFEST.json'),
        'protocol_sha256':sha(out/'PROTOCOL.json')}
write(out/'LAUNCH_OBSERVATION.json',result)
handover=Path('/root/self-audit-assets/receipts/HANDOVER.json')
before=handover.read_bytes(); before_sha=hashlib.sha256(before).hexdigest()
backup=out/('HANDOVER.before_launch.'+before_sha+'.json')
backup.write_bytes(before)
data=json.loads(before)
data['cptu_exploratory_pilot_20261031']={k:result[k] for k in ['status','observed_unix','source_commit','plan_sha256','teacher_quality_gate','teacher_test50_macro4','exploratory_authorized','backup_status','required_arms','tmux','run_root','teacher_receipt_sha256','cache_manifest_sha256','protocol_sha256']}
data['cptu_exploratory_pilot_20261031']['authorization_file']=str(out/'EXPLORATORY_TEACHER_AUTHORIZATION.json')
data['cptu_exploratory_pilot_20261031']['authorization_sha256']=sha(out/'EXPLORATORY_TEACHER_AUTHORIZATION.json')
assert handover.read_bytes()==before, 'HANDOVER changed concurrently; preserve it and retry merge'
write(handover,data)
write(out/'HANDOVER_LAUNCH_UPDATE.json',{'before_sha256':before_sha,'after_sha256':sha(handover),'backup':str(backup)})
print(json.dumps({k:result[k] for k in ['status','required_arms','gpu','source_commit','protocol_sha256']},indent=2))
