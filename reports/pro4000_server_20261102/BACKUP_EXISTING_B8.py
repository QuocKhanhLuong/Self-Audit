from pathlib import Path
import sys,json
repo=Path('/workspace/Self-Audit-kd-local-20261102');sys.path.insert(0,str(repo))
from scripts.run_b8_reproduction import DriveBackup,atomic_json,sha256
out=Path('/workspace/selfaudit-20261101/kd_local_20261102')
p=Path('/workspace/selfaudit-20261101/b8_setup/LOCKED_B8_PLAN.json')
assert sha256(p)=='1723e69380cd6657af21bcf4929a54652bffb5f6db98634d6cbd066cd80b0913'
plan=json.loads(p.read_text());root=Path(plan['run_root'])
assert sha256(root/'training/final/weights.pt')=='c710f62b007198c3f29d70ccfeb6666d7abc166b01bc8f54a594894a6f901e2b'
assert sha256(root/'raw_student/raw_student.pt')=='d76d693ecb974bcd2c83964491e8f59b4e0d0f5d55c659f80e72618a4b1d1130'
files=[(root/'training/final/weights.pt','full_checkpoint/weights.pt'),(root/'training/final/MANIFEST.json','full_checkpoint/MANIFEST.json'),(root/'raw_student/raw_student.pt','raw_student/raw_student.pt'),(root/'raw_student/RAW_STUDENT.json','raw_student/RAW_STUDENT.json'),(p,'LOCKED_PLAN.json'),(Path(plan['config']),'CONFIG.json'),(Path(plan['data']),'TRAIN_MANIFEST.json'),(root/'training/REPORT.json','TRAIN_REPORT.json'),(root/'train.log','train.log')]
files += [(Path(x['path']),'provenance/'+x['name']) for x in plan['backup_files']]
files += [(out/name,'local_continuation/'+name) for name in ['LOCAL_TEACHER_RECEIPT.json','BACKUP_DEFERRED_AUTHORIZATION.json','EVAL_STATUS.json','EVAL_COMMANDS.json','EVALUATION_PROVENANCE.json']]
for domain in ('val18','test50'):
    files += [(root/(domain+'_evaluation')/name,domain+'/'+name) for name in ['METRICS.json','RECIPE.json','PREDICTIONS_FROZEN.json']]
sums=out/'B8_SHA256SUMS.txt';sums.write_text(''.join(sha256(p)+'  '+n+'\n' for p,n in files));files.append((sums,'local_continuation/SHA256SUMS.txt'))
b=DriveBackup(plan['drive_remote'],plan['drive_parent_id'],out/'b8_backup_final.log',attempts=1,timeout_seconds=120)
verified=[];pending=[]
for path,name in files:
    try:verified.append(b.upload_verified(path,name))
    except Exception as e:pending.append({'file':name,'status':'UPLOAD_PENDING','error_type':type(e).__name__})
    result={'status':'UPLOAD_PENDING' if pending else 'RUNNING','remote_verified':False,'files':verified,'pending':pending,'local_retained':True,'teacher_bytes_modified':False}
    atomic_json(out/'B8_BACKUP_FINAL.json',result)
result.update(status='UPLOAD_PENDING' if pending else 'COMPLETED',remote_verified=not pending)
atomic_json(out/'B8_BACKUP_FINAL.json',result)
print(json.dumps(result),flush=True)
raise SystemExit(2 if pending else 0)
