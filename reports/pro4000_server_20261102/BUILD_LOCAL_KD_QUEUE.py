from pathlib import Path
import json,sys,subprocess
repo=Path('/workspace/Self-Audit-kd-local-20261102');sys.path.insert(0,str(repo))
from self_audit.server_runner import source_identity,sha256_file,atomic_json
from self_audit.server_ready_arm import load_plan,verify_teacher,cache_disk_bytes
from self_audit.server_campaign import validate_campaign
from scripts.continue_b8_transfer_pilot import build_backup_deferred_receipt
base=Path('/workspace/selfaudit-20261101');out=base/'kd_local_20261102'
source=source_identity();assert source['git_head']==sys.argv[1]
assert not (out/'CAMPAIGN.json').exists()
old=json.loads((base/'campaign_continue_20261102/state/STATE.json').read_text());assert old['status']=='completed_with_blocks'
assert json.loads((out/'EVAL_STATUS.json').read_text())['quality_gate']=='GO'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
ready=json.loads((base/'campaign_continue_20261102/READY_ARMS.json').read_text())
receipt=build_backup_deferred_receipt({'teacher_run':ready['teacher_root'],'teacher_plan_sha256':ready['b8_plan']['sha256']},out)
receipt_path=out/'LOCAL_TEACHER_RECEIPT.json';atomic_json(receipt_path,receipt)
def ppin(path):
    path=Path(path).resolve();return {'path':str(path),'sha256':sha256_file(path)}
ready.update(git_sha=source['git_head'],source_closure_sha256=source['closure_sha256'],output_root=str(out/'arms'),
             backup_deferred_authorization=ppin(out/'BACKUP_DEFERRED_AUTHORIZATION.json'),local_teacher_receipt=ppin(receipt_path))
for key,name in [('s1_template','s1_plain_kd.json'),('s2_template','s2_plain_kd.json'),('joint_template','joint_supervised.json')]:
    ready[key]=ppin(repo/'configs/server_compact'/name)
ready_path=out/'READY_ARMS.json';atomic_json(ready_path,ready);rsha=sha256_file(ready_path)
load_plan(ready_path,rsha);teacher=verify_teacher(ready)
assert teacher['quality_gate']=='GO' and teacher['remote_verified'] is False
pins=[]
def add(name,kind,path,sha=None):
    pin=ppin(path)
    if sha:assert pin['sha256']==sha,(path,sha,pin['sha256'])
    for x in pins:
        if x['path']==pin['path']:
            assert x['sha256']==pin['sha256'] and x['kind']==kind
            return x['id']
    pins.append(dict(id=name,kind=kind,**pin));return name
sources=[add('source_'+str(i),'source',repo/p,sha) for i,(p,sha) in enumerate(sorted(source['files'].items()))]
recipe_pin=add('ready_plan','recipe',ready_path,rsha)
assets=[add(key,'asset',ready[key]['path'],ready[key]['sha256']) for key in ['train','val','test','backup_deferred_authorization','local_teacher_receipt']]
teacher_root=Path(ready['teacher_root'])
for i,rel in enumerate(['training/REPORT.json','training/final/MANIFEST.json','training/final/weights.pt','raw_student/raw_student.pt','raw_student/RAW_STUDENT.json','val18_evaluation/METRICS.json','test50_evaluation/METRICS.json']):
    assets.append(add('teacher_asset_'+str(i),'asset',teacher_root/rel))
python='/venv/selfaudit-pro4000/bin/python';runner=str(repo/'scripts/run_server_ready_arm.py')
def cmd(action,role,requires,completion):
    return {'id':action.replace('-','_'),'kind':'command','role':role,'requires':requires,
        'argv':[python,runner,action,'--plan',str(ready_path),'--plan-sha256',rsha],'cwd':str(repo),
        'pin_ids':[recipe_pin,*sources,*assets],'completion':{'path':str(completion),'field':'status','equals':'COMPLETED'}}
stages=[cmd('teacher-binding','evaluation',[],out/'arms/TEACHER_BINDING.json'),cmd('cache','maintenance',['teacher_binding'],out/'arms/CACHE_READY.json')]
for arm in ('S1','S2'):
    stage=cmd(arm,'kd',['teacher_binding','cache'],out/(arm+'_TRAINING_DONE.json'))
    stage['completion'].update(field='training_complete',equals=True)
    wrapper=out/(arm+'_stage.py');original=stage['argv']
    wrapper.write_text('import subprocess,json,pathlib,sys\n'+
        'r=subprocess.run('+repr(original)+',cwd='+repr(str(repo))+')\n'+
        'p=pathlib.Path('+repr(str(out/'arms'/arm/'run/RUN_STATUS.json'))+')\n'+
        's=json.loads(p.read_text()) if p.exists() else {}\n'+
        'pathlib.Path('+repr(stage['completion']['path'])+').write_text(json.dumps(s,indent=2)+"\\n")\n'+
        'raise SystemExit(r.returncode)\n')
    wrapper_pin=add(arm.lower()+'_stage_source','source',wrapper)
    stage['argv']=[python,str(wrapper)]
    stage['training']={'steps':10000,'recipe_pin':recipe_pin,'source_pin_ids':[*sources,wrapper_pin],'asset_pin_ids':assets,
        'progress_source':{'path':str(out/'arms'/arm/'run/segments/segment_0001/TRAIN_LOG.jsonl'),'step_field':'step','step_offset':1,'step_duration_seconds_field':'timings.step_elapsed_seconds'}}
    stage['pin_ids']=[recipe_pin,*sources,wrapper_pin,*assets]
    stage['quality_gate']={'stage':'teacher_binding','metric':'Macro4','field':'metrics.Macro4','minimum':.87}
    stages.append(stage)
# All backup stages are placed after both finite training stages.
for arm in ('S1','S2'):
    stages.append({'id':arm.lower()+'_backup_eval','kind':'command','role':'backup','requires':[arm],
        'argv':[python,str(repo/'scripts/run_server_compact.py'),'backup','--run-root',str(out/'arms'/arm/'run'),
            '--rclone-destination',ready['backup_destination'],'--device','cuda'],
        'cwd':str(repo),'pin_ids':[recipe_pin,*sources,*assets],
        'completion':{'path':str(out/'arms'/arm/'run/RUN_STATUS.json'),'field':'status','equals':'COMPLETED'},
        'failure_outcomes':[{'field':'status','equals':'UPLOAD_PENDING','label':'UPLOAD_PENDING'}]})
backup_script=out/'backup_existing_b8.py';backup_pin=add('b8_final_backup_script','source',backup_script)
stages.append({'id':'b8_backup_final','kind':'command','role':'backup','requires':[],
    'argv':[python,str(backup_script)],'cwd':str(repo),'pin_ids':[backup_pin,recipe_pin,*sources,*assets],
    'completion':{'path':str(out/'B8_BACKUP_FINAL.json'),'field':'status','equals':'COMPLETED'},
    'failure_outcomes':[{'field':'status','equals':'UPLOAD_PENDING','label':'UPLOAD_PENDING'}]})
plan={'schema':'self_audit.server_campaign.v1','campaign_id':'pro4000_kd_local_20261102','campaign_mode':'training',
    'state_dir':str(out/'state'),'progress_receipts_jsonl':str(out/'progress.jsonl'),'max_wall_clock_seconds':172800,
    'poll_seconds':30,'continue_independent_on_failure':True,'gpu':{'visible_device':'0'},'pins':pins,'stages':stages,
    'env_allowlist':{'RCLONE_CONFIG':'/root/.config/selfaudit-20261101/rclone.conf'},'tmux_target':'ssh_tmux'}
validate_campaign(plan);path=out/'CAMPAIGN.json';atomic_json(path,plan)
atomic_json(out/'PREPARATION.json',{'status':'READY','source':source,'campaign_plan_sha256':sha256_file(path),
    'ready_plan_sha256':rsha,'teacher_TEST50_Macro4':teacher['metrics']['test50']['Macro4'],
    'teacher_backup_status':'UPLOAD_PENDING','stage_order':[x['id'] for x in stages],
    'estimated_cache_bytes':cache_disk_bytes(Path(ready['train']['path']),ready['train']['sha256']),
    'generator_sha256':sha256_file(__file__),'old_joint_retained_at_step':1000})
print(json.dumps({'status':'READY','plan':str(path),'sha256':sha256_file(path),'stage_order':[s['id'] for s in stages]},indent=2))
