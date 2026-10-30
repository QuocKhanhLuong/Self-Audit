import os,json,subprocess,time,hashlib,sys,threading
from pathlib import Path
ROOT=Path('/workspace/Self-Audit-cptu-20261030')
OUT=Path('/workspace/cptu-server-setup-20261030/verification_v3')
OUT.mkdir(exist_ok=False)
sys.path.insert(0,str(ROOT))
from self_audit.compact_artifacts import compact_training_source_sha256,digest
import torch,psutil
PIN=sys.argv[1]
assert subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()==PIN
assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()
source=compact_training_source_sha256()
assert source==sys.argv[2]
processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
assert not processes,'GPU has existing compute jobs: '+processes
ENV=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'); ENV['PATH']=str(Path(sys.executable).parent)+os.pathsep+ENV.get('PATH','')
receipt={'status':'RUNNING','source_git_sha':PIN,'source_sha256':source,'commands':[],
'environment':{'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'cpu_threads':1},
'scope':'software and manufactured CUDA verification only; no patient training'}
stop=threading.Event()
def write(): (OUT/'SERVER_VERIFICATION.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
def monitor():
 with (OUT/'RESOURCE_SAMPLES.jsonl').open('w') as f:
  while not stop.is_set():
   gpu=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True)
   proc=psutil.Process();procs=[proc]+proc.children(recursive=True)
   ram=0
   for p in procs:
    try:ram+=p.memory_info().rss
    except psutil.Error:pass
   f.write(json.dumps({'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'gpu_csv':gpu.stdout.strip(),'owned_process_tree_rss_bytes':ram})+'\n');f.flush();stop.wait(2)
def run(name,args,timeout=600):
 t=time.monotonic();cmd=[sys.executable,*map(str,args)]
 receipt['current_command']=name;write()
 with (OUT/(name+'.stdout')).open('w') as out,(OUT/(name+'.stderr')).open('w') as err:
  p=subprocess.run(cmd,cwd=ROOT,env=ENV,stdout=out,stderr=err,timeout=timeout)
 receipt['commands'].append({'name':name,'argv':cmd,'exit_code':p.returncode,'seconds':time.monotonic()-t,
 'stdout_sha256':digest(OUT/(name+'.stdout')),'stderr_sha256':digest(OUT/(name+'.stderr'))});write()
 print(name,p.returncode,flush=True)
 if p.returncode:raise RuntimeError(name+' failed; inspect logs')
def equal(a,b):
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
 if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
 if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b
thread=threading.Thread(target=monitor,daemon=True);thread.start()
try:
 write();run('check_repo',['scripts/check_repo.py'])
 run('original_parity',['scripts/verify_compact_original_parity.py','--out',OUT/'ORIGINAL_PARITY.json'])
 smoke=OUT/'smoke_cuda'
 run('smoke_cuda',['scripts/smoke_compact_pipeline.py','--transfer-utility','--device','cuda:0','--out',smoke])
 s=json.loads((smoke/'SMOKE_REPORT.json').read_text());cfg=smoke/'cross_patient_utility.json'
 train=smoke/'train_data/train.json';trainsha=digest(train)
 common=['train.py','--config',cfg,'--data',train,'--data-sha256',trainsha,'--synthetic','--device','cuda:0']
 run('cuda_segment1',[*common,'--stop-after-step','2','--out',OUT/'cuda_segment1'])
 cp=OUT/'cuda_segment1/final'
 run('cuda_segment2',[*common,'--resume',cp,'--resume-sha256',digest(cp/'MANIFEST.json'),'--out',OUT/'cuda_segment2'])
 original=torch.load(smoke/'cross_patient_utility/final/weights.pt',map_location='cpu',weights_only=True)
 resumed=torch.load(OUT/'cuda_segment2/final/weights.pt',map_location='cpu',weights_only=True)
 for key in ['student','optimizer','rng']:assert equal(original[key],resumed[key]),'CUDA exact resume differs: '+key
 a=[json.loads(x) for x in (smoke/'cross_patient_utility/TRAIN_LOG.jsonl').read_text().splitlines()]
 b=[json.loads(x) for x in (OUT/'cuda_segment2/TRAIN_LOG.jsonl').read_text().splitlines()]
 for x,y in zip(a,b):
  for key in ['data_schedule_sha256','status','scale','realized_displacement_norm']:assert x[key]==y[key],key
 receipt['cuda_exact_resume']='PASS'
 code='from scripts.run_transfer_utility_pilot import run_bounded_profile; import sys; r=run_bounded_profile("configs/transfer_utility/profile_32steps.json",'+repr(str(OUT/'profile_cuda'))+',device="cuda:0",synthetic=True,max_steps=32);sys.exit(0 if r["status"]=="PASS" else 2)'
 run('profile_cuda',['-c',code])
 run('pytest',['-m','pytest','-q'])
 assert source==compact_training_source_sha256(),'Source changed during verification'
 receipt.update(status='PASS',completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),patient_training='NOT_RUN_WAITING_VERIFIED_ASSETS')
except BaseException as e:
 receipt.update(status='FAIL',error=repr(e));raise
finally:
 stop.set();thread.join(timeout=5);write()
