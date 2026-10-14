import datetime,json,os,pathlib,subprocess,sys,time
root=pathlib.Path(__file__).resolve().parent
cfg=json.loads((root/'EXECUTION.json').read_text())
name=sys.argv[1]
script=['scripts/diagnose_stable_guidance.py']
commands={
'help':script+['--help'],
'math_checks':script+['probes','--out',str(root/'math_checks')],
'val_counts':script+['metrics','--evaluation',cfg['original_job']+'/run/evaluation_val_verified_grid','--out',str(root/'val_counts')],
 'train_one_batch':script+['audit','--run',cfg['original_job'],'--max-patients','1','--max-batches','1','--out',str(root/'train_one_batch')],
 'train_all':script+['audit','--run',cfg['original_job'],'--max-patients','80','--max-batches','1','--out',str(root/'train_all')],
 'tests':['-m','pytest','-q','tests/test_self_audit_stable_guidance.py','tests/test_self_audit_guidance_full.py','tests/test_guidance_final_metrics.py','tests/test_guidance_wandb_runner.py','tests/test_guidance_forensics.py','--junitxml='+str(root/'tests.xml')]
}
command=[cfg['python']]+commands[name]
env=os.environ.copy()
settings={'PYTHONPATH':cfg['source']+'/src','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
env.update(settings)
receipt={'command':command,'cwd':cfg['source'],'environment_overrides':settings,'started_at':datetime.datetime.now().astimezone().isoformat(),'status':'RUNNING'}
receipt_path=root/(name+'.receipt.json')
with receipt_path.open('x') as f: json.dump(receipt,f,indent=2)
print('START '+name,flush=True)
started=time.monotonic()
with (root/(name+'.log')).open('x') as log:
    result=subprocess.run(command,cwd=cfg['source'],env=env,stdout=log,stderr=subprocess.STDOUT)
receipt.update(exit_code=result.returncode,elapsed_seconds=time.monotonic()-started,finished_at=datetime.datetime.now().astimezone().isoformat(),status='FINISHED' if result.returncode==0 else 'FAILED')
receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
print((root/(name+'.log')).read_text()[-12000:])
sys.exit(result.returncode)
