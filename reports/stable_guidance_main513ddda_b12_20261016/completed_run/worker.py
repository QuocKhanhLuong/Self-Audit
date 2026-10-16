"""Execution-host orchestration; the archived main pipeline is unmodified."""
from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback

JOB=Path(__file__).resolve().parent
M=json.loads((JOB/'LAUNCH_MANIFEST.json').read_text())
SOURCE=JOB/'source'
STATE={'status':'STARTING','stage':'host_checks','git_commit':M['git_commit'],
       'epochs':M['epochs'],'worker_pid':os.getpid(),'started_at':time.time()}
CHILD=None

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)

def status(**values):
    STATE.update(values,updated_at=time.time());write(JOB/'STATUS.json',STATE)
    print('[WORKER]',json.dumps(STATE),flush=True)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1048576),b''):h.update(part)
    return h.hexdigest()

def source_check():
    for name,digest in M['source_sha256'].items():
        if sha(SOURCE/name)!=digest:raise ValueError('Source changed: '+name)
    for filename,key in [('SPLIT.json','split_sha256'),('CONFIG.json','config_sha256')]:
        if sha(JOB/filename)!=M[key]:raise ValueError('Input changed: '+filename)

def run(command,log=None):
    global CHILD
    print('[COMMAND]',json.dumps(command),flush=True)
    output=log.open('x') if log else None
    try:
        CHILD=subprocess.Popen(command,cwd=SOURCE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                               text=True,bufsize=1,start_new_session=True)
        status(child_pid=CHILD.pid)
        for line in CHILD.stdout:
            print(line,end='',flush=True)
            if output:output.write(line);output.flush()
        code=CHILD.wait();CHILD=None;status(child_pid=None)
        return code
    finally:
        if output:output.close()

def stopped(signum,_):
    if CHILD is not None and CHILD.poll() is None:
        os.killpg(CHILD.pid,signum)
    status(status='INTERRUPTED',signal=signum)
    raise SystemExit(128+signum)

def verified_prediction_pair():
    from self_audit_guidance.io import verify
    output=JOB/'run';lock=json.loads((output/'PREDICTIONS_LOCKED.json').read_text())
    verify(output/'student','student')
    assert sha(output/'student/FROZEN.json')==lock['student_manifest_sha256']
    assert sha(output/'RECIPE.json')==lock['recipe_sha256']
    for split in ('val','test'):
        pred=verify(output/('predictions_'+split),'student_predictions')
        assert sha(output/('predictions_'+split)/'FROZEN.json')==lock['predictions'][split]
        assert pred['student_manifest_sha256']==lock['student_manifest_sha256']
    return lock,sha(output/'student/last.pt')

def recovery():
    output=JOB/'run';current=json.loads((output/'RUN_STATUS.json').read_text())
    if current.get('failed_stage')!='evaluate_val':raise RuntimeError('Pipeline failed before recoverable final evaluation')
    logs=sorted((output/'logs').glob('evaluate_val_*.log'))
    if not logs or 'reference/native geometry mismatch; no resizing is permitted' not in logs[-1].read_text():
        raise RuntimeError('Failure is not the documented ACDC frame-affine mismatch')
    status(status='RUNNING',stage='verified_grid_evaluation')
    lock,checkpoint_sha=verified_prediction_pair()
    directory=output/'evaluation_val_verified_grid'
    command=[sys.executable,str(SOURCE/'scripts/evaluate_stable_guidance.py'),
             '--predictions',str(output/'predictions_val'),'--out',str(directory),
             '--reference-root',str(Path(M['job']).parents[1]/'data/ACDC/training'),
             '--allow-verified-acdc-frame-grid']
    receipt_dir=JOB/'evaluation_recovery';receipt_dir.mkdir(exist_ok=True)
    write(receipt_dir/'COMMAND.json',{'command':command,'original_pipeline_status':current})
    code=run(command,receipt_dir/'evaluate.log')
    if code:raise RuntimeError('Verified-grid evaluation failed; see retained log')
    from self_audit_guidance.io import verify
    ev=verify(directory,'independent_evaluation')
    assert ev['prediction_manifest_sha256']==lock['predictions']['val']
    assert verified_prediction_pair()==(lock,checkpoint_sha)
    history=json.loads((output/'student/training.json').read_text())
    assert [r['epoch'] for r in history]==list(range(1,M['epochs']+1))
    report=json.loads((directory/'METRICS.json').read_text())
    from publish_guidance_metrics import evaluation_scalars
    metrics=evaluation_scalars(report,'val')
    metrics.update({'evaluation/epoch':M['epochs'],'val/unknown_fraction':1-metrics['val/coverage']})
    payload=dict(metrics=metrics,evaluation_directory=str(directory),
                 aggregation='frame_mean_then_patient_mean_then_foreground_class_mean',
                 undefined_precision_recall='null excluded from macro, with defined-patient counts',
                 auroc_status=M['auroc'],test_status='official_unlabeled_test_export_only',
                 geometry_recovery={'policy':'exact_frame_cine_voxel_identity_no_resampling',
                                    'verified_frames':report['verified_frame_geometry_count']},
                 training={'completed_epochs':len(history),'optimizer_steps':sum(r['steps'] for r in history)},
                 original_pipeline_status=current['status'])
    write(JOB/'tracking/FINAL_METRICS.json',payload)
    receipt=dict(status='EVALUATION_RECOVERED',evaluation_directory=str(directory),
                 original_pipeline_status='BLOCKED_at_evaluate_val',training_completed_epochs=len(history),
                 predictions_changed=False,checkpoint_changed=False,threshold_changed=False,
                 metrics_report_sha256=sha(directory/'METRICS.json'),evaluated_at=time.time())
    write(receipt_dir/'RECOVERY_STATUS.json',receipt)
    original=json.loads((JOB/'tracking/WANDB_RUN.json').read_text())
    status(stage='publish_final_metrics',wandb_url=original['url'])
    import wandb
    logdir=JOB/'tracking/recovered_metrics_wandb';logdir.mkdir(exist_ok=True)
    with wandb.init(project=original['project'],entity=original['entity'],id=original['id'],
                    resume='must',mode='online',dir=str(logdir)) as tracked:
        tracked.define_metric('evaluation/epoch');tracked.define_metric('val/*',step_metric='evaluation/epoch')
        tracked.log({k:v for k,v in metrics.items() if v is not None});tracked.summary.update(metrics)
        tracked.config.update({k:M[k] for k in ('git_commit','epochs','batch_size','width','lr','seed','orientation_policy')},allow_val_change=True)
        tracked.summary['evaluation/recovery_status']='COMPLETED_VERIFIED_FRAME_GRID'
        tracked.summary['evaluation/original_status']='BLOCKED_GEOMETRY_MISMATCH'
        tracked.summary['evaluation/auroc_status']=M['auroc']
        tracked.summary['training/completed_epochs']=len(history)
        tracked.summary['training/optimizer_steps']=sum(r['steps'] for r in history)
        tracked.summary['test/status']='NOT_SCORED_OFFICIAL_ACDC_TEST_GT_UNAVAILABLE'
    write(JOB/'tracking/METRICS_PUBLISHER.json',dict(status='published_recovered_evaluation',
          run_id=original['id'],url=original['url'],published_at=time.time()))
    remote=wandb.Api(timeout=30).run('/'.join([original['entity'],original['project'],original['id']]))
    assert abs(remote.summary['val/dice']-metrics['val/dice'])<1e-10
    write(JOB/'tracking/FINAL_WANDB_VERIFIED.json',dict(status='PASS',run_state=remote.state,
          verified_at=time.time(),metrics={k:remote.summary.get(k) for k in metrics}))

def main():
    os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=M['gpu_uuid'],
       OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
       PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',WANDB_MODE='online',
       WANDB_CACHE_DIR=str(JOB/'wandb_cache'),WANDB_DATA_DIR=str(JOB/'wandb_data'),
       WANDB_CONFIG_DIR=str(JOB/'wandb_config'),WANDB_INIT_TIMEOUT='90',
       WANDB_RUN_GROUP='stable_guidance_main513ddda_50ep_val18',
       WANDB_NOTES='Unmodified main 513ddda; docs/stable_guidance_full.md; 50 epochs, 80 TRAIN/18 VAL/50 official TEST; cardinal_v1, not the historical world_left_v2 pilot.')
    os.environ.pop('PYTHONPATH',None)
    sys.path.insert(0,str(SOURCE/'src'));sys.path.insert(0,str(SOURCE/'scripts'))
    source_check()
    while True:
        free=int(subprocess.check_output(['nvidia-smi','-i',M['gpu_uuid'],'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        if free>=10000 and M['gpu_uuid'] not in processes:break
        status(status='WAITING_FOR_GPU',stage='gpu_wait',gpu_free_mib=free);time.sleep(15)
    import torch
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    assert torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 4070'
    x=torch.ones(8,device='cuda');assert x.sum().item()==8;del x;torch.cuda.empty_cache()
    write(JOB/'DEVICE_CHECK.json',dict(status='PASS',cuda_visible_devices=M['gpu_uuid'],
        torch=torch.__version__,torch_cuda=torch.version.cuda,visible_count=1,device=torch.cuda.get_device_name(0)))
    (JOB/'STARTED_AT').write_text(datetime.datetime.now().astimezone().isoformat()+'\n')
    status(status='RUNNING',stage='full_pipeline',gpu_free_mib=free)
    code=run(M['command'])
    (JOB/'EXIT_CODE').write_text(str(code)+'\n')
    pipeline=json.loads((JOB/'run/RUN_STATUS.json').read_text()) if (JOB/'run/RUN_STATUS.json').exists() else {}
    status(stage='pipeline_exited',pipeline_exit_code=code,pipeline_status=pipeline.get('status'))
    if not str(pipeline.get('status')).startswith('COMPLETED'):
        recovery()
    elif code not in (0,2):raise RuntimeError('Unexpected pipeline exit code')
    source_check()
    status(status='FINISHED',stage='complete',finished_at=time.time())
    (JOB/'FINAL_EXIT_CODE').write_text('0\n')

if __name__=='__main__':
    lock=(JOB/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stopped)
    try:main()
    except Exception as exc:
        status(status='FAILED',error=f'{type(exc).__name__}: {exc}',failed_at=time.time())
        (JOB/'FINAL_EXIT_CODE').write_text('1\n');traceback.print_exc();raise
    finally:
        (JOB/'FINISHED_AT').write_text(datetime.datetime.now().astimezone().isoformat()+'\n')
