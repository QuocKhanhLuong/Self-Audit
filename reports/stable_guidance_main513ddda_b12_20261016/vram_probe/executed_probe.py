from pathlib import Path
import argparse,contextlib,gc,json,os,signal,statistics,subprocess,sys,time,traceback

p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--pause-pid',type=int,required=True);a=p.parse_args()
job=a.job.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
m=json.loads((job/'LAUNCH_MANIFEST.json').read_text());sys.path.insert(0,str(job/'source/src'))
os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=m['gpu_uuid'],OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1')
report=dict(status='RUNNING',source_commit=m['git_commit'],parent_job=str(job),started_at=time.time(),teacher=[],student=[],gt_read=False,probe_weights_discarded=True)
def save():
    t=out/'REPORT.tmp';t.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');t.replace(out/'REPORT.json')
paused=False
def stop(signum,_):raise KeyboardInterrupt('signal '+str(signum))
for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,stop)
try:
    import numpy as np
    import torch
    from self_audit_guidance.io import read_cine,load_bank_patient
    from self_audit_guidance.teacher import FrozenCineMA
    from self_audit_guidance.student import GuidanceStudent,guidance_loss,contexts
    from self_audit_guidance.core import GuidanceConfig,make_bank
    from torch.nn import functional as F
    torch.set_num_threads(4)
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 4070'
    command=Path(f'/proc/{a.pause_pid}/cmdline').read_bytes()
    assert str(job).encode() in command and b'prepare' in command
    os.kill(a.pause_pid,signal.SIGSTOP);paused=True
    report['paused_pid']=a.pause_pid;save()
    pf=json.loads(sorted((job/'run').glob('PREFLIGHT_*.json'))[-1].read_text())
    inventory=pf['checks']['native_cine_inventory'];images=inventory['images']
    pid=max(inventory['splits']['train'],key=lambda k:images[k]['shape_xyzt'][0]*images[k]['shape_xyzt'][1])
    cine,meta=read_cine(images[pid]['raw_image']);sample=cine[:2].copy();del cine
    cmd=m['command'];ck=cmd[cmd.index('--checkpoint')+1];cfg=cmd[cmd.index('--teacher-config')+1]
    teacher=FrozenCineMA('cuda',ck,cfg,offline=True)
    report.update(device=torch.cuda.get_device_name(0),teacher_parameters=sum(v.numel() for v in teacher.model.parameters()),
                  teacher_trainable_parameters=sum(v.numel() for v in teacher.model.parameters() if v.requires_grad),
                  max_train_patient=pid,native_probe_shape=list(sample.shape),coexisting_prepare_mib=930,
                  note='Paused original prepare retains its CUDA allocations; measured peak fields refer only to this probe process. No scientific output is changed.')
    baseline=None
    for batch in [1,2,4,8]:
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        row={'tile_batch':batch};times=[]
        try:
            for repeat in range(3):
                torch.cuda.synchronize();started=time.perf_counter()
                features=teacher.extract(sample,grid=48,feature_dim=64,tile_batch=batch)
                torch.cuda.synchronize();times.append(time.perf_counter()-started)
            if baseline is None:baseline=features.copy()
            row.update(status='PASS',seconds=times,median_seconds=statistics.median(times),
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                max_abs_feature_difference=float(np.max(np.abs(features.astype(float)-baseline.astype(float)))))
        except torch.OutOfMemoryError:
            row.update(status='OOM',peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        report['teacher'].append(row);save();print('TEACHER',json.dumps(row),flush=True)
    del teacher,features;gc.collect();torch.cuda.empty_cache()
    config=GuidanceConfig(**json.loads((job/'CONFIG.json').read_text()))
    small=F.interpolate(torch.from_numpy(sample).flatten(0,1)[:,None],(48,48),mode='bilinear',align_corners=False)
    bank,bank_report=make_bank(small[:,0].reshape(*sample.shape[:2],48,48).numpy(),baseline,meta['patient_left_axis'],config)
    cases=[(pid,dict(image=sample,features=baseline,**bank)),('patient003',load_bank_patient(job/'run/bank','patient003'))]
    def student_case(pid,data,batch):
        torch.manual_seed(42);torch.cuda.reset_peak_memory_stats()
        model=GuidanceStudent(64,32).to('cuda').train();opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=1e-4)
        ids=list(np.ndindex(data['image'].shape[:2]));indices=np.array([ids[i%len(ids)] for i in range(batch)])
        t,z=indices[:,0],indices[:,1]
        x=torch.from_numpy(contexts(data['image'],indices)).to('cuda')
        f=torch.from_numpy(data['features'][t,z].astype(np.float32)).permute(0,3,1,2).to('cuda')
        q=torch.from_numpy(data['q'][t,z]).permute(0,3,1,2).to('cuda')
        w=torch.from_numpy(data['reliability'][t,z]).to('cuda')
        times=[]
        for step in range(5):
            torch.cuda.synchronize();start=time.perf_counter()
            opt.zero_grad(set_to_none=True);pred=model(x);fl,sl,active=guidance_loss(pred,f,q,w,True)
            loss=fl+sl;assert torch.isfinite(loss);loss.backward()
            if not active:
                for param in model.semantic_parameters():param.grad=None
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(norm)
            opt.step();torch.cuda.synchronize();times.append(time.perf_counter()-start)
        return dict(status='PASS',patient=pid,batch_size=batch,semantic_active=active,seconds=times,
                    median_step_seconds=statistics.median(times[2:]),samples_per_second=batch/statistics.median(times[2:]),
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    for pid,data in cases:
        for batch in [8,10,12]:
            try:row=student_case(pid,data,batch)
            except torch.OutOfMemoryError:row=dict(status='OOM',patient=pid,batch_size=batch)
            gc.collect();torch.cuda.empty_cache();report['student'].append(row);save();print('STUDENT',json.dumps(row),flush=True)
    report.update(status='COMPLETED',finished_at=time.time())
except BaseException as exc:
    report.update(status='FAILED',error=type(exc).__name__+': '+str(exc));traceback.print_exc()
finally:
    if paused:
        try:os.kill(a.pause_pid,signal.SIGCONT);report['original_prepare_resumed']=True
        except ProcessLookupError:report['original_prepare_resumed']=False
    save()
    print('FINAL',str(out/'REPORT.json'),report['status'],flush=True)
