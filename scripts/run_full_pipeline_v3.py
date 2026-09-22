#!/usr/bin/env python3
"""One-command Self-Audit v3 research pipeline: teacher -> freeze -> pseudo eval -> optional student."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time
from pathlib import Path

def log(tag,msg,fh=None):
    line=f"[{tag}] {msg}"
    print(line,flush=True)
    if fh is not None:
        fh.write(line+"\n");fh.flush()

def run_stage(name,cmd,env,fh):
    log("STAGE",f"{name} start",fh)
    log("CMD"," ".join(str(x) for x in cmd),fh)
    started=time.perf_counter()
    proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=env)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line,end="",flush=True);fh.write(line);fh.flush()
    code=proc.wait()
    elapsed=time.perf_counter()-started
    if code!=0:
        log("ERROR",f"{name} failed rc={code} after {elapsed:.1f}s",fh)
        raise subprocess.CalledProcessError(code,cmd)
    log("STAGE",f"{name} done time={elapsed:.1f}s",fh)
    return elapsed

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",choices=["acdc","mnms"],required=True)
    p.add_argument("--root",required=True)
    p.add_argument("--split-manifest",required=True)
    p.add_argument("--config",default="configs/pseudolabel_v3.json")
    p.add_argument("--out",required=True)
    p.add_argument("--device",default="cuda")
    p.add_argument("--teacher-epochs",type=int,default=3)
    p.add_argument("--student-epochs",type=int,default=10)
    p.add_argument("--batch-size",type=int,default=1)
    p.add_argument("--student-batch-size",type=int)
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--profile",choices=["compact","balanced","accurate"],default="balanced")
    p.add_argument("--max-train-batches",type=int,default=0,help="0 means full teacher epoch")
    p.add_argument("--student-max-train-batches",type=int,default=0)
    p.add_argument("--references",help="ACDC reference root; defaults to --root")
    p.add_argument("--reference-manifest",help="required for M&Ms evaluation")
    p.add_argument("--train-student",action="store_true",help="continue to student after frozen pseudo-label evaluation")
    p.add_argument("--dry-run",action="store_true")
    args=p.parse_args(argv)
    if min(args.teacher_epochs,args.student_epochs,args.batch_size,args.threads)<1:
        raise ValueError("epochs, batch size and threads must be positive")
    run=Path(args.out).resolve()
    if run.exists(): raise FileExistsError(run)
    run.mkdir(parents=True)
    log_path=run/"pipeline.log"
    env=os.environ.copy()
    repo=Path(__file__).resolve().parents[1]
    env["PYTHONPATH"]=str(repo/"src")+(os.pathsep+env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    py=sys.executable
    teacher=run/"teacher"
    evaluation=run/"evaluation_val.json"
    student=run/f"student_{args.profile}.pt"
    teacher_cmd=[py,str(repo/"scripts/train_pseudolabel_v3.py"),
        "--dataset",args.dataset,"--root",args.root,"--split-manifest",args.split_manifest,
        "--config",args.config,"--export-split","train,val","--out",str(teacher),
        "--epochs",str(args.teacher_epochs),"--batch-size",str(args.batch_size),
        "--threads",str(args.threads),"--seed",str(args.seed),"--device",args.device]
    if args.max_train_batches:
        teacher_cmd+=["--max-train-batches",str(args.max_train_batches)]
    eval_cmd=[py,str(repo/"scripts/evaluate_pseudolabel_frozen.py"),
        "--run",str(teacher),"--split","val","--out",str(evaluation)]
    if args.reference_manifest:
        eval_cmd+=["--reference-manifest",args.reference_manifest]
    elif args.dataset=="acdc":
        eval_cmd+=["--references",args.references or args.root]
    else:
        raise ValueError("M&Ms requires --reference-manifest for independent evaluation")
    sb=args.student_batch_size or args.batch_size
    student_cmd=[py,str(repo/"scripts/train_student_v3.py"),
        "--dataset",args.dataset,"--root",args.root,"--manifest",str(teacher/"FROZEN.json"),
        "--out",str(student),"--profile",args.profile,"--epochs",str(args.student_epochs),
        "--batch-size",str(sb),"--threads",str(args.threads),"--seed",str(args.seed),"--device",args.device]
    if args.student_max_train_batches:
        student_cmd+=["--max-train-batches",str(args.student_max_train_batches)]
    with log_path.open("x",encoding="utf-8") as fh:
        try:
            import torch
            gpu=torch.cuda.get_device_name(0) if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
            log("ENV",f"python={sys.version.split()[0]} torch={torch.__version__} device={args.device} gpu={gpu}",fh)
        except Exception as exc:
            log("ENV",f"torch probe failed: {exc}",fh)
        log("PIPELINE",f"dataset={args.dataset} teacher_epochs={args.teacher_epochs} student={'on' if args.train_student else 'off'} out={run}",fh)
        if args.dry_run:
            log("DRYRUN","teacher: "+" ".join(teacher_cmd),fh)
            log("DRYRUN","eval: "+" ".join(eval_cmd),fh)
            if args.train_student: log("DRYRUN","student: "+" ".join(student_cmd),fh)
            return 0
        times={}
        times["teacher"]=run_stage("teacher",teacher_cmd,env,fh)
        times["evaluation"]=run_stage("pseudo-eval",eval_cmd,env,fh)
        scores=json.loads(evaluation.read_text())
        log("METRIC",f"pseudo fg={scores.get('foreground_mean')} RV={scores.get('rv')} MYO={scores.get('myo')} LV={scores.get('lv')} known={scores.get('known_fraction')}",fh)
        if args.train_student:
            times["student"]=run_stage("student",student_cmd,env,fh)
        summary={"status":"complete","dataset":args.dataset,"teacher_dir":str(teacher),
                 "evaluation":str(evaluation),"student":str(student) if args.train_student else None,
                 "times_seconds":times,"pseudo_metrics":{k:scores.get(k) for k in ("foreground_mean","rv","myo","lv","known_fraction")}}
        (run/"PIPELINE_SUMMARY.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
        log("DONE",f"pipeline out={run} summary={run/'PIPELINE_SUMMARY.json'}",fh)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
