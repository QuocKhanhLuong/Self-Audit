#!/usr/bin/env python3
"""One-command Self-Audit v3 research pipeline: teacher -> freeze -> pseudo eval -> optional student."""
from __future__ import annotations
import argparse,codecs,json,os,subprocess,sys,time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from self_audit_pseudolabel.wandb_v3 import WandbV3Tracker
from self_audit_pseudolabel.progress_v3 import add_progress_arguments,EVENT_PREFIX,EVENT_SCHEMA

def log(tag,msg,fh=None):
    line=f"[{tag}] {msg}"
    print(line,flush=True)
    if fh is not None:
        fh.write(line+"\n");fh.flush()

def _stream_event(line, name, tracker):
    if tracker is None or not line.startswith(EVENT_PREFIX):
        return
    try:
        event=json.loads(line[len(EVENT_PREFIX):])
    except (ValueError,TypeError):
        return
    if (isinstance(event,dict) and event.get('schema')==EVENT_SCHEMA
            and event.get('stage')==name and isinstance(event.get('metrics'),dict)):
        tracker.log({name:event['metrics']})


def run_stage(name,cmd,env,fh,tracker=None):
    log("STAGE",f"{name} start",fh)
    log("CMD"," ".join(str(x) for x in cmd),fh)
    started=time.perf_counter()
    child_env=dict(env); child_env['PYTHONUNBUFFERED']='1'
    proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=child_env)
    assert proc.stdout is not None
    decoder=codecs.getincrementaldecoder('utf-8')(errors='replace')
    pending=''
    # read1 forwards available bytes, including tqdm carriage returns, without
    # waiting for a newline (or an entire epoch). Parse only complete records.
    while True:
        chunk=proc.stdout.read1(65536)
        text=decoder.decode(chunk,final=not chunk)
        if text:
            print(text,end='',flush=True)
            fh.write(text.replace('\r','\n'));fh.flush()
            pending+=text.replace('\r','\n')
            lines=pending.split('\n');pending=lines.pop()
            for line in lines: _stream_event(line,name,tracker)
        if not chunk: break
    if pending: _stream_event(pending,name,tracker)
    proc.stdout.close()
    code=proc.wait()
    elapsed=time.perf_counter()-started
    if code!=0:
        log("ERROR",f"{name} failed rc={code} after {elapsed:.1f}s",fh)
        raise subprocess.CalledProcessError(code,cmd)
    log("STAGE",f"{name} done time={elapsed:.1f}s",fh)
    return elapsed


def build_parser():
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
    add_progress_arguments(p)
    w=p.add_mutually_exclusive_group()
    w.add_argument("--wandb",dest="wandb",action="store_true",default=False,help="enable W&B tracking")
    w.add_argument("--no-wandb",dest="wandb",action="store_false",help="disable W&B tracking")
    p.add_argument("--wandb-mode",choices=["online","offline","disabled"],default="disabled")
    p.add_argument("--wandb-project",default="self-audit-v3")
    p.add_argument("--wandb-entity",default=None)
    p.add_argument("--wandb-run-name",default=None)
    return p


def _log_stage_metrics(tracker, stage, path):
    tracker.log_json(stage, path)

def main(argv=None):
    args=build_parser().parse_args(argv)
    if min(args.teacher_epochs,args.student_epochs,args.batch_size,args.threads)<1:
        raise ValueError("epochs, batch size and threads must be positive")
    if args.log_every<0: raise ValueError("log-every must be nonnegative")
    run=Path(args.out).resolve()
    if run.exists(): raise FileExistsError(run)
    run.mkdir(parents=True)
    log_path=run/"pipeline.log"
    env=os.environ.copy()
    repo=REPO_ROOT
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
    student_predictions=run/"student_predictions_val"
    student_evaluation=run/"student_evaluation_val.json"
    infer_cmd=[py,str(repo/"scripts/infer_student_v3.py"),"--source-freeze",str(teacher/"FROZEN.json"),
        "--checkpoint",str(student),"--root",args.root,"--out",str(student_predictions),
        "--split","val","--device",args.device,"--threads",str(args.threads)]
    student_eval_cmd=list(eval_cmd)
    student_eval_cmd[student_eval_cmd.index("--run")+1]=str(student_predictions)
    student_eval_cmd[student_eval_cmd.index("--out")+1]=str(student_evaluation)
    for cmd in (teacher_cmd,student_cmd):
        cmd += ['--log-every',str(args.log_every)]
    if args.no_progress:
        for cmd in (teacher_cmd,student_cmd,infer_cmd): cmd.append('--no-progress')
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
            if args.train_student:
                for name,cmd in [("student",student_cmd),("student-inference",infer_cmd),("student-eval",student_eval_cmd)]:
                    log("DRYRUN",name+": "+" ".join(cmd),fh)
            return 0
        tracker=WandbV3Tracker(
            enabled=args.wandb,
            mode=args.wandb_mode,
            project=args.wandb_project,
            entity=args.wandb_entity,
            run_name=args.wandb_run_name or run.name,
            run_dir=run/"wandb",
            config={"dataset":args.dataset,"seed":args.seed,"device":args.device,
                    "teacher_epochs":args.teacher_epochs,"student_epochs":args.student_epochs,
                    "profile":args.profile,"train_student":args.train_student},
        )
        for warning in tracker.warnings:
            log("WANDB",warning,fh)

        def tracked_stage(name,cmd):
            tracker.log({"pipeline":{"stage":name,"status":"started"}})
            elapsed=run_stage(name,cmd,env,fh,tracker=tracker)
            tracker.log({"pipeline":{"stage":name,"status":"completed","elapsed_seconds":elapsed}})
            return elapsed

        try:
            tracker.log({"pipeline":{"status":"started","dataset":args.dataset,"device":args.device}})
            times={}
            times["teacher"]=tracked_stage("teacher",teacher_cmd)
            _log_stage_metrics(tracker,"teacher_summary",teacher/"run_summary.json")
            times["evaluation"]=tracked_stage("pseudo-eval",eval_cmd)
            _log_stage_metrics(tracker,"evaluation",evaluation)
            scores=json.loads(evaluation.read_text())
            log("METRIC",f"pseudo fg={scores.get('foreground_mean')} RV={scores.get('rv')} MYO={scores.get('myo')} LV={scores.get('lv')} known={scores.get('known_fraction')}",fh)
            if args.train_student:
                times["student"]=tracked_stage("student",student_cmd)
                tracker.log_json("student_summary",student.with_suffix(".json"),include_history=False)
                times["student_inference"]=tracked_stage("student-inference",infer_cmd)
                times["student_evaluation"]=tracked_stage("student-eval",student_eval_cmd)
                _log_stage_metrics(tracker,"student_evaluation",student_evaluation)
            summary={"status":"complete","dataset":args.dataset,"teacher_dir":str(teacher),
                     "student_evaluation":str(student_evaluation) if args.train_student else None,
                     "student_predictions":str(student_predictions) if args.train_student else None,
                     "evaluation":str(evaluation),"student":str(student) if args.train_student else None,
                     "times_seconds":times,"pseudo_metrics":{k:scores.get(k) for k in ("foreground_mean","rv","myo","lv","known_fraction")}}
            (run/"PIPELINE_SUMMARY.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
            tracker.set_summary(summary)
            tracker.log({"pipeline":{"status":"complete","times_seconds":times,"pseudo_metrics":summary["pseudo_metrics"]}})
            log("DONE",f"pipeline out={run} summary={run/'PIPELINE_SUMMARY.json'}",fh)
        finally:
            tracker.finish()
            for warning in tracker.warnings:
                log("WANDB",warning,fh)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
