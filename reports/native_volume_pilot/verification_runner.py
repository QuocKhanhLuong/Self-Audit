"""Replay bounded CPU software verification into a new output directory.

Usage: python reports/native_volume_pilot/verification_runner.py [new-output-dir]
All data and weights are manufactured; this does not run patient training.
"""
import sys, json, subprocess, time, tempfile, hashlib, copy
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from self_audit.volume.checkpoint import source_sha256
from self_audit.volume.fixtures import make_volume_fixture
from scripts.volume_preflight import synthetic_runtime_config
from self_audit.volume.evaluation import load_checkpoint_student_only, load_exported_student
OUT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else ROOT/"reports/native_volume_pilot/verified"
OUT.mkdir(exist_ok=False)
SRC=source_sha256()
commands=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(name,args,cwd="/tmp",timeout=120):
    start=time.perf_counter()
    p=subprocess.run([sys.executable,*[str(v) for v in args]],cwd=cwd,text=True,capture_output=True,timeout=timeout)
    (OUT/(name+".stdout")).write_text(p.stdout)
    (OUT/(name+".stderr")).write_text(p.stderr)
    rec={"name":name,"argv":[sys.executable,*[str(v) for v in args]],"cwd":cwd,"exit_code":p.returncode,"wall_seconds":time.perf_counter()-start}
    commands.append(rec)
    (OUT/"COMMANDS.json").write_text(json.dumps(commands,indent=2)+"\n")
    print(name,p.returncode,round(rec["wall_seconds"],3),flush=True)
    if p.returncode:raise RuntimeError(name+": "+p.stderr[-2000:]+" "+p.stdout[-2000:])
    try:return json.loads(p.stdout)
    except json.JSONDecodeError:return p.stdout
run("check_repo",[ROOT/"scripts/check_repo.py"],cwd=str(ROOT))
tests=run("pytest",["-m","pytest","-q"],cwd=str(ROOT),timeout=120)
ids=["V"+f"{i:02d}" for i in range(9)]+["O01","O02","O03","S01","S02","R00","R01","R02","T00","T01","I00","I01","C00","C01"]
smokes={}
for arm in ids:
    cfg=ROOT/f"configs/native_volume/{arm}.json"
    run("preflight_"+arm,[ROOT/"scripts/volume_preflight.py","--config",cfg])
    args=[ROOT/"scripts/volume_smoke.py","--config",cfg,"--steps","2","--crop-size","32","--batch","1","--device","cpu","--max-seconds","60","--output",OUT/("smoke_"+arm+".json")]
    if arm in ("I00","I01"):args+=["--correction-config",ROOT/f"configs/native_volume/{arm}_correction.json"]
    smokes[arm]=run("smoke_"+arm,args)
profiles={}
for arm in ["V01","V04","V06"]:
    profiles[arm]=run("profile_"+arm,[ROOT/"scripts/volume_profile.py","--config",ROOT/f"configs/native_volume/{arm}.json","--iterations","10","--warmup","2","--crop-size","32","--microbatch","1","--device","cpu","--max-minutes","1","--output",OUT/("profile_"+arm+".json")])
with tempfile.TemporaryDirectory(prefix="native-volume-final-") as temp:
    tmp=Path(temp)
    base=json.loads((ROOT/"configs/native_volume/V04.json").read_text())
    cnn=json.loads((ROOT/"configs/native_volume/V06.json").read_text())
    cnn["geometry"]["slab_depth"]=3
    cf=tmp/"matchedZ3"/"V06.json";cf.parent.mkdir();cf.write_text(json.dumps(cnn))
    profiles["V06_matchedZ3"]=run("profile_V06_matchedZ3",[ROOT/"scripts/volume_profile.py","--config",cf,"--iterations","10","--warmup","2","--crop-size","32","--microbatch","1","--device","cpu","--max-minutes","1","--output",OUT/"profile_V06_matchedZ3.json"])
    (OUT/"profile_V06_matchedZ3_config.json").write_text(json.dumps(cnn,indent=2)+"\n")
    for micro,acc in [(2,1),(1,2)]:
        d=copy.deepcopy(base);d["training"]["microbatch"]=micro;d["training"]["accumulation"]=acc
        cf=tmp/f"m{micro}a{acc}"/"V04.json";cf.parent.mkdir();cf.write_text(json.dumps(d))
        name=f"profile_V04_m{micro}a{acc}"
        profiles[name]=run(name,[ROOT/"scripts/volume_profile.py","--config",cf,"--iterations","10","--warmup","2","--crop-size","32","--device","cpu","--max-minutes","1","--output",OUT/(name+".json")])
        (OUT/(name+"_config.json")).write_text(json.dumps(d,indent=2)+"\n")
    manifest,msha,geo,gsha=make_volume_fixture(tmp/"data",n_labeled=3,n_unlabeled=3,shape_tzyx=(2,5,17,19))
    c,overrides=synthetic_runtime_config(base,steps=4,crop_size=32,microbatch=1)
    c["training"].update(pseudo_threshold=0.0,pseudo_delay=0,save_every=1)
    c["optimization"].update(scheduler="cosine",lr_warmup_steps=1,min_lr_ratio=0.1)
    cfg=tmp/"train_config.json";cfg.write_text(json.dumps(c))
    (OUT/"roundtrip_config.json").write_text(json.dumps(c,indent=2)+"\n")
    common=[ROOT/"train.py","--config",cfg,"--data",manifest,"--data-sha256",msha,"--geometry",geo,"--geometry-sha256",gsha,"--device","cpu","--synthetic"]
    full=run("train_full4",common+["--out",tmp/"full","--steps","4"])
    split=run("train_stop2",common+["--out",tmp/"split","--steps","2"])
    resumed=run("train_resume4",common+["--out",tmp/"resumed","--steps","4","--resume",tmp/"split","--resume-sha256",split["checkpoint_manifest_sha256"]])
    a=torch.load(tmp/"full/weights.pt",map_location="cpu",weights_only=False)
    b=torch.load(tmp/"resumed/weights.pt",map_location="cpu",weights_only=False)
    diffs=[]
    def equal(x,y,k=""):
        if isinstance(x,torch.Tensor):
            if not isinstance(y,torch.Tensor) or not torch.equal(x,y):diffs.append(k)
        elif isinstance(x,np.ndarray):
            if not isinstance(y,np.ndarray) or not np.array_equal(x,y):diffs.append(k)
        elif isinstance(x,dict):
            if not isinstance(y,dict) or x.keys()!=y.keys():diffs.append(k+".keys");return
            for key in x:equal(x[key],y[key],k+"."+str(key))
        elif isinstance(x,(tuple,list)):
            if len(x)!=len(y):diffs.append(k+".length");return
            for i,(xx,yy) in enumerate(zip(x,y)):equal(xx,yy,k+"."+str(i))
        elif x!=y:diffs.append(k)
    equal(a,b)
    assert not diffs,diffs
    assert (tmp/"split.step000001/MANIFEST.json").exists()
    assert full["schedule_steps"]==split["schedule_steps"]==resumed["schedule_steps"]==4
    # Separate manufactured frozen-evaluation manifest, reference paths forbidden.
    train=json.loads(manifest.read_text())
    records=copy.deepcopy(train["unlabeled"][:2])
    for r in records:r["masks"]={"0":{"path":"MUST_NOT_OPEN.npy","sha256":"0"*64}}
    ev=tmp/"data/eval.json";ev.write_text(json.dumps({"schema":"self_audit.volume.evaluation.v1","role":"evaluation","split":"development","synthetic":True,"records":records}))
    gd=json.loads(geo.read_text());gd["source_manifest"]=ev.name;gd["source_manifest_sha256"]=sha(ev)
    eg=tmp/"data/eval_geometry.json";eg.write_text(json.dumps(gd))
    er=run("evaluate_image_only",[ROOT/"scripts/volume_evaluate.py","--checkpoint",tmp/"resumed","--checkpoint-sha256",resumed["checkpoint_manifest_sha256"],"--manifest",ev,"--manifest-sha256",sha(ev),"--geometry",eg,"--geometry-sha256",sha(eg),"--prediction-model","student","--synthetic","--device","cpu","--tile-size","17","--tile-overlap","2","--slab-depth","3","--slab-overlap","1","--predictions-dir",tmp/"predictions","--output",OUT/"evaluation.json"])
    assert er["labels_read"]==0
    ex=run("export_student",[ROOT/"scripts/volume_export.py","--checkpoint",tmp/"resumed","--checkpoint-sha256",resumed["checkpoint_manifest_sha256"],"--out",tmp/"export","--device","cpu","--synthetic"])
    original,_=load_checkpoint_student_only(tmp/"resumed",resumed["checkpoint_manifest_sha256"])
    exported,meta=load_exported_student(tmp/"export",sha(tmp/"export/MANIFEST.json"))
    x=torch.rand(1,3,3,33,35);v=torch.ones(1,3,33,35,dtype=torch.bool)
    with torch.no_grad():assert torch.equal(original(x,v),exported(x,v))
    rt={"status":"PASS","evidence":"CPU_SYNTHETIC_ONLY","payload_differences":diffs,"planned_horizon":4,"stop":2,"resume_target":4,"intermediate_save_every_verified":True,"label_reads_image_only":er["labels_read"],"export_bitwise_logits":True,"export_teacher_included":meta["teacher_included"],"export_external_teacher_required":meta["external_teacher_required"],"export_pretrained_weights_required":meta["pretrained_weights_required_at_load"],"fixture_manifest_sha256":msha,"fixture_geometry_sha256":gsha,"full_checkpoint_sha256":full["checkpoint_manifest_sha256"],"resumed_checkpoint_sha256":resumed["checkpoint_manifest_sha256"],"export_manifest_sha256":sha(tmp/"export/MANIFEST.json")}
    (OUT/"ROUNDTRIP.json").write_text(json.dumps(rt,indent=2)+"\n")
END=source_sha256()
receipt={"status":"PASS","evidence":"CPU_SYNTHETIC_SOFTWARE_ONLY","source_sha256_before":SRC,"source_sha256_after":END,"source_stable":SRC==END,"pytest_output":tests,"smoke_arms":{k:v["status"] for k,v in smokes.items()},"profile_receipts":list(profiles),"command_count":len(commands),"NOT_RUN":["PRO4000/CUDA/VRAM","real DINO weights","real patient TRAIN smoke","patient Dice","human annotation time","long training","novelty/efficacy"]}
(OUT/"FINAL_VALIDATION.json").write_text(json.dumps(receipt,indent=2)+"\n")
assert SRC==END,"Source changed during final gate"
print(json.dumps(receipt,indent=2),flush=True)
