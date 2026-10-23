"""Replay fixed round-two research probes in an isolated source/report copy."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,tempfile,time
ROOT=Path(__file__).resolve().parents[4]
REPORT=Path("reports/research/20261023_astra_ultra_round2")
PARENT=Path("reports/research/20261023_astra_ultra")
NAMES=["A_solver_qualification.py","B_geometry.py","C_endpoint_headroom.py","C_endpoint_nonidentifiability.py","E_identification.py","F_realizable_replay.py"]
OUTPUTS={name:[name.removesuffix(".py")+".json"] for name in NAMES}
OUTPUTS.update({name:[name.removesuffix(".py")+".json"] for name in ["A_moment_audit.py","A_review_F.py","B_review_E.py","E_review_C.py"]})
OUTPUTS["F_review_A.py"]=["F_A_rational_validation.json"]
FINAL="--final" in sys.argv[1:]
REFRESH="--refresh" in sys.argv[1:]
FINAL=FINAL or REFRESH
INITIAL=ROOT/REPORT/"evidence/coordinator_replay/RECEIPT.json"
previous=json.loads(INITIAL.read_text()) if FINAL else None
skipped=[]
if FINAL:
    prior={x["name"]:x for x in previous["commands"]}
    skipped=[n for n in NAMES if prior[n]["exit_code"]==0 and prior[n]["sha256"]==hashlib.sha256((ROOT/REPORT/"probes"/n).read_bytes()).hexdigest()]
    NAMES=[n for n in NAMES if n not in skipped]
    NAMES += ["A_moment_audit.py","A_review_F.py","B_review_E.py","E_review_C.py","F_review_A.py"]
if REFRESH:
    last_path=ROOT/REPORT/"evidence/coordinator_replay_final/RECEIPT.json"
    last=json.loads(last_path.read_text())
    checked={x["name"]:x for x in previous["commands"]+last["commands"]}
    all_names=list(checked)
    NAMES=[n for n in all_names if checked[n]["exit_code"]!=0 or checked[n]["sha256"]!=hashlib.sha256((ROOT/REPORT/"probes"/n).read_bytes()).hexdigest()]
    skipped=[n for n in all_names if n not in NAMES]
OUT=ROOT/REPORT/("evidence/coordinator_replay_refresh" if REFRESH else "evidence/coordinator_replay_final" if FINAL else "evidence/coordinator_replay")
OUT.mkdir(parents=True,exist_ok=True)
gitdir=subprocess.check_output(["git","rev-parse","--absolute-git-dir"],cwd=ROOT,text=True).strip()
receipt={"scope":"Manufactured CPU and saved-count arithmetic only; fixed F fixture replay, no repeated discovery or patient/GPU work","python":sys.version,"executable":sys.executable,"parent_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"sources":{},"commands":[]}
receipt["phase"]="post_review_changed_and_new_probes" if FINAL else "initial_six_primary_probes"
if REFRESH:
    receipt["phase"]="final_changed_probe_refresh"
    receipt["previous_post_review_receipt_sha256"]=hashlib.sha256(last_path.read_bytes()).hexdigest()
receipt["unchanged_primary_probes_already_passed"]=skipped
if FINAL:
    receipt["initial_receipt_sha256"]=hashlib.sha256(INITIAL.read_bytes()).hexdigest()
    receipt["evidence_copy_scope"]="Only JSON outputs of commands executed in this phase are retained; dependencies remain in the parent evidence directory."
with tempfile.TemporaryDirectory(prefix="self-audit-round2-replay-") as directory:
    work=Path(directory)
    for source in sorted((ROOT/"self_audit").glob("*.py"))+[ROOT/"train.py"]:
        relative=source.relative_to(ROOT); target=work/relative
        target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(source.read_bytes())
        receipt["sources"][str(relative)]=hashlib.sha256(source.read_bytes()).hexdigest()
    if FINAL:
        assert receipt["sources"]==previous["sources"], "Source changed: initial replay cannot justify skipped probes"
    for folder in [PARENT,REPORT]:
        shutil.copytree(ROOT/folder,work/folder,ignore=shutil.ignore_patterns("__pycache__","coordinator_replay","coordinator_replay_final","coordinator_replay_refresh"))
    env={**os.environ,"GIT_DIR":gitdir,"GIT_WORK_TREE":str(work),"PYTHONDONTWRITEBYTECODE":"1"}
    for name in NAMES:
        script=work/REPORT/"probes"/name
        start=time.monotonic()
        r=subprocess.run([sys.executable,str(script)],cwd=work,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (OUT/(name+".log")).write_text(r.stdout)
        receipt["commands"].append({"name":name,"sha256":hashlib.sha256(script.read_bytes()).hexdigest(),"exit_code":r.returncode,"seconds":time.monotonic()-start,"log":name+".log"})
        print(name,"exit",r.returncode,flush=True)
    for name in NAMES:
        for filename in OUTPUTS[name]:
            path=work/REPORT/"evidence"/filename
            if path.exists():
                shutil.copyfile(path,OUT/path.name)
receipt["status"]="PASS" if all(x["exit_code"]==0 for x in receipt["commands"]) else "FAIL"
(OUT/"RECEIPT.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(receipt["status"])
raise SystemExit(0 if receipt["status"]=="PASS" else 1)
