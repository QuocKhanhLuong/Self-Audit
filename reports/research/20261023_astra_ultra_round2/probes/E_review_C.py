#!/usr/bin/env python3
"""Independent standard-library recomputation for E review of C; saved CSV only."""
import csv,hashlib,io,itertools,json,subprocess
from fractions import Fraction as F
from pathlib import Path
root=Path(__file__).resolve().parents[1]
raw=subprocess.check_output(["git","show","8960b6baac4c6600553d900c303f192c2fcfa4f0:results/PATIENT_METRICS.csv"])
rows=list(csv.DictReader(io.StringIO(raw.decode())))
by={(r["run"].removeprefix("L8_S42_"),r["patient"],r["class_name"]):r for r in rows}
patients=sorted({r["patient"] for r in rows});classes=("RV","MYO","LV")
methods=("supervised","supervised_extra","ema_full","ema_block")
assert len(rows)==288 and len(patients)==18
score=lambda m,p:sum(float(by[m,p,c]["dice_mean_frames"]) for c in classes)/3
out={"status":"PASS","input_sha256":hashlib.sha256(raw).hexdigest(),"oracles":[],"coupling":{}}
sets=(("ema_block","supervised"),("ema_block","ema_full"),methods)
expected=json.loads((root/"evidence/C_endpoint_headroom.json").read_text())
for options,record in zip(sets,expected["oracles"]):
 constant=max(sum(score(m,p) for p in patients)/18 for m in options)
 oracle=sum(max(score(m,p) for m in options) for p in patients)/18
 relax=sum(max(float(by[m,p,c]["dice_mean_frames"]) for m in options) for p in patients for c in classes)/54
 assert abs(constant-record["best_constant_foreground_dice"])<1e-14
 assert abs(oracle-record["ORACLE_retrospective_whole_patient_foreground_dice"])<1e-14
 assert abs(relax-record["RELAXATION_independent_patient_class_score_maximum"])<1e-14
 out["oracles"].append({"methods":options,"constant":constant,"whole_patient":oracle,"class_relaxation":relax})
for other,record in zip(("supervised","ema_full"),expected["paired_tradeoffs"]):
 xlo=xhi=ylo=yhi=0
 for p in patients:
  block,comp=by["ema_block",p,"RV"],by[other,p,"RV"]
  a,b=int(block["TP"]),int(comp["TP"])
  n=a+int(block["FN"])
  assert n==b+int(comp["FN"])
  xlo+=max(0,b-a);xhi+=min(b,n-a);ylo+=max(0,a-b);yhi+=min(a,n-b)
 result={"wrong_to_correct":[xlo,xhi],"correct_to_wrong":[ylo,yhi]}
 assert result["wrong_to_correct"]==record["class"]["RV"]["block_wrong_comparator_correct_count_bounds"]
 assert result["correct_to_wrong"]==record["class"]["RV"]["block_correct_comparator_wrong_count_bounds"]
 out["coupling"][other]=result
bg=[]
for other,record in zip(("supervised","ema_full"),expected["BG_RV_endpoint_overlap_bounds"]):
 lo=hi=0
 for p in patients:
  a=json.loads(by["ema_block",p,"BG"]["confusion_4x5"])[0]
  b=json.loads(by[other,p,"BG"]["confusion_4x5"])[0]
  assert sum(a)==sum(b)
  lo+=max(0,a[1]+b[0]-sum(a));hi+=min(a[1],b[0])
 assert [lo,hi]==record["block_BG_to_RV_and_comparator_BG_count_bounds"]
 bg.append({"comparator":other,"bounds":[lo,hi]})
out["BG_RV_bounds"]=bg
risk=lambda u,v:((1-u)**2+v*v)/4
s=(F(4,5),F(1,10));a=(F(9,10),F(1,20));h=(F(3,5),F(1,5))
assert risk(*s)==F(1,80) and risk(*a)==F(1,320) and risk(*h)==F(1,20)
for k in range(8):
 t=F(1,2**k)
 assert risk(s[0]-t/5,s[1]+t/10)-risk(*s)>F(1,10**8)
out["rational_counterexample"]={"three_risks_verified":True,"all_eight_scales_reject_at_1e_minus8":True,
 "not_production_realizability":True}
out["claim_scope"]="Independent arithmetic only; finite endpoint selector and marginal coupling bounds; not patient execution or new-method headroom."
(root/"evidence/E_review_C.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
