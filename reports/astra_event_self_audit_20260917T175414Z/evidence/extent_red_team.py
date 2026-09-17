"""Evaluation-only counterexample. GT constructs oracle masks; no optimization."""
import json
from pathlib import Path
import sys
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation, distance_transform_edt
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/"src"))
from self_audit_event.data import load_acdc
from self_audit_event.reference_free_auditor import ReferenceFreeAuditor, AuditObservation
from self_audit_event.evaluation import json_safe

torch.set_num_threads(4)
root=Path(__file__).resolve().parent
identity=json.loads((root/"acdc_data_identity.json").read_text())
datasets,_=load_acdc(identity["root"],identity["manifest"],image_size=64)
q=ReferenceFreeAuditor().eval() # quality() uses only fixed anchor; random learned head never used
rows=[]
with torch.no_grad():
    for sample in datasets["val"]:
        x=sample["image"][None];y=sample["mask"].numpy();m=y==2
        if not m.any(): continue
        # One fixed 4-connected erosion/dilation at resized resolution, not a severity search.
        eroded=binary_erosion(m);dilated=binary_dilation(m)
        nearest=distance_transform_edt(m,return_distances=False,return_indices=True)
        erosion=y.copy();erosion[m & ~eroded]=y[tuple(nearest[:,m & ~eroded])]
        dilation=y.copy();dilation[dilated]=2
        variants=[y,erosion,dilation]
        logits=torch.stack([torch.nn.functional.one_hot(torch.from_numpy(a).long(),4).permute(2,0,1).float()*20-10 for a in variants])
        scores=q.quality(AuditObservation(x.expand(3,-1,-1,-1),logits)).tolist()
        row={"patient_id":sample["patient_id"],"case_id":sample["case_id"],"slice_idx":sample["slice_idx"]}
        for j,name in enumerate(["erosion","dilation"],1):
            mm=variants[j]==2
            row[name]={"rf_delta":scores[j]-scores[0],"myocardium_dice":float(2*(mm&m).sum()/(mm.sum()+m.sum())),"area_ratio":float(mm.sum()/m.sum())}
        rows.append(row)
summary={}
for name in ["erosion","dilation"]:
    vals=[r[name] for r in rows]
    patients={}
    for r in rows: patients.setdefault(r["patient_id"],[]).append(r[name])
    summary[name]={"evaluated_slices":len(vals),"patient_count":len(patients),
       "rf_prefers_wrong_mask_fraction":float(np.mean([v["rf_delta"]>0 for v in vals])),
       "mean_rf_delta":float(np.mean([v["rf_delta"] for v in vals])),
       "mean_myocardium_dice":float(np.mean([v["myocardium_dice"] for v in vals])),
       "patient_mean_rf_delta":{p:float(np.mean([v["rf_delta"] for v in a])) for p,a in patients.items()}}
out={"status":"EVALUATION_ONLY_ORACLE_RED_TEAM", "training_or_threshold_changes":False,
     "note":"GT masks construct a known-correct reference ONLY for this analysis. The fixed score never receives a separate reference/target, corruption name, or ID. This is not natural-transition validity or a slice-independent CI. One-pixel operation at 64x64.",
     "summary":summary,"rows":rows}
(root/"extent_red_team.json").write_text(json.dumps(json_safe(out),indent=2,allow_nan=False))
print(json.dumps(summary,indent=2))
