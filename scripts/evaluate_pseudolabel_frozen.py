#!/usr/bin/env python3
"""Independent post-freeze ACDC evaluator."""
from __future__ import annotations
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from self_audit_pseudolabel.freeze import sha256_file

def load_gt(path):
    import nibabel as nib
    img=nib.as_closest_canonical(nib.load(str(path))); arr=np.asarray(img.dataobj,dtype=np.int16)
    if arr.ndim!=3: raise ValueError(f"Expected 3-D GT, got {arr.shape}")
    return np.transpose(arr,(2,1,0))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--run",required=True); ap.add_argument("--references",required=True); ap.add_argument("--out",required=True)
    args=ap.parse_args(); run=Path(args.run); payload=json.loads((run/"FROZEN.json").read_text())
    for e in payload["entries"]:
        path=run/e["path"]
        if sha256_file(path)!=e["sha256"]: raise ValueError(f"Frozen output changed: {path}")
    refs=Path(args.references); counts=defaultdict(lambda:{c:[0,0] for c in (1,2,3)}); scored=0
    for e in payload["entries"]:
        patient=e["patient_id"]; t=int(e["t"]); z=int(e["z"])
        gt_path=None
        for pat in (f"{patient}_frame{t+1:02d}_gt.nii.gz",f"{patient}_frame{t+1:02d}_gt.nii"):
            hits=list(refs.rglob(pat))
            if hits: gt_path=hits[0]; break
        if gt_path is None: continue
        gt=load_gt(gt_path); pred=np.load(run/e["path"])["pseudo_label"]
        if z>=gt.shape[0]: raise IndexError(f"z {z} outside {gt_path} shape {gt.shape}")
        truth=gt[z]
        if pred.shape!=truth.shape: raise ValueError(f"Shape mismatch {pred.shape} vs {truth.shape}")
        scored+=1
        for c in (1,2,3):
            inter=int(((pred==c)&(truth==c)).sum()); denom=int((pred==c).sum()+(truth==c).sum())
            counts[patient][c][0]+=2*inter; counts[patient][c][1]+=denom
    rows=[]; class_values={1:[],2:[],3:[]}
    for patient,pc in sorted(counts.items()):
        dice={}
        for c in (1,2,3):
            num,den=pc[c]; d=1.0 if den==0 else num/den; dice[c]=d; class_values[c].append(d)
        rows.append({"patient":patient,"rv":dice[1],"myo":dice[2],"lv":dice[3],"foreground_mean":sum(dice.values())/3})
    result={"scored_slices":scored,"patients":len(rows),
        "foreground_mean":float(np.mean([r["foreground_mean"] for r in rows])) if rows else None,
        "rv":float(np.mean(class_values[1])) if rows else None,"myo":float(np.mean(class_values[2])) if rows else None,
        "lv":float(np.mean(class_values[3])) if rows else None,"patient_rows":rows}
    Path(args.out).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
