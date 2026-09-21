#!/usr/bin/env python3
"""Bounded image-only training/export entrypoint for Self-Audit v3 pseudo-labels."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from self_audit_pseudolabel.data_v3 import discover_acdc_full_cine,discover_mnms_full_cine,CineSliceDataset
from self_audit_pseudolabel.system_v3 import CinePseudoTeacher,UNKNOWN
from self_audit_pseudolabel.trainer_v3 import ProgressiveTeacherTrainer,ProgressiveConfig
from self_audit_pseudolabel.freeze import export_pseudo_npz,write_freeze_manifest
from self_audit_pseudolabel.consistency import consistency_gate

def move_batch(batch,device):
    out=dict(batch)
    for key in ("prev","cur","nxt"): out[key]=out[key].to(device=device,dtype=torch.float32)
    return out

def flush_patient(rows,out_root,dataset,use_consistency=True):
    if not rows: return []
    patient=rows[0]["patient_id"]; tmax=max(r["t"] for r in rows)+1; zmax=max(r["z"] for r in rows)+1
    _,c,h,w=rows[0]["soft"].shape
    soft=torch.zeros(tmax,zmax,c,h,w); valid=torch.zeros(tmax,zmax,h,w,dtype=torch.bool)
    for r in rows:
        soft[r["t"],r["z"]]=r["soft"][0]; valid[r["t"],r["z"]]=r["valid"][0]
    if use_consistency:
        soft,valid=consistency_gate(soft,valid)
    labels=soft.argmax(2); labels=torch.where(valid,labels,torch.full_like(labels,UNKNOWN))
    entries=[]
    for r in rows:
        t,z=r["t"],r["z"]; rel=Path("pseudo")/patient/f"t{t:03d}_z{z:03d}.npz"; path=out_root/rel
        sha=export_pseudo_npz(path,pseudo_label=labels[t,z].numpy(),valid=valid[t,z].numpy(),
            soft_label=soft[t,z].numpy(),metadata={"patient_id":patient,"t":t,"z":z,"dataset":dataset,
            "consistency_gate":bool(use_consistency)})
        entries.append({"path":str(rel),"sha256":sha,"patient_id":patient,"t":t,"z":z,
            "valid_fraction":float(valid[t,z].float().mean())})
    return entries

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",choices=["acdc","mnms"],required=True); p.add_argument("--root",required=True); p.add_argument("--out",required=True)
    p.add_argument("--epochs",type=int,default=1); p.add_argument("--batch-size",type=int,default=4); p.add_argument("--lr",type=float,default=1e-3)
    p.add_argument("--device",default="cpu"); p.add_argument("--max-train-batches",type=int,default=0); p.add_argument("--seed",type=int,default=42)
    p.add_argument("--no-consistency",action="store_true")
    args=p.parse_args()
    if args.epochs<1: raise ValueError("epochs must be positive")
    torch.manual_seed(args.seed)
    records=discover_acdc_full_cine(args.root) if args.dataset=="acdc" else discover_mnms_full_cine(args.root)
    ds=CineSliceDataset(records,cache_records=2); loader=DataLoader(ds,batch_size=args.batch_size,shuffle=True,num_workers=0)
    device=torch.device(args.device); teacher=CinePseudoTeacher().to(device)
    optim=torch.optim.AdamW(teacher.parameters(),lr=args.lr,weight_decay=1e-4)
    trainer=ProgressiveTeacherTrainer(teacher,optim,ProgressiveConfig())
    out_root=Path(args.out); out_root.mkdir(parents=True,exist_ok=False)
    journal=[]
    for epoch in range(args.epochs):
        teacher.train()
        for step,batch in enumerate(loader):
            if args.max_train_batches and step>=args.max_train_batches: break
            losses,accepted,_=trainer.train_batch(move_batch(batch,device))
            journal.append({"epoch":epoch,"step":step,"accepted_regions":accepted,**losses})
    (out_root/"train_metrics.json").write_text(json.dumps(journal,indent=2),encoding="utf-8")
    torch.save({"model":teacher.state_dict(),"prototype_bank":trainer.bank.prototypes.cpu(),
        "prototype_counts":trainer.bank.counts.cpu(),"args":vars(args)},out_root/"teacher.pt")

    export_loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=0); teacher.eval(); entries=[]; rows=[]; current=None
    for batch in export_loader:
        pred,_,_=trainer.infer_batch(move_batch(batch,device))
        patient=str(batch["patient_id"][0]); t=int(batch["t"][0]); z=int(batch["z"][0])
        if current is not None and patient!=current:
            entries.extend(flush_patient(rows,out_root,args.dataset,not args.no_consistency)); rows=[]
        current=patient
        rows.append({"patient_id":patient,"t":t,"z":z,"soft":pred["soft_label"].cpu(),"valid":pred["valid"].cpu()})
    entries.extend(flush_patient(rows,out_root,args.dataset,not args.no_consistency))
    write_freeze_manifest(out_root,entries,{"dataset":args.dataset,"seed":args.seed,"epochs":args.epochs,
        "batch_size":args.batch_size,"manual_mask_input":False,"consistency_gate":not args.no_consistency})
    print(json.dumps({"status":"complete","samples":len(ds),"exports":len(entries),"out":str(out_root)},indent=2))

if __name__=="__main__": main()
