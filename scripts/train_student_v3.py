#!/usr/bin/env python3
"""Train final adaptive annotator only from frozen pseudo-label pixels."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from self_audit_pseudolabel.data_v3 import discover_acdc_full_cine,discover_mnms_full_cine,CineSliceDataset
from self_audit_pseudolabel.freeze import sha256_file
from self_audit_pseudolabel.system_v3 import AdaptiveAnnotationStudent,pseudo_supervision_loss

class FrozenPseudoDataset(Dataset):
    def __init__(self,root,dataset,manifest):
        records=discover_acdc_full_cine(root) if dataset=="acdc" else discover_mnms_full_cine(root)
        self.base=CineSliceDataset(records,cache_records=2); self.freeze_root=Path(manifest).parent
        payload=json.loads(Path(manifest).read_text()); self.rows=[]; lookup={}
        for idx,(ri,t,z) in enumerate(self.base.index):
            lookup[(self.base.records[ri].patient_id,int(t),int(z))]=idx
        for e in payload["entries"]:
            path=self.freeze_root/e["path"]
            if sha256_file(path)!=e["sha256"]: raise ValueError(f"Frozen pseudo hash mismatch: {path}")
            key=(e["patient_id"],int(e["t"]),int(e["z"]))
            if key not in lookup: raise KeyError(f"Pseudo sample not in cine dataset: {key}")
            self.rows.append((lookup[key],path))
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        idx,path=self.rows[i]; sample=self.base[idx]; p=np.load(path)
        return {"image":sample["cur"],"target":torch.from_numpy(p["pseudo_label"].astype(np.int64)),
                "valid":torch.from_numpy(p["valid"].astype(bool))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",choices=["acdc","mnms"],required=True)
    ap.add_argument("--root",required=True); ap.add_argument("--manifest",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--epochs",type=int,default=10); ap.add_argument("--batch-size",type=int,default=8)
    ap.add_argument("--lr",type=float,default=1e-3); ap.add_argument("--device",default="cpu")
    ap.add_argument("--profile",choices=["compact","balanced","accurate"],default="balanced"); ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args(); torch.manual_seed(args.seed)
    ds=FrozenPseudoDataset(args.root,args.dataset,args.manifest); dl=DataLoader(ds,batch_size=args.batch_size,shuffle=True,num_workers=0)
    device=torch.device(args.device); model=AdaptiveAnnotationStudent().to(device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    hist=[]
    for epoch in range(args.epochs):
        model.train()
        for step,b in enumerate(dl):
            x=b["image"].to(device).float(); y=b["target"].to(device); v=b["valid"].to(device)
            out=model(x,profile=args.profile); loss=pseudo_supervision_loss(out,y,v)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            hist.append({"epoch":epoch,"step":step,"loss":float(loss.detach().cpu()),"valid_fraction":float(v.float().mean())})
    outp=Path(args.out); outp.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"model":model.state_dict(),"args":vars(args),"manifest":str(args.manifest)},outp)
    outp.with_suffix(".json").write_text(json.dumps(hist,indent=2))
    print(json.dumps({"status":"complete","samples":len(ds),"checkpoint":str(outp)},indent=2))
if __name__=="__main__": main()
