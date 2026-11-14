#!/usr/bin/env python3
"""Create deterministic before/after overlays for the locked L8 masks."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def run(manifest: Path, expected_sha: str, output: Path, *, seed: int=42) -> dict:
    """Load only the 16 labeled L8 frames and save paired overlays/identity."""
    from self_audit.candidate_fuse.data import FuseData
    from self_audit.candidate_fuse.augmentation import StrongInplaneAugment
    data=FuseData(manifest,expected_sha,role="train",synthetic=False)
    if len(data._labeled_ids)!=8: raise ValueError("preview requires locked L8 TRAIN manifest")
    aug=StrongInplaneAugment(seed=seed)
    output.mkdir(parents=True,exist_ok=False)
    records=[]
    selected=sorted(data.labeled_keys())[:12]
    for index,key in enumerate(selected):
        frame=data.load(key,labeled=True)
        before=frame.image.detach().cpu(); target=frame.target.detach().cpu()
        state_before=aug.state_dict()["generator_state"].numpy().tobytes()
        after, transformed_target, geometry=aug.teacher_view(before,target,frame.geometry)
        state_after=aug.state_dict()["generator_state"].numpy().tobytes()
        if tuple(after.shape)!=tuple(before.shape) or geometry != frame.geometry:
            raise ValueError("augmentation changed native shape/geometry")
        if transformed_target is None or not torch_equal_labels(transformed_target):
            raise ValueError("augmentation produced invalid categorical labels")
        name=f"{index:02d}_{key[0]}_frame{key[1]:03d}.png"
        # One panel file per frame, with paired source/transformed central slices.
        _paired_overlay(before,target,after,transformed_target,output/name,
                        f"{key[0]} frame {key[1]} | seed {seed}")
        records.append({"patient_id":key[0],"frame_index":key[1],"overlay":name,
                        "transform_identity":{"seed":seed,
                        "rng_state_before_sha256":hashlib.sha256(state_before).hexdigest(),
                        "rng_state_after_sha256":hashlib.sha256(state_after).hexdigest(),
                        "image_after_sha256":hashlib.sha256(after.contiguous().numpy().tobytes()).hexdigest(),
                        "geometry_before":frame.geometry,"geometry_after":geometry}})
    if len(records)!=12: raise ValueError("expected exactly 12 deterministic L8 ED/ES preview frames")
    receipt={"schema":"self_audit.fuse_aug_preview.v1","status":"COMPLETED",
        "evidence_status":"REAL_DATA_VISUAL_QA","seed":seed,"manifest_sha256":data.manifest_sha256,
        "frames":records}
    (output/"PREVIEW.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    return receipt

def torch_equal_labels(mask):
    import torch
    return mask.dtype==torch.long and bool(torch.isfinite(mask.float()).all()) and not bool(((mask<0)|(mask>3)).any())

def _paired_overlay(image0,mask0,image1,mask1,path,title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    a=np.asarray(image0).squeeze(); b=np.asarray(image1).squeeze(); m0=np.asarray(mask0).squeeze(); m1=np.asarray(mask1).squeeze()
    z=a.shape[0]//2; fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,im,m,lab in zip(axes,(a,b),(m0,m1),("before","augmented")):
        ax.imshow(im[z],cmap="gray",vmin=0,vmax=1); c=np.ma.masked_where(m[z]==0,m[z]); ax.imshow(c,cmap="viridis",alpha=.42,vmin=1,vmax=3); ax.set_title(lab); ax.axis("off")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path,dpi=120); plt.close(fig)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--config",type=Path); p.add_argument("--manifest",type=Path); p.add_argument("--manifest-sha256"); p.add_argument("--out",required=True,type=Path); p.add_argument("--seed",type=int,default=42); a=p.parse_args(argv)
    if a.config:
        config=json.loads(a.config.read_text()); data=config["data"]
        if (config.get("schema") != "self_audit.candidate_fuse.v2"
                or config.get("experiment") != {"track":"A","dataset":"ACDC"}
                or config.get("augmentation") != {"policy":"inplane_strong_v1"}):
            p.error("--config must be the locked A-aug v2 config")
        a.manifest=Path(data["train_manifest"]); a.manifest_sha256=data["train_manifest_sha256"]
    if not a.manifest or not a.manifest_sha256: p.error("provide --config or both --manifest and --manifest-sha256")
    print(json.dumps(run(a.manifest,a.manifest_sha256,a.out,seed=a.seed),indent=2))
if __name__=="__main__": raise SystemExit(main())
