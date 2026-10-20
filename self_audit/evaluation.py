"""Freeze full-cine probabilities before opening development masks."""
from pathlib import Path
import json
import numpy as np
import torch
from .data import TrainingData, write_json
from .model import digest
from .engine import load_checkpoint


def confusion_matrix(y, prediction):
    if y.shape != prediction.shape or not np.isin(y, [0,1,2,3,255]).all() or not np.isin(prediction,[0,1,2,3,255]).all():
        raise ValueError("invalid evaluation arrays")
    valid = y != 255
    return np.bincount((y[valid].astype(int)*5 + np.where(prediction[valid]==255,4,prediction[valid])).astype(int),minlength=20).reshape(4,5)


def class_metrics(cm):
    rows=[]
    for k,name in enumerate(["BG","RV","MYO","LV"]):
        tp=int(cm[k,k]); gt=int(cm[k].sum()); pred=int(cm[:,k].sum())
        rows.append({"class":name,"TP":tp,"FP":pred-tp,"FN":gt-tp,"reference_present":bool(gt),
                     "dice":2*tp/(gt+pred) if gt else None,
                     "precision":tp/pred if pred else None,"recall":tp/gt if gt else None,
                     "prediction_count":pred,"reference_count":gt})
    return rows


def evaluate(checkpoint, checkpoint_sha, data_manifest, data_sha, out, device="cuda", threshold=None):
    manifest=Path(data_manifest).resolve(); out=Path(out).resolve()
    if digest(manifest)!=data_sha: raise ValueError("evaluation manifest hash mismatch")
    m=json.loads(manifest.read_text())
    if m.get("role") not in {"development","external_frozen"} or m.get("schema")!="self_audit.data.v2":
        raise ValueError("separate evaluation manifest required")
    model,meta=load_checkpoint(checkpoint,checkpoint_sha,device)
    ids=[r['id'] for r in m['patients']]
    if len(set(ids))!=len(ids) or set(ids)&set(meta['labeled_ids']+meta['unlabeled_ids']):
        raise ValueError("evaluation patient overlap with TRAIN")
    if bool(m.get('synthetic',False))!=meta['synthetic']: raise ValueError("synthetic/patient mismatch")
    if out.exists() or out.is_relative_to(manifest.parent) or manifest.parent.is_relative_to(out):
        raise ValueError("new output outside evaluation data required")
    if threshold is not None and not 0<=threshold<=1: raise ValueError("invalid threshold")
    out.mkdir(parents=True);pred_dir=out/'predictions';pred_dir.mkdir(); inventory=[]
    resolve=lambda p:Path(p) if Path(p).is_absolute() else manifest.parent/p
    recipe={"checkpoint_sha256":checkpoint_sha,"data_sha256":data_sha,"threshold":threshold,
            "scope":m.get('scope',m['role']),"prediction_model":"student","synthetic":meta['synthetic']}
    write_json(out/'RECIPE.json',recipe)
    with torch.inference_mode():
        for r in m['patients']:
            ip=resolve(r['image'])
            if digest(ip)!=r['image_sha256']: raise ValueError("evaluation image hash mismatch")
            x=np.load(ip,mmap_mode='r',allow_pickle=False)
            if x.ndim!=4 or not np.isfinite(x).all(): raise ValueError("invalid evaluation image")
            op=pred_dir/(r['id']+'.npy');t,z,h,w=x.shape
            prob=np.lib.format.open_memmap(op,mode='w+',dtype='float32',shape=(t,z,4,h,w))
            for ti in range(t):
                for zi in range(z):
                    logits=model(TrainingData.context(x,ti,zi).to(device))
                    prob[ti,zi]=logits.softmax(1)[0].cpu().numpy()
            prob.flush();del prob
            inventory.append({"id":r['id'],"file":op.name,"sha256":digest(op)})
    write_json(out/'PREDICTIONS_FROZEN.json',{"recipe_sha256":digest(out/'RECIPE.json'),"files":inventory})
    # No mask array has been opened above this line.
    rows=[];total=np.zeros((4,5),dtype=np.int64)
    for r,pr in zip(m['patients'],inventory):
        op=pred_dir/pr['file']
        if digest(op)!=pr['sha256']: raise ValueError("frozen prediction changed")
        ps=np.load(op,mmap_mode='r',allow_pickle=False)
        for ti,mr in r['masks'].items():
            mp=resolve(mr['path'])
            if digest(mp)!=mr['sha256']: raise ValueError("reference hash mismatch")
            y=np.load(mp,allow_pickle=False);p=np.asarray(ps[int(ti)])
            raw=p.argmax(1); cm=confusion_matrix(y,raw);total+=cm
            row={"patient":r['id'],"frame":int(ti),"raw":class_metrics(cm),"raw_confusion":cm.tolist()}
            if threshold is not None:
                lab=np.where(p.max(1)>=threshold,raw,255)
                row['gated']=class_metrics(confusion_matrix(y,lab))
                row['coverage']=float((lab[y!=255]!=255).mean())
            rows.append(row)
    per_patient={}
    for pid in ids:
        subset=[r for r in rows if r['patient']==pid]
        per_patient[pid]={}
        for j,name in enumerate(['BG','RV','MYO','LV']):
            vals=[r['raw'][j]['dice'] for r in subset if r['raw'][j]['dice'] is not None]
            per_patient[pid][name]=float(np.mean(vals)) if vals else None
    macro={}
    for name in ['BG','RV','MYO','LV']:
        vals=[p[name] for p in per_patient.values() if p[name] is not None]
        macro[name]={"dice":float(np.mean(vals)) if vals else None,"present_patients":len(vals)}
    result={"scope":recipe['scope'],"synthetic":meta['synthetic'],"macro_patient":macro,
            "patients":per_patient,"frames":rows,"pooled_confusion":total.tolist(),
            "physical_boundary_metrics":"NOT_COMPUTED; no spacing certification in v2 cache",
            "warning":"Audit risk is TRAIN optimization evidence, not this evaluation or a clinical guarantee."}
    write_json(out/'METRICS.json',result)
    return result
