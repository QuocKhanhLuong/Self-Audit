#!/usr/bin/env python3
"""Independent native-grid evaluator. All frozen artifacts verified BEFORE reference reads."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from self_audit_pseudolabel.freeze import verify_frozen,safe_path

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',required=True); ap.add_argument('--references')
    ap.add_argument('--reference-manifest',help='Explicit patient_id,t,path,label_map entries; required for M&Ms')
    ap.add_argument('--split',choices=['train','val','test'],default='val'); ap.add_argument('--out',required=True)
    args=ap.parse_args(argv); root=Path(args.run)
    print(f"[EVAL] verify freeze={root/'FROZEN.json'} split={args.split}",flush=True)
    payload=verify_frozen(root/'FROZEN.json')  # no nibabel/reference access before this boundary
    cfg=payload['config']; inventory={r['patient_id']:r for r in cfg['export_records'] if r['split']==args.split}
    if not inventory: raise ValueError('requested split is not in this frozen export')
    if args.split!='train' and set(inventory)&set(cfg['producer_patient_ids']): raise ValueError('evaluation patient leakage')
    entries={(e['patient_id'],e['t'],e['z']):e for e in payload['entries'] if e['split']==args.split}
    references=[]
    if args.reference_manifest:
        references=json.loads(Path(args.reference_manifest).read_text())['references']
        references=[r for r in references if r['patient_id'] in inventory]
    elif cfg['dataset']=='acdc' and args.references:
        for pid,r in inventory.items():
            if r.get('ed_index') is None or r.get('es_index') is None:
                raise ValueError('missing declared ED/ES; provide an explicit reference manifest')
            for frame in sorted(set([r['ed_index'],r['es_index']])):
                hits=sorted(Path(args.references).rglob(f'{pid}_frame{frame:02d}_gt.nii'))+sorted(Path(args.references).rglob(f'{pid}_frame{frame:02d}_gt.nii.gz'))
                if len(hits)!=1: raise ValueError(f'expected one reference for {pid} frame {frame}, found {len(hits)}')
                references.append({'patient_id':pid,'t':frame-1,'path':str(hits[0]),'label_map':{'0':0,'1':1,'2':2,'3':3}})
    else:
        raise ValueError('provide explicit --reference-manifest with verified phase and label mapping')
    if not references or set(r['patient_id'] for r in references)!=set(inventory): raise ValueError('incomplete reference patient coverage')
    import nibabel as nib
    totals={pid:{c:[0,0,0] for c in (1,2,3)} for pid in inventory}; known={pid:[0,0] for pid in inventory}
    seen=set(); scored=0
    for ref in references:
        pid,t=ref['patient_id'],int(ref['t']); info=inventory[pid]
        if (pid,t) in seen: raise ValueError('duplicate reference frame')
        seen.add((pid,t)); im=nib.load(ref['path']); truth=np.asarray(im.dataobj)
        if truth.ndim==4:
            if 'reference_frame_index' not in ref: raise ValueError('4-D reference requires explicit frame index')
            truth=truth[:,:,:,int(ref['reference_frame_index'])]
        if truth.shape!=tuple(info['shape'][:3]) or not np.allclose(im.affine,info['affine'],atol=1e-4,rtol=0):
            raise ValueError('native shape/affine mismatch; no implicit canonical reorientation')
        if not np.isfinite(truth).all() or not np.equal(truth,np.floor(truth)).all(): raise ValueError('invalid reference labels')
        mapping={int(k):int(v) for k,v in ref['label_map'].items()}
        if set(mapping.values())!={0,1,2,3} or mapping.get(0)!=0: raise ValueError('explicit complete label mapping required')
        if not set(np.unique(truth)).issubset(mapping): raise ValueError('unmapped reference class')
        mapped=np.zeros_like(truth,dtype=np.uint8)
        for k,v in mapping.items(): mapped[truth==k]=v
        for z in range(mapped.shape[2]):
            key=(pid,t,z)
            if key not in entries: raise ValueError('missing frozen prediction for a reference slice')
            with np.load(safe_path(root,entries[key]['path']),allow_pickle=False) as data: pred=data['pseudo_label']
            target=mapped[:,:,z].T
            if pred.shape!=target.shape: raise ValueError('native slice shape mismatch')
            for c in (1,2,3):
                totals[pid][c][0]+=int(((pred==c)&(target==c)).sum())
                totals[pid][c][1]+=int((pred==c).sum()); totals[pid][c][2]+=int((target==c).sum())
            known[pid][0]+=int((pred!=255).sum()); known[pid][1]+=pred.size; scored+=1
    rows=[]
    for pid,counts in sorted(totals.items()):
        dice={c:(None if pc+gc==0 else 2*tp/(pc+gc)) for c,(tp,pc,gc) in counts.items()}
        values=[v for v in dice.values() if v is not None]
        rows.append({'patient':pid,'rv':dice[1],'myo':dice[2],'lv':dice[3],
                     'foreground_mean':float(np.mean(values)) if values else None,
                     'known_fraction':known[pid][0]/max(known[pid][1],1)})
    vals=[r['foreground_mean'] for r in rows if r['foreground_mean'] is not None]
    class_means={}
    for key in ('rv','myo','lv'):
        values=[r[key] for r in rows if r[key] is not None]
        class_means[key]=float(np.mean(values)) if values else None
    known_values=[r['known_fraction'] for r in rows]
    result={'split':args.split,'evaluation_kind':'in_sample' if args.split=='train' else 'held_out_development',
        'manifest_id':payload['manifest_id'],'patients':len(rows),'scored_slices':scored,
        'foreground_mean':float(np.mean(vals)) if vals else None,
        'rv':class_means['rv'],'myo':class_means['myo'],'lv':class_means['lv'],
        'known_fraction':float(np.mean(known_values)) if known_values else None,'patient_rows':rows,
        'empty_class_policy':'exclude_both_empty','UNKNOWN_policy':'missed_reference_anatomy',
        'teacher_ready':'NOT_AUTOMATICALLY_CERTIFIED','bounded_training':cfg.get('bounded_training',False)}
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as f: json.dump(result,f,indent=2,allow_nan=False)
    def fmt(v): return 'n/a' if v is None else f'{v:.4f}'
    print(f"[EVAL] patients={len(rows)} slices={scored} fg={fmt(result['foreground_mean'])} "
          f"RV={fmt(result['rv'])} MYO={fmt(result['myo'])} LV={fmt(result['lv'])} "
          f"known={fmt(result['known_fraction'])} out={out}",flush=True)
    return result
if __name__=='__main__': main()
