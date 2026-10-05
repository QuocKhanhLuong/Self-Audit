#!/usr/bin/env python3
"""Independent phase-aware native-grid scoring after complete freeze verification."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from self_audit_pseudolabel.freeze import verify_frozen, safe_path, strict_index, sha256_file


def metric_row(counts, known):
    dice = [None if p+g == 0 else 2*tp/(p+g) for tp,p,g in counts]
    values = [d for d in dice if d is not None]
    return dict(zip(('rv','myo','lv'), dice),
                foreground_mean=float(np.mean(values)) if values else None,
                known_fraction=known[0]/max(known[1], 1))


def mean_rows(rows):
    result = {}
    for key in ('rv','myo','lv','foreground_mean','known_fraction'):
        values = [r[key] for r in rows if r[key] is not None]
        result[key] = float(np.mean(values)) if values else None
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', required=True)
    ap.add_argument('--references')
    ap.add_argument('--reference-manifest', help='Explicit references with phase and zero-based integer indices')
    ap.add_argument('--split', choices=['train','val','test'], default='val')
    ap.add_argument('--allow-unknown-spatial-units', action='store_true',
                    help='Explicitly permit both grids declaring unknown units; recorded in result')
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    root = Path(args.run)
    # Verify even generic payload hashes before reporting a missing semantic role.
    payload = verify_frozen(root/'FROZEN.json', expected_role=('teacher_freeze','student_predictions'))
    cfg = payload['config']
    inventory = {r['patient_id']:r for r in cfg['export_records'] if r['split']==args.split}
    if not inventory:
        raise ValueError('requested split is not in this frozen export')
    if args.split!='train' and set(inventory)&set(cfg['producer_patient_ids']):
        raise ValueError('evaluation patient leakage')
    entries = {(e['patient_id'],e['t'],e['z']):e for e in payload['entries'] if e['split']==args.split}
    protocol = {'required_phases':['ED','ES'], 'phase_indices':'zero_based'}
    references = []
    reference_manifest_sha = None
    if args.reference_manifest:
        reference_manifest_sha = sha256_file(args.reference_manifest)
        body = json.loads(Path(args.reference_manifest).read_text())
        protocol = body.get('protocol', protocol)
        references = [dict(r) for r in body['references'] if r['patient_id'] in inventory]
    elif cfg['dataset']=='acdc' and args.references:
        for pid,info in inventory.items():
            for phase,field in [('ED','ed_index'),('ES','es_index')]:
                frame = info.get(field)
                if type(frame) is not int or not 1<=frame<=info['shape'][3]:
                    raise ValueError('missing/invalid one-based ED/ES; provide explicit phase manifest')
                hits = list(Path(args.references).rglob(f'{pid}_frame{frame:02d}_gt.nii'))
                hits += list(Path(args.references).rglob(f'{pid}_frame{frame:02d}_gt.nii.gz'))
                if len(hits)!=1:
                    raise ValueError(f'expected one reference for {pid} {phase}')
                references.append({'patient_id':pid, 'phase':phase, 't':frame-1, 'path':str(hits[0]),
                                   'label_map':{'0':0,'1':1,'2':2,'3':3}})
    else:
        raise ValueError('provide independent references with verified phase and label mapping')
    phases = protocol.get('required_phases')
    if phases!=['ED','ES'] or protocol.get('phase_indices','zero_based')!='zero_based':
        raise ValueError('supported evaluation protocol requires ED and ES, zero-based indices')
    seen = set()
    seen_phases = {pid:set() for pid in inventory}
    # Validate declarations before opening any reference, including invalid ref indices.
    for ref in references:
        pid = ref['patient_id']; info = inventory[pid]
        t = strict_index(ref.get('t'), 't', info['shape'][3])
        if 'reference_frame_index' in ref:
            strict_index(ref['reference_frame_index'], 'reference_frame_index')
        if 'phase' not in ref:
            candidates = [phase for phase,field in [('ED','ed_index'),('ES','es_index')]
                          if type(info.get(field)) is int and info[field]-1==t]
            if len(candidates)!=1: raise ValueError('explicit ED/ES phase required')
            ref['phase'] = candidates[0]
        phase = ref['phase']
        if phase not in phases or phase in seen_phases[pid] or (pid,t) in seen:
            raise ValueError('duplicate/invalid reference phase or frame')
        field = {'ED':'ed_index','ES':'es_index'}[phase]
        if info.get(field) is not None and (type(info[field]) is not int or info[field]-1!=t):
            raise ValueError('reference phase disagrees with locked cine metadata')
        seen_phases[pid].add(phase); seen.add((pid,t))
    if not references or any(s!=set(phases) for s in seen_phases.values()):
        raise ValueError('incomplete reference phase coverage: require ED+ES for every patient')
    import nibabel as nib
    totals = {pid:np.zeros((3,3),dtype=np.int64) for pid in inventory}
    known = {pid:np.zeros(2,dtype=np.int64) for pid in inventory}
    frame_rows = []; reference_rows = []; scored = 0
    for ref in references:
        pid,t,phase = ref['patient_id'],ref['t'],ref['phase']; info = inventory[pid]
        digest = sha256_file(ref['path'])
        im = nib.load(ref['path'])
        if len(im.shape)==4:
            index = strict_index(ref.get('reference_frame_index'), 'reference_frame_index', im.shape[3])
            truth = np.asarray(im.dataobj[:,:,:,index])
        elif len(im.shape)==3 and 'reference_frame_index' not in ref:
            truth = np.asarray(im.dataobj)
        else:
            raise ValueError('reference must be 3-D or explicitly indexed 4-D')
        unit = im.header.get_xyzt_units()[0]
        if unit!=info['spatial_unit'] or (unit=='unknown' and not args.allow_unknown_spatial_units):
            raise ValueError('native spatial unit mismatch/unknown')
        if truth.shape!=tuple(info['shape'][:3]) or not np.allclose(im.affine,info['affine'],atol=1e-4,rtol=0):
            raise ValueError('native shape/affine mismatch; no implicit canonical reorientation')
        if not np.isfinite(truth).all() or not np.equal(truth,np.floor(truth)).all():
            raise ValueError('invalid reference labels')
        raw = ref['label_map']
        if any(not isinstance(k,str) or not k.isdecimal() or str(int(k))!=k or type(v) is not int for k,v in raw.items()):
            raise ValueError('label mapping requires integer class IDs')
        mapping = {int(k):v for k,v in raw.items()}
        if set(mapping.values())!={0,1,2,3} or mapping.get(0)!=0 or not set(np.unique(truth)).issubset(mapping):
            raise ValueError('explicit complete label mapping required')
        mapped = np.zeros_like(truth,dtype=np.uint8)
        for k,v in mapping.items(): mapped[truth==k]=v
        counts = np.zeros((3,3),dtype=np.int64); frame_known = np.zeros(2,dtype=np.int64)
        for z in range(mapped.shape[2]):
            key = (pid,t,z)
            if key not in entries: raise ValueError('missing frozen prediction for reference slice')
            with np.load(safe_path(root,entries[key]['path']),allow_pickle=False) as data:
                pred = data['pseudo_label']
            target = mapped[:,:,z].T
            if pred.shape!=target.shape: raise ValueError('native slice shape mismatch')
            for i,c in enumerate((1,2,3)):
                counts[i] += [int(((pred==c)&(target==c)).sum()),int((pred==c).sum()),int((target==c).sum())]
            frame_known += [int((pred!=255).sum()),pred.size]; scored+=1
        if sha256_file(ref['path'])!=digest: raise ValueError('reference changed during scoring')
        totals[pid]+=counts; known[pid]+=frame_known
        frame_rows.append({'patient':pid,'phase':phase,'t':t,**metric_row(counts,frame_known)})
        reference_rows.append({**ref,'sha256':digest,'spatial_unit':unit,'shape':list(im.shape),
                               'affine':im.affine.tolist()})
    rows = [{'patient':pid,**metric_row(totals[pid],known[pid])} for pid in sorted(inventory)]
    result = {'split':args.split,'evaluation_kind':{'train':'in_sample','val':'held_out_development','test':'held_out_test'}[args.split],
              'test_untouched':'NOT_CERTIFIED_BY_CODE','artifact_role':cfg['artifact_role'],
              'manifest_id':payload['manifest_id'],'patients':len(rows),'scored_slices':scored,
              **mean_rows(rows),'patient_rows':rows,'frame_rows':frame_rows,
              'estimand':'patient_macro_of_phase_pooled_counts',
              'phase_macro':mean_rows(frame_rows),
              'phase_metrics':{p:mean_rows([r for r in frame_rows if r['phase']==p]) for p in phases},
              'reference_rows':reference_rows,'reference_manifest_sha256':reference_manifest_sha,
              'protocol':protocol,'allow_unknown_spatial_units':args.allow_unknown_spatial_units,
              'empty_class_policy':'exclude_both_empty','UNKNOWN_policy':'missed_reference_anatomy',
              'teacher_ready':'NOT_AUTOMATICALLY_CERTIFIED','bounded_training':cfg.get('bounded_training',False)}
    verify_frozen(root/'FROZEN.json',expected_role=cfg['artifact_role'])
    if reference_manifest_sha and sha256_file(args.reference_manifest)!=reference_manifest_sha:
        raise ValueError('reference manifest changed during evaluation')
    out = Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as f: json.dump(result,f,indent=2,allow_nan=False)
    print(f"[EVAL] patients={len(rows)} slices={scored} foreground_mean={result['foreground_mean']} out={out}",flush=True)
    return result

if __name__=='__main__': main()
