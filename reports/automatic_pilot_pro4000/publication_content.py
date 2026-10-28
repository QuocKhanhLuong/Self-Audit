import sys,json,hashlib
from pathlib import Path
kind,path=sys.argv[1:];p=Path(path);r=json.loads(p.read_text());x={'kind':kind,'server_artifact':str(p),'artifact_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'base_commit':'93f7c7a1f9606a68955f849dd580b90462ad329e'}
if kind=='training':
 assert r['status']=='COMPLETED' and not r['synthetic'];x.update(report=r,quality='NOT_EVALUATED_IN_THIS_TRAINING_RECEIPT',checkpoint=str(p.parent/'final'),batch_size=128,full_ema_microbatch_size=40 if r['arm']=='full_ema' else 0,comparison=p.parent.parent.name)
elif kind=='dataset':
 x['receipt']=r;x['duplicate_audit']=json.loads((p.parent/'cross_dataset_voxel_duplicates.json').read_text());x['limitation']='Original splits/files unchanged. Exact-voxel duplicate M&Ms IDs quarantined in derived manifests; missing temporal spacing remains unknown.'
elif kind=='decision':
 x['decision_gate']=r['decision_gate'];x['verified_context']=r.get('verified_context');x['actual_phase_seconds']=r.get('actual_phase_seconds');x['arms']={a:{k:m.get(k) for k in ['foreground_patient_macro','macro_patient','accounting','prediction_coverage','scoring_coverage']} for a,m in r.get('arms',{}).items()};x['patient_metric_artifacts']='Full per-patient metrics and failures are retained under reports/automatic_pilot_pro4000/PATIENT_METRICS.csv, FAILURE_PATIENTS.csv and runs/pro4000/evaluation_*/METRICS.json';x['limitations']='Single seed, configured pilot; Drive TEST50 previously exposed, not a new confirmatory cohort; references not challenge certified. Full-EMA includes encoder updates in executed base code. See local report, evaluation index and frozen prediction hashes.'
else:raise ValueError(kind)
print(json.dumps(x,indent=2,allow_nan=False))
