#!/usr/bin/env python3
"""Read only published text evidence; output owned JSON. No model/data imports."""
import ast
import hashlib
import json
import re
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
E = ROOT / 'reports/stable_guidance_world_v2_20261014'
OUT = Path(__file__).with_name('PROVENANCE_METRICS_checks.json')
BASE = '7d1c6a04064cda473d1eccfbed818392c37b1a37'
TRAIN_BASE = 'd5a45f31481f0a702a1f9d5d418889fe704e9105'
PIN = 'b937e3b746608725e43d3bc3cef8861666d152dc'
C = ('BG', 'RV', 'MYO', 'LV')
M = ('dice', 'iou', 'precision', 'recall')
checks = []
def check(name, condition, detail=None):
    checks.append(dict(name=name, passed=bool(condition), detail=detail))
def js(p): return json.loads((E/p).read_text())
def sha(b): return hashlib.sha256(b).hexdigest()
def file_sha(p): return sha((E/p).read_bytes())
def mean(v):
    v=[x for x in v if x is not None]
    return sum(v)/len(v) if v else None
def git_bytes(spec):
    return subprocess.run(['git','show',spec],cwd=ROOT,capture_output=True,check=True).stdout
def close(a,b):
    if a is None or b is None:return a is b
    if isinstance(a,dict):return a.keys()==b.keys() and all(close(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(close(x,y) for x,y in zip(a,b))
    if isinstance(a,(float,int)) and isinstance(b,(float,int)):return abs(a-b)<=1e-12
    return a==b

manifest=js('MANIFEST.json'); publication=js('PUBLICATION.json')
files={r['published_path']:r for r in manifest['files']}
check('HEAD pinned',subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()==PIN)
check('unique manifest paths',len(files)==len(manifest['files']))
file_results=[]
for p,r in files.items():
    b=(E/p).read_bytes(); hash_ok=sha(b)==r['published_sha256'];size_ok=len(b)==r['published_bytes']
    flag_ok=r['sanitized']==(r['original_sha256']!=r['published_sha256'])
    check('published file '+p,hash_ok and size_ok and flag_ok)
    check('pinned publication bytes '+p,b==git_bytes(PIN+':'+str((E/p).relative_to(ROOT))))
    file_results.append(dict(path=p,hash_verified=hash_ok,bytes_verified=size_ok,sanitized=r['sanitized'],original_sha256=r['original_sha256'],published_sha256=r['published_sha256']))
check('PUBLICATION copied count',publication['copied_files']==len(files))
check('PUBLICATION copied bytes',publication['copied_bytes']==sum(r['published_bytes'] for r in files.values()))
all_paths={str(p.relative_to(E)) for p in E.rglob('*') if p.is_file()}
unlisted=sorted(all_paths-set(files))
for p in unlisted:check('pinned nonmanifest publication '+p,(E/p).read_bytes()==git_bytes(PIN+':'+str((E/p).relative_to(ROOT))))
sanitization_scan={}
for name,pattern in {'absolute_home':r'(?<!\$\{)(?:/home/|/Users/)[A-Za-z0-9_.-]+','email':r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}','gpu_uuid':r'GPU-[0-9a-fA-F]{8}-[0-9a-fA-F-]{27,}','credential_assignment':r'(?i)(?:api[_-]?key|access[_-]?token|password)\s*[=:]\s*["\']?[A-Za-z0-9_-]{16,}'}.items():
    hits={p:len(re.findall(pattern,(E/p).read_text())) for p in all_paths}
    sanitization_scan[name]={p:n for p,n in hits.items() if n}
# Never expose matched values in the output. Pattern matches require interpretation.

# Reconstruct each patch target in memory from the git base, validating every context/deletion line.
patch_path='run_evidence/launcher_integration.patch';patch=(E/patch_path).read_text().splitlines(keepends=True)
patched={};i=0
while i<len(patch):
    assert patch[i].startswith('--- a/');path=patch[i][6:].rstrip('\n');i+=1
    assert patch[i]=='+++ b/'+path+'\n';i+=1
    original=git_bytes(BASE+':'+path).decode().splitlines(keepends=True);result=[];cursor=0
    while i<len(patch) and not patch[i].startswith('--- a/'):
        h=re.fullmatch(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@.*\n',patch[i]);assert h,patch[i]
        old_start,old_n,new_start,new_n=int(h[1])-1,int(h[2] or 1),int(h[3])-1,int(h[4] or 1)
        result.extend(original[cursor:old_start]);cursor=old_start;assert len(result)==new_start
        removed=added=0;i+=1
        while i<len(patch) and not patch[i].startswith(('@@','--- a/')):
            line=patch[i];i+=1
            if line[0] in ' -':assert original[cursor]==line[1:];cursor+=1;removed+=1
            if line[0] in ' +':result.append(line[1:]);added+=1
        assert (removed,added)==(old_n,new_n)
    result.extend(original[cursor:]);patched[path]=''.join(result).encode()
launch=js('run_evidence/LAUNCH_MANIFEST.json');summary=js('summary.json')
check('launcher patch digest',file_sha(patch_path)==launch['local_patch_sha256']==summary['local_launcher_patch_sha256'])
source_results=[]
for p in sorted((E/'executed_source').rglob('*')):
    if not p.is_file():continue
    rel=str(p.relative_to(E/'executed_source'));target=rel.removesuffix('.txt')
    expected=patched[target] if target in patched else git_bytes(BASE+':'+target)
    eq=expected==p.read_bytes();check('base plus patch source '+rel,eq)
    source_results.append(dict(path=rel,base_commit=BASE,patched=target in patched,sha256=sha(p.read_bytes()),matches_base_plus_patch=eq))
for path,h in launch['scientific_source_sha256'].items():check('launch source '+path,file_sha('executed_source/'+path)==h)

recipe=js('run_evidence/run/RECIPE.json');split=js('run_evidence/SPLIT.json');eff=js('run_evidence/run/EFFECTIVE_SPLIT.json')
bank=js('run_evidence/run/bank/FROZEN.json');student=js('run_evidence/run/student/FROZEN.json')
lock=js('run_evidence/run/PREDICTIONS_LOCKED.json');ev=js('run_evidence/run/evaluation_val/FROZEN.json')
metrics=js('run_evidence/run/evaluation_val/METRICS.json')
pf=js('run_evidence/run/PREFLIGHT_1791920657499107190.json');inv=pf['checks']['native_cine_inventory']
base_diag=js('baseline_forensics/train_all/DIAGNOSTICS.json');baseline_source=[]
for role,rec in base_diag['source_receipts'].items():
    for path,h in rec.items():
        actual=sha(git_bytes(TRAIN_BASE+':src/self_audit_guidance/'+path));equal=actual==h
        baseline_source.append(dict(role=role,path=path,matches_d5a45f3=equal,recorded=h,git_sha256=actual))
        check('baseline source '+role+'/'+path,equal)
# Bind original hashes recorded by publisher, not hashes of sanitized copies.
bindings=[]
def bind(label,recorded,target):
    r=files[target];ok=recorded==r['original_sha256']
    bindings.append(dict(label=label,target=target,recorded_sha256=recorded,original_sha256=r['original_sha256'],published_sha256=r['published_sha256'],matches_original=ok,matches_published=recorded==r['published_sha256']))
    check('original binding '+label,ok)
bind('recipe config',recipe['config_sha256'],'run_evidence/CONFIG.json')
bind('recipe split',recipe['split_sha256'],'run_evidence/SPLIT.json')
bind('effective source split',eff['provenance']['source_sha256'],'run_evidence/SPLIT.json')
bind('bank effective split',bank['source_split_sha256'],'run_evidence/run/EFFECTIVE_SPLIT.json')
bind('preflight effective split',inv['source_split_sha256'],'run_evidence/run/EFFECTIVE_SPLIT.json')
bind('student bank',student['bank_manifest_sha256'],'run_evidence/run/bank/FROZEN.json')
bind('lock recipe',lock['recipe_sha256'],'run_evidence/run/RECIPE.json')
bind('lock student',lock['student_manifest_sha256'],'run_evidence/run/student/FROZEN.json')
for role in ['bank','student','predictions_val','predictions_test','evaluation_val']:
    p='run_evidence/run/'+role+'/FROZEN.json';d=js(p)
    for filename,h in d['files'].items():
        target='run_evidence/run/'+role+'/'+filename
        if target in files:bind(role+' file '+filename,h,target)
    if 'source_sha256' in d:check(role+' source map',d['source_sha256']==recipe['source_sha256'])
for s in ['val','test']:
    p='run_evidence/run/predictions_'+s+'/FROZEN.json';pred=js(p)
    bind('lock '+s,lock['predictions'][s],p)
    bind(s+' student',pred['student_manifest_sha256'],'run_evidence/run/student/FROZEN.json')
    check(s+' checkpoint',pred['checkpoint_sha256']==student['files']['last.pt']==summary['checkpoint_sha256'])
    check(s+' split',pred['patients']==eff[s] and pred['locked_splits']==recipe['splits'])
    check(s+' image identity',all(v['image_sha256']==inv['images'][pid]['image_sha256'] for pid,v in pred['images'].items()))
bind('evaluation prediction',ev['prediction_manifest_sha256'],'run_evidence/run/predictions_val/FROZEN.json')
bind('metrics prediction',metrics['prediction_manifest_sha256'],'run_evidence/run/predictions_val/FROZEN.json')
bind('new diagnostics metrics',js('run_evidence/post_training/val_counts/DIAGNOSTICS.json')['metrics_sha256'],'run_evidence/run/evaluation_val/METRICS.json')
check('same effective splits',all(eff[s]==split[s+'_patients']==recipe['splits'][s]==inv['splits'][s]==base_diag['resolved_recipe']['splits'][s]==bank['locked_splits'][s] for s in ['train','val','test']))
check('disjoint IDs',len(set(sum([eff[s] for s in ['train','val','test']],[])))==148)
check('bank patients',bank['patients']==eff['train'])
check('same config values',recipe['config']==js('run_evidence/CONFIG.json')==js('run_evidence/run/bank/config.json')==bank['config']==base_diag['resolved_config'])
check('bank image identity',all(v['image_sha256']==inv['images'][pid]['image_sha256'] for pid,v in bank['image_records'].items()))
check('raw image hashes distinct across included cohort',len({v['image_sha256'] for v in inv['images'].values()})==148)

check('baseline new TRAIN image hashes',all(v['geometry']['image_sha256']==bank['image_records'][pid]['image_sha256'] for pid,v in base_diag['patients'].items()))
check('recipe source map verified',all(file_sha('executed_source/src/self_audit_guidance/'+p)==h for p,h in recipe['source_sha256'].items()))
check('baseline resolved recipe config',base_diag['resolved_recipe']['config']==recipe['config'])
# Numeric recomputation from tp/fp/fn only. No cached frame scores are inputs.
def frame_value(r):
    tp,fp,fn=[r[k] for k in ('tp','fp','fn')]
    assert all(type(v) is int and v>=0 for v in (tp,fp,fn))
    nums=(2*tp,tp,tp,tp);dens=(2*tp+fp+fn,tp+fp+fn,tp+fp,tp+fn)
    return {m:n/d if d else None for m,n,d in zip(M,nums,dens)}
def reduce_rows(rows):
    by_frame=defaultdict(dict);by_patient=defaultdict(lambda:defaultdict(list));flat=[]
    for r in rows:
        key=(r['patient_id'],r['frame_index']);assert r['class_name'] not in by_frame[key]
        by_frame[key][r['class_name']]=r;v=frame_value(r)
        by_patient[r['patient_id']][r['class_name']].append(v)
        flat.append({**{k:r[k] for k in ['patient_id','frame_index','class_name','stratum','tp','fp','fn']},**v})
    coverage=defaultdict(list)
    for (pid,frame),g in by_frame.items():
        assert set(g)==set(C)
        total=sum(r['tp']+r['fn'] for r in g.values());accepted=sum(r['tp']+r['fp'] for r in g.values())
        assert 0<=accepted<=total
        coverage[pid].append(accepted/total)
    per_patient={pid:{c:{m:mean(r[m] for r in group[c]) for m in M} for c in C} for pid,group in sorted(by_patient.items())}
    classes={c:{m:mean(p[c][m] for p in per_patient.values()) for m in M} for c in C}
    return dict(patient_count=len(per_patient),frames=len(by_frame),classes=classes,foreground={m:mean(classes[c][m] for c in C[1:]) for m in M},coverage=mean(mean(v) for v in coverage.values()),per_patient=per_patient,patient_coverage={p:mean(v) for p,v in sorted(coverage.items())},defined_patients={c:{m:sum(p[c][m] is not None for p in per_patient.values()) for m in M} for c in C},defined_frames={c:{m:sum(frame_value(r)[m] is not None for r in rows if r['class_name']==c) for m in M} for c in C},totals={c:{k:sum(r[k] for r in rows if r['class_name']==c) for k in ['tp','fp','fn']} for c in C},frame_metrics=flat)
rows={};recomputed={}
for label,path in [('baseline','baseline_forensics/val_counts/DIAGNOSTICS.json'),('new','run_evidence/post_training/val_counts/DIAGNOSTICS.json')]:
    d=js(path);rows[label]=d['patient_frame_counts'];rec=reduce_rows(rows[label]);recomputed[label]={'overall':rec,'strata':{s:reduce_rows([r for r in rows[label] if r['stratum']==s]) for s in ['aligned','verified_recovery']}}
    for scope in ['overall','aligned','verified_recovery']:
        want=d['overall'] if scope=='overall' else d['strata'][scope];got=rec if scope=='overall' else recomputed[label]['strata'][scope]
        check(label+' '+scope+' published reduction',all(close(got[k],want[k]) for k in ['patient_count','frames','classes','foreground','coverage','per_patient','defined_patients']))
    check(label+' frame cached numbers',all(close(frame_value(r),{m:r[m] for m in M}) for r in rows[label]))
index={label:{(r['patient_id'],r['frame_index'],r['class_name']):r for r in rr} for label,rr in rows.items()}
check('paired frame/class identities',index['baseline'].keys()==index['new'].keys())
check('paired reference counts and strata',all(a['tp']+a['fn']==index['new'][k]['tp']+index['new'][k]['fn'] and a['stratum']==index['new'][k]['stratum'] for k,a in index['baseline'].items()))
for f in metrics['frames']:
    check('new evaluator counts '+f['patient_id']+'/'+str(f['frame_index']),all(all(f['metrics']['classes'][c][k]==index['new'][(f['patient_id'],f['frame_index'],c)][k] for k in ['tp','fp','fn']) for c in C))
    check('new frame total '+f['patient_id']+'/'+str(f['frame_index']),sum(f['metrics']['classes'][c]['tp']+f['metrics']['classes'][c]['fn'] for c in C)==f['metrics']['pixels'])

b=recomputed['baseline']['overall'];n=recomputed['new']['overall']
def quantile(v,p):
    v=sorted(v);x=(len(v)-1)*p;i=int(x);return v[i] if i==len(v)-1 else v[i]+(v[i+1]-v[i])*(x-i)
def describe(v):
    return dict(n=len(v),mean=mean(v),median=statistics.median(v),min=min(v),max=max(v),q10=quantile(v,.1),q25=quantile(v,.25),q75=quantile(v,.75),q90=quantile(v,.9),positive=sum(x>1e-15 for x in v),negative=sum(x< -1e-15 for x in v),zero=sum(abs(x)<=1e-15 for x in v))
patient_rows=[]
for pid in b['per_patient']:
    row={'patient_id':pid,'stratum':next(r['stratum'] for r in rows['baseline'] if r['patient_id']==pid)}
    for m in ['dice','iou','recall','coverage']:
        bv=b['patient_coverage'][pid] if m=='coverage' else mean(b['per_patient'][pid][c][m] for c in C[1:])
        nv=n['patient_coverage'][pid] if m=='coverage' else mean(n['per_patient'][pid][c][m] for c in C[1:])
        row[m]={'baseline':bv,'new':nv,'delta_percentage_points':100*(nv-bv)}
    patient_rows.append(row)
paired={}
for s in ['overall','aligned','verified_recovery']:
    rr=[r for r in patient_rows if s=='overall' or r['stratum']==s]
    paired[s]={m:describe([r[m]['delta_percentage_points'] for r in rr]) for m in ['dice','iou','recall','coverage']}
dice_sorted=sorted(patient_rows,key=lambda r:r['dice']['delta_percentage_points'],reverse=True)
loo=[{'excluded_patient':r['patient_id'],'mean_dice_delta_pp':mean(x['dice']['delta_percentage_points'] for x in patient_rows if x!=r)} for r in patient_rows]
outlier={
    'leave_one_out':loo,
    'exclude_largest_positive_deltas':[dict(k=k,excluded_patients=[r['patient_id'] for r in dice_sorted[:k]],mean_remaining_dice_delta_pp=mean(r['dice']['delta_percentage_points'] for r in dice_sorted[k:])) for k in [1,2,3]],
    'cohort_dice_distribution':{label:{**describe([mean(v[c]['dice'] for c in C[1:]) for v in rec['per_patient'].values()]),'patients_below_1_percent':sum(mean(v[c]['dice'] for c in C[1:])<.01 for v in rec['per_patient'].values()),'patients_zero_foreground_dice':sum(mean(v[c]['dice'] for c in C[1:])==0 for v in rec['per_patient'].values())} for label,rec in [('baseline',b),('new',n)]}
}
# Prespecified here as descriptive support intersections, never threshold selection.
common={}
for cls in ['MYO','LV']:
    keys=[k for k,a in index['baseline'].items() if k[2]==cls and a['tp']+a['fp']>0 and index['new'][k]['tp']+index['new'][k]['fp']>0]
    vals={label:{pid:mean(frame_value(index[label][k])['precision'] for k in keys if k[0]==pid) for pid in sorted({k[0] for k in keys})} for label in ['baseline','new']}
    common[cls]={'patient_count':len(vals['new']),'frame_count':len(keys),'keys':[[k[0],k[1]] for k in keys],'per_patient':vals,'baseline':mean(vals['baseline'].values()),'new':mean(vals['new'].values())}
strict_ids=[pid for pid in b['per_patient'] if all(index[label][k]['tp']+index[label][k]['fp']>0 for label in ['baseline','new'] for k in index[label] if k[0]==pid and k[2] in ['MYO','LV'])]
strict={label:{c:mean(mean(frame_value(r)['precision'] for k,r in index[label].items() if k[0]==pid and k[2]==c) for pid in strict_ids) for c in ['MYO','LV']} for label in ['baseline','new']}
precision={
    'published_macro':{'baseline':b['foreground']['precision'],'new':n['foreground']['precision'],'delta_pp':100*(n['foreground']['precision']-b['foreground']['precision'])},
    'MYO_LV_only_original_defined_sets':{'baseline':mean(b['classes'][c]['precision'] for c in ['MYO','LV']),'new':mean(n['classes'][c]['precision'] for c in ['MYO','LV'])},
    'class_specific_identical_frame_support':{'classes':common,'baseline':mean(common[c]['baseline'] for c in common),'new':mean(common[c]['new'] for c in common)},
    'both_classes_both_frames_defined_in_both_runs':{'patients':strict_ids,'patient_count':len(strict_ids),'per_class':strict,'baseline':mean(strict['baseline'].values()),'new':mean(strict['new'].values())},
    'limits':'Conditional on jointly retained prediction support; not full-cohort precision, RV precision, matched pixel support, causal effect or generalization evidence.'
}
for v in precision.values():
    if isinstance(v,dict) and 'baseline' in v and 'new' in v:v['delta_pp']=100*(v['new']-v['baseline'])
precision['decomposition_pp']={
    'baseline_RV_class_omission_effect':100*(precision['MYO_LV_only_original_defined_sets']['baseline']-b['foreground']['precision']),
    'MYO_LV_change_with_run_specific_support':precision['MYO_LV_only_original_defined_sets']['delta_pp']}
strict_patient_precision=[]
for pid in strict_ids:
    values={label:mean(mean(frame_value(r)['precision'] for k,r in index[label].items() if k[0]==pid and k[2]==c) for c in ['MYO','LV']) for label in ['baseline','new']}
    strict_patient_precision.append({'patient_id':pid,**values,'delta_pp':100*(values['new']-values['baseline'])})
precision['strict_common_patient_deltas']=strict_patient_precision
precision['strict_common_delta_summary_pp']=describe([r['delta_pp'] for r in strict_patient_precision])
# Training/log receipts, no execution of the archived suite.
history=js('run_evidence/run/student/training.json')
logrows=[];stages=[]
for line in (E/'run_evidence/launcher.log').read_text().splitlines():
    if line.startswith('[STUDENT] '):logrows.append(ast.literal_eval(line[len('[STUDENT] '):]))
    if line.startswith('[STAGE] '):stages.append(line.split()[1])
check('50 epoch records',len(history)==len(logrows)==50)
check('history equals launcher epoch rows',history==logrows)
check('exact runtime stage order',stages==['preflight','prepare','train','export_val','export_test','evaluate_val'])
check('optimizer steps sum',sum(r['steps'] for r in history)==summary['training']['optimizer_steps']==127650)
check('semantic active steps sum',sum(r['semantic_diagnostics']['active_batches'] for r in history)==summary['training']['semantic_updates']==110461)
check('first last epoch summary',history[0]==summary['training']['first_epoch'] and history[-1]==summary['training']['last_epoch'])
# Verify suite counts from archived JUnit XML without running code.
import xml.etree.ElementTree as ET
suite=[]
for p in ['run_evidence/launch_tests.xml','baseline_forensics/tests.xml']:
    root=ET.fromstring((E/p).read_text());cases=root.findall('.//testcase');failed=[x for x in cases if x.find('failure') is not None or x.find('error') is not None]
    suite.append(dict(path=p,testcases=len(cases),failures=len(failed)))
    check('archived suite '+p,len(cases)==(122 if p.startswith('run_evidence') else 120) and not failed)

result=dict(schema='self_audit.provenance_metrics_independent_recomputation.v1',status='PASS' if all(c['passed'] for c in checks) else 'FINDINGS',evidence_commit=PIN,executed_base_commit=BASE,baseline_training_commit=TRAIN_BASE,scope='VAL saved-count arithmetic and text provenance only; no image, mask, weight or prediction arrays read; no TEST scores; no inference or training',script_sha256=sha(Path(__file__).read_bytes()),checks=checks,publication=dict(files=file_results,copied_count=len(files),copied_bytes=sum(r['published_bytes'] for r in files.values()),sanitized_count=sum(r['sanitized'] for r in files.values()),unlisted_files_pinned_by_git=unlisted,sanitization_pattern_hits=sanitization_scan,excluded_artifacts_count=len(publication['excluded_binary_artifacts'])),source_identity=dict(executed_source=source_results,patch_targets=sorted(patched),baseline_training_receipts=baseline_source),original_hash_bindings=bindings,missing_original_payloads=dict(checkpoint=student['files']['last.pt'],source_archive=launch['source_base_archive_sha256'],baseline_evaluation=js('baseline_forensics/val_counts/DIAGNOSTICS.json')['metrics_sha256']),cohort=dict(counts={s:len(eff[s]) for s in ['train','val','test']},excluded_val=split['provenance']['exclusions'],config_equal_values_baseline_new=True,baseline_config_hash=base_diag['resolved_recipe']['config_sha256'],new_config_hash=recipe['config_sha256']),recomputed=recomputed,paired_patient_metrics=patient_rows,paired_delta_summaries=paired,outlier_sensitivity=outlier,precision_comparisons=precision,training_receipts=dict(epochs=len(history),stages=stages,tests=suite),interpretation_limits=['Hash matches verify current published bytes and recorded original-identifier consistency, not absent payload bytes or authenticity of runtime claims.','Baseline original evaluation is absent; arithmetic starts from hash-bound published diagnostic counts.','Baseline/new reference class totals agree for every patient/frame/class, but equal totals cannot prove identical masks.','Patient counts are units of descriptive pairing; no pixel-IID intervals or tests.','No raw geometry recovery verification, no checkpoint replay, no TEST scoring, no statistical or causal claim.'])
OUT.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps({'status':result['status'],'checks':len(checks),'failed':[c for c in checks if not c['passed']],'manifest_files':len(files),'sanitized':result['publication']['sanitized_count'],'output':str(OUT)},indent=2))
