"""Evidence-only geometry attribution; no models, data arrays, GPU, or TEST scores.

Run: rtk proxy python -B reports/guidance_research_continuation_20261014/GEOMETRY_ATTRIBUTION_CHECK.py
Outputs only GEOMETRY_ATTRIBUTION_* beside this file. Requires existing NumPy.
"""
from pathlib import Path
import ast
import csv
import difflib
import hashlib
import json
import re
import subprocess
from collections import Counter
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
E = ROOT / 'reports/stable_guidance_world_v2_20261014'
S = E / 'executed_source'
OLD = 'd5a45f31481f0a702a1f9d5d418889fe704e9105'
BASE = '7d1c6a04064cda473d1eccfbed818392c37b1a37'
HEAD = 'b937e3b746608725e43d3bc3cef8861666d152dc'
CLASSES = ['BG', 'RV', 'MYO', 'LV']

def sha(data):
    return hashlib.sha256(data).hexdigest()

def git_blob(commit, rel):
    p = subprocess.run(['rtk', 'proxy', 'git', 'show', commit+':'+rel], cwd=ROOT, capture_output=True)
    return None if p.returncode else p.stdout

def read(rel):
    return json.loads((E/rel).read_text())

def write(name, obj):
    (OUT/('GEOMETRY_ATTRIBUTION_'+name+'.json')).write_text(json.dumps(obj, indent=2, sort_keys=True)+'\n')

actual_head = subprocess.check_output(['rtk','proxy','git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
assert actual_head == HEAD
before = {str(p.relative_to(E)): sha(p.read_bytes()) for p in E.rglob('*') if p.is_file()}
old = read('baseline_forensics/train_all/DIAGNOSTICS.json')
bank = read('run_evidence/run/bank/FROZEN.json')
report = read('run_evidence/run/bank/REPORT.json')
recipe = read('run_evidence/run/RECIPE.json')
launch = read('run_evidence/LAUNCH_MANIFEST.json')
history = read('run_evidence/run/student/training.json')
assert set(old['patients']) == set(bank['patients']) == set(bank['image_records'])
assert bank['patient_reports'] == report['patient_reports']
assert all(v['recorded_bank_report']['class_pixels'] == [c['hard_accepted'] for c in v['classes']] for v in old['patients'].values())

# Execute only the pure affine helper extracted by AST; no package import / IO / arrays.
tree = ast.parse((S/'src/self_audit_guidance/experimental_geometry.py').read_text())
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'world_left_geometry')
namespace = {'np': np}
exec(compile(ast.Module(body=[fn], type_ignores=[]), '<executed geometry helper>', 'exec'), namespace)
geometry = namespace['world_left_geometry']
patients = []
checks = {}
for pid, m in bank['image_records'].items():
    o = old['patients'][pid]; g = m['world_left_geometry']; a = np.asarray(m['affine'])
    assert o['geometry'] == g
    assert g['image_sha256'] == m['image_sha256']
    assert o['legacy_patient_left_axis'] == m['patient_left_axis'] == '+x'
    regenerated = geometry(a, m['shape_xyzt'][1::-1], [48,48], sform_code=g['sform_code'], qform_code=g['qform_code'])
    assert all(g[k] == v for k,v in regenerated.items())
    projection = -a[0,:2] / np.linalg.norm(a[:3,:3],axis=0)[:2]
    assert np.argmax(abs(projection)) == 0 and projection[0] >= .9
    if g['valid']:
        assert g['sform_code'] == 2 and g['qform_code'] == 0
        assert np.array_equal(a, g['sform']) and np.array_equal(a, np.diag([-1.,-1.,1.,1.]))
        assert m['spatial_unit'] == 'unknown' and g['qform'] is None
        assert g['left_vector_yx'][0] == 0 and g['left_vector_yx'][1] > 0
    else:
        assert g['sform_code'] == g['qform_code'] == 0 and g['sform'] is g['qform'] is None
        assert m['spatial_unit'] == 'mm' and g['left_vector_yx'] == [0.,0.]
    op=o['recorded_bank_report']; npat=bank['patient_reports'][pid]
    assert op['graph'] == npat['graph']
    assert op['anatomy']['wall_cavity_hypotheses'] == npat['anatomy']['wall_cavity_hypotheses']
    patients.append(dict(patient_id=pid, reason=g['reason'], sform_code=g['sform_code'], qform_code=g['qform_code'],
                         spatial_unit=m['spatial_unit'], old_pixels=op['class_pixels'], new_pixels=npat['class_pixels'],
                         delta_pixels=[n-o for n,o in zip(npat['class_pixels'],op['class_pixels'])],
                         old_rv_hypotheses=op['anatomy']['rv_hypotheses'], new_rv_hypotheses=npat['anatomy']['rv_hypotheses']))
checks['patient_records'] = patients
checks['strata'] = {}
for reason in ['CODED_WORLD_LEFT','NO_CODED_WORLD_GEOMETRY']:
    rows = [r for r in patients if r['reason']==reason]; ids=[r['patient_id'] for r in rows]
    d=dict(patients=len(rows), ids=ids)
    for mode in ['old','new']:
        d[mode+'_pixels']=[sum(r[mode+'_pixels'][c] for r in rows) for c in range(4)]
        d[mode+'_class_patients']=[sum(r[mode+'_pixels'][c]>0 for r in rows) for c in range(4)]
        d[mode+'_rv_hypotheses']=sum(r[mode+'_rv_hypotheses'] for r in rows)
    d['delta_pixels']=[n-o for n,o in zip(d['new_pixels'],d['old_pixels'])]
    d['old_soft_summaries']={k:[sum(old['patients'][i]['classes'][c][k] for i in ids) for c in range(4)]
        for k in ['r0_argmax_informative','q_argmax','soft_weight_pixels','soft_weight_on_unknown','target_mass']}
    d['new_raw_vote_mass']=[sum(bank['patient_reports'][i]['anatomy']['raw_vote_mass_per_class'][c] for i in ids) for c in range(4)]
    d['patients_with_changed_hard_counts']=[r['patient_id'] for r in rows if any(r['delta_pixels'])]
    checks['strata'][reason]=d
assert checks['strata']['CODED_WORLD_LEFT']['delta_pixels']==[0,0,0,0]
assert checks['strata']['NO_CODED_WORLD_GEOMETRY']['delta_pixels']==[192,-4322,53,0]
checks['all_pixels']=sum(v['pixels'] for v in old['patients'].values())
checks['old_unknown_pixels']=sum(v['unknown_pixels'] for v in old['patients'].values())
checks['new_unknown_pixels']=checks['all_pixels']-sum(sum(v['class_pixels']) for v in bank['patient_reports'].values())
checks['all_geometries_recomputed']=True
checks['all_image_hashes_match_old_header_audit']=True
checks['all_graph_count_reports_equal']=True
checks['all_wall_cavity_hypothesis_counts_equal']=True
checks['new_bank_npz_payloads_present']=sum((E/'run_evidence/run/bank'/(i+'.npz')).exists() for i in bank['patients'])
checks['full_array_identity']='BLOCKED: old and new per-array hashes/arrays absent; count equality is not pixel identity'

# Available full-bank target mass compared to complete per-epoch diagnostic sum.
mass_old=[sum(v['classes'][c]['target_mass'] for v in old['patients'].values()) for c in range(4)]
mass_new=history[0]['semantic_diagnostics']['per_class_target_mass']
checks['weighted_Q_mass']={'old_full_bank_reported':mass_old,'new_epoch1_reported':mass_new,
    'delta':[n-o for n,o in zip(mass_new,mass_old)],
    'relative_delta_percent':[100*(n-o)/o for n,o in zip(mass_new,mass_old)],
    'new_epoch_range':[[min(h['semantic_diagnostics']['per_class_target_mass'][c] for h in history),max(h['semantic_diagnostics']['per_class_target_mass'][c] for h in history)] for c in range(4)],
    'caveat':'Equivalent full pixel sums under complete batching and all-class permission; NumPy per-patient and GPU per-batch reductions differ in order/grouping; new per-patient Q and weights are absent.'}
checks['semantic_updates']={'old':old['semantic_updates_recorded'],'new':sum(h['semantic_diagnostics']['active_batches'] for h in history)}
checks['recipe_equal_keys']=[k for k in sorted(recipe) if k in old['resolved_recipe'] and recipe[k]==old['resolved_recipe'][k]]
checks['recipe_differences']={k:{'old':old['resolved_recipe'].get(k),'new':recipe.get(k)} for k in sorted(set(recipe)|set(old['resolved_recipe'])) if recipe.get(k)!=old['resolved_recipe'].get(k)}

# Exact git blob checks and textual diffs without changing checkout or temp source.
diffs=[];checks['source_comparisons']={}
for commit in [OLD,BASE]:
    comparisons={}
    for f in sorted(S.rglob('*')):
        if not f.is_file():continue
        rel=str(f.relative_to(S)); source_rel=rel[:-4] if rel.endswith('.py.txt') else rel
        prev=git_blob(commit,source_rel); now=f.read_bytes()
        comparisons[source_rel]={'base_sha256':sha(prev) if prev is not None else None,'executed_sha256':sha(now),
                                 'status':'MISSING_AT_BASE' if prev is None else 'EQUAL' if prev==now else 'DIFF'}
        if prev is not None and prev!=now:
            diffs.extend(difflib.unified_diff(prev.decode().splitlines(True),now.decode().splitlines(True),fromfile=commit+':'+source_rel,tofile='executed_source/'+rel))
    checks['source_comparisons'][commit]=comparisons
checks['baseline_source_receipts_match_d5a45f3']={stage:{name:sha(git_blob(OLD,'src/self_audit_guidance/'+name))==h for name,h in values.items()} for stage,values in old['source_receipts'].items()}
assert all(all(values.values()) for values in checks['baseline_source_receipts_match_d5a45f3'].values())
checks['executed_source_matches_launch_hashes']={name:sha((S/name).read_bytes())==h for name,h in launch['scientific_source_sha256'].items()}
assert all(checks['executed_source_matches_launch_hashes'].values())
checks['unchanged_scientific_symbols']={}
for file,names in {'core.py':['build_graph','anchored_solve','_cluster'],
                   'student.py':['Block','GuidanceStudent','guidance_loss','contexts','predict_cine']}.items():
    rel='src/self_audit_guidance/'+file
    oldtext=git_blob(OLD,rel).decode();newtext=(S/rel).read_text()
    for name in names:
        snippets=[]
        for source in [oldtext,newtext]:
            node=next(n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
            snippets.append(ast.get_source_segment(source,node))
        assert snippets[0]==snippets[1]
        checks['unchanged_scientific_symbols'][file+':'+name]=sha(snippets[0].encode())
patch=(E/'run_evidence/launcher_integration.patch').read_text()
checks['launcher_patch_sha256']=sha(patch.encode())
checks['launcher_patch_matches_declared_sha256']=checks['launcher_patch_sha256']==launch['local_patch_sha256']
# Strict unified patch replay in memory, including context, hunk and final-source checks.
lines=patch.splitlines(True); pos=0; replay={}
while pos<len(lines):
    assert lines[pos].startswith('--- a/'); rel=lines[pos][6:].strip();pos+=1
    assert lines[pos].strip()=='+++ b/'+rel;pos+=1
    src=git_blob(BASE,rel).decode().splitlines(True);result=[];cur=0
    while pos<len(lines) and not lines[pos].startswith('--- a/'):
        match=re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@',lines[pos]);assert match
        start=int(match[1])-1; result.extend(src[cur:start]);cur=start;pos+=1;old_n=new_n=0
        while pos<len(lines) and not lines[pos].startswith(('@@','--- a/')):
            line=lines[pos];pos+=1;prefix=line[0];body=line[1:]
            if prefix in ' -':assert src[cur]==body;cur+=1;old_n+=1
            if prefix in ' +':result.append(body);new_n+=1
        assert old_n==int(match[2] or 1) and new_n==int(match[4] or 1)
    result.extend(src[cur:]); target=S/(rel+'.txt' if rel.startswith('tests/') else rel)
    replay[rel]=''.join(result)==target.read_text();assert replay[rel]
checks['patch_replayed_exactly_against_7d1c6a0']=replay
(OUT/'GEOMETRY_ATTRIBUTION_SOURCE_DIFF.txt').write_text(''.join(diffs))

# CPU-only geometric identity checks, mathematical checks rather than clinical tests.
synthetic=[]
for angle in [0., np.pi/4, np.pi/2, 3*np.pi/4]:
    a=np.eye(4); a[:2,:2]=[[np.cos(angle)*2,-np.sin(angle)*3],[np.sin(angle)*2,np.cos(angle)*3]]
    a[:3,3]=[13,-7,4]
    g=geometry(a,[192,256],[48,48],sform_code=1,qform_code=0)
    if angle==np.pi/4:
        assert max(abs(-a[0,:2]/np.linalg.norm(a[:3,:3],axis=0)[:2]))<.9 and g['valid']
    u=np.array([7.,11.]);v=np.array([29.,36.]);sy,sx=4,256/48
    def world(p):return a @ [sx*p[1]+(sx-1)/2,sy*p[0]+(sy-1)/2,3,1]
    lhs=np.asarray(g['left_vector_yx'])@(u-v); rhs=-(world(u)-world(v))[0]
    assert abs(lhs-rhs)<1e-12
    synthetic.append(dict(angle=float(angle),difference_error=float(abs(lhs-rhs))))
assert not geometry(np.eye(4),[192,256],[48,48],sform_code=0,qform_code=0)['valid']
checks['synthetic_covector_identity']=synthetic
checks['synthetic_uncoded_abstains']=True

# Exposed VAL counts only; no reference/prediction arrays and no TEST scores.
old_val=read('baseline_forensics/val_counts/DIAGNOSTICS.json')
new_val=read('run_evidence/post_training/val_counts/DIAGNOSTICS.json')
def mean(x):
    x=[v for v in x if v is not None]
    return sum(x)/len(x) if x else None
val={}
for st in ['aligned','verified_recovery']:
    val[st]={}
    for name,obj in [('old',old_val),('new',new_val)]:
        rows=[r for r in obj['patient_frame_counts'] if r['stratum']==st]
        ids=sorted({r['patient_id'] for r in rows});groups={}
        for c in CLASSES:
            cr=[r for r in rows if r['class_name']==c]
            def metric(r,k):
                t,f,n=r['tp'],r['fp'],r['fn']; num,den={'dice':(2*t,2*t+f+n),'recall':(t,t+n),'precision':(t,t+f)}[k]
                return num/den if den else None
            groups[c]={k:mean([mean([metric(r,k) for r in cr if r['patient_id']==i]) for i in ids]) for k in ['dice','precision','recall']}
            groups[c].update(predicted_pixels=sum(r['tp']+r['fp'] for r in cr),true_positive_pixels=sum(r['tp'] for r in cr),reference_pixels=sum(r['tp']+r['fn'] for r in cr),predicted_frames=sum(r['tp']+r['fp']>0 for r in cr))
        val[st][name]=dict(patient_ids=ids,patients=len(ids),classes=groups,foreground_dice=mean([groups[c]['dice'] for c in CLASSES[1:]]))
    assert val[st]['old']['patient_ids']==val[st]['new']['patient_ids']
    assert all(val[st]['old']['classes'][c]['reference_pixels']==val[st]['new']['classes'][c]['reference_pixels'] for c in CLASSES)
val['scope']='DERIVED from exposed VAL counts; reference-informed development; frame alignment strata are not proven TRAIN header-code strata.'
write('VAL_STRATA',val)
after={str(p.relative_to(E)):sha(p.read_bytes()) for p in E.rglob('*') if p.is_file()}
assert before==after
checks.update(status='VERIFIED_EVIDENCE_REDUCTIONS_AND_CPU_GEOMETRY_ONLY',head=HEAD,baseline_training_sha=OLD,new_base_sha=BASE,
              class_order=CLASSES,evidence_files_sha256=before,evidence_unchanged_during_check=True,reference_arrays_read=False,
              test_scores_read=False,training_or_model_inference_run=False)
write('CHECKS',checks)
with (OUT/'GEOMETRY_ATTRIBUTION_PATIENTS.csv').open('w') as stream:
    w=csv.DictWriter(stream,fieldnames=list(patients[0]));w.writeheader();w.writerows(patients)
print(json.dumps({k:checks[k] for k in ['status','all_pixels','old_unknown_pixels','new_unknown_pixels','weighted_Q_mass','semantic_updates','patch_replayed_exactly_against_7d1c6a0','launcher_patch_matches_declared_sha256','evidence_unchanged_during_check']},indent=2))
