"""Report-only reductions of published text evidence; stdlib, no arrays/model/GT.

Run from any cwd: rtk proxy python <this file>
Writes only LEARNING_MECHANISM_* siblings. Assertions check evidence arithmetic,
not clinical accuracy, training reproducibility, or original binary integrity.
"""
from pathlib import Path
import ast
import csv
import hashlib
import json
import math
import statistics as st
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
E = ROOT / 'reports/stable_guidance_world_v2_20261014'
SHA = 'b937e3b746608725e43d3bc3cef8861666d152dc'
BASE = 'd5a45f31481f0a702a1f9d5d418889fe704e9105'
CLASSES = ['BG', 'RV', 'MYO', 'LV']
read = lambda p: json.loads((E / p).read_text())
digest = lambda b: hashlib.sha256(b).hexdigest()
b = read('baseline_forensics/train_all/DIAGNOSTICS.json')
n = read('run_evidence/run/bank/REPORT.json')
h = read('run_evidence/run/student/training.json')
recipe = read('run_evidence/run/RECIPE.json')
student = read('run_evidence/run/student/REPORT.json')
summary = read('summary.json')
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == SHA
assert set(b['patients']) == set(n['patient_reports']) == set(recipe['splits']['train'])
assert len(b['patients']) == 80 and len(h) == 50

def distribution(xs):
    xs = list(xs)
    return dict(n=len(xs), min=min(xs), median=st.median(xs), mean=st.mean(xs), max=max(xs))

def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

manifest = read('MANIFEST.json')['files']
hash_errors = [r['published_path'] for r in manifest
               if digest((E/r['published_path']).read_bytes()) != r['published_sha256']]
assert not hash_errors
source_checks = {}
for name, expected in recipe['source_sha256'].items():
    actual = digest((E/'executed_source/src/self_audit_guidance'/name).read_bytes())
    source_checks[name] = dict(actual=actual, expected=expected, equal=actual == expected)
assert all(x['equal'] for x in source_checks.values())
base_source = subprocess.check_output(['git','show', BASE+':src/self_audit_guidance/student.py'], cwd=ROOT)
assert digest(base_source) == b['source_receipts']['student']['student.py']
old_ast, new_ast = ast.parse(base_source), ast.parse((E/'executed_source/src/self_audit_guidance/student.py').read_text())
def symbol(tree, name):
    return ast.dump(next(x for x in tree.body if getattr(x,'name',None)==name), include_attributes=False)
unchanged = {name: symbol(old_ast,name)==symbol(new_ast,name)
             for name in ['Block','GuidanceStudent','guidance_loss','contexts','load_student','predict_cine']}
assert all(unchanged.values())
patch_hash = digest((E/'run_evidence/launcher_integration.patch').read_bytes())
assert patch_hash == summary['local_launcher_patch_sha256']

per_class = []
for pid,p in b['patients'].items():
    for c,k in enumerate(CLASSES):
        old = p['classes'][c]; new = n['patient_reports'][pid]
        assert old['hard_accepted'] == p['recorded_bank_report']['class_pixels'][c]
        assert old['soft_weight_pixels'] == old['hard_accepted'] + old['soft_weight_on_unknown']
        per_class.append(dict(patient_id=pid,class_name=k,geometry=p['geometry']['reason'],
            baseline_r0_informative_argmax=old['r0_argmax_informative'],baseline_q_argmax=old['q_argmax'],
            baseline_hard=old['hard_accepted'],baseline_soft_positive_argmax=old['soft_weight_pixels'],
            baseline_soft_positive_on_unknown=old['soft_weight_on_unknown'],
            baseline_channel_target_mass=old['target_mass'],
            new_hard=new['class_pixels'][c],hard_delta=new['class_pixels'][c]-old['hard_accepted'],
            new_raw_vote_mass=new['anatomy']['raw_vote_mass_per_class'][c],
            new_per_patient_soft_support='NOT_RECORDED'))
write_csv('LEARNING_MECHANISM_PATIENT_CLASS.csv',per_class)
aggregates = {}
for c,k in enumerate(CLASSES):
    rows = [r for r in per_class if r['class_name']==k]
    totals = {key:sum(r[key] for r in rows) for key in rows[0]
              if key not in ['patient_id','class_name','geometry','new_per_patient_soft_support']}
    totals.update(baseline_hard_patients=sum(r['baseline_hard']>0 for r in rows),
        new_hard_patients=sum(r['new_hard']>0 for r in rows),
        baseline_soft_patients=sum(r['baseline_soft_positive_argmax']>0 for r in rows),
        baseline_soft_without_hard_patients=sum(r['baseline_soft_positive_argmax']>0 and not r['baseline_hard'] for r in rows),
        baseline_hard_distribution=distribution(r['baseline_hard'] for r in rows),
        new_hard_distribution=distribution(r['new_hard'] for r in rows),
        baseline_top5_hard_fraction=sum(sorted([r['baseline_hard'] for r in rows],reverse=True)[:5])/totals['baseline_hard'],
        new_top5_hard_fraction=sum(sorted([r['new_hard'] for r in rows],reverse=True)[:5])/totals['new_hard'])
    totals['baseline_confidence_on_q_argmax_class'] = sum(
        p['classes'][c]['q_argmax']*(p['classes'][c]['mean_target_confidence'] or 0)
        for p in b['patients'].values())/totals['baseline_q_argmax']
    totals['baseline_entropy_on_q_argmax_class'] = sum(
        p['classes'][c]['q_argmax']*(p['classes'][c]['mean_target_entropy'] or 0)
        for p in b['patients'].values())/totals['baseline_q_argmax']
    aggregates[k] = totals
assert [aggregates[k]['baseline_hard_patients']>=2 for k in CLASSES] == b['allowed_classes']
assert [aggregates[k]['new_hard_patients'] for k in CLASSES] == student['class_patients'] == n['class_patients']
assert [aggregates[k]['new_hard'] for k in CLASSES] == summary['bank']['class_pixels']
bank = {key:sum(p[key] for p in b['patients'].values()) for key in
        ['pixels','unknown_pixels','initial_informative_pixels','accepted_initially_neutral',
         'weight_sum','zero_reliability_pixels','permission_veto_pixels']}
bank['positive_weight_pixels'] = sum(a['baseline_soft_positive_argmax'] for a in aggregates.values())
bank['soft_unknown_pixels'] = sum(a['baseline_soft_positive_on_unknown'] for a in aggregates.values())
bank['pooled_weighted_entropy'] = sum(p['weight_sum']*p['weighted_target_entropy'] for p in b['patients'].values())/bank['weight_sum']
bank['new_hard_total'] = sum(a['new_hard'] for a in aggregates.values())
bank['new_pixels'] = sum(p['graph']['nodes'] for p in n['patient_reports'].values())
bank['new_unknown_pixels'] = bank['new_pixels']-bank['new_hard_total']
assert bank['positive_weight_pixels'] == bank['pixels']-bank['zero_reliability_pixels']
assert bank['soft_unknown_pixels'] == bank['positive_weight_pixels']-(bank['pixels']-bank['unknown_pixels'])

batch_rows=[]
for pid,p in b['patients'].items():
    assert len(p['batches']) == 1
    x=p['batches'][0]; sem=x['semantic']; grad=x['shared_encoder_gradients']
    assert abs(sem['ce']-sem['entropy']-sem['kl']) < 1e-6
    assert not grad['weights_updated']
    batch_rows.append(dict(patient_id=pid,active=sem['active'],ce=sem['ce'],entropy=sem['entropy'],kl=sem['kl'],
        weight_sum=sem['weight_sum'],denominator=sem['denominator'],
        **{k:grad[k] for k in ['feature_norm','semantic_norm','cosine']},**x['fidelity_not_accuracy'],
        **{f'{gate}_{key}':v for gate,counts in x['gates'].items() for key,v in counts.items()
           if key not in ['argmax_counts','representation_only']}))
write_csv('LEARNING_MECHANISM_BASELINE_BATCHES.csv',batch_rows)
batch_summary = dict(count=len(batch_rows),active=sum(r['active'] for r in batch_rows),
    scalar_distributions={k:distribution(r[k] for r in batch_rows) for k in batch_rows[0]
                          if k not in ['patient_id','active'] and not k.startswith(('coarse_','native_','projected_'))},
    fidelity={k:distribution(r[k] for r in batch_rows) for k in p['batches'][0]['fidelity_not_accuracy']},
    negative_gradient_cosines=sum(r['cosine']<0 for r in batch_rows),
    nonzero_feature_gradient_batches=sum(r['feature_norm']>0 for r in batch_rows),
    nonzero_semantic_gradient_batches=sum(r['semantic_norm']>0 for r in batch_rows),
    gates={gate:{key:sum(p['batches'][0]['gates'][gate][key] for p in b['patients'].values())
                for key in ['pixels','accepted','probability_rejected','margin_rejected','permission_rejected']}
           for gate in ['coarse','native','projected_control']},
    argmax_counts={gate:[sum(p['batches'][0]['gates'][gate]['argmax_counts'][c] for p in b['patients'].values())
                        for c in range(4)] for gate in ['coarse','native','projected_control']},
    per_class_ce=[st.mean(p['batches'][0]['semantic']['per_class_ce'][c] for p in b['patients'].values()) for c in range(4)])
overlap=[]
for pid,p in b['patients'].items():
    x=p['batches'][0]; ng=x['gates']['native']; pg=x['gates']['projected_control']
    N=ng['pixels']; A=ng['accepted']; B=pg['accepted']
    product=N*x['fidelity_not_accuracy']['native_projected_gated_disagreement']
    D=round(product)
    # Stored rate was bool.float().mean(); counts <2**24 and rate rounding
    # uncertainty <0.5 count here. Rounded counts are DERIVED, not raw receipts.
    assert N < 2**24 and abs(product-D)<0.01 and N*2**-24 < 0.5
    lo=max(0,A+B-N,math.ceil((A+B-D)/2));hi=min(A,B,A+B-D)
    assert 0<=lo<=hi
    overlap.append(dict(patient_id=pid,pixels=N,native_accepted=A,projected_accepted=B,
        reconstructed_disagreement=D,reconstruction_error=abs(product-D),
        overlap_lower=lo,overlap_upper=hi,native_only_lower=A-hi,native_only_upper=A-lo))
batch_summary['support_overlap']={k:sum(r[k] for r in overlap) for k in overlap[0]
    if k not in ['patient_id','reconstruction_error']}
batch_summary['support_overlap']['max_count_rounding_residual']=max(r['reconstruction_error'] for r in overlap)
batch_summary['denominator_clamped_patients']=[r['patient_id'] for r in batch_rows if r['weight_sum']<1e-8]
write_csv('LEARNING_MECHANISM_SUPPORT_OVERLAP.csv',overlap)

epoch_rows=[]
for row in h:
    d=row['semantic_diagnostics'];a=d['active_fraction']
    assert d['batches']==row['steps']==2553
    assert abs(d['ce']-d['entropy']-d['kl'])<1e-7
    assert abs(d['ce']-row['semantic_loss'])<1e-7
    assert abs(d['active_batch_ce']*a-d['ce'])<1e-12
    epoch_rows.append(dict(epoch=row['epoch'],batches=d['batches'],active_batches=d['active_batches'],
        active_fraction=a,feature_loss=row['feature_loss'],ce=d['ce'],entropy=d['entropy'],kl=d['kl'],
        active_ce=d['active_batch_ce'],active_entropy=d['entropy']/a,active_kl=d['kl']/a,
        **{f'target_mass_{k}':d['per_class_target_mass'][c] for c,k in enumerate(CLASSES)},
        **{f'ce_{k}':d['per_class_ce'][c] for c,k in enumerate(CLASSES)}))
write_csv('LEARNING_MECHANISM_EPOCHS.csv',epoch_rows)
new_active=sum(r['active_batches'] for r in epoch_rows)
old_active=sum(r['active_batches'] for r in b['fixed_bank_activation_schedule'])
assert new_active == student['semantic_updates'] == summary['training']['semantic_updates']
assert old_active == b['semantic_updates_recorded']
epochs=dict(optimizer_steps=sum(r['batches'] for r in epoch_rows),new_eligible_steps=new_active,
    baseline_eligible_steps=old_active,new_active_fraction=new_active/127650,baseline_active_fraction=old_active/127650,
    first=epoch_rows[0],last=epoch_rows[-1],
    active_fraction_distribution=distribution(r['active_fraction'] for r in epoch_rows),
    kl_upward_transitions=sum(epoch_rows[i]['kl']>epoch_rows[i-1]['kl'] for i in range(1,50)),
    active_kl_upward_transitions=sum(epoch_rows[i]['active_kl']>epoch_rows[i-1]['active_kl'] for i in range(1,50)),
    last10_mean={k:st.mean(r[k] for r in epoch_rows[-10:]) for k in ['ce','entropy','kl','active_ce','active_entropy','active_kl']},
    max_entropy_identity_error=max(abs(r['ce']-r['entropy']-r['kl']) for r in epoch_rows),
    per_channel_target_mass_range={k:distribution(r[f'target_mass_{k}'] for r in epoch_rows) for k in CLASSES})
for k in CLASSES:
    aggregates[k]['baseline_signed_centered_mass'] = aggregates[k]['baseline_channel_target_mass'] - bank['weight_sum']/4
    aggregates[k]['new_epoch50_signed_centered_mass_before_activation'] = epoch_rows[-1][f'target_mass_{k}'] - sum(
        epoch_rows[-1][f'target_mass_{c}'] for c in CLASSES)/4
strata={}
for reason in sorted({r['geometry'] for r in per_class}):
    rows=[r for r in per_class if r['geometry']==reason]
    strata[reason]=dict(patients=len({r['patient_id'] for r in rows}),
        baseline_hard=[sum(r['baseline_hard'] for r in rows if r['class_name']==k) for k in CLASSES],
        new_hard=[sum(r['new_hard'] for r in rows if r['class_name']==k) for k in CLASSES],
        new_raw_votes=[sum(r['new_raw_vote_mass'] for r in rows if r['class_name']==k) for k in CLASSES])
assert strata['NO_CODED_WORLD_GEOMETRY']['new_raw_votes'][1] == 0
checks=dict(status='VERIFIED_TEXT_ARITHMETIC_DERIVED_FROM_REPORTED_MEASUREMENTS',source_sha=SHA,
    source_manifest_files_verified=len(manifest),source_hashes=source_checks,baseline_student_sha256=digest(base_source),
    baseline_unchanged_symbols=unchanged,launcher_patch_sha256=patch_hash,
    bank=bank,classes=aggregates,geometry_strata=strata,baseline_sampled_batches=batch_summary,epochs=epochs,
    limitations=['No original arrays, checkpoints, masks or model forwards used',
                 'Published hashes verify copied bytes, not sanitized original artifacts',
                 'New per-patient soft-bank counts and native/control diagnostics absent',
                 'Eligibility counters do not measure actual parameter updates',
                 'No TEST scoring or threshold/checkpoint selection'])
(OUT/'LEARNING_MECHANISM_CHECKS.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(dict(status=checks['status'],manifest_files_verified=len(manifest),
    patient_class_rows=len(per_class),sampled_batch_rows=len(batch_rows),epoch_rows=len(epoch_rows),
    old_active=old_active,new_active=new_active),indent=2))
