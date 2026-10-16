#!/usr/bin/env python3
"""Read published JSON/source only; no arrays, model, patient images, GT or network."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
E = ROOT / 'reports/stable_guidance_main513ddda_b12_20261016'
R = E / 'completed_run/run'
H = ROOT / 'reports/stable_guidance_world_v2_20261014/run_evidence/run/bank/FROZEN.json'
inputs = {}

def read(path):
    raw = path.read_bytes()
    inputs[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)

bank = read(R / 'bank/FROZEN.json')
report = read(R / 'bank/REPORT.json')
historical = read(H)
evaluation = read(R / 'evaluation_val_verified_grid/METRICS.json')
recovery = read(E / 'completed_run/evaluation_recovery/RECOVERY_STATUS.json')
counts = [sum(r['class_pixels'][c] for r in report['patient_reports'].values()) for c in range(4)]
nodes = sum(r['graph']['nodes'] for r in report['patient_reports'].values())
orientation = {'train': dict(Counter(r['patient_left_axis'] for r in bank['image_records'].values()))}
for split in ('val', 'test'):
    manifest = read(R / f'predictions_{split}/FROZEN.json')
    orientation[split] = dict(Counter(r['patient_left_axis'] for r in manifest['images'].values()))

header_strata = {}
for pid, current in bank['image_records'].items():
    old = historical['image_records'][pid]
    assert current['image_sha256'] == old['image_sha256']
    assert current['affine'] == old['affine']
    assert current['shape_xyzt'] == old['shape_xyzt']
    assert all(current['affine'][i][j] == 0 for i in range(3) for j in range(3) if i != j)
    geometry = old['world_left_geometry']
    assert geometry['image_sha256'] == current['image_sha256']
    key = f"sform{geometry['sform_code']}_qform{geometry['qform_code']}"
    group = header_strata.setdefault(key, {'patients': [], 'current_hard_class_counts': [0]*4})
    group['patients'].append(pid)
    for c, count in enumerate(report['patient_reports'][pid]['class_pixels']):
        group['current_hard_class_counts'][c] += count

def sampled_native_sites(n, output=48):
    sites = set()
    for i in range(output):
        x = (i + .5) * n/output - .5
        lower = math.floor(x)
        sites.add(max(0, min(n-1, lower)))
        if x-lower > 1e-12:
            sites.add(max(0, min(n-1, lower+1)))
    return len(sites)

sampling = {}
for pid, r in bank['image_records'].items():
    w, h = r['shape_xyzt'][:2]
    touched = sampled_native_sites(w) * sampled_native_sites(h)
    sampling[pid] = {'native_hw': [h,w], 'directly_sampled_native_logit_sites': touched,
                     'native_sites': h*w, 'fraction': touched/(h*w)}

def softmax(x):
    z = [math.exp(v-max(x)) for v in x]
    return [v/sum(z) for v in z]

# Same coarse logits (averaged by a two-pixel bilinear query), different native gates.
base = [2.0, 0.0, 0.0, 0.0]
native_a = [4.0, 0.0, 0.0, 0.0]
native_b = [0.0, 0.0, 0.0, 0.0]
assert [(a+b)/2 for a,b in zip(native_a,native_b)] == base
coarse_probability = softmax(base)[0]
assert coarse_probability >= .7
assert softmax(native_a)[0] >= .7 and softmax(native_b)[0] < .7

# Cardinal side predicate and exact world-left covector can disagree on an oblique grid.
theta = math.radians(20)
a00, a01 = -math.cos(theta), math.sin(theta)
delta_yx = [10., 1.]
world_left_delta = -a01*delta_yx[0] - a00*delta_yx[1]
cardinal_delta = delta_yx[1]
assert abs(a00) >= .9 and cardinal_delta > 0 and world_left_delta < 0

rv = {key: sum(row['metrics']['classes']['RV'][key] for row in evaluation['frames'])
      for key in ('tp','fp','fn','predicted_pixels','reference_pixels')}
assert rv['tp'] + rv['fp'] == rv['predicted_pixels']
assert evaluation['verified_frame_geometry_count'] == 14
assert not any(recovery[k] for k in ('predictions_changed','checkpoint_changed','threshold_changed'))
for name in ('core','io','teacher','student','pipeline','evaluate'):
    p = E / f'completed_run/source/src/self_audit_guidance/{name}.py'
    inputs[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()

out = {
    'status': 'PASS_METADATA_AND_PURE_CPU_MATH_ONLY',
    'scope': 'Published JSON/source; no real arrays or model execution; historical headers joined by recorded raw-image SHA only',
    'source_commit': '513ddda8cec3a166f13cfe68fe6379cab014a1a7',
    'class_order': ['BG','RV','MYO','LV'], 'class_patients': report['class_patients'],
    'bank_hard_counts': counts, 'bank_nodes': nodes, 'hard_accepted_fraction': sum(counts)/nodes,
    'orientation': orientation, 'header_strata_sha_matched_80': header_strata,
    'current_train_affines_axis_aligned': True,
    'anatomy_sums': {k: sum(r['anatomy'][k] for r in report['patient_reports'].values())
                     for k in next(iter(report['patient_reports'].values()))['anatomy']},
    'rv_val_saved_counts': rv,
    'rv_frames_with_prediction': sum(r['metrics']['classes']['RV']['predicted_pixels']>0 for r in evaluation['frames']),
    'rv_frames_with_tp': sum(r['metrics']['classes']['RV']['tp']>0 for r in evaluation['frames']),
    'verified_frame_geometry_count': evaluation['verified_frame_geometry_count'],
    'native_semantic_interpolation_support': sampling,
    'native_semantic_interpolation_fraction_summary': {
        'min': min(x['fraction'] for x in sampling.values()),
        'max': max(x['fraction'] for x in sampling.values()),
        'mean_patient': sum(x['fraction'] for x in sampling.values())/len(sampling)},
    'synthetic_native_gate_counterexample': {'coarse_probability': coarse_probability,
        'native_probabilities': [softmax(native_a)[0],softmax(native_b)[0]],
        'coarse_loss_identical': True, 'claim': 'Native gate is not determined by interpolated-logit loss; no real-model effect measured'},
    'synthetic_oblique_cardinal_counterexample': {'degrees':20,'old_rule_accepts':True,
        'centroid_delta_yx':delta_yx,'cardinal_left_delta':cardinal_delta,'exact_world_left_delta':world_left_delta,
        'claim':'Possible cardinal side error; current metadata has no such oblique affine'},
    'input_sha256': inputs,
}
target = ROOT / 'reports/convergence_council_20261016/checks/geometry.json'
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(out, indent=2, sort_keys=True, allow_nan=False)+'\n')
print(json.dumps({k:out[k] for k in ('status','bank_hard_counts','orientation','rv_val_saved_counts','native_semantic_interpolation_fraction_summary')}, indent=2))
