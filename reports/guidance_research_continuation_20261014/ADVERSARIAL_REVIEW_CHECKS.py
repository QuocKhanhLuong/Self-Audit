"""Independent stdlib arithmetic review; no arrays, model, training, or GT access.

Read already-published VAL confusion counts as exposed development evidence.
Write only the owned ADVERSARIAL_REVIEW_CHECKS.json sibling.
"""
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct

OUT = Path(__file__).resolve().parent
E = OUT.parent / 'stable_guidance_world_v2_20261014'
READS = {}


def read(path):
    data = path.read_bytes()
    READS[str(path.relative_to(OUT.parent))] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def mean(xs):
    vals = [v for v in xs if v is not None]
    return statistics.mean(vals) if vals else None


def f32(x):
    return struct.unpack('f', struct.pack('f', x))[0]


manifest = read(E / 'MANIFEST.json')
for item in manifest['files']:
    data = (E / item['published_path']).read_bytes()
    assert hashlib.sha256(data).hexdigest() == item['published_sha256']
    assert len(data) == item['published_bytes']

base = read(E / 'baseline_forensics/train_all/DIAGNOSTICS.json')
bank = read(E / 'run_evidence/run/bank/FROZEN.json')
history = read(E / 'run_evidence/run/student/training.json')
raw_counts = [read(E / p)['patient_frame_counts'] for p in (
    'baseline_forensics/val_counts/DIAGNOSTICS.json',
    'run_evidence/post_training/val_counts/DIAGNOSTICS.json')]
classes = ['BG', 'RV', 'MYO', 'LV']
metric_runs = []
for rows in raw_counts:
    patients = sorted({r['patient_id'] for r in rows})
    assert len(rows) == 144 and len(patients) == 18
    values = {}
    sums = {}
    for c in classes:
        selected = [r for r in rows if r['class_name'] == c]
        sums[c] = {k: sum(r[k] for r in selected) for k in ('tp', 'fp', 'fn')}
        for p in patients:
            subset = [r for r in selected if r['patient_id'] == p]
            assert len(subset) == 2
            vals = {}
            for name in ('dice', 'recall', 'precision'):
                frame_vals = []
                for r in subset:
                    tp, fp, fn = r['tp'], r['fp'], r['fn']
                    num, den = {'dice': (2*tp, 2*tp+fp+fn),
                                'recall': (tp, tp+fn),
                                'precision': (tp, tp+fp)}[name]
                    frame_vals.append(num/den if den else None)
                vals[name] = mean(frame_vals)
            values[p, c] = vals
    class_macro = {c: {m: mean(values[p, c][m] for p in patients)
                      for m in ('dice', 'recall', 'precision')} for c in classes}
    patient_dice = {p: mean(values[p, c]['dice'] for c in classes[1:]) for p in patients}
    coverage = {}
    for p in patients:
        frames = sorted({r['frame_index'] for r in rows if r['patient_id'] == p})
        ratios = []
        for f in frames:
            selected = [r for r in rows if r['patient_id'] == p and r['frame_index'] == f]
            ratios.append(sum(r['tp']+r['fp'] for r in selected)/sum(r['tp']+r['fn'] for r in selected))
        coverage[p] = mean(ratios)
    metric_runs.append(dict(class_macro=class_macro, patient_dice=patient_dice,
                            coverage=mean(coverage.values()), totals=sums,
                            foreground_dice=mean(patient_dice.values()),
                            foreground_precision=mean(class_macro[c]['precision'] for c in classes[1:])))

old, new = metric_runs
deltas = {p: 100*(new['patient_dice'][p]-old['patient_dice'][p]) for p in old['patient_dice']}
precision_omission = mean(old['class_macro'][c]['precision'] for c in ('MYO', 'LV'))
assert new['totals']['RV'] == {'tp': 0, 'fp': 0, 'fn': 230281}
paired = dict(mean_pp=mean(deltas.values()), median_pp=statistics.median(deltas.values()),
              positive=sum(x>0 for x in deltas.values()), negative=sum(x<0 for x in deltas.values()),
              excluding_patient098_pp=mean(x for p, x in deltas.items() if p != 'patient098'),
              deltas_pp=deltas,
              precision_omit_baseline_rv_pp=100*(precision_omission-old['foreground_precision']),
              precision_remaining_change_pp=100*(new['foreground_precision']-precision_omission))

strata = {}
for pid, record in bank['image_records'].items():
    g = record['world_left_geometry']
    name = 'coded' if g['valid'] else 'uncoded'
    st = strata.setdefault(name, dict(patients=0, old=[0]*4, new=[0]*4,
                                     patient_count_identity=True))
    st['patients'] += 1
    a = [r['hard_accepted'] for r in base['patients'][pid]['classes']]
    b = bank['patient_reports'][pid]['class_pixels']
    st['old'] = [x+y for x, y in zip(st['old'], a)]
    st['new'] = [x+y for x, y in zip(st['new'], b)]
    st['patient_count_identity'] &= a == b
    if name == 'coded':
        assert (g['sform_code'], g['qform_code'], g['spatial_unit']) == (2, 0, 'unknown')
        assert record['affine'] == [[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]
        assert g['left_vector_yx'] == [0, record['shape_xyzt'][0]/48]
    else:
        assert g['sform_code'] == g['qform_code'] == 0
        assert g['left_vector_yx'] == [0, 0]
        assert b[1] == 0
assert strata['coded']['patients'] == 52 and strata['uncoded']['patients'] == 28
assert strata['coded']['patient_count_identity']

overlap_rows = []
for pid, item in base['patients'].items():
    assert len(item['batches']) == 1
    r = item['batches'][0]
    a, b = r['gates']['native']['accepted'], r['gates']['projected_control']['accepted']
    n = r['gates']['native']['pixels']
    frac = r['fidelity_not_accuracy']['native_projected_gated_disagreement']
    d = round(n*frac)
    # Check not just small residual: this integer's float32 ratio equals the
    # recorded float exactly, and adjacent integers cannot have the same ratio.
    assert n < 2**24 and f32(d/n) == frac
    assert all(f32(k/n) != frac for k in (d-1, d+1))
    lo = max(0, a+b-n, math.ceil((a+b-d)/2))
    hi = min(a, b, a+b-d)
    assert lo <= hi
    for i in (lo, hi):
        different_in_overlap = d-a-b+2*i
        assert 0 <= different_in_overlap <= i
        assert a+b-i <= n
    overlap_rows.append(dict(patient=pid, pixels=n, native=a, projected=b, disagreement=d,
                             rounding_residual=abs(n*frac-d), intersection_low=lo, intersection_high=hi))
overlap = {k: sum(r[k] for r in overlap_rows) for k in (
    'pixels', 'native', 'projected', 'disagreement', 'intersection_low', 'intersection_high')}
overlap.update(native_only_low=overlap['native']-overlap['intersection_high'],
               native_only_high=overlap['native']-overlap['intersection_low'],
               max_rounding_residual=max(r['rounding_residual'] for r in overlap_rows),
               all_80_float32_counts_unique=True, all_80_endpoints_feasible=True)
overlap['native_only_fractions'] = [overlap[k]/overlap['native'] for k in ('native_only_low','native_only_high')]
overlap['pooled_disagreement'] = overlap['disagreement']/overlap['pixels']
assert (overlap['native'],overlap['projected'],overlap['disagreement']) == (215428,121350,119330)

last = history[-1]['semantic_diagnostics']
epochs = dict(steps=sum(r['steps'] for r in history),
              eligible=sum(r['semantic_diagnostics']['active_batches'] for r in history),
              max_ce_identity_residual=max(abs(r['semantic_diagnostics']['ce']-r['semantic_diagnostics']['entropy']-r['semantic_diagnostics']['kl']) for r in history),
              last_active_batch_mean_kl=last['kl']/last['active_fraction'],
              last_entropy_fraction=last['entropy']/last['ce'],
              patient024=base['patients']['patient024']['batches'][0]['semantic'])

def entropy(q):
    return -sum(x*math.log(x) for x in q if x)


def softmax(z):
    es = [math.exp(x-max(z)) for x in z]
    return [x/sum(es) for x in es]


# Root single-pixel BG counterexample needs an active-batch qualifier.
qs = [[.4,.2,.2,.2], [.1,.2,.5,.2]]
ws = [1-entropy(q)/math.log(4) for q in qs]
active = any(max(range(4),key=q.__getitem__) != 0 and w>0 for q,w in zip(qs,ws))
assert active and not (max(range(4),key=qs[0].__getitem__) != 0)
denom = max(sum(ws), 1e-8)
grad = ws[0]/denom*(.25-qs[0][1])
def loss(z):
    return -sum(w*sum(t*math.log(p) for t,p in zip(q,softmax(zz)))
                for q,w,zz in zip(qs,ws,z))/denom
eps = 1e-5
plus, minus = [[0.]*4 for _ in qs], [[0.]*4 for _ in qs]
plus[0][1], minus[0][1] = eps, -eps
finite = (loss(plus)-loss(minus))/(2*eps)
assert grad>0 and abs(grad-finite)<1e-9
active_fixture = dict(targets=qs, weights=ws, active=active,
                      rv_mass=sum(w*q[1] for w,q in zip(ws,qs)),
                      first_pixel_rv_logit_gradient=grad, finite_difference=finite,
                      teacher_match_kl=0, teacher_match_accepted_pixels=0,
                      root_bg_only_single_pixel_active=False)
assert all(max(q)<.7 for q in qs)

# A route-weighted corruption witness: almost all counted anchors may be good,
# while a neutral target routes its nonrestart mass entirely to a bad hub.
alpha = .35
wrong = [alpha/4, alpha/4+1-alpha, alpha/4, alpha/4]
wrong_h = 1-alpha
wrong_reliability = wrong_h*(1-entropy(wrong)/math.log(4))
assert wrong[1]>.7 and wrong_reliability>.05
corruption = dict(correct_anchors=99, wrong_anchors=1,
                  target_initial_uniform=True, target_routes_to_wrong_anchor=True,
                  target_true_class=0, target_distribution=wrong,
                  propagated_strength=wrong_h, reliability=wrong_reliability,
                  route_weighted_bad_fraction=1, accepted_label=1,
                  scope='Stochastic-matrix theorem counterexample, not an executed repository graph')

kappa = .6*math.log(.6/.7)+.4*math.log(.4/.3)
theory = dict(kappa=kappa,
              conditional_active_kl_over_kappa=epochs['last_active_batch_mean_kl']/kappa,
              warning='Illustration only: online batch KL is not final/native divergence; even substituting it gives an upper bound capped at one.')
restart = .35
prow = [.02, .97, .01]
rs = [[.25]*4, [.995, .005/3, .005/3, .005/3],
      [.005/3, .995, .005/3, .005/3]]
kernel = [[restart*(i==j)+(1-restart)*prow[j] for j in range(3)] for i in range(3)]
sol = [[sum(kernel[i][j]*rs[j][c] for j in range(3)) for c in range(4)] for i in range(3)]
assert max(abs(sol[i][c]-(restart*rs[i][c]+(1-restart)*sum(prow[j]*sol[j][c] for j in range(3)))) for i in range(3) for c in range(4)) < 1e-14
h = kernel[0][1]+kernel[0][2]
lower = (1-h)/4+kernel[0][1]*.995
rel = h*(1-entropy(sol[0])/math.log(4))
margin = sol[0][0]-max(sol[0][1:])
assert sol[0][0]>.7 and margin>.2 and rel>.05
assert lower-.01-1e-6>.7
spatial_ceiling = (restart/4+(1-restart)*(1-.2))/(1-(1-restart)*.2)
assert spatial_ceiling<.7
theory['neutral_wrong_source_fixture'] = dict(q=sol, h=h, b=kernel[0][2]/h,
    lower=lower, reliability=rel, margin=margin, spatial_only_neutral_ceiling=spatial_ceiling,
    general_stochastic_family_only=True, actual_bank_graph_realizability='NOT_ESTABLISHED')
result = dict(status='PASS_INDEPENDENT_ARITHMETIC_AND_CONDITIONAL_COUNTEREXAMPLES_ONLY',
              source_sha='b937e3b746608725e43d3bc3cef8861666d152dc',
              manifest_files_verified=len(manifest['files']),
              source_measurement_status='Published historical measurements, not replayed',
              metric_runs=metric_runs, paired=paired, geometry=strata,
              overlap=overlap, overlap_rows=overlap_rows, epochs=epochs,
              active_soft_mass_counterexample=active_fixture,
              route_corruption_counterexample=corruption, theory=theory,
              inputs_sha256=READS,
              reviewed_drafts_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in OUT.iterdir() if p.suffix in ('.md','.json') and not p.name.startswith('ADVERSARIAL_REVIEW')},
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              no_training=True, no_model_execution=True, no_reference_arrays_read=True,
              official_test_scored=False, val_exposed_development=True)
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('status','manifest_files_verified','paired','overlap','active_soft_mass_counterexample','theory')},indent=2))
