"""CPU-only arithmetic checks; no project imports, data, models, or training.

Run from any directory with Python 3; writes only the same-prefix JSON receipt.
Numerical examples supplement, and do not replace, the report's proofs.
"""
import hashlib
import json
import math
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE.parent / "stable_guidance_world_v2_20261014"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def simplex(rng, k=4):
    xs = [rng.expovariate(1) for _ in range(k)]
    return [x / sum(xs) for x in xs]


def kl(q, p):
    return sum(x * math.log(x / y) for x, y in zip(q, p) if x)


def binary_kl(a, b):
    return kl([a, 1-a], [b, 1-b])


def softmax(z):
    e = [math.exp(x-max(z)) for x in z]
    return [x/sum(e) for x in e]


def restart(p, r, alpha=.35):
    q = [row[:] for row in r]
    for _ in range(10000):
        n = [[alpha*r[i][c]+(1-alpha)*sum(p[i][j]*q[j][c] for j in range(len(p)))
              for c in range(len(r[0]))] for i in range(len(p))]
        if max(abs(n[i][c]-q[i][c]) for i in range(len(p)) for c in range(len(r[0]))) < 1e-14:
            return n
        q = n
    raise AssertionError("contraction did not converge")


def main():
    rng = random.Random(20261014)
    checks = {}
    tau, delta = .7, .1
    witnessed = 0
    for _ in range(20000):
        q, p = simplex(rng), simplex(rng)
        if max(q) <= tau-delta and max(p) >= tau:
            witnessed += 1
            assert kl(q, p) + 1e-12 >= binary_kl(tau-delta, tau) >= 2*delta**2-1e-12
    checks['common_grid_binary_kl'] = {'trials': 20000, 'eligible_pairs': witnessed, 'kappa': binary_kl(.6, .7), 'pinsker': .02}
    # Zero weighted KL leaves any amount of zero-weight population unconstrained.
    checks['zero_weight_counterexample'] = {'weighted_kl': 0, 'zero_weight_fraction': .99, 'possible_unweighted_wrong_fraction': .99}
    # Four native logit vectors average to log(q); all native pixels can be confident.
    q = [.4, .3, .2, .1]
    logits = [[math.log(q[c])+4*(4*(c == i)-1) for c in range(4)] for i in range(4)]
    coarse = softmax([sum(row[c] for row in logits)/4 for c in range(4)])
    native_conf = [max(softmax(row)) for row in logits]
    assert abs(kl(q, coarse)) < 1e-14 and min(native_conf) > .999
    checks['grid_counterexample'] = {'q': q, 'coarse_p': coarse, 'coarse_kl': kl(q, coarse), 'native_confidences': native_conf}
    # A below-threshold initial node becomes accepted with imperfect soft anchors.
    p = [[1/3]*3 for _ in range(3)]
    r = [[.625,.125,.125,.125],[.985,.005,.005,.005],[.985,.005,.005,.005]]
    out = restart(p, r)
    h = .35*.5+.65*(.5+1+1)/3
    lower = (1-h)/4+h*(1-.015)
    assert r[0][0] < .7 and out[0][0] >= lower > .7+.02
    checks['conditional_positive_expansion'] = {'scope': 'weak_informative_anchor_crosses_hard_threshold_not_neutral_expansion', 'initial_probability': r[0][0], 'final_probability': out[0][0], 'h': h, 'eta': .015, 'lower_bound': lower, 'synthetic_only': True}
    # Genuine neutral-node expansion with one wrong source, on a general P.
    # This P is not a realizable unit-loop spatial-only repository graph.
    alpha = .35
    p = [[.02, .97, .01] for _ in range(3)]
    r = [[.25]*4, [.995, .005/3, .005/3, .005/3],
         [.005/3, .995, .005/3, .005/3]]
    truth = [0, 0, 0]  # Synthetic labels only: source 2 points to the wrong class.
    assert all(min(row) >= 0 and abs(sum(row)-1) < 1e-14 for row in p+r)
    pp = [[sum(p[i][k]*p[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    assert max(abs(pp[i][j]-p[i][j]) for i in range(3) for j in range(3)) < 1e-14
    kernel = [[alpha*(i == j)+(1-alpha)*p[i][j] for j in range(3)] for i in range(3)]
    kernel_iterative = restart(p, [[float(i == j) for j in range(3)] for i in range(3)])
    assert max(abs(kernel[i][j]-kernel_iterative[i][j]) for i in range(3) for j in range(3)) < 1e-12
    q_exact = [[sum(kernel[i][j]*r[j][c] for j in range(3)) for c in range(4)] for i in range(3)]
    q_iterative = restart(p, r)
    assert max(abs(q_exact[i][c]-q_iterative[i][c]) for i in range(3) for c in range(4)) < 1e-12
    assert max(r[0])-min(r[0]) == 0
    assert sum(max(range(4), key=lambda c: r[j][c]) != truth[j] for j in (1,2)) == 1
    h = kernel[0][1]+kernel[0][2]
    bad = kernel[0][2]/h
    eta = .005
    lower = (1-h)/4+h*(1-bad)*(1-eta)
    q0 = q_exact[0]
    entropy = -sum(x*math.log(x) for x in q0)
    reliability = h*(1-entropy/math.log(4))
    entropy_upper = -lower*math.log(lower)-(1-lower)*math.log((1-lower)/3)
    rel_lower = h*(1-entropy_upper/math.log(4))
    margin = sorted(q0)[-1]-sorted(q0)[-2]
    h_iterative = restart(p, [[0.,1.],[1.,0.],[1.,0.]])[0][0]
    assert abs(h-h_iterative) < 1e-12
    assert abs(lower-.7180975) < 1e-12 and abs(q0[0]-.7181083333333333) < 1e-12
    assert q0[0] >= lower > .7 and margin > .2 and reliability >= rel_lower > .05
    assert lower-.01-1e-6 > .7 and 2*(lower-.01-1e-6)-1 > .2
    accepted = [max(row) >= .7 and sorted(row)[-1]-sorted(row)[-2] >= .2 for row in q_exact]
    assert accepted == [True, True, False]
    assert all(not accepted[i] or max(range(4), key=lambda c: q_exact[i][c]) == truth[i] for i in range(3))
    spatial_self_min = 1/(1+4)
    spatial_neutral_ceiling = (alpha/4+(1-alpha)*(1-spatial_self_min))/(1-(1-alpha)*spatial_self_min)
    assert p[0][0] < spatial_self_min and spatial_neutral_ceiling < .7
    checks['neutral_expansion_with_one_wrong_source'] = {
        'scope': 'general_row_stochastic_restart_family_not_run_graph_workaround',
        'simplex_and_idempotence_verified': True, 'alpha': alpha, 'P': p, 'R': r,
        'synthetic_truth': truth, 'informative_source_count': 2, 'wrong_source_count': 1,
        'neutral_node': 0, 'K_row_0': kernel[0], 'h': h, 'bad_influence_fraction': bad,
        'eta': eta, 'lower_bound': lower, 'Q_row_0': q0, 'margin': margin,
        'entropy_nats': entropy, 'reliability': reliability, 'reliability_lower_bound': rel_lower,
        'probability_margin_reliability_gates': [True, True, True],
        'same_grid_student_deviation_budget': .01, 'teacher_solver_error_budget': 1e-6,
        'perfect_student_accepted_nodes': [i for i,a in enumerate(accepted) if a],
        'uniform_node_measure_newly_accepted_neutral_mass': 1/3,
        'perfect_student_accepted_error_mass': 0,
        'unit_loop_spatial_only_min_self_transition': spatial_self_min,
        'unit_loop_spatial_only_neutral_probability_ceiling': spatial_neutral_ceiling,
        'synthetic_only': True}
    # Noise in anchors and in graph has additive coordinate perturbation bound.
    for _ in range(1000):
        n = 4
        p = [simplex(rng) for _ in range(n)]
        p0 = [simplex(rng) for _ in range(n)]
        r = [simplex(rng) for _ in range(n)]
        r0 = [simplex(rng) for _ in range(n)]
        eps = max(sum(abs(x-y) for x,y in zip(a,b))/2 for a,b in zip(p,p0))
        eta = max(abs(r[i][c]-r0[i][c]) for i in range(n) for c in range(4))
        a,b = restart(p,r),restart(p0,r0)
        err = max(abs(a[i][c]-b[i][c]) for i in range(n) for c in range(4))
        assert err <= eta+.65/.35*eps+1e-11
    checks['perturbation_bound'] = {'trials': 1000, 'passed': True}
    c,beta,w0,d,D = .3,.002,.1,.02,2.4e-7
    e = D/(2*w0*d*d)
    checks['risk_expansion_example'] = {'candidate_mass':c,'semantic_bad_mass':beta,'deviation_mass_bound':e,'coverage_lower':c-beta-e,'risk_upper':(beta+e)/(c-beta-e),'synthetic_only': True}
    summary = json.loads((EVIDENCE/'summary.json').read_text())
    sem = summary['training']['last_epoch']['semantic_diagnostics']
    checks['reported_epoch_arithmetic'] = {'ce_minus_entropy_minus_kl':sem['ce']-sem['entropy']-sem['kl'], 'mean_active_batch_recorded_kl_including_any_denominator_floor':sem['kl']/sem['active_fraction'], 'normalized_kl_measure_interpretation': 'Equals this active-batch scalar mean only when no active denominator is floored', 'native_unknown_fraction_reported':summary['unknown_fraction']}
    assert abs(checks['reported_epoch_arithmetic']['ce_minus_entropy_minus_kl']) < 1e-7
    # Dividing by active fraction always recovers the mean active recorded scalar.
    # With a denominator floor, the total weighted measure M can be smaller.
    floor_q = [.1,.4,.3,.2]
    floor_kl = [kl(floor_q, [.1,.7,.1,.1]), kl(floor_q, [.4,.1,.2,.3]), 0.]
    floor_weights = [5e-9, 2e-8, 0.]
    floor_active = [1,1,0]
    scaled = [a*w/max(w,1e-8) for a,w in zip(floor_active, floor_weights)]
    recorded = [s*k for s,k in zip(scaled, floor_kl)]
    epoch = sum(recorded)/3
    active_fraction = sum(floor_active)/3
    active_mean = sum(recorded)/sum(floor_active)
    measure_mass = sum(scaled)/3
    assert abs(epoch/active_fraction-active_mean) < 1e-14
    assert measure_mass < active_fraction and abs(epoch/measure_mass-active_mean) > 1e-3
    checks['active_batch_mean_with_floor'] = {'active_weights': floor_weights[:2],
        'recorded_scalars': recorded, 'epoch_mean': epoch, 'active_fraction': active_fraction,
        'mean_active_recorded_scalar': active_mean, 'epoch_divided_by_active_fraction': epoch/active_fraction,
        'weighted_measure_mass': measure_mass, 'normalized_measure_kl': epoch/measure_mass,
        'identity_including_floor_verified': True, 'synthetic_only': True}
    source = EVIDENCE/'executed_source/src/self_audit_guidance'
    provenance = {}
    for manifest in ['run_evidence/run/RECIPE.json','run_evidence/run/student/FROZEN.json']:
        expected = json.loads((EVIDENCE/manifest).read_text())['source_sha256']
        matches = {name:sha(source/name) == value for name,value in expected.items()}
        assert all(matches.values())
        provenance[manifest] = matches
    patch = EVIDENCE/'run_evidence/launcher_integration.patch'
    assert sha(patch) == summary['local_launcher_patch_sha256']
    result = {'status':'PASS_SYNTHETIC_AND_ARTIFACT_ARITHMETIC_ONLY', 'seed':20261014,
              'snapshot_sha':'b937e3b746608725e43d3bc3cef8861666d152dc',
              'source_hash_matches':provenance,'launcher_patch_sha256':sha(patch),
              'checks':checks,'limitations':['Not a semantic or population certificate','No final-checkpoint fidelity rerun','No dataset, checkpoint, target-mask, TEST scoring, or training access']}
    dest = HERE/'THEORY_PRIOR_ART_CHECKS.json'
    dest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'check_groups':len(checks),'receipt':str(dest)}))


if __name__ == '__main__':
    main()
