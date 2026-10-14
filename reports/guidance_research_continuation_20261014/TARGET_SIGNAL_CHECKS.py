"""Small deterministic algebra checks; no project imports, data, GT, or training."""
import hashlib
import json
import math
from pathlib import Path
import random
import struct

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE.parent / 'stable_guidance_world_v2_20261014'


def close(a, b, tol=1e-12):
    assert abs(a-b) <= tol, (a, b, tol)


def matmul(a, b):
    return [[sum(x*y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def softmax(x):
    v = [math.exp(t-max(x)) for t in x]
    return [t/sum(v) for t in v]


def ce(q, p):
    return -sum(x*math.log(y) for x, y in zip(q, p))


def main():
    checks = {}
    alpha, u = .35, [.25]*4
    p = [[.02, .97, .01]]*3  # General family, not the run's spatial graph.
    k = [[alpha*(i == j)+(1-alpha)*p[i][j] for j in range(3)] for i in range(3)]
    assert matmul(p, p) == p
    s = [0., .8, .6]
    v = [u, [.01, .97, .01, .01], [.01, .01, .97, .01]]
    r = [[(1-t)*a+t*b for a, b in zip(u, row)] for t, row in zip(s, v)]
    mass = matmul(k, [[t*x for x in row] for t, row in zip(s, v)])
    h = [sum(x*y for x, y in zip(row, s)) for row in k]
    embedded_rhs = [[t*x for x in row]+[1-t] for t, row in zip(s, v)]
    embedded_solution = matmul(k, embedded_rhs)
    q = matmul(k, r)
    z = [[x/t for x in row] for t, row in zip(h, mass)]
    for i in range(3):
        close(sum(k[i]), 1); close(sum(mass[i]), h[i]); close(sum(z[i]), 1)
        close(sum(embedded_rhs[i]), 1); close(embedded_solution[i][-1], 1-h[i])
        for c in range(4):
            close(embedded_solution[i][c], mass[i][c])
            close(q[i][c], (1-h[i])*.25+h[i]*z[i][c])
            for d in range(4):
                close(q[i][c]-q[i][d], h[i]*(z[i][c]-z[i][d]))
    checks['mixture_simplex_ordering'] = {'P': p, 'K': k, 's': s, 'h': h,
        'M': mass, 'Q': q, 'Z': z, 'five_channel_embedding_checked': True,
        'scope': 'general_family_synthetic_only'}

    # Fixed original weights and denominator, both active and inactive, with floor.
    logits = [.2, -.7, .8, .1]
    pred = softmax(logits)
    close(ce(q[0], pred), (1-h[0])*ce(u, pred)+h[0]*ce(z[0], pred))
    gradient_errors = []
    for active in (0, 1):
        for weight in (0., 1e-13, .4):
            scale = active*weight/max(weight, 1e-8)
            difference = scale*(ce(z[0], pred)-ce(q[0], pred))
            close(difference, scale*(1-h[0])*(ce(z[0], pred)-ce(u, pred)))
            for c in range(4):
                def delta_at(offset):
                    x = logits.copy(); x[c] += offset
                    pr = softmax(x)
                    return scale*(ce(z[0], pr)-ce(q[0], pr))
                numerical = (delta_at(1e-5)-delta_at(-1e-5))/2e-5
                exact = scale*(q[0][c]-z[0][c])
                gradient_errors.append(abs(numerical-exact))
                close(numerical, exact, 1e-10)
    # A rounded, non-unit target requires the row-mass term in the derivative.
    rounded = [x*1.000001 for x in z[0]]
    for c in range(4):
        def f(offset):
            x = logits.copy(); x[c] += offset
            return ce(rounded, softmax(x))-ce(q[0], softmax(x))
        exact = (sum(rounded)-sum(q[0]))*pred[c]+q[0][c]-rounded[c]
        close((f(1e-5)-f(-1e-5))/2e-5, exact, 1e-10)
    checks['loss_gradient'] = {'max_finite_difference_error': max(gradient_errors),
        'active_inactive_zero_weight_and_floor_checked': True, 'nonunit_row_mass_checked': True}

    # Common valid-h mask in BOTH arms, retaining full original denominator.
    weights, valid = [.2, .4, .1], [True, False, True]
    denominator = max(sum(weights), 1e-8)
    full_q = sum(w*ce(row, pred) for w, row in zip(weights, q))/denominator
    masked_q = sum(w*ce(row, pred) for w, row, ok in zip(weights, q, valid) if ok)/denominator
    masked_z = sum(w*ce(row, pred) for w, row, ok in zip(weights, z, valid) if ok)/denominator
    delta = sum(weights[i]*(1-h[i])*(ce(z[i], pred)-ce(u, pred))
                for i in range(3) if valid[i])/denominator
    close(masked_z-masked_q, delta)
    assert full_q > masked_q and sum(w for w, ok in zip(weights, valid) if ok) < denominator
    checks['common_mask_original_denominator'] = {'original_full_Q_loss': full_q,
        'paired_masked_Q_loss': masked_q, 'paired_masked_Z_loss': masked_z,
        'full_original_denominator': denominator, 'excluded_weight_mass': .4,
        'excluded_weight_fraction': .4/denominator, 'original_activation_held_fixed': True,
        'masked_Q_is_not_full_baseline': True}

    # Same soft evidence and operator, two possible semantic worlds.
    wrong_v = [.01, .01, .97, .01]
    exposure = .1
    wrong_q = [(1-exposure)*.25+exposure*x for x in wrong_v]
    reliability = exposure*(1-ce(wrong_q, wrong_q)/math.log(4))
    assert reliability > 1e-8 and max(wrong_q) < .7
    grad = [a-b for a, b in zip(wrong_q, wrong_v)]
    close(grad[2], -.648)
    assert max(wrong_v) > .7 and sorted(wrong_v)[-1]-sorted(wrong_v)[-2] > .2
    correct_v = [.01, .97, .01, .01]
    good_q = [(1-exposure)*.25+exposure*x for x in correct_v]
    assert max(good_q) < .7 < max(correct_v)
    checks['wrong_name_and_positive_fixture'] = {'P': [[1.]], 'alpha': alpha,
        'mass': -math.log1p(-exposure), 'h': exposure, 'V_wrong': wrong_v,
        'Q_wrong': wrong_q, 'Z_wrong': wrong_v, 'truth': 'RV',
        'original_reliability': reliability, 'original_permission_assumed': True,
        'original_batch_active': True, 'p_equals_Q_gradient_Q': [0.]*4,
        'gradient_Z': grad, 'correct_world_V': correct_v,
        'scope': 'synthetic_general_family_not_measured_anatomy'}

    # Route-weighted corruption: positive margin despite one wrong source above.
    good_fraction = k[0][1]*s[1]/h[0]
    lower = good_fraction*.97
    close(z[0][1], lower + k[0][2]*s[2]/h[0]*.01)
    assert z[0][1] >= lower > .7+.02
    assert 2*lower-1-2*.02 > .2
    checks['conditional_margin'] = {'bad_route_fraction': 1-good_fraction,
        'good_source_error': .03, 'lower_Z_true': lower,
        'student_same_grid_sup_error_budget': .02, 'neutral_node_Q': q[0],
        'neutral_node_Z': z[0], 'source_zero_is_truly_neutral': s[0] == 0}

    # Missing source strength is not identifiable from the mixed R0.
    ambiguous = []
    for strength, votes in [(.2, [1., 0., 0., 0.]), (.6, [.5, 1/6, 1/6, 1/6])]:
        ambiguous.append([(1-strength)*.25+strength*x for x in votes])
    for a, b in zip(*ambiguous): close(a, b)
    f32 = lambda x: struct.unpack('f', struct.pack('f', x))[0]
    tiny = 1e-9
    rounded_q = [f32((1-tiny)*.25+tiny*x) for x in wrong_v]
    assert rounded_q == u
    conflict = [0., .5, .5, 0.]
    cq = [(1-.4)*.25+.4*x for x in conflict]
    close(cq[1], cq[2]); close(conflict[1], conflict[2])
    checks['nonidentifiability_zero_tiny_and_conflict'] = {'same_R0': ambiguous,
        'distinct_strengths': [.2, .6], 'tiny_h': tiny, 'float32_Q_loses_contrast': rounded_q,
        'zero_h_M': [0.]*4, 'zero_h_Z': 'undefined_keep_Q', 'conflicting_Z': conflict}
    rng = random.Random(20261014)
    for _ in range(1000):
        hh = rng.uniform(.01, 1); eps_h, eps_m = hh/10, hh/20
        mm = hh*rng.random()
        dh, dm = rng.uniform(-eps_h, eps_h), rng.uniform(-eps_m, eps_m)
        error = abs((mm+dm)/(hh+dh)-mm/hh)
        assert error <= (eps_m+eps_h)/(hh-eps_h)
    checks['ratio_error_bound'] = {'trials': 1000, 'passed': True}
    naive_ess = 1/sum(x*x for x in [.1]*10)
    grouped_ess = 1/sum(x*x for x in [1.])
    close(naive_ess, 10); close(grouped_ess, 1)
    checks['correlated_vote_ess'] = {'ten_equal_duplicate_event_naive_ess': naive_ess,
        'one_fully_correlated_group_ess': grouped_ess, 'not_independent_patient_count': True}

    source = EVIDENCE/'executed_source/src/self_audit_guidance'
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    provenance = {}
    for manifest in ['run_evidence/run/RECIPE.json', 'run_evidence/run/student/FROZEN.json']:
        expected = json.loads((EVIDENCE/manifest).read_text())['source_sha256']
        provenance[manifest] = {name: sha(source/name) == digest for name, digest in expected.items()}
        assert all(provenance[manifest].values())
    summary = json.loads((EVIDENCE/'summary.json').read_text())
    patch_sha = sha(EVIDENCE/'run_evidence/launcher_integration.patch')
    assert patch_sha == summary['local_launcher_patch_sha256']
    result = {'status': 'PASS_SYNTHETIC_AND_SOURCE_IDENTITY_ONLY',
        'snapshot_sha': 'b937e3b746608725e43d3bc3cef8861666d152dc',
        'source_hash_matches': provenance, 'launcher_patch_sha256': patch_sha,
        'checks': checks, 'live_Q_reproduction': 'BLOCKED_NOT_RUN',
        'limits': ['No real tensor or gradient rerun', 'No anatomy correctness evidence',
                  'No production imports or edits, dataset/weights/GT access, training, GPU, or TEST scoring']}
    dest = HERE/'TARGET_SIGNAL_CHECKS.json'
    dest.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'status': result['status'], 'check_groups': len(checks), 'receipt': str(dest)}))


if __name__ == '__main__':
    main()
