"""Tiny mathematical/source checks only: no data, fitting, training, or tuning."""
from fractions import Fraction as F
from itertools import combinations, product
from pathlib import Path
import importlib.util
import json
import sys
import numpy as np
import torch

ROOT = Path('/tmp/self-audit-scientific-20261012-d5a45f3')
OUT = Path(__file__).parent

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result

core = module('theory_core', 'src/self_audit_guidance/core.py')
cqa = module('theory_cqa', 'src/self_audit_pseudolabel/cqa_v3.py')
alpha, beta = F(7, 20), F(13, 20)
kappa = alpha * (1 + beta / 7 + beta**2 / 7)
upper = 1 - F(3, 4) * kappa
assert upper == F(22313, 32000) and upper < F(7, 10)
graph_count = node_count = 0
maximum_neutral = 0.0
minimum_diagonal = 1.0
for n in range(1, 6):
    edges = list(combinations(range(n), 2))
    for bits in product((0, 1), repeat=len(edges)):
        W = np.eye(n)
        for (i, j), b in zip(edges, bits):
            W[i, j] = W[j, i] = b
        P = W / W.sum(1, keepdims=True)
        K = float(alpha) * np.linalg.inv(np.eye(n) - float(beta) * P)
        minimum_diagonal = min(minimum_diagonal, float(K.diagonal().min()))
        # Worst possible class probability at each neutral node: all other
        # restarts are pure in the same class. This dominates every simplex R0.
        maximum_neutral = max(maximum_neutral, float((1 - .75 * K.diagonal()).max()))
        assert (K.diagonal() >= float(kappa) - 1e-12).all()
        graph_count += 1
        node_count += n

# Distance bound is sharp in the unrestricted directed family, not claimed
# sharp for the repository's symmetric weighted graph family.
P = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=float)
R = np.full((3, 4), .25); R[2] = [0, 0, 0, 1]
Q, receipt = core.anchored_solve(P, R, tolerance=1e-10)
expected = [.25 + .75 * .65**2, .25 + .75 * .65, 1]
assert np.allclose(Q[:, 3], expected, atol=1e-10)
assert np.max(np.abs(Q - float(alpha) * np.linalg.solve(np.eye(3)-float(beta)*P, R))) <= receipt['error_bound'] + 1e-15
assert np.allclose(core.anchored_solve(np.eye(2), np.full((2, 4), .25))[0], .25)
assert np.allclose(core.anchored_solve(P, R, alpha=1)[0], R)

# Exercise six non-self neighbors using the actual default graph constructor:
# three frames, unique static one-hot location features, genuine mutual matches.
image = np.zeros((3, 1, 3, 3), dtype=np.float32)
features = np.broadcast_to(np.eye(9, dtype=np.float32).reshape(1, 1, 3, 3, 9), (3, 1, 3, 3, 9)).copy()
actual_P, graph_metrics = core.build_graph(image, features, core.GuidanceConfig())
center = 13
assert actual_P.getrow(center).nnz - 1 == 6
actual_R = np.zeros((27, 4)); actual_R[:, 3] = 1; actual_R[center] = .25
actual_Q, actual_receipt = core.anchored_solve(actual_P, actual_R, tolerance=1e-10)
assert float(actual_Q[center].max()) < float(upper)
clique_P = np.ones((7, 7))/7
clique_K = float(alpha)*np.linalg.inv(np.eye(7)-float(beta)*clique_P)
assert np.all(clique_K.diagonal() >= float(kappa))

# Numerical arithmetic at the configured thresholds, not a parameter sweep.
p = .7
h_lower = (p - .25) / .75
entropy_upper = -p*np.log(p) - (1-p)*np.log((1-p)/3)
reliability_lower = h_lower*(1-entropy_upper/np.log(4))
assert reliability_lower > .05 and 2*p - 1 > .2

# Fixed-relation uniform logits have exactly zero coassignment gradient.
logits = torch.zeros((1, 4, 2, 3), requires_grad=True)
ids = torch.tensor([[[1, 2, 3], [1, 2, 3]]], dtype=torch.long)
relations = torch.tensor([[0, 1, 2, 1], [0, 2, 3, 0]], dtype=torch.long)
loss = cqa.relation_distillation_loss(logits.softmax(1), ids, relations)
gradient, = torch.autograd.grad(loss, logits)
assert float(gradient.abs().max()) == 0.0 and float(loss.detach()) > 0

# A fixed two-fold scalar counterexample: each original region's fold means
# are +e/-e in opposite order; pooling improves both fold risks by 3e^2.
ids_np = np.array([[1, 1, 2, 2], [1, 1, 2, 2]])
parity = (np.indices(ids_np.shape).sum(0) % 2).ravel()
atoms = {a: np.flatnonzero(ids_np.ravel() == a) for a in (1, 2)}
e = .01
v = np.where(parity == 0, e, -e)
v[atoms[2]] *= -1
images = np.stack([v.reshape(ids_np.shape)]*2)
delta = cqa._risk_delta(images, atoms, {1, 2}, parity, 2)
assert np.allclose(delta, -3*e**2)

# Finite coloring counterexamples and a feasible control.
def colorable(n, k, positives, negatives):
    return any(all(c[a] == c[b] for a,b in positives)
               and all(c[a] != c[b] for a,b in negatives)
               for c in product(range(k), repeat=n))
assert not colorable(3, 3, [(0,1),(1,2)], [(0,2)])
assert not colorable(3, 2, [], [(0,1),(1,2),(0,2)])
assert colorable(3, 2, [(0,1)], [(1,2)])

result = dict(
    scope='MATHEMATICAL_SANITY_CHECKS_NOT_SEGMENTATION_EVIDENCE',
    source_sha='d5a45f31481f0a702a1f9d5d418889fe704e9105',
    graph_count=graph_count, neutral_node_cases=node_count,
    exact_kappa=str(kappa), exact_neutral_upper=str(upper),
    neutral_upper=float(upper), enumerated_maximum_neutral=maximum_neutral,
    enumerated_minimum_resolvent_diagonal=minimum_diagonal,
    distance_chain_q=Q[:,3].tolist(), solver_receipt=receipt,
    actual_graph_metrics=graph_metrics, actual_graph_center_neighbors=6,
    actual_graph_neutral_max_probability=float(actual_Q[center].max()),
    actual_graph_solver_receipt=actual_receipt,
    seven_node_clique_neutral_max=float((1-.75*clique_K.diagonal()).max()),
    reliability_lower_at_probability_070=float(reliability_lower),
    fixed_relation_uniform_loss=float(loss.detach()),
    fixed_relation_uniform_max_gradient=float(gradient.abs().max()),
    two_fold_merge_risk_deltas=delta.tolist(),
    controls=['uniform', 'disconnected', 'alpha=1', 'directed chain',
              'conflicting relations', 'insufficient prototypes', 'feasible relations'],
    neural_training=False, datasets_read=False,
)
(OUT/'MATH_CHECKS.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
