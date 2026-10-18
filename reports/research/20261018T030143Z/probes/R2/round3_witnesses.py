"""Tiny manufactured counterexamples; no project imports, data, or fitting."""
import hashlib
import json
import math
import platform
from pathlib import Path


def observe(p):
    return [(p[0] + p[1]) / 2, (p[2] + p[3]) / 2]


def wrong_graph_residual(p):
    return [p[0] - p[1], p[2] - p[3]]


p_star = [.9, .3, .7, .1]
q = observe(p_star)
p_hat = [q[0], q[0], q[1], q[1]]
v_columns = [[1., 1., 0., 0.], [0., 0., 1., 1.]]
hv_columns = [observe(col) for col in v_columns]
determinant = hv_columns[0][0] * hv_columns[1][1] - hv_columns[1][0] * hv_columns[0][1]
assert determinant == 1.
residual = math.sqrt(sum((a - b) ** 2 for a, b in zip(observe(p_hat), q)))
graph_residual = math.sqrt(sum(a * a for a in wrong_graph_residual(p_hat)))
error = math.sqrt(sum((a - b) ** 2 for a, b in zip(p_hat, p_star)))
hard_errors = sum((a > .5) != (b > .5) for a, b in zip(p_hat, p_star))
assert residual == graph_residual == 0.
assert math.isclose(error, .6) and hard_errors == 2

# A fitted constant observation map ignores p even after its parameters freeze.
b = list(q)
def learned_observation(p):
    return list(b)

alternative = [.2, .8, .8, .2]
assert learned_observation(p_star) == learned_observation(alternative) == q
frozen_map_response = [a - c for a, c in zip(learned_observation(p_star), learned_observation(alternative))]
assert frozen_map_response == [0., 0.]

receipt = {
    'status': 'EXECUTED_MANUFACTURED_ONLY',
    'python': platform.python_version(),
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'wrong_graph': {
        'truth_foreground_probability': p_star,
        'observations': q,
        'unique_wrong_graph_solution': p_hat,
        'H_V_determinant': determinant,
        'teacher_residual': residual,
        'graph_residual': graph_residual,
        'foreground_channel_unnormalized_L2_error': error,
        'full_binary_vector_unnormalized_L2_error': math.sqrt(2) * error,
        'hard_error_count': hard_errors,
        'pixel_count': len(p_star),
    },
    'learned_observation_bypass': {
        'definition': 'F_b(p)=b; b=q; frozen b',
        'first_p': p_star,
        'second_p': alternative,
        'observation_difference': frozen_map_response,
        'derivative_wrt_p': 'zero everywhere',
    },
    'limits': 'Counterexamples only. No patient anatomy, real teacher model, or learned upsampler was evaluated.',
}
print(json.dumps(receipt, indent=2))
