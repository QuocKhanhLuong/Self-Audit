"""Four manufactured algebra controls; no neural training or patient evidence."""
import json
import math


def close(a, b):
    assert abs(a - b) < 1e-12, (a, b)


results = {}
# Repeated diagonal ridge filters equal one transformed-kernel ridge filter.
# Fixed manufactured penalties, not a reproduction of data-derived tolerances.
eigenvalues, penalties, target = (0.3, 1.0, 3.0), (0.2, 0.4, 0.1), (1.0, -0.5, 0.7)
rows = []
for d, z in zip(eigenvalues, target):
    repeated = z
    for c in penalties:
        repeated *= d / (d + c)
    shrink = math.prod(d / (d + c) for c in penalties)
    transformed = penalties[0] * shrink / (1 - shrink)
    one_pass = transformed / (transformed + penalties[0]) * z
    close(repeated, one_pass)
    rows.append({"eigenvalue": d, "combined_filter": shrink,
                 "repeated_prediction": repeated, "one_pass_prediction": one_pass})
results["spectral_filter_one_pass"] = rows

p0, proposal, alpha, rounds = 0.2, 0.8, 0.3, 4
p = p0
for _ in range(rounds):
    p = (1 - alpha) * p + alpha * proposal
mix = 1 - (1 - alpha) ** rounds
one_pass = (1 - mix) * p0 + mix * proposal
close(p, one_pass)
results["fixed_proposal_relaxation"] = {"rounds": rounds, "final_mix": mix,
    "iterated": p, "one_pass": one_pass}

theta_a, theta_b = (1.0, 1.0), (-1.0, -1.0)
output_average = (math.prod(theta_a) + math.prod(theta_b)) / 2
average_theta = tuple((a + b) / 2 for a, b in zip(theta_a, theta_b))
weight_average_output = math.prod(average_theta)
assert output_average == 1.0 and weight_average_output == 0.0
results["parameter_vs_output_average"] = {"output_average": output_average,
    "weight_average_output": weight_average_output, "model": "f(a,b)=a*b"}

p, target, step = 0.7, 0.3, 0.1
gradient_probability = p - target
proposal = p - step * gradient_probability
jacobian = p * (1 - p)  # scalar sigmoid parameterization
direct_parameter_gradient = gradient_probability * jacobian
transfer_parameter_gradient = (p - proposal) * jacobian / step
close(direct_parameter_gradient, transfer_parameter_gradient)
results["direct_energy_squared_probability_transfer"] = {
    "direct_parameter_gradient": direct_parameter_gradient,
    "scaled_transfer_parameter_gradient": transfer_parameter_gradient,
    "same_point_only": True}

print(json.dumps({"status": "MANUFACTURED_ALGEBRA_ONLY", "passed_checks": len(results),
                  "checks": results}, indent=2, sort_keys=True, allow_nan=False))
