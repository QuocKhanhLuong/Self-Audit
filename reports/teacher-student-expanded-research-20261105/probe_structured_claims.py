"""Exact/small synthetic counterexamples; no MRI, model or patient evaluation."""
import json
from fractions import Fraction as F
from itertools import combinations, product
from pathlib import Path
import numpy as np

def pair_prob(law, i, j):
    return sum(p for partition, p in law if partition[i] == partition[j])

law_a = [((0, 0, 0), F(1, 2)), ((0, 1, 2), F(1, 2))]
law_b = [((0, 0, 0), F(1, 4)), ((0, 0, 1), F(1, 4)),
         ((0, 1, 0), F(1, 4)), ((0, 1, 1), F(1, 4))]
pairs_a = [pair_prob(law_a, *pair) for pair in combinations(range(3), 2)]
pairs_b = [pair_prob(law_b, *pair) for pair in combinations(range(3), 2)]
triple_a = sum(p for z, p in law_a if len(set(z)) == 1)
triple_b = sum(p for z, p in law_b if len(set(z)) == 1)
assert pairs_a == pairs_b and triple_a != triple_b

def semantic_posterior(law):
    """Four named labels, same image likelihood at sites0/1, query site2."""
    numerator = denominator = F(0)
    for partition, mass in law:
        k = len(set(partition))
        for region_labels in product(range(4), repeat=k):
            labels = [region_labels[z] for z in partition]
            likelihood = 4**((labels[0] == 0) + (labels[1] == 0))
            weight = mass * F(likelihood, 4**k)
            denominator += weight
            if labels[2] == 0:
                numerator += weight
    return numerator / denominator

post_a, post_b = semantic_posterior(law_a), semantic_posterior(law_b)
assert (post_a, post_b) == (F(61, 100), F(139, 250))

# A fixed coarse partition destroys a one-pixel MYO-like separator.
# Here binary foreground is only a toy separator, not real anatomy.
y = np.array([0, 0, 1, 0, 0, 0])
cells = [np.array([0, 1, 2]), np.array([3, 4, 5])]
best_correct = sum(np.bincount(y[cell], minlength=2).max() for cell in cells)
best_dice = max(2 * np.sum((np.repeat([a, b], 3) == 1) & (y == 1)) /
                (np.sum(np.repeat([a, b], 3) == 1) + np.sum(y == 1))
                for a in [0, 1] for b in [0, 1])
assert best_correct == 5 and best_dice == .5

# Reducing a Markov operator generally does not preserve its resolvent.
# Uniform measure is stationary here; C merges states0/1 but they have
# different escape probabilities to state2, so the partition is not lumpable.
p = np.array([[.9, .1, 0], [.1, .1, .8], [0, .8, .2]])
c = np.array([[1., 0], [1., 0], [0., 1]])
mu = np.eye(3) / 3
r = np.linalg.solve(c.T @ mu @ c, c.T @ mu)
pk = r @ p @ c
alpha = .8
projected_response = r @ np.linalg.solve(np.eye(3)-alpha*p, c)
reduced_response = np.linalg.inv(np.eye(2)-alpha*pk)
gap = float(np.linalg.norm(projected_response-reduced_response))
assert gap > .01

result = {
    "status": "FIXTURE_ONLY",
    "pairwise_not_joint": {"pairwise_a": list(map(str,pairs_a)),
        "pairwise_b": list(map(str,pairs_b)), "triple_a": str(triple_a),
        "triple_b": str(triple_b), "meaning": "Pairwise equality does not identify a partition law. No Dice superiority follows."},
    "joint_law_changes_executable_marginal": {"posterior_a": str(post_a),
        "posterior_b": str(post_b), "meaning": "With identical soft image evidence and four named region-label factors, exact circuit marginal differs. This does not show that a student can learn the useful law."},
    "coarse_partition_ceiling": {"best_pixel_accuracy": float(best_correct/len(y)),
        "best_foreground_dice": float(best_dice), "meaning": "Every classifier constant on these cells misses fine support; an adaptive split or detail renderer is required."},
    "project_then_solve_is_not_solve_then_project": {"frobenius_gap": gap,
        "projected_response": projected_response.tolist(), "reduced_response": reduced_response.tolist(),
        "meaning": "Teacher operator coarsening requires response validation or a lumpability assumption."},
    "not_tested": ["patient Dice", "new student performance", "teacher quality", "GPU latency", "novelty"]
}
Path(__file__).with_name("PROBE_RESULTS.json").write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps(result, indent=2))
