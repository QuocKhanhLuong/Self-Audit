"""Constructive arithmetic checks only; does not import or evaluate Self-Audit."""

import json
from pathlib import Path


def main():
    previous = [0, 1, 2, 0, 1, 2]
    factual = [1, 2, 2, 0, 0, 1]
    truth = [0, 2, 2, 1, 1, 0]
    regress = [i for i, (a, b, y) in enumerate(zip(previous, factual, truth)) if a == y and b != y]
    fixes = [i for i, (a, b, y) in enumerate(zip(previous, factual, truth)) if a != y and b == y]
    oracle = [previous[i] if i in regress else factual[i] for i in range(len(truth))]
    assert all(oracle[i] == truth[i] for i in regress)
    assert all(oracle[i] == factual[i] for i in fixes)
    assert all(oracle[i] == factual[i] for i in range(len(truth)) if i not in regress)
    selected = {0, 1, 2, 3, 5}
    masked = [previous[i] if i in selected else factual[i] for i in range(len(truth))]
    net_delta = sum(int(n == y) - int(b == y) for n, b, y in zip(masked, factual, truth))
    signed_count = len(selected.intersection(regress)) - len(selected.intersection(fixes))
    assert net_delta == signed_count

    g, factual_margin, cf_margin = 0.2, -0.2, 0.3
    final_margin = (1 - g) * factual_margin + g * cf_margin
    assert cf_margin > 0 and final_margin < 0

    def loss(q):
        return (q[0] - 1) ** 2 + (q[1] - 1) ** 2

    def protected_margin(q):
        return 0.1 - 2 * q[0]

    choices = {"factual": (0, 0), "full": (1, 1), "half": (0.5, 0.5), "missed_feasible": (0, 1)}
    evaluations = {
        name: {"coordinates": q, "restoration_loss": loss(q), "protected_margin": protected_margin(q),
               "feasible": protected_margin(q) >= 0.05}
        for name, q in choices.items()
    }
    assert not evaluations["full"]["feasible"] and not evaluations["half"]["feasible"]
    assert evaluations["missed_feasible"]["feasible"] and loss((0, 1)) < loss((0, 0))
    # grad at P=(0,0): grad L=(-2,-2), grad lambda*||Q-P||^2=(0,0).
    gradients = {str(lam): [-2 + 2 * lam * 0, -2 + 2 * lam * 0] for lam in (0, 0.1, 1, 10)}
    assert all(value == [-2, -2] for value in gradients.values())

    result = {
        "evidence_type": "constructed arithmetic examples; not trained-model or medical results",
        "oracle_rollback": {"previous": previous, "factual": factual, "gt_offline_only": truth,
                            "true_regress_indices": regress, "true_fix_indices": fixes, "oracle": oracle,
                            "regress_repair_fraction": 1.0, "previous_fix_destroyed": 0},
        "gate_can_suppress_repair": {"gate": g, "factual_pairwise_margin": factual_margin,
                                     "counterfactual_pairwise_margin": cf_margin, "final_pairwise_margin": final_margin,
                                     "required_gate_strictly_greater_than": -factual_margin / (cf_margin - factual_margin)},
        "signed_masked_rollback_identity": {"selected_indices": sorted(selected), "net_correct_delta": net_delta,
                                           "selected_true_regress_minus_selected_true_fix": signed_count},
        "initial_gradients_by_lambda": gradients,
        "feasible_direction_missed_by_two_checks": evaluations,
    }
    destination = Path(__file__).with_suffix(".json")
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "checks": 5, "output": str(destination)}, indent=2))


if __name__ == "__main__":
    main()
