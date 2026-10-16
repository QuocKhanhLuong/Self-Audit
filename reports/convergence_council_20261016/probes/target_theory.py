"""Text-only receipts and bounded synthetic algebra; no model or patient arrays."""
from pathlib import Path
import hashlib
import json
import math


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "reports/stable_guidance_main513ddda_b12_20261016/completed_run"
OUT = Path(__file__).resolve().parents[1] / "checks/target_theory.json"


def entropy(p):
    return -sum(x * math.log(x) for x in p if x > 0)


def accepted(p):
    top = sorted(p)
    return top[-1] >= .7 and top[-1] - top[-2] >= .2


def main():
    frozen = json.loads((RUN / "run/bank/FROZEN.json").read_text())
    bank = json.loads((RUN / "run/bank/REPORT.json").read_text())
    convergence = json.loads((RUN / "analysis/semantic_convergence/REPORT.json").read_text())
    source_hashes = {}
    for name in ("core.py", "student.py"):
        actual = hashlib.sha256((RUN / "source/src/self_audit_guidance" / name).read_bytes()).hexdigest()
        assert actual == frozen["source_sha256"][name]
        source_hashes[name] = actual
    ps = list(bank["patient_reports"].values())
    counts = [sum(p["class_pixels"][c] for p in ps) for c in range(4)]
    nodes = sum(p["graph"]["nodes"] for p in ps)
    aggregate = dict(nodes=nodes, class_pixels=counts, class_patients=bank["class_patients"],
                     allowed_classes=[n >= 2 for n in bank["class_patients"]],
                     hard_coverage=sum(counts)/nodes, hard_foreground_coverage=sum(counts[1:])/nodes,
                     anchor_supported_pixels=sum(p["anatomy"]["anchor_supported_pixels"] for p in ps),
                     no_foreground_patients=sum(p["status"] == "NO_FOREGROUND_SUPPORT" for p in ps),
                     all_bank_solvers_converged=all(p["solver"]["converged"] for p in ps))
    e = convergence["last_epoch"]
    mass = e["uniform_prediction_ce"] / math.log(4)
    normalized_entropy = e["target_entropy_floor"] / mass
    gate_entropy_max = entropy([.7, .1, .1, .1])
    entropy_bound = dict(effective_training_measure_mass=mass,
                         normalized_weighted_entropy=normalized_entropy,
                         max_entropy_given_qmax_at_least_point7=gate_entropy_max,
                         approximate_lower_bound_weighted_mass_qmax_below_point7=
                         (normalized_entropy-gate_entropy_max)/(math.log(4)-gate_entropy_max),
                         scope="Derived from published float64 audit of float32 targets; approximate normalization. Not voxel coverage.")
    # Stochastic two-node graph, equal unit self/neighbor edges; P^2=P.
    # Source 0 is truly neutral. The sole source 1 incorrectly votes RV.
    alpha = .35
    p = [[.5, .5], [.5, .5]]
    k = [[alpha * (i == j) + (1-alpha)*p[i][j] for j in range(2)] for i in range(2)]
    strengths = [0., .9]
    source_vote = [0., 1., 0., 0.]
    h = [sum(k[i][j]*strengths[j] for j in range(2)) for i in range(2)]
    q = [[(1-hi)*.25 + hi*v for v in source_vote] for hi in h]
    r0 = [[(1-s)*.25+s*v for v in source_vote] for s in strengths]
    residual = max(abs(q[i][c] - alpha*r0[i][c] - (1-alpha)*sum(p[i][j]*q[j][c] for j in range(2)))
                   for i in range(2) for c in range(4))
    assert residual < 1e-14
    assert not accepted(q[0]) and accepted(source_vote)
    reliability = [hi*(1-entropy(qi)/math.log(4)) for hi, qi in zip(h, q)]
    assert reliability[0] > 0
    # Gradient wrt logits is prediction minus target. At p=Q, full-Q CE has zero
    # gradient while a switch to Z amplifies the wrong RV identity.
    delta = [q[0][c]-source_vote[c] for c in range(4)]
    assert delta[1] < 0 and all(delta[c] > 0 for c in (0, 2, 3))
    wrong_source = dict(graph=p, kernel=k, strengths=strengths, h=h, q=q,
                        z=[source_vote, source_vote], reliability=reliability,
                        fixed_point_residual=residual, q_export_accept=[accepted(x) for x in q],
                        z_export_accept=[True, True], gradient_at_p_equals_q_for_z=delta,
                        scope="Synthetic algebra in stochastic graph family; no claim of anatomical-anchor realizability.")
    # Perfectly fitting a full-support target need not yield any exported labels.
    perfect_q = [.15, .55, .15, .15]
    assert not accepted(perfect_q)
    # Full-Q soft CE can move a globally disallowed channel upward if its
    # current prediction is below neutral mass, despite a permitted argmax.
    permitted_argmax_q = [.7, .1, .1, .1]
    prediction = [.8, .01, .09, .1]
    unsupported_channel_gradient = prediction[1] - permitted_argmax_q[1]
    assert unsupported_channel_gradient < 0
    result = dict(status="PASS_TEXT_RECEIPTS_AND_SYNTHETIC_ALGEBRA_ONLY", executed_source_commit="513ddda8cec3a166f13cfe68fe6379cab014a1a7",
                  source_hashes=source_hashes, aggregate=aggregate, entropy_gate_bound=entropy_bound,
                  wrong_source=wrong_source, perfect_fit_abstention=dict(q=perfect_q, kl=0., accepted=False),
                  hypothetical_disallowed_channel=dict(gradient=unsupported_channel_gradient,
                     current_run_applicable=False), patient_arrays_read=False, gpu_used=False, training_run=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
