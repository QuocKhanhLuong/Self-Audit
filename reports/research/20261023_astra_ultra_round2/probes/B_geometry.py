"""Research-only exact-head and convex-geometry probes. No patient/GPU claim."""
from pathlib import Path
from unittest.mock import patch
import copy
import hashlib
import json
import platform
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import numpy as np
import scipy
import torch
from torch.nn import functional as F
import self_audit.control as control


def quadratic(A, b, e, x, ridge):
    x = np.asarray(x, dtype=float)
    risks = e - b @ x + .5 * np.einsum("i,gij,j->g", x, A, x)
    return {"risks": risks.tolist(), "mean_risk": float(risks.mean()),
            "penalty": float(.5 * ridge * x @ x),
            "objective": float(risks.mean() + .5 * ridge * x @ x),
            "gradient": (A.mean(0) @ x - b.mean(0) + ridge*x).tolist()}


def scalar_ray(A, b, ridge):
    return control.scalar_allocation(A.sum((1, 2)), b.sum(1), ridge=A.shape[1]*ridge)


def exact_box_certificate(A, b, e, x, ridge):
    q = quadratic(A, b, e, x, ridge)
    g = np.asarray(q["gradient"])
    # Separability of the linear supporting hyperplane over the entire box.
    minimum_support = float(np.minimum(g, 0).sum() - g @ x)
    assert minimum_support >= -1e-14
    assert np.all(np.asarray(q["risks"]) <= e + 1e-14)
    return {"minimum_supporting_hyperplane_on_box": minimum_support, **q}


def finite_head_case(name, dz_rows, exact_joint):
    torch.manual_seed(23023)
    array = np.asarray(dz_rows)
    if array.ndim == 2:
        array = array[:, :, None]
    pixels = array.shape[2]
    head = torch.nn.Conv2d(pixels, 4, 1).double()
    with torch.no_grad():
        head.weight.zero_(); head.bias.zero_()
    features = torch.eye(pixels, dtype=torch.float64)[None, :, None, :]
    labels = torch.ones(1, 1, pixels, dtype=torch.long)
    directions = [(torch.tensor(dz, dtype=torch.float64)[:, :, None, None],
                   torch.zeros(4, dtype=torch.float64)) for dz in array]
    p = head(features).softmax(1).detach()
    weights, targets, _ = control.audit_weights(labels)
    dz = torch.stack([F.conv2d(features, *d) for d in directions])
    dp = p[None] * (dz - (p[None] * dz).sum(2, keepdim=True))
    A, b, e = control.response_moments(p, dp, weights, targets)
    ridge = 1e-6
    xj = np.asarray(exact_joint, float)
    scalar = scalar_ray(A, b, ridge)
    xs = np.full(len(directions), scalar)
    certificate = exact_box_certificate(A, b, e, xj, ridge)

    def run(x, tolerance, backtracks, injected=True):
        model = copy.deepcopy(head)
        before = copy.deepcopy(model.state_dict())
        rng = torch.get_rng_state().clone()
        def selector(actual_A, actual_b, actual_e, **kwargs):
            assert np.allclose(actual_A, A, rtol=0, atol=1e-14)
            assert np.allclose(actual_b, b, rtol=0, atol=1e-14)
            qr = quadratic(A, b, e, x, ridge)
            return control.Allocation(np.asarray(x).copy(), "RESEARCH_EXACT_FIXED_COEFFICIENTS",
                                      qr["objective"], np.array(qr["risks"]))
        if injected:
            with patch.object(control, "solve_allocation", selector):
                report = control.controlled_head_step(model, features, labels, directions,
                    [f"manufactured_source_{i}" for i in range(len(directions))],
                    ridge=ridge, budget=0., tolerance=tolerance, backtracks=backtracks)
        else:
            report = control.controlled_head_step(model, features, labels, directions,
                [f"manufactured_source_{i}" for i in range(len(directions))],
                ridge=ridge, budget=0., tolerance=tolerance, backtracks=backtracks)
        unchanged = all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
        return {"control": report, "head_unchanged": unchanged,
                "rng_unchanged": bool(torch.equal(torch.get_rng_state(), rng)),
                "probabilities": model(features).softmax(1).detach().flatten().tolist()}

    result = {"name": name, "scope": "manually constructed candidate directions; actual existing finite head guard",
              "dz": dz_rows, "A": A.tolist(), "b": b.tolist(), "e": e.tolist(), "ridge": ridge,
              "exact_joint": xj.tolist(), "exact_scalar": scalar,
              "joint_box_certificate": certificate,
              "scalar_local": quadratic(A, b, e, xs, ridge),
              "joint_default_guard": run(xj, 1e-8, 8),
              "scalar_default_guard": run(xs, 1e-8, 8),
              "joint_zero_tolerance": run(xj, 0., 12),
              "scalar_zero_tolerance": run(xs, 0., 12),
              "source_solver_joint_default_guard": run(xj, 1e-8, 8, injected=False)}
    if name == "local_joint_wins_finite_joint_rejects":
        assert certificate["mean_risk"] < result["scalar_local"]["mean_risk"]
        assert result["joint_default_guard"]["control"]["scale"] == 0
        assert result["joint_zero_tolerance"]["control"]["scale"] == 0
        assert result["scalar_default_guard"]["control"]["scale"] == 1
        assert result["scalar_zero_tolerance"]["control"]["scale"] == 1
        assert result["source_solver_joint_default_guard"]["control"]["scale"] == 0
        assert result["joint_default_guard"]["head_unchanged"]
        assert np.all(np.array(result["scalar_default_guard"]["control"]["after"]) < e)
        perturbed = []
        for a2 in [1e-5, 1e-4, 1e-3, .005]:
            x = np.array([1., a2])
            local = quadratic(A, b, e, x, ridge)
            actual = run(x, 1e-8, 8)
            assert np.all(np.asarray(local["risks"]) <= e)
            assert actual["control"]["scale"] == 0
            perturbed.append({"alpha": x.tolist(), "objective_gap_to_exact_joint": local["objective"]-certificate["objective"],
                              "local": local, "actual": actual})
        result["feasible_near_optimal_perturbations"] = perturbed
    else:
        j = result["joint_default_guard"]["control"]
        s = result["scalar_default_guard"]["control"]
        assert j["scale"] == s["scale"] == 1
        if name == "symmetry_no_finite_benefit":
            assert np.array_equal(xj, xs)
            assert np.array_equal(j["after"], s["after"])
        else:
            assert np.mean(j["after"]) < np.mean(s["after"])
    return result


def local_cases():
    rows = []
    for label, residual in [("symmetry_no_benefit", [.4, .4]), ("asymmetry_strict_benefit", [.9, .1])]:
        V = np.eye(2); r = np.array(residual); ridge = .1
        A = np.array([V.T@V]); b = np.array([V.T@r]); e = np.array([.5*r@r])
        xj = np.clip(np.linalg.solve(A.mean(0)+ridge*np.eye(2), b.mean(0)), 0, 1)
        xs = np.full(2, scalar_ray(A,b,ridge))
        j = exact_box_certificate(A,b,e,xj,ridge); s = quadratic(A,b,e,xs,ridge)
        rows.append({"name":label, "scope":"consistent manufactured quadratic moments, not patient data",
                     "V":V.tolist(),"r":r.tolist(),"ridge":ridge,"joint_alpha":xj.tolist(),"scalar_alpha":xs.tolist(),
                     "joint":j,"scalar":s})
        if label == "symmetry_no_benefit": assert np.allclose(xj, xs, atol=1e-14)
        else: assert j["objective"] < s["objective"]
    k = 2e-6
    A = np.array([np.diag([2.5,.5])])*k
    b = np.array([[1.92,.5]])*k
    V = np.diag(np.sqrt([2.5,.5]))*np.sqrt(k)
    r = np.array([1.92/np.sqrt(2.5),.5/np.sqrt(.5)])*np.sqrt(k)
    e = np.array([.5*r@r]); ridge=1e-6
    xj = np.linalg.solve(A[0]+ridge*np.eye(2),b[0]); xs=np.full(2,scalar_ray(A,b,ridge))
    j=exact_box_certificate(A,b,e,xj,ridge); s=quadratic(A,b,e,xs,ridge)
    assert j["objective"] < s["objective"] and j["mean_risk"] > s["mean_risk"]
    rows.append({"name":"regularized_objective_better_mean_risk_worse","V":V.tolist(),"r":r.tolist(),
                 "ridge":ridge,"joint_alpha":xj.tolist(),"scalar_alpha":xs.tolist(),"joint":j,"scalar":s})
    return rows


def portfolio_noncomponentwise():
    labels = torch.ones(1,1,1,dtype=torch.long)
    weights, targets, _ = control.audit_weights(labels)
    choices={"base":[.25,.25,.25,.25],"accepted_S":[.2,.6,.1,.1],"accepted_J":[.1,.65,.15,.1]}
    scored={k:control.risks(torch.tensor(p,dtype=torch.float64).reshape(1,4,1,1),weights,targets).tolist()
            for k,p in choices.items()}
    selected=min(scored,key=lambda k:np.mean(scored[k]))
    assert selected=="accepted_J" and scored[selected][1]>scored["accepted_S"][1]
    return {"scope":"valid probability endpoints illustrating selection implication only; not solver outputs",
            "probabilities":choices,"risks":scored,"selected_by_actual_mean":selected}


def main():
    torch.set_num_threads(1)
    report={"scope":"research-only CPU float64; no patient, pretrained, GPU or real pseudo-gradient evidence",
            "python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"torch":torch.__version__,
            "probe_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "production_source_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in [ROOT/"self_audit/control.py",ROOT/"self_audit/decoder_control.py"]},
            "local_cases":local_cases(),
            "finite_reversal":finite_head_case("local_joint_wins_finite_joint_rejects",
                                                [[1.,0.,-.5,-.5],[-.8,.1,.35,.35]],[1.,0.]),
            "finite_strict_benefit":finite_head_case("local_joint_wins_finite_joint_wins",
                                                     [[-.1,.3,-.1,-.1],[.3,-.1,-.1,-.1]],[1.,0.]),
            "finite_no_benefit":finite_head_case("symmetry_no_finite_benefit",
                                                 [[[-.1,0.],[.3,0.],[-.1,0.],[-.1,0.]],
                                                  [[0.,-.1],[0.,.3],[0.,-.1],[0.,-.1]]],[1.,1.]),
            "portfolio_noncomponentwise":portfolio_noncomponentwise()}
    out=ROOT/"reports/research/20261023_astra_ultra_round2/evidence/B_geometry.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"output":str(out),"checks":"PASS","local_cases":3,"finite_cases":3,
                      "near_optimal_perturbations":4,"portfolio_cases":1}))


if __name__=="__main__": main()
