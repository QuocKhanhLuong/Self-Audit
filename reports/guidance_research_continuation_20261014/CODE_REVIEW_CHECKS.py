"""Independent local CPU mathematical checks; zero optimizer steps or real data."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
from scipy import sparse
import torch

from experiments.guidance_signal_20261014 import evidence, pilot, runtime, stagewise


def dense_oracle():
    rng = np.random.default_rng(63901)
    worst = dict(M=0., h=0., Q=0., Z=0.)
    cases = []
    for n in (1, 2, 7, 11):
        for alpha in (.05, .35, 1.):
            P = rng.uniform(size=(n, n)); P /= P.sum(1, keepdims=True)
            s = rng.uniform(size=n)
            s[0] = 0.
            if n > 1:
                s[1] = 1.
            V = rng.dirichlet(np.ones(4), n)
            got, receipt = evidence.decompose(sparse.csr_matrix(P), s, V, alpha=alpha)
            K = np.linalg.solve(np.eye(n)-(1-alpha)*P, alpha*np.eye(n))
            M, h = K @ (s[:, None]*V), K @ s
            Q = K @ ((1-s[:, None])*.25+s[:, None]*V)
            eps = receipt["linear_solver"]["error_bound"]
            for key, expected in (("M", M), ("h", h), ("Q", Q)):
                error = float(np.max(np.abs(got[key]-expected)))
                worst[key] = max(worst[key], error)
                assert error <= (1.25 if key == "Q" else 1.)*eps+2e-14
            mask = got["valid"]
            if mask.any():
                error = np.max(np.abs(got["Z"][mask]-M[mask]/h[mask, None]), axis=1)
                assert np.all(error <= got["z_error_bound"][mask]+2e-14)
                worst["Z"] = max(worst["Z"], float(error.max()))
            assert np.isnan(got["Z"][~mask]).all()
            cases.append(dict(nodes=n, alpha=alpha, valid=int(mask.sum()), error_bound=eps))
    return dict(cases=cases, max_absolute_error=worst,
                oracle="dense inverse, independent of iterative implementation")


def no_new_names():
    # High source exposure, but no informative named signal.
    P = sparse.csr_matrix([[.75, .25], [.25, .75]])
    got, _ = evidence.decompose(P, np.array([.9, .6]), np.full((2, 4), .25))
    np.testing.assert_allclose(got["Z"], .25, rtol=0, atol=1e-12)
    assert got["valid"].all()
    uniform_h = got["h"].tolist()
    # Removing the uniform mixture retains the supplied (possibly wrong) name.
    wrong = np.array([[.01, .01, .97, .01]])
    got, _ = evidence.decompose(sparse.eye(1), np.array([.1]), wrong)
    assert got["Q"].argmax(-1).item() == got["Z"].argmax(-1).item() == 2
    return dict(uniform_valid_h=uniform_h, wrong_source_name_retained=True,
                anatomical_claim=False)


def r0_nonidentifiability():
    u = np.full(4, .25)
    a = .8*u+.2*np.array([1., 0., 0., 0.])
    b = .4*u+.6*np.array([.5, 1/6, 1/6, 1/6])
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-15)
    return dict(shared_R0=a.tolist(), different_strengths=[.2, .6],
                raw_history_not_identified=True)


def exact_gradient_difference():
    dtype = torch.float32
    h = torch.tensor([.1, .6, .99, 0.], dtype=dtype).reshape(1, 1, 2, 2)
    z = torch.tensor([[.01, .01, .97, .01], [.1, .3, .5, .1],
                      [.8, .1, .05, .05], [.25]*4], dtype=dtype).T.reshape(1, 4, 2, 2)
    q = (1-h)*.25+h*z
    weights = torch.tensor([.2, .05, .01, 0.], dtype=dtype).reshape(1, 2, 2)
    valid = weights > 0
    z = z.clone(); z[:, :, 1, 1] = float("nan")
    logits = torch.tensor([-.2, .1, .3, -.5], dtype=dtype).reshape(1, 4, 1, 1).repeat(1, 1, 2, 2)
    logits.requires_grad_()
    terms = pilot.semantic_terms(logits, q, z, weights, valid)
    grad_q = torch.autograd.grad(terms["Q"]["ce"], logits, retain_graph=True)[0]
    grad_z = torch.autograd.grad(terms["Z"]["ce"], logits, retain_graph=True)[0]
    expected = torch.zeros_like(logits)
    expected.permute(0, 2, 3, 1)[valid] = (
        weights[valid, None]/weights.sum()*(q.permute(0, 2, 3, 1)[valid]-z.permute(0, 2, 3, 1)[valid]))
    err = float((grad_z-grad_q-expected).abs().max())
    assert err < 1e-7
    assert torch.isfinite(grad_z).all() and torch.equal(grad_z[:, :, 1, 1], torch.zeros(1, 4))
    for arm in ("full_Q", "Q", "Z"):
        assert abs(float((terms[arm]["ce"]-terms[arm]["entropy"]-terms[arm]["kl"]).detach())) < 3e-7
    return dict(max_gradient_difference_error=err, invalid_nan_consumption=False,
                named_RV_counterexample=dict(target=[.1, .3, .5, .1], above_uniform=.05, winning_margin=0.))


def denominator_and_activation():
    rt = runtime.load_runtime()
    # This tiny weight lies below the denominator clamp. It must not be silently
    # converted into an unweighted mean or treated as semantic inactivity.
    q = torch.tensor([.1, .4, .3, .2]).reshape(1, 4, 1, 1)
    w = torch.tensor([[[2e-10]]])
    logits = torch.zeros_like(q, requires_grad=True)
    terms = pilot.semantic_terms(logits, q, q, w, torch.ones_like(w, dtype=torch.bool))
    expected = (2e-10/1e-8)*np.log(4)
    assert terms["active"] and abs(float(terms["Q"]["ce"].detach())-expected) < 1e-8
    teacher = torch.ones(1, 2, 1, 1)
    output = dict(logits=logits, features=teacher.clone().requires_grad_())
    # Use the actual archived loss signature and feature-key contract.
    _, original, active = rt.student.guidance_loss(output, teacher, q, w, True)
    assert active and torch.allclose(original, terms["full_Q"]["ce"], atol=1e-8, rtol=0)
    qb = torch.tensor([.6, .1, .2, .1]).reshape(1, 4, 1, 1)
    inactive = pilot.semantic_terms(logits, qb, q, w, torch.ones_like(w, dtype=torch.bool))
    assert not inactive["active"]
    assert all(float(inactive[a][k].detach()) == 0 for a in ("Q", "Z", "full_Q") for k in ("ce", "entropy", "kl"))
    try:
        pilot.check_excluded_mass(torch.tensor([1e-8]), torch.tensor([False]))
    except ValueError:
        pass
    else:
        raise AssertionError("100% exclusion with unit normalized weight was accepted")
    return dict(clamped_expected_ce=expected, measured_ce=float(terms["Q"]["ce"].detach()),
                inactive_Q_activation_retained=True, total_1e_8_all_excluded_rejected=True)


def plan_tamper_precedes_dependencies():
    original = {"steps": 64, "seed": 42, "patient_ids": ["train_a"]}
    changed = {**original, "steps": 65}
    old = pilot._dependencies
    def forbid():
        raise AssertionError("artifact/optimizer dependency loaded before plan digest rejection")
    pilot._dependencies = forbid
    try:
        try:
            pilot.run_pilot("absent", "absent", "absent", "absent", changed, "absent",
                            expected_plan_sha=pilot.json_hash(original))
        except ValueError as exc:
            assert "plan SHA" in str(exc)
        else:
            raise AssertionError("allowed-range plan mutation accepted")
    finally:
        pilot._dependencies = old
    return dict(allowed_range_budget_mutation_rejected_before_dependencies=True)


def stagewise_logprob_underflow():
    q = np.array([[[[[1., 0., 0., 0.]]]]])
    p = np.array([[[[[0., 1., 0., 0.]]]]])
    logp = np.array([[[[[-1000., 0., -1000., -1000.]]]]])
    row = stagewise.fidelity_scalars("synthetic", q, p, np.ones(q.shape[:-1]), [True]*4, logp)[0]
    assert row["kl_q_to_student"] == 1000.
    return dict(log_softmax_underflow_kl=row["kl_q_to_student"], clipped_log_proxy_rejected=True)


def main():
    report = dict(scope="INDEPENDENT_LOCAL_CPU_SYNTHETIC", optimizer_steps=0,
                  real_patient_data_read=False, original_artifact_replay=False,
                  reference_masks_read=False, remote_access=False,
                  versions=dict(python=sys.version.split()[0], numpy=np.__version__, torch=torch.__version__),
                  checks={}, source_sha256={})
    for path in sorted((ROOT/"experiments/guidance_signal_20261014").rglob("*.py")):
        report["source_sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for check in (dense_oracle, no_new_names, r0_nonidentifiability, exact_gradient_difference,
                  denominator_and_activation, plan_tamper_precedes_dependencies, stagewise_logprob_underflow):
        try:
            report["checks"][check.__name__] = dict(status="PASS", **check())
        except Exception as exc:
            report["checks"][check.__name__] = dict(status="FAIL", error=repr(exc))
    report["status"] = "PASS_SOFTWARE_ONLY" if all(x["status"] == "PASS" for x in report["checks"].values()) else "FAIL"
    path = Path(__file__).with_suffix(".json")
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(status=report["status"], checks=report["checks"]), allow_nan=False))
    return int(report["status"] == "FAIL")


if __name__ == "__main__":
    raise SystemExit(main())
