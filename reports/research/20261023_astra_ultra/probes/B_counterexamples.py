"""Bounded manufactured CPU theory probes; no patient or efficacy evidence."""
from pathlib import Path
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
from self_audit.control import (
    audit_weights, controlled_head_step, response_moments, risks,
    scalar_allocation, solve_allocation,
)


def make_head(probabilities):
    p = torch.tensor(probabilities, dtype=torch.float64)
    head = torch.nn.Conv2d(p.shape[1], 4, 1, bias=True).double()
    with torch.no_grad():
        head.weight.copy_(p.log()[:, :, None, None])
        head.bias.zero_()
    features = torch.eye(p.shape[1], dtype=torch.float64)[None, :, None, :]
    return head, features


def hard_dice(probabilities, labels, cls=1):
    pred = probabilities.argmax(1) == cls
    truth = labels == cls
    return 2 * int((pred & truth).sum()) / int(pred.sum() + truth.sum())


def brier_dice_counterexample():
    initial = [[.46, .46, .88, .88], [.52, .52, .1, .1], [.01] * 4, [.01] * 4]
    final = [[.19, .51, .89, .89], [.8, .48, .1, .1], [.005] * 4, [.005] * 4]
    head, features = make_head(initial)
    labels = torch.tensor([[[1, 1, 0, 0]]])
    p0 = head(features).softmax(1).detach()
    delta = torch.tensor(final, dtype=torch.float64).log() - torch.tensor(initial, dtype=torch.float64).log()
    report = controlled_head_step(head, features, labels,
        [(delta[:, :, None, None], torch.zeros(4, dtype=torch.float64))],
        ["manufactured_direction"], method="joint_guard", tolerance=0.)
    p1 = head(features).softmax(1).detach()
    result = {"scope": "manually constructed affine-head displacement, not a generated pseudo-gradient",
              "dice_before": hard_dice(p0, labels), "dice_after": hard_dice(p1, labels),
              "probabilities_before": p0.tolist(), "probabilities_after": p1.tolist(), "control": report}
    assert report["scale"] == 1. and result["dice_after"] < result["dice_before"]
    assert np.all(np.array(report["after"]) < np.array(report["before"]))
    return result


def tangent_boundary_counterexample():
    head, features = make_head([[.25], [.25], [.25], [.25]])
    labels = torch.tensor([[[1]]])
    base = {k: v.detach().clone() for k, v in head.state_dict().items()}
    direction = (torch.tensor([1., 0., -.5, -.5], dtype=torch.float64)[:, None, None, None],
                 torch.zeros(4, dtype=torch.float64))
    report = controlled_head_step(head, features, labels, [direction], ["manufactured_direction"],
                                 method="joint_guard", tolerance=0., backtracks=12)
    unchanged = all(torch.equal(v, base[k]) for k, v in head.state_dict().items())
    assert report["alpha"][0] > .9 and report["scale"] == 0. and unchanged
    assert all(row["risks"][0] > report["before"][0] for row in report["trials"])
    return {"scope": "exact softmax head; correct-class derivative zero and adverse second derivative",
            "head_bitwise_unchanged": unchanged, "control": report}


def physically_realizable_small_scale():
    rows = []
    for magnitude in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        head, features = make_head([[.25], [.25], [.25], [.25]])
        labels = torch.tensor([[[1]]])
        p = head(features).softmax(1).detach()
        weights, targets, _ = audit_weights(labels)
        dz = torch.tensor([-1., 3., -1., -1.], dtype=torch.float64)[None, None, :, None, None] * magnitude
        dp = p[None] * (dz - (p[None] * dz).sum(2, keepdim=True))
        A, b, e = response_moments(p, dp, weights, targets)
        joint = solve_allocation(A, b, e, ridge=1e-6, budget=0., source_ids=["class1"])
        exact = scalar_allocation(A[:, 0, 0], b[:, 0], ridge=1e-6, budget=0.)
        def objective(x):
            return float((e - b[:, 0] * x + .5 * A[:, 0, 0] * x * x).mean() + .5e-6 * x*x)
        f0, fs, fj = objective(0.), objective(exact), objective(float(joint.alpha[0]))
        rows.append({"magnitude": magnitude, "A": A.tolist(), "b": b.tolist(), "e": e.tolist(),
                     "joint_alpha": joint.alpha.tolist(), "scalar_alpha": exact, "joint_status": joint.status,
                     "objective_at_zero": f0, "joint_objective_gain": f0-fj,
                     "exact_scalar_objective_gain": f0-fs,
                     "retained_gain_fraction": (f0-fj)/(f0-fs) if f0>fs else None})
    assert any(r["scalar_alpha"] > .01 and r["joint_alpha"][0] < 1e-5 for r in rows)
    return rows


def consistent_moment_validation_gap():
    # PSD alone does not guarantee that (A,b,e) can arise from a common V,r.
    result = solve_allocation(np.zeros((1, 1, 1)), np.array([[1.]]), np.array([.5]), ridge=1e-6)
    assert result.local_risks[0] < 0
    return {"scope": "standalone API malformed input only; internal response_moments is consistent",
            "A": [[[0.]]], "b": [[1.]], "e": [.5], "alpha": result.alpha.tolist(),
            "status": result.status, "local_risks": result.local_risks.tolist()}


def main():
    torch.set_num_threads(1)
    output = ROOT / "reports/research/20261023_astra_ultra/evidence/B_counterexamples.json"
    report = {
        "scope": "CPU float64 manufactured analysis; no patient data, pretrained weights or GPU",
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "scipy": scipy.__version__,
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [ROOT / "self_audit/control.py", ROOT / "self_audit/decoder_control.py"]},
        "brier_down_dice_down": brier_dice_counterexample(),
        "tangent_boundary": tangent_boundary_counterexample(),
        "physical_small_scale": physically_realizable_small_scale(),
        "moment_contract_gap": consistent_moment_validation_gap(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "checks": "PASS", "cases": 4,
                      "physical_scale_cases": len(report["physical_small_scale"])}))


if __name__ == "__main__":
    main()
