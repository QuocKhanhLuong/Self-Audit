"""Tiny manufactured R2 counterexamples. No models, patients, or optimization runs."""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import torch
from torch.nn import functional as F


torch.set_num_threads(1)
torch.manual_seed(20261018)
torch.set_default_dtype(torch.float64)


def ce(logits, q):
    return -(q * logits.log_softmax(-1)).sum(-1)


def run():
    report = {"status": "MANUFACTURED_CPU_ONLY", "seed": 20261018,
              "torch": torch.__version__, "python": platform.python_version()}
    k = 16
    D = torch.ones(1, k) / k
    U = torch.ones(k, 1)
    P = torch.eye(k) - U @ D
    report["split"] = {
        "DU_error": float((D @ U - torch.eye(1)).abs().max()),
        "DP_error": float((D @ P).abs().max()),
        "P_idempotence_error": float((P @ P - P).abs().max()),
        "rank_P": int(torch.linalg.matrix_rank(P)),
    }

    # A transport has more rows but cannot add teacher observation rank.
    H = torch.zeros(4, 16)
    for i in range(4):
        H[i, 4 * i:4 * i + 4] = 0.25
    R = torch.eye(4).repeat_interleave(4, dim=0)
    report["resized_rank"] = {"teacher_rank": int(torch.linalg.matrix_rank(H)),
                               "resized_rank": int(torch.linalg.matrix_rank(R @ H)),
                               "resized_rows": len(R)}

    # Softmax and average do not commute even when logit detail is exactly centered.
    a = torch.tensor([2.0, 0.0])
    d = torch.tensor([[2.0, 0.0], [-2.0, 0.0]])
    l = a + d
    report["probability_mismatch"] = {
        "mean_logit_error": float((l.mean(0) - a).abs().max()),
        "softmax_mean_logits": a.softmax(-1).tolist(),
        "mean_softmax_logits": l.softmax(-1).mean(0).tolist(),
    }

    # Jensen holds with uniform weights and constant q; gauge remains invisible.
    q = torch.tensor([0.8, 0.2])
    theta = torch.tensor(0.0, requires_grad=True)
    varying = torch.stack([torch.stack([theta, theta * 0]),
                           torch.stack([-theta, theta * 0])])
    weighted = (ce(varying, q) * torch.tensor([0.9, 0.1])).sum()
    grad_weighted = torch.autograd.grad(weighted, theta)[0]
    gauge = torch.tensor([[2.0, 2.0], [-2.0, -2.0]])
    report["jensen"] = {
        "constant_q_uniform_gap": float(ce(l, q).mean() - ce(a, q)),
        "gauge_gap": float(ce(a + gauge, q).mean() - ce(a, q)),
        "nonuniform_weight_detail_gradient_at_zero": float(grad_weighted),
    }

    # P uses padded pixels too. A real-only mean need not remain zero.
    v = torch.tensor([1.0, -1.0, -1.0, -1.0])
    centered = v - v.mean()
    report["padding"] = {"padded_mean": float(centered.mean()),
                         "first_valid_pixel_mean": float(centered[:1].mean())}

    s = torch.full((16,), -1.0)
    s[0] = 1.0
    bounded = s - s.mean()
    report["capacity_limit"] = {"cell_size": 16, "bound_b": 1.0,
                                "max_projected_detail_limit": float(bounded.max()),
                                "within_class_range_limit": float(bounded.max()-bounded.min()),
                                "strict_finite_tanh": True}

    # K enforces two image-derived constant regions. H must distinguish their means.
    K = torch.tensor([[1., -1., 0., 0.], [0., 0., 1., -1.]])
    H_bad = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
    H_good = torch.tensor([[0.5, 0.5, 0., 0.], [0., 0., 0.5, 0.5]])
    A_bad = torch.cat([H_bad, K])
    A_good = torch.cat([H_good, K])
    truth = torch.tensor([0.9, 0.9, 0.1, 0.1])
    swapped = 1 - truth
    noise = torch.tensor([0.02, -0.01, 0.003, -0.004])
    obs = A_good @ truth + noise
    estimate = torch.linalg.lstsq(A_good, obs).solution
    sigma = torch.linalg.svdvals(A_good).min()
    report["joint_observation"] = {
        "rank_without_region_identification": int(torch.linalg.matrix_rank(A_bad)),
        "rank_with_region_identification": int(torch.linalg.matrix_rank(A_good)),
        "permutation_observation_difference": float((A_bad @ truth - A_bad @ swapped).norm()),
        "sigma_min_good": float(sigma),
        "noise_norm": float(noise.norm()),
        "recovery_error": float((estimate-truth).norm()),
        "noise_over_sigma_bound": float(noise.norm()/sigma),
    }

    # Near dependent observations can identify a solution but remain unstable.
    sensitivity = []
    for eps in (1., 0.1, 0.01, 0.001):
        A = torch.tensor([[1., 0.], [1., eps]])
        perturb = torch.tensor([0., 0.001])
        sensitivity.append({"epsilon": eps,
                            "sigma_min": float(torch.linalg.svdvals(A).min()),
                            "solution_perturbation": float(torch.linalg.solve(A, perturb).norm())})
    report["uniqueness_without_stability"] = sensitivity

    # If teacher says .6 but independent image says truth .8, exact fit has fixed bias.
    teacher, image_truth = 0.6, 0.8
    soft = (teacher + image_truth) / 2
    trust = min(teacher + 0.2, max(teacher - 0.2, image_truth))
    report["hard_soft_trust_scalar"] = {"truth": image_truth, "teacher": teacher,
                                       "hard_error": abs(teacher-image_truth),
                                       "equal_precision_soft_error": abs(soft-image_truth),
                                       "oracle_valid_radius_trust_error": abs(trust-image_truth)}

    # Direct coarse gradient excludes detail parameter, but shared state may alter it.
    shared = torch.tensor(1., requires_grad=True)
    w = torch.tensor(0.5, requires_grad=True)
    loss = (shared-2)**2
    g_shared, g_w = torch.autograd.grad(loss, (shared, w), allow_unused=True)
    new_shared = shared.detach() - 0.1*g_shared
    report["shared_gradient"] = {"detail_parameter_gradient_is_none": g_w is None,
                                 "nonzero_head_detail_before": float((w*shared).detach()),
                                 "nonzero_head_detail_after": float((w*new_shared).detach()),
                                 "zero_head_detail_after": float((0*new_shared).detach())}

    precision = {}
    source = torch.randn(1, 4, 8, 8) * 11
    for dtype in (torch.float64, torch.float32, torch.float16, torch.bfloat16):
        z = source.to(dtype)
        detail = z - F.avg_pool2d(z, 4, 4).repeat_interleave(4, -2).repeat_interleave(4, -1)
        residual = F.avg_pool2d(detail, 4, 4)
        precision[str(dtype)] = float(residual.abs().max())
    report["finite_precision_DP_residual"] = precision
    assert report["resized_rank"]["teacher_rank"] == report["resized_rank"]["resized_rank"]
    assert report["jensen"]["constant_q_uniform_gap"] > 0
    assert abs(report["jensen"]["gauge_gap"]) < 1e-12
    assert report["joint_observation"]["recovery_error"] <= report["joint_observation"]["noise_over_sigma_bound"] + 1e-12
    path = Path(__file__)
    report["script_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    run()
