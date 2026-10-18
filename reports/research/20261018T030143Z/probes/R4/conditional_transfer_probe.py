"""Manufactured finite-distribution checks, not patient or learning evidence.

Only Python standard library; no production imports, files, weights, or data.
Run from this directory; prints its complete deterministic receipt.
"""
from __future__ import annotations

import json
import math


def close(x, y, tol=1e-12):
    assert abs(x - y) <= tol, (x, y)


def brier(p, eta):
    # Expected full two-class Brier loss, not scalar squared error.
    return 2 * ((p - eta) ** 2 + eta * (1 - eta))


def average(xs):
    return sum(xs) / len(xs)


def main():
    out = {"status": "MANUFACTURED_ENUMERATION_ONLY", "checks": {}}
    # W changes the fitted conditional mean even with unbiased unweighted T.
    eta = 0.5
    targets, weights = (0.3, 0.7), (1.0, 3.0)
    weighted = sum(t * w for t, w in zip(targets, weights)) / sum(weights)
    close(weighted, 0.6)
    close(average(targets), eta)
    out["checks"]["selection_bias"] = {
        "unweighted_optimum": average(targets), "weighted_optimum": weighted,
        "weighted_optimum_excess_brier": brier(weighted, eta) - brier(eta, eta),
    }

    # Nuisance target variation: Y mean is unchanged by hidden coin.
    eta, targets = 0.7, (0.5, 0.9)
    mean = average(targets)
    teacher = average([brier(t, eta) for t in targets])
    student = brier(mean, eta)
    variance = average([2 * (t - mean) ** 2 for t in targets])
    close(teacher - student, variance)
    out["checks"]["nuisance_denoising"] = {
        "teacher_brier": teacher, "student_brier": student,
        "improvement": teacher - student, "conditional_target_variance": variance,
    }

    # Identical target values, but hidden coin now determines the true class law.
    etas = (0.5, 0.9)
    teacher = average([brier(t, e) for t, e in zip(targets, etas)])
    student = average([brier(mean, e) for e in etas])
    covariance = average([2 * (t - mean) * (e - mean) for t, e in zip(targets, etas)])
    close(teacher - student, variance - 2 * covariance)
    close(student - teacher, variance)
    out["checks"]["privileged_information_loss"] = {
        "teacher_brier": teacher, "student_brier": student,
        "student_worsening": student - teacher,
        "same_observed_targets_as_nuisance_case": True,
    }

    # Hidden perfectly predictive bit erased by masking: no capacity fixes it.
    masked_mean = average((0.0, 1.0))
    close(masked_mean, 0.5)
    out["checks"]["erased_cue"] = {
        "full_view_brier": 0.0, "masked_best_brier": brier(masked_mean, 0.5),
        "masked_minimum_ce": math.log(2),
    }

    # Uniform dilution with nonconstant h and identical student-visible V.
    z, h, w = (0.9, 0.2), (0.1, 0.8), (1.0, 2.0)
    q = tuple((1 - hi) * 0.5 + hi * zi for hi, zi in zip(h, z))
    q_mean = sum(wi * qi for wi, qi in zip(w, q)) / sum(w)
    z_weighted = sum(wi * hi * zi for wi, hi, zi in zip(w, h, z)) / sum(wi * hi for wi, hi in zip(w, h))
    h_bar = sum(wi * hi for wi, hi in zip(w, h)) / sum(w)
    close(q_mean, (1 - h_bar) * 0.5 + h_bar * z_weighted)
    out["checks"]["q_z_decomposition"] = {
        "q_optimum": q_mean, "wh_weighted_z_optimum": z_weighted,
        "effective_h": h_bar, "unweighted_z_mean": average(z),
    }

    # Noisy biased proposal contraction, exact output mixing, no optimizer.
    eta, p, rho, bias, sigma, alpha = 0.7, 0.9, 0.5, 0.02, 0.03, 0.4
    proposal = tuple(eta + rho * (p - eta) + bias + n for n in (-sigma, sigma))
    updated = tuple((1 - alpha) * p + alpha * a for a in proposal)
    observed = average([(v - eta) ** 2 for v in updated])
    predicted = ((1 - alpha + alpha * rho) * (p - eta) + alpha * bias) ** 2 + alpha ** 2 * sigma ** 2
    close(observed, predicted)
    assert observed < (p - eta) ** 2
    out["checks"]["noisy_biased_feedback"] = {
        "initial_scalar_squared_error": (p - eta) ** 2,
        "updated_mean_scalar_squared_error": observed, "formula": predicted,
        "truth_parameters_manufactured_and_latent_in_real_setting": True,
    }

    # Common bias is unchanged by arbitrarily many identical history copies.
    repeated = [0.8] * 100
    close(average(repeated), 0.8)
    out["checks"]["history_common_bias"] = {
        "copies": len(repeated), "eta": 0.2, "history_mean": average(repeated),
        "squared_bias": (average(repeated) - 0.2) ** 2,
    }

    # Exact distillation and sharpening are different operations.
    p, eta = 0.6, 0.0
    hard = float(p > 0.5)
    assert brier(hard, eta) > brier(p, eta)
    out["checks"]["hardening_error_amplification"] = {
        "before_brier": brier(p, eta), "after_brier": brier(hard, eta),
        "self_ce_before": -p * math.log(p) - (1 - p) * math.log(1 - p),
        "hard_target_ce_after": 0.0,
    }

    # Coarse averages cannot distinguish opposite native label arrangements.
    native_a, native_b = (1.0, 0.0), (0.0, 1.0)
    close(average(native_a), average(native_b))
    out["checks"]["coarse_nullspace"] = {
        "native_a": native_a, "native_b": native_b,
        "identical_coarse_probability": average(native_a),
    }
    out["passed_checks"] = len(out["checks"])
    print(json.dumps(out, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
