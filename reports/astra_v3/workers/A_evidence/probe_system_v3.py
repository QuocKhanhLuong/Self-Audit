"""Diagnostic and synthetic probes for CinePseudoTeacher and AdaptiveAnnotationStudent.

Evaluates:
1. Intermediate tensor shapes, softmax axes, resize modes, gradients.
2. Motion branch: intensity differences vs optical flow, temporal swap, zero motion.
3. Appearance branch: z-channel permutation, reconstruction head behavior.
4. Prototype dynamics: collapse, assignment entropy, permutation across seeds.
5. Semantic head and evidence: unseeded random logits, evidence_logits intervention, UNKNOWN rate.
6. Boundary mixture: prototype blending vs spatial validity gating.
7. Gradient flow: backprop through teacher and student.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

from self_audit_pseudolabel.system_v3 import (
    AdaptiveAnnotationStudent,
    CinePseudoTeacher,
    UNKNOWN,
    pool_regions,
    pseudo_supervision_loss,
)

EVIDENCE_DIR = Path(__file__).resolve().parent


def probe_trace_shapes_and_resizing():
    """Traces full forward pass intermediate shapes and interpolation modes."""
    results = {}
    torch.manual_seed(42)
    teacher = CinePseudoTeacher(width=24, appearance_dim=48, motion_dim=16, fused_dim=64, k=12)
    teacher.eval()

    B, C_in, H, W = 2, 3, 64, 64
    cur = torch.randn(B, C_in, H, W)
    prev = torch.randn(B, C_in, H, W)
    nxt = torch.randn(B, C_in, H, W)

    # 1. Appearance encoder
    h = teacher.appearance.e2(teacher.appearance.e1(teacher.appearance.e0(cur)))
    f_app = teacher.appearance.out(h)
    recon_interp = F.interpolate(f_app, size=cur.shape[-2:], mode="bilinear", align_corners=False)
    recon = teacher.appearance.recon(recon_interp)

    # 2. Motion branch
    p, c, n = prev[:, 1:2], cur[:, 1:2], nxt[:, 1:2]
    raw_mot = torch.cat([c - p, n - c, (c - p).abs(), (n - c).abs()], 1)
    f_mot = teacher.motion.net(raw_mot)

    # 3. Fusion
    fused_cat = torch.cat([f_app, f_mot], 1)
    fused = teacher.fuse(fused_cat)

    # 4. Prototypes
    f_norm = F.normalize(fused, dim=1)
    p_norm = F.normalize(teacher.regions.p, dim=1)
    proto_logits = torch.einsum("bchw,kc->bkhw", f_norm, p_norm) / teacher.regions.temperature
    q = proto_logits.softmax(1)

    # 5. Pooling
    r = pool_regions(q, fused, cur[:, 1:2], f_mot)

    # 6. Semantic logits
    logits = teacher.semantic(r)
    prob = logits.softmax(-1)

    # 7. Validity and dense projections
    top2 = prob.topk(2, -1).values
    min_prob = 0.70
    min_margin = 0.20
    valid_region = (top2[..., 0] >= min_prob) & ((top2[..., 0] - top2[..., 1]) >= min_margin)
    dense_prob_low = torch.einsum("bkhw,bkc->bchw", q, prob)
    dense_valid_low = torch.einsum("bkhw,bk->bhw", q, valid_region.to(q.dtype)) >= 0.5
    dense_prob = F.interpolate(dense_prob_low, cur.shape[-2:], mode="bilinear", align_corners=False)
    dense_valid = F.interpolate(dense_valid_low[:, None].float(), cur.shape[-2:], mode="nearest")[:, 0].bool()
    label = dense_prob.argmax(1)
    pseudo = torch.where(dense_valid, label, torch.full_like(label, UNKNOWN))

    results["cur_shape"] = list(cur.shape)
    results["f_app_shape"] = list(f_app.shape)
    results["recon_shape"] = list(recon.shape)
    results["raw_motion_shape"] = list(raw_mot.shape)
    results["f_motion_shape"] = list(f_mot.shape)
    results["fused_shape"] = list(fused.shape)
    results["proto_q_shape"] = list(q.shape)
    results["q_softmax_axis"] = 1
    results["r_pooled_shape"] = list(r.shape)
    results["semantic_logits_shape"] = list(logits.shape)
    results["semantic_softmax_axis"] = -1
    results["valid_region_shape"] = list(valid_region.shape)
    results["dense_prob_low_shape"] = list(dense_prob_low.shape)
    results["dense_valid_low_shape"] = list(dense_valid_low.shape)
    results["dense_prob_shape"] = list(dense_prob.shape)
    results["dense_valid_shape"] = list(dense_valid.shape)
    results["pseudo_shape"] = list(pseudo.shape)
    results["dense_prob_interpolate_mode"] = "bilinear (align_corners=False)"
    results["dense_valid_interpolate_mode"] = "nearest"

    # Test variable spatial resolutions
    resolution_tests = {}
    for test_res in [(32, 32), (64, 64), (128, 128), (216, 216), (224, 224), (33, 33), (35, 35)]:
        th, tw = test_res
        c_test = torch.randn(1, 3, th, tw)
        p_test = torch.randn(1, 3, th, tw)
        n_test = torch.randn(1, 3, th, tw)
        try:
            out_test = teacher(p_test, c_test, n_test)
            status = "OK"
            out_pseudo_shape = list(out_test["pseudo_label"].shape)
        except Exception as e:
            status = f"FAILED: {type(e).__name__}: {e}"
            out_pseudo_shape = None
        resolution_tests[f"{th}x{tw}"] = {
            "status": status,
            "pseudo_shape": out_pseudo_shape,
        }
    results["resolution_compatibility"] = resolution_tests
    return results


def probe_motion_interventions():
    """Tests zero motion, temporal swap, and channel shuffling on MotionBranch."""
    results = {}
    torch.manual_seed(42)
    teacher = CinePseudoTeacher(width=24, appearance_dim=48, motion_dim=16, fused_dim=64, k=12)
    teacher.eval()

    cur = torch.randn(1, 3, 64, 64)
    # Synthetic moving disk to simulate myocardium displacement
    prev = torch.roll(cur, shifts=-2, dims=-1)  # shifted horizontally
    nxt = torch.roll(cur, shifts=2, dims=-1)

    with torch.no_grad():
        out_base = teacher(prev, cur, nxt)
        m_base = out_base["motion_features"]

        # 1. Zero motion (prev == cur == nxt)
        out_zero = teacher(cur, cur, cur)
        m_zero = out_zero["motion_features"]
        # Check if zero motion produces nonzero features due to conv bias / norm
        zero_norm = float(m_zero.norm().item())
        zero_std = float(m_zero.std().item())
        zero_max_diff = float((m_base - m_zero).abs().max().item())

        # 2. Temporal swap (swap prev and nxt)
        out_swap = teacher(nxt, cur, prev)
        m_swap = out_swap["motion_features"]
        swap_diff_l1 = float((m_base - m_swap).abs().mean().item())
        swap_rel_diff = float((m_base - m_swap).norm() / m_base.norm())

        # 3. Channel shuffle in motion branch raw input
        p, c, n = prev[:, 1:2], cur[:, 1:2], nxt[:, 1:2]
        raw = torch.cat([c - p, n - c, (c - p).abs(), (n - c).abs()], 1)
        raw_shuffled = raw[:, [1, 0, 3, 2]]  # swap forward/backward differences
        m_shuffled = teacher.motion.net(raw_shuffled)
        shuf_diff_l1 = float((m_base - m_shuffled).abs().mean().item())

        # 4. Pure intensity scaling / flicker (no spatial displacement, just global brightness jump)
        cur_bright = cur.clone()
        cur_bright[:, 1:2] = cur_bright[:, 1:2] * 1.5
        out_flicker = teacher(cur, cur_bright, cur)
        m_flicker = out_flicker["motion_features"]
        flicker_diff_from_zero = float((m_flicker - m_zero).norm().item())

    results["zero_motion_feature_norm"] = zero_norm
    results["zero_motion_feature_spatial_std"] = zero_std
    results["zero_vs_base_max_abs_diff"] = zero_max_diff
    results["temporal_swap_relative_l2_diff"] = swap_rel_diff
    results["temporal_swap_mean_abs_diff"] = swap_diff_l1
    results["channel_shuffle_mean_abs_diff"] = shuf_diff_l1
    results["pure_flicker_motion_norm_vs_zero"] = flicker_diff_from_zero
    results["motion_is_optical_flow"] = False
    results["motion_is_pixel_difference"] = True
    return results


def probe_appearance_z_interventions():
    """Tests z-channel permutation and appearance reconstruction."""
    results = {}
    torch.manual_seed(42)
    teacher = CinePseudoTeacher(width=24, appearance_dim=48, motion_dim=16, fused_dim=64, k=12)
    teacher.eval()

    cur = torch.randn(1, 3, 64, 64)
    prev = torch.roll(cur, -1, -1)
    nxt = torch.roll(cur, 1, -1)

    with torch.no_grad():
        out_base = teacher(prev, cur, nxt)
        app_base = out_base["appearance_features"]
        recon_base = out_base["reconstruction"]

        # Permute z channels: [z-1, z, z+1] -> [z+1, z, z-1] (reflection along z)
        cur_z_flipped = cur[:, [2, 1, 0], :, :]
        prev_z_flipped = prev[:, [2, 1, 0], :, :]
        nxt_z_flipped = nxt[:, [2, 1, 0], :, :]
        out_z_flip = teacher(prev_z_flipped, cur_z_flipped, nxt_z_flipped)
        app_z_flip = out_z_flip["appearance_features"]
        z_flip_rel_diff = float((app_base - app_z_flip).norm() / app_base.norm())

        # Permute center slice: put z+1 in center: [z-1, z+1, z]
        cur_z_scrambled = cur[:, [0, 2, 1], :, :]
        out_z_scrambled = teacher(prev, cur_z_scrambled, nxt)
        z_scramble_rel_diff = float((app_base - out_z_scrambled["appearance_features"]).norm() / app_base.norm())

        # Reconstruction correlation with center slice
        recon_center_err = float(F.mse_loss(recon_base, cur[:, 1:2]).item())

    results["z_flipped_relative_l2_diff"] = z_flip_rel_diff
    results["z_scrambled_relative_l2_diff"] = z_scramble_rel_diff
    results["recon_mse_initial"] = recon_center_err
    return results


def probe_evidence_and_unknown_behavior():
    """Tests UNKNOWN abstention rate with unseeded, random, and strong evidence_logits."""
    results = {}
    torch.manual_seed(42)
    teacher = CinePseudoTeacher(width=24, appearance_dim=48, motion_dim=16, fused_dim=64, k=12)
    teacher.eval()

    B, H, W = 4, 64, 64
    cur = torch.randn(B, 3, H, W)
    prev = torch.roll(cur, -1, -1)
    nxt = torch.roll(cur, 1, -1)

    with torch.no_grad():
        # 1. Default forward (evidence_logits = None, random semantic head)
        out_none = teacher(prev, cur, nxt)
        valid_none = out_none["valid"]
        valid_fraction_none = float(valid_none.float().mean().item())
        unknown_fraction_none = float((out_none["pseudo_label"] == UNKNOWN).float().mean().item())
        prob_max_mean_none = float(out_none["semantic_prob"].max(-1).values.mean().item())

        # 2. Random evidence logits (small magnitude, N(0, 1))
        ev_random = torch.randn(B, 12, 4)
        out_rand = teacher(prev, cur, nxt, evidence_logits=ev_random)
        valid_fraction_rand = float(out_rand["valid"].float().mean().item())
        unknown_fraction_rand = float((out_rand["pseudo_label"] == UNKNOWN).float().mean().item())

        # 3. Strong evidence logits (+10 on a designated class per prototype)
        ev_strong = torch.zeros(B, 12, 4)
        # Assign prototypes to classes 0..3 cyclically
        for k_idx in range(12):
            ev_strong[:, k_idx, k_idx % 4] = 10.0
        out_strong = teacher(prev, cur, nxt, evidence_logits=ev_strong)
        valid_fraction_strong = float(out_strong["valid"].float().mean().item())
        unknown_fraction_strong = float((out_strong["pseudo_label"] == UNKNOWN).float().mean().item())
        classes_present_strong = sorted(torch.unique(out_strong["pseudo_label"]).tolist())

        # 4. Sensitivity to min_prob and min_margin thresholds
        threshold_sweep = {}
        for min_p, min_m in [(0.50, 0.10), (0.60, 0.15), (0.70, 0.20), (0.80, 0.25), (0.90, 0.30)]:
            out_thresh = teacher(prev, cur, nxt, min_prob=min_p, min_margin=min_m)
            valid_frac = float(out_thresh["valid"].float().mean().item())
            threshold_sweep[f"p{min_p}_m{min_m}"] = {
                "valid_fraction": valid_frac,
                "unknown_fraction": 1.0 - valid_frac,
            }

    results["unseeded_default"] = {
        "valid_fraction": valid_fraction_none,
        "unknown_fraction": unknown_fraction_none,
        "mean_max_prototype_prob": prob_max_mean_none,
    }
    results["random_evidence_n01"] = {
        "valid_fraction": valid_fraction_rand,
        "unknown_fraction": unknown_fraction_rand,
    }
    results["strong_cyclic_evidence"] = {
        "valid_fraction": valid_fraction_strong,
        "unknown_fraction": unknown_fraction_strong,
        "classes_present": classes_present_strong,
    }
    results["threshold_sensitivity"] = threshold_sweep
    return results


def probe_prototype_collapse_and_permutation():
    """Diagnoses prototype utilization, entropy, and permutation across initializations."""
    results = {}
    seeds = [1, 2, 3, 42, 100]
    B, H, W = 2, 64, 64
    cur = torch.randn(B, 3, H, W)
    prev = torch.roll(cur, -1, -1)
    nxt = torch.roll(cur, 1, -1)

    seed_reports = {}
    for s in seeds:
        torch.manual_seed(s)
        teacher = CinePseudoTeacher(width=24, appearance_dim=48, motion_dim=16, fused_dim=64, k=12).eval()
        with torch.no_grad():
            out = teacher(prev, cur, nxt)
            q = out["region_prob"]  # [B, K, H/4, W/4]
            # Spatial mass per prototype
            mass = q.sum(dim=(-2, -1))  # [B, K]
            mean_mass = mass.mean(dim=0).tolist()
            # Active prototypes (mass > 5% of average equal share)
            equal_share = (H // 4) * (W // 4) / 12.0
            active_count = int((mass.mean(0) > (0.1 * equal_share)).sum().item())
            # Spatial assignment entropy per pixel: -sum(q * log(q + 1e-8))
            entropy = -(q * (q + 1e-8).log()).sum(dim=1).mean().item()
            # Max possible entropy is log(12) = 2.4849
            norm_entropy = float(entropy / torch.tensor(12.0).log().item())

        seed_reports[f"seed_{s}"] = {
            "active_prototypes": active_count,
            "mean_mass": [round(m, 2) for m in mean_mass],
            "norm_entropy": round(norm_entropy, 4),
        }
    results["seed_sensitivity"] = seed_reports
    return results


def probe_boundary_mixture_and_dense_validity():
    """Evaluates whether boundary pixels with 50/50 prototype mixture are incorrectly validated."""
    results = {}
    B, K, H, W = 1, 12, 16, 16

    # Create synthetic assignment: prototype 0 dominates left half, prototype 1 dominates right half
    # Center column has 50% prototype 0 and 50% prototype 1
    q = torch.zeros(B, K, H, W)
    q[0, 0, :, :8] = 0.95
    q[0, 1, :, :8] = 0.05
    q[0, 0, :, 8:] = 0.05
    q[0, 1, :, 8:] = 0.95
    # Transition at boundary column 7 and 8
    q[0, 0, :, 7] = 0.50
    q[0, 1, :, 7] = 0.50
    q[0, 0, :, 8] = 0.50
    q[0, 1, :, 8] = 0.50

    # Case A: Both prototype 0 and 1 are high-confidence (e.g. proto 0 is MYO, proto 1 is LV)
    prob_A = torch.zeros(B, K, 4)
    prob_A[0, 0, 2] = 0.90  # Proto 0: 90% MYO
    prob_A[0, 0, 0] = 0.10
    prob_A[0, 1, 3] = 0.90  # Proto 1: 90% LV
    prob_A[0, 1, 0] = 0.10

    # Evaluate dense_valid_low and dense_prob_low
    top2_A = prob_A.topk(2, -1).values
    valid_region_A = (top2_A[..., 0] >= 0.70) & ((top2_A[..., 0] - top2_A[..., 1]) >= 0.20)
    dense_prob_low_A = torch.einsum("bkhw,bkc->bchw", q, prob_A)
    dense_valid_low_A = torch.einsum("bkhw,bk->bhw", q, valid_region_A.to(q.dtype)) >= 0.5

    # Check the boundary column 7
    border_valid = bool(dense_valid_low_A[0, 0, 7].item())
    border_probs = dense_prob_low_A[0, :, 0, 7].tolist()
    border_margin = abs(border_probs[2] - border_probs[3])

    results["boundary_pixel_marked_valid"] = border_valid
    results["boundary_pixel_class_probs"] = [round(p, 4) for p in border_probs]
    results["boundary_pixel_top2_margin"] = round(border_margin, 4)
    results["mixture_dilution_flaw_detected"] = (border_valid and border_margin < 0.05)
    return results


def probe_gradient_flow():
    """Tests gradient flow to all teacher and student parameters."""
    results = {}
    torch.manual_seed(42)
    teacher = CinePseudoTeacher(width=8, appearance_dim=16, motion_dim=8, fused_dim=24, k=8)
    student = AdaptiveAnnotationStudent(width=16, window_k=4)

    cur = torch.randn(1, 3, 32, 32, requires_grad=True)
    prev = torch.roll(cur, -1, -1)
    nxt = torch.roll(cur, 1, -1)

    # 1. Forward teacher
    teacher_out = teacher(prev, cur, nxt)
    pseudo = teacher_out["pseudo_label"]
    valid = teacher_out["valid"]

    # 2. Forward student
    student_out = student(cur, profile="balanced")

    # 3. Compute pseudo_supervision_loss
    # Force some pixels to be valid to test backward pass
    valid_synthetic = torch.zeros_like(pseudo, dtype=torch.bool)
    valid_synthetic[:, 4:28, 4:28] = True
    target_synthetic = torch.zeros_like(pseudo)
    target_synthetic[:, 4:28, 4:28] = 1

    loss = pseudo_supervision_loss(student_out, target_synthetic, valid_synthetic)
    loss.backward()

    # Check student gradients
    student_grads = {}
    for name, param in student.named_parameters():
        if param.grad is not None:
            student_grads[name] = float(param.grad.norm().item())
        else:
            student_grads[name] = None

    # Check teacher gradients: Does student loss flow to teacher?
    # Because target was pseudo.detach() / frozen, teacher params should have NO gradients.
    teacher_grads_from_student = {}
    for name, param in teacher.named_parameters():
        teacher_grads_from_student[name] = float(param.grad.norm().item()) if param.grad is not None else None

    # 4. Now test direct hypothetical loss on teacher's soft_label
    teacher.zero_grad()
    soft_label = teacher(prev, cur, nxt)["soft_label"]
    toy_loss = soft_label[:, 1].mean()
    toy_loss.backward()

    teacher_grads_direct = {}
    for name, param in teacher.named_parameters():
        teacher_grads_direct[name] = float(param.grad.norm().item()) if param.grad is not None else None

    results["student_has_nonzero_grads"] = all(g is not None and g > 0 for g in student_grads.values())
    results["teacher_grads_from_student_loss"] = teacher_grads_from_student
    results["teacher_grads_direct_loss"] = teacher_grads_direct
    results["pseudo_is_detached_integer"] = (pseudo.dtype == torch.long)
    return results


def run_all_probes():
    print("Running system_v3 synthetic probes...")
    all_results = {}
    all_results["trace_shapes"] = probe_trace_shapes_and_resizing()
    all_results["motion"] = probe_motion_interventions()
    all_results["appearance"] = probe_appearance_z_interventions()
    all_results["evidence_and_unknown"] = probe_evidence_and_unknown_behavior()
    all_results["prototype_dynamics"] = probe_prototype_collapse_and_permutation()
    all_results["boundary_mixture"] = probe_boundary_mixture_and_dense_validity()
    all_results["gradient_flow"] = probe_gradient_flow()

    out_file = EVIDENCE_DIR / "probe_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {out_file}")
    return all_results


if __name__ == "__main__":
    run_all_probes()
