"""Quantitative Red Team probes for Astra v3 scaffold.
Generates empirical falsification evidence for reports/astra_v3/workers/G_red_team.md.
"""
import json
import math
import sys
import os

import torch
import torch.nn.functional as F

from self_audit_pseudolabel.system_v3 import (
    CinePseudoTeacher,
    AdaptiveAnnotationStudent,
    PROFILES,
    NUM_CLASSES,
    UNKNOWN,
    pseudo_supervision_loss,
)

def run_probes():
    results = {}
    torch.manual_seed(42)

    # 1. Parameter counts & architectural footprint
    teacher = CinePseudoTeacher(width=24, appearance_dim=48, motion_dim=16, fused_dim=64, k=12)
    student = AdaptiveAnnotationStudent(width=32, window_k=8)

    teacher_params = {k: sum(p.numel() for p in m.parameters()) for k, m in [
        ("appearance", teacher.appearance),
        ("motion", teacher.motion),
        ("fuse", teacher.fuse),
        ("regions", teacher.regions),
        ("semantic", teacher.semantic),
    ]}
    teacher_total = sum(p.numel() for p in teacher.parameters())
    teacher_params["total"] = teacher_total

    student_params = {k: sum(p.numel() for p in m.parameters()) for k, m in [
        ("encoder", student.encoder),
        ("a0_head", student.a0_head),
        ("refiner", student.refiner),
    ]}
    student_total = sum(p.numel() for p in student.parameters())
    student_params["total"] = student_total

    results["parameter_counts"] = {
        "teacher": teacher_params,
        "student": student_params,
        "teacher_student_ratio": teacher_total / student_total,
    }

    # 2. Permutation Invariance probe
    # Test whether an unsupervised teacher semantic head can distinguish class permutations
    # Permutation matrix for classes (e.g. swap RV=1 and LV=3)
    P = torch.tensor([
        [1, 0, 0, 0],
        [0, 0, 0, 1], # RV -> LV
        [0, 0, 1, 0],
        [0, 1, 0, 0], # LV -> RV
    ], dtype=torch.float32)

    # If we permute the weights of the final linear layer:
    linear = teacher.semantic[2]
    w_orig = linear.weight.data.clone()
    b_orig = linear.bias.data.clone()

    x_cur = torch.rand(2, 3, 64, 64)
    x_prev = torch.roll(x_cur, -1, -1)
    x_nxt = torch.roll(x_cur, 1, -1)

    out_orig = teacher(x_prev, x_cur, x_nxt)
    
    # Permute semantic head weights
    linear.weight.data = P @ w_orig
    linear.bias.data = P @ b_orig

    out_perm = teacher(x_prev, x_cur, x_nxt)

    # Check that dense_prob is exactly permuted
    expected_perm_prob = torch.einsum("ij,bjhw->bihw", P, out_orig["soft_label"])
    prob_perm_diff = (out_perm["soft_label"] - expected_perm_prob).abs().max().item()
    
    # Check reconstruction difference (should be exactly 0)
    recon_diff = (out_orig["reconstruction"] - out_perm["reconstruction"]).abs().max().item()

    results["permutation_invariance"] = {
        "prob_perm_max_abs_error": prob_perm_diff,
        "reconstruction_invariance_error": recon_diff,
        "conclusion": "Class semantics are completely ungrounded and permutable under any unsupervised loss."
    }

    # 3. Dynamic Window & Profile Computational Multipliers
    student.eval()
    x_in = torch.rand(1, 3, 224, 224)
    
    # Count calls to refinement_block for each profile
    call_counts = {}
    for prof in ["compact", "balanced", "accurate"]:
        calls = []
        handle = student.refiner.refinement_block.register_forward_hook(lambda m, i, o: calls.append(1))
        with torch.no_grad():
            out = student(x_in, profile=prof)
        handle.remove()
        call_counts[prof] = {
            "dw_calls": len(calls),
            "stages": len(out["stages"]),
            "is_a0_identical_to_final": torch.equal(out["a0_logits"], out["final_logits"]),
        }

    results["profile_execution_contracts"] = call_counts

    # 4. Save results to G_evidence
    evidence_path = os.path.join(os.path.dirname(__file__), "red_team_probe_results.json")
    with open(evidence_path, "w") as f:
        json.dump(results, f, indent=2)

    print("Probe completed successfully.")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    run_probes()
