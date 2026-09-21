"""Worker B empirical verification of supervision, identifiability, and historical receipts.
Runs purely on synthetic data and metadata inspection. Does not open manual GT arrays.
"""
from __future__ import annotations

import json
import math
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from self_audit_maskfree.data.firewall import assert_image_only, forbidden_reason, MaskAccessError
from self_audit_pseudolabel.system_v3 import CinePseudoTeacher, NUM_CLASSES, UNKNOWN
from shared_benchmark.adapter import _background, _resolve_enclosure, _resolve_rv
from shared_benchmark.region_graph import build_region_graph
from shared_benchmark.semantic_contract import BG, RV, MYO, LV, VOID


def probe_v3_permutation_symmetry():
    torch.manual_seed(42)
    teacher = CinePseudoTeacher(width=8, appearance_dim=16, motion_dim=8, fused_dim=24, k=4).eval()
    
    # Synthetic inputs
    prev = torch.rand(1, 3, 32, 32)
    cur = torch.rand(1, 3, 32, 32)
    nxt = torch.rand(1, 3, 32, 32)
    
    with torch.no_grad():
        out_base = teacher(prev, cur, nxt, min_prob=0.0, min_margin=0.0)
        base_probs = out_base["soft_label"] # [1, 4, 32, 32]
        
        # Test permutation equivariance
        perm = [1, 2, 3, 0] # non-trivial permutation
        perm_matrix = torch.eye(4)[perm] # [4, 4]
        
        # Clone model and permute semantic layer 2 weights and bias
        teacher_perm = CinePseudoTeacher(width=8, appearance_dim=16, motion_dim=8, fused_dim=24, k=4).eval()
        teacher_perm.load_state_dict(teacher.state_dict())
        with torch.no_grad():
            teacher_perm.semantic[2].weight.copy_(perm_matrix @ teacher.semantic[2].weight)
            teacher_perm.semantic[2].bias.copy_(perm_matrix @ teacher.semantic[2].bias)
            
        out_perm = teacher_perm(prev, cur, nxt, min_prob=0.0, min_margin=0.0)
        perm_probs = out_perm["soft_label"]
        
        # The predicted probabilities under teacher_perm must match exactly the permuted probabilities of teacher
        expected_perm_probs = base_probs[:, perm, :, :]
        max_diff = (perm_probs - expected_perm_probs).abs().max().item()
        
    return {
        "permutation_equivariance_exact": bool(max_diff < 1e-6),
        "max_diff": max_diff,
        "base_class_probs_spatial_mean": base_probs.mean(dim=(0, 2, 3)).tolist(),
        "permuted_class_probs_spatial_mean": perm_probs.mean(dim=(0, 2, 3)).tolist(),
        "unseeded_top1_mean": out_base["semantic_prob"].max(dim=-1).values.mean().item(),
        "unseeded_valid_fraction_default_gate": (out_base["semantic_prob"].max(dim=-1).values >= 0.70).float().mean().item(),
    }


def probe_topology_adapter():
    # Build a synthetic short-axis partition:
    # 0: Border / Background
    # 1: Myocardium ring (enclosing cavity)
    # 2: Left Ventricle cavity (inside ring)
    # 3: Right Ventricle crescent (adjacent to myocardium ring exterior)
    
    H, W = 64, 64
    partition = np.zeros((H, W), dtype=np.int32)
    
    y, x = np.ogrid[:H, :W]
    cy, cx = 32, 32
    dist_sq = (y - cy) ** 2 + (x - cx) ** 2
    
    # Myocardium ring: radius 8 to 16
    myo_mask = (dist_sq >= 8**2) & (dist_sq < 16**2)
    # LV cavity: radius < 8
    lv_mask = dist_sq < 8**2
    # RV crescent: adjacent on patient-right (x < 32)
    rv_mask = (dist_sq >= 16**2) & (dist_sq < 22**2) & (x < 30)
    
    partition[myo_mask] = 1
    partition[lv_mask] = 2
    partition[rv_mask] = 3
    # rest is 0 (touches border)
    
    graph = build_region_graph(partition)
    semantic = np.full((H, W), VOID, dtype=np.uint8)
    assignments = {}
    assignment_reasons = {}
    unresolved_reasons = {}
    
    bg_stat = _background(graph, semantic, assignments, assignment_reasons, unresolved_reasons)
    enc_stat = _resolve_enclosure(graph, semantic, assignments, assignment_reasons, unresolved_reasons)
    rv_stat = _resolve_rv(graph, semantic, assignments, assignment_reasons, unresolved_reasons)
    
    assigned_roles = {comp_key: role for comp_key, role in assignments.items()}
    
    return {
        "bg_status": bg_stat,
        "enclosure_status": enc_stat,
        "rv_status": rv_stat,
        "assignments": assigned_roles,
        "reasons": assignment_reasons,
        "correct_resolution": (
            assignments.get("c0_p0") == BG and
            assignments.get("c1_p0") == MYO and
            assignments.get("c2_p0") == LV and
            assignments.get("c3_p0") == RV
        )
    }


def probe_firewall_rules():
    test_cases = [
        ("patient001/patient001_frame01.nii.gz", True),
        ("patient001/patient001_frame01_gt.nii.gz", False),
        ("patient001/masks/patient001_frame01.nii.gz", False),
        ("patient001/patient001_manual.nii.gz", False),
        ("patient001/Info.cfg", False),
        ("patient001/diagnosis.csv", False),
        ("patient001/patient001_crop128.nii.gz", True),  # Path looks valid even if pre-cropped!
    ]
    results = {}
    for path_str, should_pass in test_cases:
        reason = forbidden_reason(path_str)
        is_allowed = (reason is None)
        results[path_str] = {
            "allowed": is_allowed,
            "expected_allowed": should_pass,
            "reason": reason,
            "passed_contract": is_allowed == should_pass
        }
    return results


def summarize_historical_receipts():
    receipts_summary = {
        "candidate_c_synthetic_flow": {
            "path": "reports/candidate_c/synthetic_flow_receipt.json",
            "mode": "candidate_c",
            "status": "PASS",
            "elapsed_seconds": 18.574937105178833,
            "acdc_replay_failures": 0,
            "mnms_replay_failures": 0,
            "acdc_feasible_solves": 4,
            "acdc_fallback_solves": 4,
            "acdc_invalidations": 15,
            "ordinary_real_evidence_attempts": 8,
            "real_evidence_attempts": 16,
        },
        "candidate_c_post_cleanup": {
            "path": "reports/candidate_c/post_cleanup_test_receipt.json",
            "exit_code": 0,
            "elapsed_seconds": 60.1942982673645,
            "file_count": 72,
            "critical_imports": 4,
        },
        "candidate_c_final_pytest": {
            "path": "reports/candidate_c/final_test_receipt.json",
            "exit_code": 0,
            "elapsed_seconds": 498.69305396080017,
            "compiled_files_count": 21,
            "critical_imports": 7,
        },
        "maskfree150_validation_receipt": {
            "path": "reports/maskfree150/validation_receipt.json",
            "commit": "7607eeac7b185998af030af57208a0902632706a",
            "focused_checks": {"tests": 45, "errors": 0, "failures": 0, "time": 16.279},
            "reference_checks": {"tests": 5, "errors": 0, "failures": 0, "time": 13.506},
            "software_probes_acdc": {"units": 16, "verification_rows": 16, "scientific_outcome": "label_generation_produced_valid_foreground"},
            "software_probes_mnms": {"units": 16, "verification_rows": 16, "scientific_outcome": "label_generation_produced_valid_foreground"},
            "real_execution": {"acdc_150": "NOT STARTED", "mnms_150": "NOT STARTED", "rtx4070_physical_batch8": "NOT STARTED"},
        },
        "maskfree150_local_baseline_profile": {
            "path": "reports/maskfree150/local_baseline/profile_report.json",
            "units": 160,
            "candidates": 640,
            "fits": 640,
            "semantic_unresolved": 159,
            "accepted_edits": 64,
            "score_calls": 5866,
            "unresolved_rate": 159 / 160.0,
        }
    }
    return receipts_summary


def main():
    print("Running Worker B probes...")
    perm_res = probe_v3_permutation_symmetry()
    print("V3 permutation probe:", perm_res)
    
    topo_res = probe_topology_adapter()
    print("Topology adapter probe:", topo_res)
    
    fw_res = probe_firewall_rules()
    print("Firewall probe:", fw_res)
    
    receipts = summarize_historical_receipts()
    print("Historical receipts summary prepared.")
    
    full_evidence = {
        "v3_permutation_symmetry": perm_res,
        "topology_adapter": topo_res,
        "firewall_rules": fw_res,
        "historical_receipts": receipts,
    }
    
    out_path = Path("reports/astra_v3/workers/B_evidence/supervision_identifiability_evidence.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(full_evidence, indent=2), encoding="utf-8")
    print(f"Evidence written to {out_path}")


if __name__ == "__main__":
    main()
