"""Decisive proposed regression tests for CinePseudoTeacher and AdaptiveAnnotationStudent.

These tests demonstrate both the existing system behavior/flaws and verify
the proposed fixes (e.g. dense probability margin gating, motion invariants).
"""
import pytest
import torch
import torch.nn.functional as F

from self_audit_pseudolabel.system_v3 import (
    AdaptiveAnnotationStudent,
    CinePseudoTeacher,
    UNKNOWN,
    pool_regions,
    pseudo_supervision_loss,
)


def _cine(batch=1, h=32, w=32):
    torch.manual_seed(17)
    cur = torch.rand(batch, 3, h, w)
    prev = torch.roll(cur, -1, -1)
    nxt = torch.roll(cur, 1, -1)
    return prev, cur, nxt


def test_teacher_abstains_100_percent_without_evidence_seed():
    """Confirms unseeded teacher abstains (valid=0, pseudo=255) across batch.

    Because an unseeded semantic MLP outputs near-uniform probabilities (~0.25),
    none exceed min_prob=0.70. The teacher abstains 100% rather than producing
    corrupted pseudo-labels.
    """
    teacher = CinePseudoTeacher(width=8, appearance_dim=16, motion_dim=8, fused_dim=24, k=8).eval()
    prev, cur, nxt = _cine(batch=2, h=32, w=32)
    with torch.no_grad():
        out = teacher(prev, cur, nxt)
    # Valid should be identically False
    assert not out["valid"].any(), "Unseeded teacher unexpectedly generated valid predictions without evidence!"
    assert (out["pseudo_label"] == UNKNOWN).all(), "All pixels must be UNKNOWN when valid is False"


def test_strong_evidence_enables_valid_pseudolabels():
    """Confirms teacher produces valid pseudo-labels when evidence_logits is provided."""
    teacher = CinePseudoTeacher(width=8, appearance_dim=16, motion_dim=8, fused_dim=24, k=8).eval()
    prev, cur, nxt = _cine(batch=1, h=32, w=32)
    ev = torch.zeros(1, 8, 4)
    # Give strong prior for prototypes 0, 1, 2, 3
    for k in range(8):
        ev[:, k, k % 4] = 15.0
    with torch.no_grad():
        out = teacher(prev, cur, nxt, evidence_logits=ev)
    assert out["valid"].any(), "Strong evidence must activate valid regions"
    assert set(torch.unique(out["pseudo_label"]).tolist()).issubset({0, 1, 2, 3, UNKNOWN})
    # At least some non-unknown classes must be present
    present = [c for c in torch.unique(out["pseudo_label"]).tolist() if c != UNKNOWN]
    assert len(present) > 0, "Non-unknown classes must be predicted under strong evidence"


def test_motion_branch_is_pixel_difference_not_optical_flow():
    """Proves MotionBranch detects static intensity flickering as motion."""
    teacher = CinePseudoTeacher(width=8, appearance_dim=16, motion_dim=8, fused_dim=24, k=8).eval()
    cur = torch.ones(1, 3, 32, 32) * 0.5
    # True static scene
    static_motion = teacher.motion(cur, cur, cur)

    # Static scene with global flicker (contrast change, zero displacement)
    flicker_cur = cur * 1.5
    flicker_motion = teacher.motion(cur, flicker_cur, cur)

    # If this were optical flow, displacement would be zero.
    # Because it is intensity difference, flicker generates huge spurious motion features.
    diff = (flicker_motion - static_motion).norm().item()
    assert diff > 1.0, f"Expected large spurious difference from intensity flicker, got {diff}"


def test_reconstruction_head_is_isolated_from_semantic_gradient():
    """Confirms appearance reconstruction head receives zero gradient from semantic loss."""
    teacher = CinePseudoTeacher(width=8, appearance_dim=16, motion_dim=8, fused_dim=24, k=8)
    prev, cur, nxt = _cine(batch=1, h=32, w=32)
    out = teacher(prev, cur, nxt)
    # Direct loss on semantic soft label
    loss = out["soft_label"][:, 0].mean()
    loss.backward()

    assert teacher.appearance.recon.weight.grad is None, "Recon head must have NULL grad from semantic loss"
    assert teacher.appearance.recon.bias.grad is None, "Recon head bias must have NULL grad from semantic loss"


def test_boundary_mixture_dilution_vulnerability():
    """Proves that region-level gating permits ambiguous 50/50 mixture pixels at boundaries.

    Proposed regression test: verify that a pixel-level dense margin check is required
    to reject ambiguous boundary pixels.
    """
    B, K, H, W = 1, 4, 8, 8
    q = torch.zeros(B, K, H, W)
    # Pixel (0, 0) is a 50/50 mix of Proto 0 (MYO) and Proto 1 (LV)
    q[0, 0, 0, 0] = 0.5
    q[0, 1, 0, 0] = 0.5
    q[0, :, 1:, :] = 0.25

    # Proto 0 is 100% MYO (class 2), Proto 1 is 100% LV (class 3)
    prob = torch.zeros(B, K, 4)
    prob[0, 0, 2] = 1.0
    prob[0, 1, 3] = 1.0

    # System v3 region-level validity check
    top2 = prob.topk(2, -1).values
    valid_region = (top2[..., 0] >= 0.70) & ((top2[..., 0] - top2[..., 1]) >= 0.20)
    dense_valid_low = torch.einsum("bkhw,bk->bhw", q, valid_region.to(q.dtype)) >= 0.5
    dense_prob_low = torch.einsum("bkhw,bkc->bchw", q, prob)

    # Flaw demonstration: region-level check marks pixel valid despite 50/50 ambiguity
    assert dense_valid_low[0, 0, 0].item() is True, "Flaw: region-level gating flags boundary pixel as valid"
    assert dense_prob_low[0, 2, 0, 0].item() == 0.5
    assert dense_prob_low[0, 3, 0, 0].item() == 0.5

    # Proposed fix: Dense margin check rejects this ambiguous pixel
    dense_top2 = dense_prob_low.topk(2, dim=1).values
    dense_margin_valid = (dense_top2[:, 0] >= 0.70) & ((dense_top2[:, 0] - dense_top2[:, 1]) >= 0.20)
    assert dense_margin_valid[0, 0, 0].item() is False, "Dense margin check must reject 50/50 boundary mixture"
