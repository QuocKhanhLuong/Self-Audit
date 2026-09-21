"""Pytest suite for Worker C Dynamic Window and Student audit reproducibility."""

import pytest
import torch
import torch.nn.functional as F

from self_audit.models.dynamic_window import DynamicWindowAttention
from self_audit.models.annotation_expert import AnnotationExpert, entropy_from_logits
from self_audit_pseudolabel.system_v3 import (
    AdaptiveAnnotationStudent,
    PROFILES,
    DeploymentEncoder,
    pseudo_supervision_loss,
    NUM_CLASSES,
)


def test_exact_compact_equals_a0():
    """Verify that compact profile produces output identically equal to a0."""
    student = AdaptiveAnnotationStudent(width=16, window_k=4).eval()
    x = torch.randn(2, 3, 64, 64)
    out = student(x, profile="compact")

    assert out["final_logits"] is out["a0_logits"], "final_logits must be the identical tensor as a0_logits"
    assert torch.equal(out["final_logits"], out["a0_logits"]), "final_logits must equal a0_logits bit-for-bit"
    assert len(out["stages"]) == 1
    assert out["stages"][0] is out["a0_logits"]
    assert len(out["window_metadata"]) == 0


def test_single_encoder_pass_across_all_profiles():
    """Verify that the deployment encoder is evaluated exactly once across all profiles."""
    student = AdaptiveAnnotationStudent(width=16, window_k=4).eval()
    x = torch.randn(1, 3, 64, 64)

    for prof in ["compact", "balanced", "accurate"]:
        pass_count = 0
        def hook(mod, inp, outp):
            nonlocal pass_count
            pass_count += 1

        h = student.encoder.register_forward_hook(hook)
        _ = student(x, profile=prof)
        h.remove()
        assert pass_count == 1, f"Encoder must be called exactly once for profile {prof}, but was called {pass_count} times"


def test_feature_only_audit_evidence_ablation():
    """Verify feature_only mode removes audit evidence but preserves channel shapes."""
    student = AdaptiveAnnotationStudent(width=16, window_k=4).eval()
    assert student.refiner.audit_conditioning == "feature_only"

    x = torch.randn(1, 3, 64, 64)
    recorded_inputs = []
    def hook(mod, inp, outp):
        recorded_inputs.append(inp[0].detach().clone())

    h = student.refiner.input_projection[0].register_forward_hook(hook)
    _ = student(x, profile="balanced")
    h.remove()

    inp = recorded_inputs[0]
    # Input channels: 16 (feat) + 4 (logits) + 1 (entropy) + 3 (audit) = 24
    assert inp.shape[1] == 16 + 4 + 1 + 3
    audit_slice = inp[:, 21:24]
    assert torch.all(audit_slice == 0.0), "Audit evidence channels must be strictly zeroed in feature_only mode"


def test_logits_entropy_image_conditioning_retained():
    """Verify that shared features, previous logits, and Shannon entropy are active inputs."""
    student = AdaptiveAnnotationStudent(width=16, window_k=4).eval()
    x = torch.randn(1, 3, 64, 64)
    recorded_inputs = []
    def hook(mod, inp, outp):
        recorded_inputs.append(inp[0].detach().clone())

    h = student.refiner.input_projection[0].register_forward_hook(hook)
    out = student(x, profile="balanced")
    h.remove()

    inp = recorded_inputs[0]
    feat_slice = inp[:, :16]
    logits_slice = inp[:, 16:20]
    entropy_slice = inp[:, 20:21]

    assert (feat_slice.abs() > 0).any(), "Shared image features must be non-zero"
    assert (logits_slice.abs() > 0).any(), "Previous logits must be non-zero"
    assert (entropy_slice.abs() > 0).any(), "Entropy must be non-zero"
    assert (entropy_slice >= 0.0).all() and (entropy_slice <= 1.0).all(), "Entropy must be in [0, 1]"

    # Verify entropy formula invariant
    a0 = out["a0_logits"]
    expected_entropy = F.interpolate(entropy_from_logits(a0), size=feat_slice.shape[-2:], mode="bilinear", align_corners=False)
    assert torch.allclose(entropy_slice, expected_entropy, atol=1e-5), "Entropy channel must match normalized Shannon entropy"


def test_stage_dependent_depth():
    """Verify stage-dependent depth: turn 0 has depth=1, turn 1 has depth=2."""
    student = AdaptiveAnnotationStudent(width=16, window_k=4).eval()
    x = torch.randn(1, 3, 64, 64)

    dw_invocations = 0
    def hook(mod, inp, outp):
        nonlocal dw_invocations
        dw_invocations += 1

    h = student.refiner.refinement_block.register_forward_hook(hook)
    _ = student(x, profile="accurate")
    h.remove()

    # Turn 0: depth = min(0+1, 3) = 1
    # Turn 1: depth = min(1+1, 3) = 2
    # Total calls to DynamicWindowAttention = 1 + 2 = 3
    assert dw_invocations == 3, f"Expected 3 DynamicWindowAttention invocations in accurate profile, got {dw_invocations}"


def test_deterministic_turns():
    """Verify that turns and stages are compile-time deterministic constants."""
    student = AdaptiveAnnotationStudent(width=16, window_k=4).eval()
    x = torch.randn(1, 3, 64, 64)

    assert PROFILES["compact"].turns == 0
    assert PROFILES["balanced"].turns == 1
    assert PROFILES["accurate"].turns == 2

    out_c = student(x, profile="compact")
    assert len(out_c["stages"]) == 1

    out_b = student(x, profile="balanced")
    assert len(out_b["stages"]) == 2

    out_a = student(x, profile="accurate")
    assert len(out_a["stages"]) == 3


def test_all_gradient_paths_connected_no_detaches():
    """Verify that all backward gradients flow continuously from A2 to A1, A0, and encoder without detaches."""
    student = AdaptiveAnnotationStudent(width=16, window_k=4).train()
    x = torch.randn(2, 3, 64, 64, requires_grad=True)

    out = student(x, profile="accurate")
    a0 = out["a0_logits"]
    a1 = out["stages"][1]
    a2 = out["final_logits"]

    target = torch.randint(0, NUM_CLASSES, (2, 64, 64))
    loss_a2 = F.cross_entropy(a2, target)

    # Check autograd graph connectivity
    grad_a1 = torch.autograd.grad(loss_a2, a1, retain_graph=True)[0]
    grad_a0 = torch.autograd.grad(loss_a2, a0, retain_graph=True)[0]

    assert grad_a1 is not None and grad_a1.abs().sum() > 0, "A1 must receive gradient from A2"
    assert grad_a0 is not None and grad_a0.abs().sum() > 0, "A0 must receive gradient from A2"

    loss_a2.backward()
    for name, p in student.encoder.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, f"Encoder param {name} must receive gradient from A2"

    for name, p in student.a0_head.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, f"A0 head param {name} must receive gradient from A2"


def test_later_stage_gradients_optimize_a0_headroom():
    """Verify that joint training degrades A0 standalone accuracy vs detached A0 baseline."""
    torch.manual_seed(42)
    B, C, H, W = 4, 3, 32, 32
    x = torch.randn(B, C, H, W)
    y = torch.randint(0, NUM_CLASSES, (B, H, W))
    valid = torch.ones(B, H, W, dtype=torch.bool)

    # Model 1: Standard joint training
    m1 = AdaptiveAnnotationStudent(width=8, window_k=2)
    opt1 = torch.optim.AdamW(m1.parameters(), lr=1e-2)
    for _ in range(15):
        opt1.zero_grad()
        out1 = m1(x, profile="accurate")
        loss1 = pseudo_supervision_loss(out1, y, valid, a0_weight=0.25)
        loss1.backward()
        opt1.step()

    # Model 2: Detached A0 training (stop-gradient into A0)
    m2 = AdaptiveAnnotationStudent(width=8, window_k=2)
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-2)
    for _ in range(15):
        opt2.zero_grad()
        feat = m2.encoder(x)
        a0 = F.interpolate(m2.a0_head(feat), x.shape[-2:], mode="bilinear", align_corners=False)
        # Detach a0 and feat before passing to refiner
        logits = a0.detach()
        feat_det = feat.detach()
        for turn in range(2):
            out = m2.refiner(feat_det, logits, turn_index=turn)
            logits = out.candidate_logits
        loss2 = F.cross_entropy(logits, y) + 1.0 * F.cross_entropy(a0, y)
        loss2.backward()
        opt2.step()

    with torch.no_grad():
        ce_a0_joint = F.cross_entropy(m1(x, profile="compact")["final_logits"], y).item()
        ce_a0_detached = F.cross_entropy(m2(x, profile="compact")["final_logits"], y).item()

    # Detached A0 has lower cross-entropy on A0 (better standalone quality)
    assert ce_a0_detached < ce_a0_joint, (
        f"Detached A0 should preserve A0 quality better than joint training. "
        f"Got detached={ce_a0_detached:.4f}, joint={ce_a0_joint:.4f}"
    )
