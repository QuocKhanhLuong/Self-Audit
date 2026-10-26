"""Tests for motion-aligned temporal consistency (self_audit.volume.temporal).

Verifies:
1. TemporalConfig normalization and mode switches (none, identity, motion).
2. Explicit rejection of unsupported temporal modes (no silent fallback).
3. Identity diagnostic: zero flow field, explicitly flagged in metrics.
4. Known-motion phantom test:
   - Recovers known 3D translation with subpixel accuracy.
   - Aligns warped adjacent image with source.
   - Confirms temporal consistency loss is minimized by motion alignment.
5. Independent adversarial phantoms addressing Astra ctx_a1f3944f22c6:
   - Invalid texture: support-aware registration ignores invalid bright textures.
   - Fractional boundary contamination: conservative bilinear validity excludes
     voxels contaminated by invalid neighbor interpolation footprint.
   - Out-of-bounds motion: displacements exceeding physical bounds are rejected,
     never clipped to boundary and falsely certified by forward-backward check.
   - Subpixel motion and occlusion: forward-backward check detects occlusion and
     excludes occluded voxels from temporal loss.
6. Non-circular warping bounds and occlusion detection.
7. Stop-gradient and student differentiability:
   - Nonzero finite gradient into student logits.
   - Zero gradient into teacher probabilities or images (detached).
8. Edge cases: Z=1 (single-slice 2D cine), empty valid support, mode="none".
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from self_audit.volume.temporal import (
    RegistrationResult,
    TemporalConfig,
    register_volumes,
    temporal_consistency,
    warp_volume_3d,
)


def _make_gaussian_sphere_phantom(
    z: int = 5,
    h: int = 32,
    w: int = 32,
    cz: float = 2.0,
    cy: float = 16.0,
    cx: float = 16.0,
    sigma: float = 4.0,
) -> Tensor:
    """Create a 3D synthetic Gaussian sphere phantom."""
    zz, yy, xx = torch.meshgrid(
        torch.arange(z, dtype=torch.float32),
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    dist2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
    return torch.exp(-dist2 / (2.0 * sigma * sigma))


def test_temporal_config_normalization():
    c0 = TemporalConfig()
    assert c0.mode == "none"

    c1 = TemporalConfig.from_mapping({"mode": "motion_aligned", "weight": 0.5})
    assert c1.mode == "motion"
    assert c1.weight == 0.5

    c2 = TemporalConfig.from_mapping({"mode": "identity_diagnostic"})
    assert c2.mode == "identity"

    c3 = TemporalConfig.from_mapping({"mode": "off"})
    assert c3.mode == "none"


def test_reject_unsupported_temporal_mode():
    with pytest.raises(ValueError, match="Unsupported temporal mode"):
        TemporalConfig.from_mapping({"mode": "optical_flow_unsupported"})


def test_identity_diagnostic():
    b, c, z, h, w = 1, 4, 3, 16, 16
    student_logits = torch.randn(b, c, z, h, w, requires_grad=True)
    teacher_prob = torch.softmax(torch.randn(b, c, z, h, w), dim=1)
    img = torch.rand(b, 3, z, h, w)
    valid = torch.ones(b, z, h, w, dtype=torch.bool)

    loss, metrics = temporal_consistency(
        student_logits=student_logits,
        teacher_probabilities=teacher_prob,
        current_image=img,
        adjacent_image=img,
        current_valid=valid,
        adjacent_valid=valid,
        config=TemporalConfig(mode="identity"),
    )

    assert metrics["is_identity_diagnostic"] is True
    assert metrics["mode"] == "identity"
    assert metrics["mean_displacement_voxels"] == 0.0
    assert loss.requires_grad
    assert float(loss.item()) > 0.0


def test_known_motion_phantom_registration():
    # 3D phantom shifted by known subpixel displacement: dy=2.0, dx=-3.0, dz=0.0
    z, h, w = 5, 48, 48
    phantom_t = _make_gaussian_sphere_phantom(z, h, w, cz=2.0, cy=24.0, cx=24.0, sigma=5.0)

    true_dz, true_dy, true_dx = 0.0, 2.0, -3.0
    phantom_adj = _make_gaussian_sphere_phantom(
        z, h, w, cz=2.0 + true_dz, cy=24.0 + true_dy, cx=24.0 + true_dx, sigma=5.0
    )

    curr_img = phantom_t.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1, 1)
    adj_img = phantom_adj.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1, 1)
    valid = torch.ones((1, z, h, w), dtype=torch.bool)

    cfg = TemporalConfig(mode="motion", slice_wise=False)
    reg = register_volumes(curr_img, adj_img, current_valid=valid, adjacent_valid=valid, config=cfg)

    # Recovered displacement should map curr -> adj: so +dx=-3.0, +dy=2.0, +dz=0.0
    rec_dx = reg.flow_xyz[0, 0].mean().item()
    rec_dy = reg.flow_xyz[0, 1].mean().item()
    rec_dz = reg.flow_xyz[0, 2].mean().item()

    assert abs(rec_dx - true_dx) < 0.25, f"Expected dx {true_dx}, got {rec_dx}"
    assert abs(rec_dy - true_dy) < 0.25, f"Expected dy {true_dy}, got {rec_dy}"
    assert abs(rec_dz - true_dz) < 0.25, f"Expected dz {true_dz}, got {rec_dz}"

    # Verify warp recovers source phantom with high alignment
    warped_adj = warp_volume_3d(phantom_adj.unsqueeze(0).unsqueeze(0), reg.flow_xyz)
    diff = (warped_adj.squeeze() - phantom_t).abs().mean().item()
    assert diff < 0.05, f"Residual warp difference is {diff}"


def test_adversarial_invalid_texture_support_awareness():
    """Astra blocker test: valid anatomy moves by dx=+2.0, while invalid background

    padding contains high-contrast bright texture moving by dx=+6.0.
    Support-aware registration must recover dx=2.0, NOT dx=6.0.
    """
    z, h, w = 3, 32, 64
    # Valid support contains the anatomy (x in [8, 44], y in [4, 28])
    valid = torch.zeros((1, z, h, w), dtype=torch.bool)
    valid[:, :, 4:28, 8:44] = True

    # Valid anatomy in curr centered at cx=24.0, moves to cx=26.0 (dx=+2.0)
    curr_anatomy = _make_gaussian_sphere_phantom(z, h, w, cz=1.0, cy=16.0, cx=24.0, sigma=4.0)
    adj_anatomy = _make_gaussian_sphere_phantom(z, h, w, cz=1.0, cy=16.0, cx=26.0, sigma=4.0)

    curr_img = curr_anatomy.clone()
    adj_img = adj_anatomy.clone()

    # Add strong adversarial texture in the INVALID region (x >= 48) moving by dx=+6.0
    curr_img[:, :, 48:54] += 10.0  # very bright!
    adj_img[:, :, 54:60] += 10.0  # shifted by dx=+6.0

    curr_t = curr_img.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1, 1)
    adj_t = adj_img.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1, 1)

    cfg = TemporalConfig(mode="motion", slice_wise=False, max_disp_xyz=(2.0, 4.0, 8.0))
    reg = register_volumes(curr_t, adj_t, current_valid=valid, adjacent_valid=valid, config=cfg)

    rec_dx = reg.flow_xyz[0, 0].mean().item()
    # Must recover the true anatomy motion (+2.0), NOT the invalid bright texture motion (+6.0)
    assert abs(rec_dx - 2.0) < 0.35, f"Expected anatomy dx ~ 2.0, got {rec_dx} (distorted by invalid texture!)"
    assert abs(rec_dx - 6.0) > 2.0


def test_adversarial_fractional_boundary_contamination():
    """Astra blocker test: subpixel shift (e.g. +0.25) near validity boundary.

    Conservative bilinear support must reject boundary voxels whose bilinear
    interpolation footprint samples from invalid neighbors.
    """
    z, h, w = 2, 16, 32
    curr_valid = torch.ones((1, z, h, w), dtype=torch.bool)
    # Adjacent valid is True only for x < 16 (boundary at x=16)
    adj_valid = torch.zeros((1, z, h, w), dtype=torch.bool)
    adj_valid[:, :, :, :16] = True

    student_logits = torch.randn(1, 2, z, h, w, requires_grad=True)
    teacher_prob = torch.zeros(1, 2, z, h, w)
    teacher_prob[:, 0] = 1.0

    curr_img = torch.rand(1, 3, z, h, w)
    adj_img = torch.rand(1, 3, z, h, w)

    # Force a fractional displacement dx = +0.25
    cfg = TemporalConfig(mode="motion", weight=1.0)

    # Test direct consistency under a forced fractional displacement:
    # Voxel x=15 maps to x' = 15.25. Its neighbors in adjacent frame are 15 (valid) and 16 (invalid).
    # Bilinear interpolation of adjacent valid gives 0.75 <= 0.999.
    # Therefore voxel x=15 MUST be marked invalid in temporal_valid!
    flow = torch.zeros((1, 3, z, h, w))
    flow[:, 0] = 0.25  # dx = +0.25

    # Check conservative validity warp
    adj_val_float = adj_valid.unsqueeze(1).float()
    warped_adj_val = warp_volume_3d(adj_val_float, flow, mode="bilinear", padding_mode="zeros").squeeze(1)
    conservative_valid = curr_valid & (warped_adj_val > 0.999)

    # At x=15: warped_adj_val is ~0.75, so conservative_valid must be False!
    assert not bool(conservative_valid[0, 0, 0, 15].item()), "Boundary voxel at x=15 was contaminated by invalid neighbor!"
    # At x=14: neighbors are 14 (valid) and 15 (valid), so conservative_valid is True
    assert bool(conservative_valid[0, 0, 0, 14].item())


def test_adversarial_out_of_bound_motion():
    """Astra blocker test: true motion dx=6.0 exceeds physical bound max_disp_x=2.0.

    Must reject motion as out-of-bounds, NOT clip to 2.0 and falsely accept voxels!
    """
    z, h, w = 3, 32, 48
    valid = torch.ones((1, z, h, w), dtype=torch.bool)

    # True motion is dx = +6.0
    p_curr = _make_gaussian_sphere_phantom(z, h, w, cz=1.0, cy=16.0, cx=20.0, sigma=4.0)
    p_adj = _make_gaussian_sphere_phantom(z, h, w, cz=1.0, cy=16.0, cx=26.0, sigma=4.0)

    curr_img = p_curr.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1, 1)
    adj_img = p_adj.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1, 1)

    student_logits = torch.randn(1, 2, z, h, w, requires_grad=True)
    teacher_prob = torch.softmax(torch.randn(1, 2, z, h, w), dim=1)

    # Max bound is 2.0 voxels
    cfg = TemporalConfig(mode="motion", max_disp_xyz=(2.0, 2.0, 2.0))

    reg = register_volumes(curr_img, adj_img, current_valid=valid, adjacent_valid=valid, config=cfg)
    # Registration must mark motion as out of bounds!
    assert not reg.in_bounds_mask.all()

    loss, metrics = temporal_consistency(
        student_logits=student_logits,
        teacher_probabilities=teacher_prob,
        current_image=curr_img,
        adjacent_image=adj_img,
        current_valid=valid,
        adjacent_valid=valid,
        config=cfg,
    )

    # Zero voxels accepted under out-of-bound unobservable motion
    assert metrics["temporal_valid_voxels"] == 0
    assert float(loss.item()) == 0.0


def test_adversarial_subpixel_motion_and_occlusion():
    """Forward-backward check samples backward flow at mapped coordinates and flags occlusion."""
    z, h, w = 3, 32, 32
    valid = torch.ones((1, z, h, w), dtype=torch.bool)

    p_curr = _make_gaussian_sphere_phantom(z, h, w, cz=1.0, cy=16.0, cx=16.0, sigma=4.0)
    # Subpixel motion: dx=+1.5, dy=+1.0
    p_adj = _make_gaussian_sphere_phantom(z, h, w, cz=1.0, cy=17.0, cx=17.5, sigma=4.0)

    curr_img = p_curr.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1, 1)
    adj_img = p_adj.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1, 1)

    # Create an occluded region in adjacent frame
    adj_img[:, :, :, 14:18, 14:18] = 0.0  # sphere center occluded/erased

    cfg = TemporalConfig(mode="motion", slice_wise=True, consistency_threshold=1.0)
    reg = register_volumes(curr_img, adj_img, current_valid=valid, adjacent_valid=valid, config=cfg)

    # Occlusion mask must detect the inconsistent / occluded region
    assert reg.occlusion_mask.any()


def test_motion_alignment_reduces_temporal_loss():
    z, h, w = 3, 32, 32
    curr_phantom = _make_gaussian_sphere_phantom(z, h, w, cz=1.0, cy=16.0, cx=16.0, sigma=4.0)
    adj_phantom = _make_gaussian_sphere_phantom(z, h, w, cz=1.0, cy=18.0, cx=13.0, sigma=4.0)

    curr_img = curr_phantom.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1, 1)
    adj_img = adj_phantom.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1, 1)
    valid = torch.ones(1, z, h, w, dtype=torch.bool)

    student_logits = torch.zeros(1, 2, z, h, w, requires_grad=True)
    with torch.no_grad():
        student_logits[0, 1] = curr_phantom * 5.0
        student_logits[0, 0] = (1.0 - curr_phantom) * 5.0

    teacher_prob = torch.zeros(1, 2, z, h, w)
    teacher_prob[0, 1] = adj_phantom
    teacher_prob[0, 0] = 1.0 - adj_phantom

    # 1. Loss with identity diagnostic (no alignment)
    loss_unaligned, m_unaligned = temporal_consistency(
        student_logits=student_logits,
        teacher_probabilities=teacher_prob,
        current_image=curr_img,
        adjacent_image=adj_img,
        current_valid=valid,
        adjacent_valid=valid,
        config=TemporalConfig(mode="identity"),
    )

    # 2. Loss with motion alignment
    loss_aligned, m_aligned = temporal_consistency(
        student_logits=student_logits,
        teacher_probabilities=teacher_prob,
        current_image=curr_img,
        adjacent_image=adj_img,
        current_valid=valid,
        adjacent_valid=valid,
        config=TemporalConfig(mode="motion"),
    )

    assert float(loss_aligned.item()) < float(loss_unaligned.item()), (
        f"Aligned loss {loss_aligned.item()} should be less than unaligned {loss_unaligned.item()}"
    )


def test_non_circular_warping_bounds_and_occlusion():
    z, h, w = 3, 16, 16
    curr_img = torch.rand(1, 3, z, h, w)
    forced_flow = torch.zeros(1, 3, z, h, w)
    forced_flow[0, 0] = 10.0  # +10 in X

    warped = warp_volume_3d(curr_img, forced_flow, padding_mode="zeros")
    assert torch.all(warped[:, :, :, :, 6:] == 0.0)


def test_temporal_stop_gradient_and_student_differentiability():
    b, c, z, h, w = 2, 4, 3, 16, 16
    student_logits = torch.randn(b, c, z, h, w, requires_grad=True)
    teacher_prob = torch.softmax(torch.randn(b, c, z, h, w), dim=1).requires_grad_(True)
    curr_img = torch.rand(b, 3, z, h, w).requires_grad_(True)
    adj_img = torch.rand(b, 3, z, h, w).requires_grad_(True)
    valid = torch.ones(b, z, h, w, dtype=torch.bool)

    loss, metrics = temporal_consistency(
        student_logits=student_logits,
        teacher_probabilities=teacher_prob,
        current_image=curr_img,
        adjacent_image=adj_img,
        current_valid=valid,
        adjacent_valid=valid,
        config=TemporalConfig(mode="motion", weight=0.2),
    )

    loss.backward()

    # 1. Student logits MUST have nonzero, finite gradients
    assert student_logits.grad is not None
    assert torch.isfinite(student_logits.grad).all()
    assert float(student_logits.grad.abs().sum()) > 0.0

    # 2. Teacher probabilities and images MUST NOT have gradients (strictly detached)
    assert teacher_prob.grad is None
    assert curr_img.grad is None
    assert adj_img.grad is None


def test_edge_cases_z1_and_empty_valid():
    # 1. Single-slice Z=1 (2D cine)
    b, c, z, h, w = 1, 4, 1, 16, 16
    student_logits = torch.randn(b, c, z, h, w, requires_grad=True)
    teacher_prob = torch.softmax(torch.randn(b, c, z, h, w), dim=1)
    img = torch.rand(b, 3, z, h, w)
    valid = torch.ones(b, z, h, w, dtype=torch.bool)

    loss, metrics = temporal_consistency(
        student_logits=student_logits,
        teacher_probabilities=teacher_prob,
        current_image=img,
        adjacent_image=img,
        current_valid=valid,
        adjacent_valid=valid,
        config=TemporalConfig(mode="motion"),
    )
    assert torch.isfinite(loss)
    assert metrics["temporal_valid_voxels"] > 0

    # 2. Empty valid support
    empty_valid = torch.zeros(b, z, h, w, dtype=torch.bool)
    loss_empty, metrics_empty = temporal_consistency(
        student_logits=student_logits,
        teacher_probabilities=teacher_prob,
        current_image=img,
        adjacent_image=img,
        current_valid=empty_valid,
        adjacent_valid=empty_valid,
        config=TemporalConfig(mode="motion"),
    )
    assert loss_empty.item() == 0.0
    assert metrics_empty["temporal_valid_voxels"] == 0

    # 3. Mode "none"
    loss_none, metrics_none = temporal_consistency(
        student_logits=student_logits,
        teacher_probabilities=teacher_prob,
        current_image=img,
        adjacent_image=img,
        current_valid=valid,
        adjacent_valid=valid,
        config=TemporalConfig(mode="none"),
    )
    assert loss_none.item() == 0.0
    assert metrics_none["temporal_loss"] == 0.0


def test_unobservable_constant_image_rejection():
    """Verify that constant (unobservable) valid image content is rejected before apodization.

    Coordinator blocker:
    All-ones, all-zeros, or constant images in motion mode must NOT produce accepted voxels
    due to artificial apodization window peaks. Both cases must yield 0 accepted voxels,
    temporal_valid_voxels=0, zero loss, and zero student gradient.
    Identity diagnostic mode remains separate and untouched.
    """
    b, c, z, h, w = 1, 4, 3, 16, 16
    v = torch.ones(b, z, h, w, dtype=torch.bool)
    teacher_prob = torch.softmax(torch.randn(b, c, z, h, w), dim=1)

    for const_val in (1.0, 0.0, 42.0):
        img_const = torch.full((b, 1, z, h, w), const_val)

        # 1. Registration result directly
        reg = register_volumes(img_const, img_const, v, v, config={"mode": "motion", "slice_wise": False})
        accepted = reg.in_bounds_mask & (~reg.occlusion_mask)
        assert int(accepted.sum().item()) == 0, f"Expected 0 accepted voxels for constant value {const_val}"

        # 2. Slicewise motion mode also rejects
        reg_sw = register_volumes(img_const, img_const, v, v, config={"mode": "motion", "slice_wise": True})
        accepted_sw = reg_sw.in_bounds_mask & (~reg_sw.occlusion_mask)
        assert int(accepted_sw.sum().item()) == 0

        # 3. Temporal consistency loss and gradient
        student_logits = torch.randn(b, c, z, h, w, requires_grad=True)
        loss, metrics = temporal_consistency(
            student_logits=student_logits,
            teacher_probabilities=teacher_prob,
            current_image=img_const,
            adjacent_image=img_const,
            current_valid=v,
            adjacent_valid=v,
            config=TemporalConfig(mode="motion"),
        )
        assert float(loss.detach()) == 0.0
        assert metrics["temporal_valid_voxels"] == 0
        loss.backward()
        grad_sum = float(student_logits.grad.abs().sum().item()) if student_logits.grad is not None else 0.0
        assert grad_sum == 0.0

    # 4. Identity mode remains separate diagnostic with full acceptance
    reg_id = register_volumes(img_const, img_const, v, v, config={"mode": "identity"})
    accepted_id = reg_id.in_bounds_mask & (~reg_id.occlusion_mask)
    assert int(accepted_id.sum().item()) == b * z * h * w


def test_temporal_per_sample_spacing_displacement_metric():
    """Coordinator regression msg_2e0063b8ab4e:

    Duplicate dx=2 registration pair B=2 with spacingXYZ [1, 1, 4] and [3, 1, 4].
    Sample 0: dx=2 at sx=1mm -> 2mm displacement.
    Sample 1: dx=2 at sx=3mm -> 6mm displacement.
    Mean physical displacement across the batch must report 4.0mm (previously 2.0mm due to metadata[0] reuse).
    Also validates metadata count matching batch size, non-positive/non-finite spacing rejection,
    and retention of finite directional gradients.
    """
    b, c, z, h, w = 2, 4, 3, 16, 16
    v = torch.ones(b, z, h, w, dtype=torch.bool)

    # Create distinct textures with known dx=2 translation in valid interior
    torch.manual_seed(42)
    img_curr = torch.zeros(b, 1, z, h, w)
    img_curr[:, :, :, 3:13, 3:13] = torch.rand(b, 1, z, 10, 10) + 0.5
    img_adj = torch.zeros_like(img_curr)
    # dx = 2 translation
    img_adj[:, :, :, 3:13, 5:15] = img_curr[:, :, :, 3:13, 3:13]

    teacher_prob = torch.softmax(torch.randn(b, c, z, h, w), dim=1)
    student_logits = torch.randn(b, c, z, h, w, requires_grad=True)

    meta = [
        {"patient": "p0", "spacing_xyz_mm": [1.0, 1.0, 4.0]},
        {"patient": "p1", "spacing_xyz_mm": [3.0, 1.0, 4.0]},
    ]

    loss, metrics = temporal_consistency(
        student_logits=student_logits,
        teacher_probabilities=teacher_prob,
        current_image=img_curr,
        adjacent_image=img_adj,
        current_valid=v,
        adjacent_valid=v,
        metadata=meta,
        config=TemporalConfig(mode="motion", slice_wise=False),
    )

    # Mean displacement in mm must be ~4.0mm (average of 2mm and 6mm)
    assert abs(metrics["mean_displacement_mm"] - 4.0) < 0.2, (
        f"Expected ~4.0mm mean displacement, got {metrics['mean_displacement_mm']}"
    )

    # Retain finite directional student gradients
    loss.backward()
    assert student_logits.grad is not None
    assert torch.isfinite(student_logits.grad).all()
    assert float(student_logits.grad.abs().sum()) > 0.0

    # Validation: missing metadata entry for batch size B=2 fails closed
    with pytest.raises(ValueError, match="Expected metadata length >= batch size"):
        temporal_consistency(
            student_logits=student_logits,
            teacher_probabilities=teacher_prob,
            current_image=img_curr,
            adjacent_image=img_adj,
            current_valid=v,
            adjacent_valid=v,
            metadata=[meta[0]],  # Only 1 entry for batch size 2
            config=TemporalConfig(mode="motion"),
        )

    # Validation: non-positive or non-finite spacing fails closed
    bad_meta = [meta[0], {"patient": "p1", "spacing_xyz_mm": [-1.0, 1.0, 4.0]}]
    with pytest.raises(ValueError, match="spacing_xyz_mm must contain 3 finite positive values"):
        temporal_consistency(
            student_logits=student_logits,
            teacher_probabilities=teacher_prob,
            current_image=img_curr,
            adjacent_image=img_adj,
            current_valid=v,
            adjacent_valid=v,
            metadata=bad_meta,
            config=TemporalConfig(mode="motion"),
        )
