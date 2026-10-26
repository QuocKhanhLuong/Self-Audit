"""Tests for deterministic physical region learning (self_audit.volume.regions).

Verifies:
1. RegionManager initialization, configuration parsing, and modes (none, image, ema).
2. Determinism: identical inputs yield bitwise identical region labels and hashes.
3. Physical spacing awareness: anisotropic voxel spacing scales distance metrics.
4. Missing physical metadata fails closed (no silent isotropic 1mm default).
5. Boundary downweighting & rejection from image gradients and teacher class transitions without GT.
6. Differentiable region consistency loss:
   - Nonzero, finite gradients into student logits.
   - Zero gradients into teacher probabilities or features (detached).
   - Zero loss when predictions inside each region are perfectly uniform.
7. Caching and refresh cadence:
   - P1 cache defect repro: step 0 patient A vs step 1 patient B does NOT reuse A targets.
   - Patient B and crop changes cannot reuse A.
   - Interior voxel change is NOT missed by full tensor hashing (no sparse skip bug).
   - Support and spacing changes cause cache miss.
   - Same sample within cadence interval hits cache and reuses targets.
   - Cadence expiration triggers fresh target computation.
   - Different microbatches at the same outer step have distinct cache keys.
8. State serialization, bounded cache eviction, and corruption detection:
   - state_dict and load_state_dict round-trip preserves exact state.
   - Corrupted hash or tensor integrity raises ValueError.
   - Bounded cache evicts oldest entries beyond max_cache_size.
9. Edge cases: empty support, single slice Z=1, boundary values.
"""

from __future__ import annotations

import copy
import collections
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from self_audit.volume.regions import (
    RegionConfig,
    RegionManager,
    RegionTargets,
    compute_boundary_weights,
    compute_cache_hash,
    compute_physical_gradient,
    compute_sample_identity_hash,
    generate_deterministic_supervoxels_3d,
    region_consistency_loss,
)


def _make_meta(
    patient: str = "patient001",
    frame: int = 0,
    spacing: list[float] | None = None,
    crop: dict | None = None,
) -> list[dict]:
    return [
        {
            "patient": patient,
            "frame": frame,
            "z_indices": [0, 1],
            "spacing_xyz_mm": spacing if spacing is not None else [1.25, 1.25, 8.0],
            "crop": crop or {"top": 0, "left": 0, "height": 16, "width": 16},
            "source_image_sha256": "fake_sha256_" + patient,
            "current_affine_xyz_to_world_mm": np.eye(4).tolist(),
        }
    ]


def test_region_config_normalization():
    # Test default
    c0 = RegionConfig()
    assert c0.mode == "none"

    # Test aliases
    c1 = RegionConfig.from_mapping({"mode": "image_only", "weight": 0.2})
    assert c1.mode == "image"
    assert c1.weight == 0.2

    c2 = RegionConfig.from_mapping({"mode": "ema_derived", "target_num_regions": 16})
    assert c2.mode == "ema"
    assert c2.target_num_regions == 16

    c3 = RegionConfig.from_mapping({"mode": "off"})
    assert c3.mode == "none"


def test_deterministic_supervoxels():
    # Synthetic 3D slab
    z, h, w = 3, 24, 24
    np.random.seed(42)
    img = np.random.uniform(0.1, 0.9, size=(z, h, w)).astype(np.float32)
    valid = np.ones((z, h, w), dtype=bool)
    spacing_iso = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    labels1, n1 = generate_deterministic_supervoxels_3d(img, valid, spacing_iso, target_num_regions=8)
    labels2, n2 = generate_deterministic_supervoxels_3d(img, valid, spacing_iso, target_num_regions=8)

    # Must be 100% bitwise identical
    assert n1 == n2
    assert np.array_equal(labels1, labels2)
    assert n1 > 0
    # Every valid voxel must be assigned a non-negative cluster ID
    assert np.all(labels1 >= 0)
    assert np.max(labels1) == n1 - 1


def test_physical_spacing_awareness():
    # If Z spacing is 10mm and XY is 1mm, clustering should favor in-plane grouping over cross-slice
    z, h, w = 4, 32, 32
    img = np.zeros((z, h, w), dtype=np.float32)
    valid = np.ones((z, h, w), dtype=bool)

    # Anisotropic spacing: Z is 10x larger than XY
    spacing_aniso = np.array([10.0, 1.0, 1.0], dtype=np.float32)
    spacing_iso = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    labels_aniso, n_aniso = generate_deterministic_supervoxels_3d(img, valid, spacing_aniso, target_num_regions=16)
    labels_iso, n_iso = generate_deterministic_supervoxels_3d(img, valid, spacing_iso, target_num_regions=16)

    # The partitions should be noticeably different due to physical distance scaling
    assert not np.array_equal(labels_aniso, labels_iso)


def test_reject_missing_or_invalid_physical_metadata():
    rm = RegionManager(RegionConfig(mode="image", target_num_regions=8))
    b, c, z, h, w = 1, 3, 2, 16, 16
    img = torch.rand(b, c, z, h, w)
    valid = torch.ones(b, z, h, w, dtype=torch.bool)

    # 1. Missing metadata entirely
    with pytest.raises(ValueError, match="Missing physical metadata"):
        rm.targets(img, valid, metadata=None, step=0)

    # 2. Metadata without spacing_xyz_mm
    with pytest.raises(ValueError, match="Missing or invalid spacing_xyz_mm"):
        rm.targets(img, valid, metadata=[{"patient": "p1"}], step=0)

    # 3. Non-positive or non-finite spacing
    with pytest.raises(ValueError, match="Missing or invalid spacing_xyz_mm"):
        rm.targets(img, valid, metadata=[{"spacing_xyz_mm": [1.0, -2.0, 1.0]}], step=0)

    with pytest.raises(ValueError, match="Missing or invalid spacing_xyz_mm"):
        rm.targets(img, valid, metadata=[{"spacing_xyz_mm": [1.0, float("nan"), 1.0]}], step=0)


def test_boundary_downweighting_image_and_teacher():
    z, h, w = 2, 20, 20
    img = np.zeros((z, h, w), dtype=np.float32)
    # Create sharp vertical edge in the middle of X
    img[:, :, 10:] = 1.0
    valid = np.ones((z, h, w), dtype=bool)
    spacing = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    cfg = RegionConfig(mode="image", boundary_threshold=0.3, boundary_downweight=0.1)
    weights = compute_boundary_weights(img, valid, spacing, probabilities_czyx=None, config=cfg)

    # Voxels near the edge x=9,10 should have downweighted weights
    assert np.all(weights[:, :, :8] == 1.0)
    assert np.any(weights[:, :, 9:11] == 0.1)

    # Now test with teacher predictions showing confident class transition
    probs = np.zeros((4, z, h, w), dtype=np.float32)
    probs[0, :, :, :10] = 0.95  # Class 0 on left
    probs[1, :, :, 10:] = 0.95  # Class 1 on right
    weights_ema = compute_boundary_weights(img, valid, spacing, probabilities_czyx=probs, config=cfg)
    assert np.any(weights_ema[:, :, 9:11] == 0.1)


def test_p1_cache_defect_repro_and_fix():
    """Reproduction of Astra ctx_a1f3944f22c6 P1 defect:

    In old code: refresh_every=5, step 0 Patient A (zeros, prob class 1) was cached.
    At step 1 Patient B (ones, prob class 3) with same shape returned same A object / class 1.
    Fix verifies that Step 1 returns distinct Patient B targets with prob class 3.
    """
    cfg = RegionConfig(mode="ema", refresh_every=5, target_num_regions=4)
    rm = RegionManager(cfg)

    b, c, z, h, w = 1, 3, 2, 16, 16
    img_a = torch.zeros(b, c, z, h, w)
    valid_a = torch.ones(b, z, h, w, dtype=torch.bool)
    meta_a = _make_meta(patient="patient_A")
    prob_a = torch.zeros(b, 4, z, h, w)
    prob_a[:, 1] = 1.0  # Class 1

    img_b = torch.ones(b, c, z, h, w)
    valid_b = torch.ones(b, z, h, w, dtype=torch.bool)
    meta_b = _make_meta(patient="patient_B")
    prob_b = torch.zeros(b, 4, z, h, w)
    prob_b[:, 3] = 1.0  # Class 3

    # Step 0: Patient A
    targets_a = rm.targets(img_a, valid_a, meta_a, probabilities=prob_a, step=0)
    assert targets_a.step == 0
    assert targets_a.target_probabilities is not None
    assert float(targets_a.target_probabilities[:, 1].mean().item()) == 1.0

    # Step 1: Patient B (same shape, within cadence window 1 < 5)
    targets_b = rm.targets(img_b, valid_b, meta_b, probabilities=prob_b, step=1)
    assert targets_b.step == 1
    # Must NOT reuse Patient A!
    assert targets_b is not targets_a
    assert targets_b.cache_hash != targets_a.cache_hash
    assert targets_b.target_probabilities is not None
    assert float(targets_b.target_probabilities[:, 3].mean().item()) == 1.0
    assert float(targets_b.target_probabilities[:, 1].mean().item()) == 0.0


def test_patient_and_crop_changes_cannot_reuse():
    cfg = RegionConfig(mode="image", refresh_every=5, target_num_regions=4)
    rm = RegionManager(cfg)

    b, c, z, h, w = 1, 3, 2, 16, 16
    img = torch.rand(b, c, z, h, w)
    valid = torch.ones(b, z, h, w, dtype=torch.bool)

    meta1 = _make_meta(patient="patient_01", crop={"top": 0, "left": 0, "height": 16, "width": 16})
    targets1 = rm.targets(img, valid, meta1, step=0)

    # 1. Different patient
    meta_patient2 = _make_meta(patient="patient_02", crop={"top": 0, "left": 0, "height": 16, "width": 16})
    targets_patient2 = rm.targets(img, valid, meta_patient2, step=1)
    assert targets_patient2 is not targets1
    assert targets_patient2.cache_hash != targets1.cache_hash

    # 2. Same patient, different crop
    meta_crop2 = _make_meta(patient="patient_01", crop={"top": 4, "left": 4, "height": 16, "width": 16})
    targets_crop2 = rm.targets(img, valid, meta_crop2, step=2)
    assert targets_crop2 is not targets1
    assert targets_crop2.cache_hash != targets1.cache_hash


def test_interior_voxel_change_not_missed_by_sparse_hash():
    """Verify that changing a single interior voxel (missed by sparse stride) causes hash mismatch."""
    cfg = RegionConfig(mode="image", refresh_every=5, target_num_regions=4)
    rm = RegionManager(cfg)

    b, c, z, h, w = 1, 3, 4, 32, 32
    img1 = torch.zeros(b, c, z, h, w)
    valid = torch.ones(b, z, h, w, dtype=torch.bool)
    meta = _make_meta()

    targets1 = rm.targets(img1, valid, meta, step=0)

    # Modify an interior voxel at odd indices that sparse strides (e.g. // 4) would skip
    img2 = img1.clone()
    img2[0, 1, 1, 7, 11] = 0.85

    targets2 = rm.targets(img2, valid, meta, step=1)
    assert targets2 is not targets1
    assert targets2.cache_hash != targets1.cache_hash


def test_support_and_spacing_changes():
    cfg = RegionConfig(mode="image", refresh_every=5, target_num_regions=4)
    rm = RegionManager(cfg)

    b, c, z, h, w = 1, 3, 2, 16, 16
    img = torch.rand(b, c, z, h, w)
    valid1 = torch.ones(b, z, h, w, dtype=torch.bool)
    meta1 = _make_meta(spacing=[1.0, 1.0, 8.0])

    targets1 = rm.targets(img, valid1, meta1, step=0)

    # 1. Support change
    valid2 = valid1.clone()
    valid2[0, 0, 0, 0] = False
    targets_val = rm.targets(img, valid2, meta1, step=1)
    assert targets_val is not targets1

    # 2. Spacing change
    meta2 = _make_meta(spacing=[2.0, 2.0, 10.0])
    targets_spacing = rm.targets(img, valid1, meta2, step=2)
    assert targets_spacing is not targets1


def test_same_sample_cadence_valid_reuse_and_refresh():
    cfg = RegionConfig(mode="ema", refresh_every=3, target_num_regions=4)
    rm = RegionManager(cfg)

    b, c, z, h, w = 1, 3, 2, 16, 16
    img = torch.rand(b, c, z, h, w)
    valid = torch.ones(b, z, h, w, dtype=torch.bool)
    meta = _make_meta()

    prob0 = torch.zeros(b, 4, z, h, w)
    prob0[:, 0] = 1.0

    # Step 0: Interval 0
    t0 = rm.targets(img, valid, meta, probabilities=prob0, step=0)

    # Step 1: Same sample, teacher weights slightly updated, same interval (1 // 3 == 0)
    prob1 = prob0.clone()
    prob1[:, 0] = 0.9
    prob1[:, 1] = 0.1
    t1 = rm.targets(img, valid, meta, probabilities=prob1, step=1)

    # Cadence preserved: identical cached targets returned, NOT invalidated by new prob1
    assert t1 is t0
    assert t1.cache_hash == t0.cache_hash

    # Step 3: Same sample, but step advances to interval 1 (3 // 3 == 1)
    prob3 = torch.zeros(b, 4, z, h, w)
    prob3[:, 2] = 1.0
    t3 = rm.targets(img, valid, meta, probabilities=prob3, step=3)

    assert t3 is not t0
    assert t3.step == 3
    assert t3.cache_hash != t0.cache_hash
    assert float(t3.target_probabilities[:, 2].mean().item()) == 1.0


def test_multi_microbatch_same_step_isolation():
    """Verify that different microbatches at the same outer step do not collide."""
    cfg = RegionConfig(mode="image", refresh_every=5, target_num_regions=4)
    rm = RegionManager(cfg)

    b, c, z, h, w = 1, 3, 2, 16, 16
    valid = torch.ones(b, z, h, w, dtype=torch.bool)

    img_micro0 = torch.zeros(b, c, z, h, w)
    meta_micro0 = _make_meta(patient="patient_micro0")

    img_micro1 = torch.ones(b, c, z, h, w)
    meta_micro1 = _make_meta(patient="patient_micro1")

    # Outer step 10, microbatch 0
    t_micro0 = rm.targets(img_micro0, valid, meta_micro0, step=10)
    # Outer step 10, microbatch 1
    t_micro1 = rm.targets(img_micro1, valid, meta_micro1, step=10)

    assert t_micro1 is not t_micro0
    assert t_micro1.cache_hash != t_micro0.cache_hash


def test_bounded_cache_eviction():
    cfg = RegionConfig(mode="image", refresh_every=10, max_cache_size=4)
    rm = RegionManager(cfg, max_cache_size=4)

    b, c, z, h, w = 1, 3, 2, 16, 16
    valid = torch.ones(b, z, h, w, dtype=torch.bool)

    # Add 10 distinct samples
    hashes = []
    for i in range(10):
        img = torch.full((b, c, z, h, w), float(i))
        meta = _make_meta(patient=f"p_{i}")
        t = rm.targets(img, valid, meta, step=0)
        hashes.append(t.cache_hash)

    # Cache should never exceed bounded capacity of 4
    assert len(rm._cache) == 4
    # The oldest entries should have been evicted
    assert hashes[0] not in rm._cache
    assert hashes[-1] in rm._cache


def test_cache_serialization_and_corruption_detection():
    cfg = RegionConfig(mode="image", refresh_every=5, target_num_regions=4)
    rm1 = RegionManager(cfg)

    b, c, z, h, w = 1, 3, 2, 16, 16
    img = torch.rand(b, c, z, h, w)
    valid = torch.ones(b, z, h, w, dtype=torch.bool)
    meta = _make_meta()

    t = rm1.targets(img, valid, meta, step=2)
    state = rm1.state_dict()

    # 1. Valid resume
    rm2 = RegionManager(cfg)
    rm2.load_state_dict(state)
    assert rm2._last_step == 2
    assert rm2._cache_hash == rm1._cache_hash
    assert rm2._cached_targets is not None
    assert torch.equal(rm2._cached_targets.region_labels, t.region_labels)

    # 2. Corrupted hash length in state
    bad_state1 = copy.deepcopy(state)
    bad_state1["cache_hash"] = "short_corrupted_hash"
    rm_bad = RegionManager(cfg)
    with pytest.raises(ValueError, match="Corrupted cache state"):
        rm_bad.load_state_dict(bad_state1)

    # 3. Corrupted weights out of range [0, 1]
    bad_state2 = copy.deepcopy(state)
    bad_state2["cached_targets"]["region_weights"][0, 0, 0, 0] = 5.0
    with pytest.raises(ValueError, match="Corrupted cache target"):
        rm_bad.load_state_dict(bad_state2)


def test_differentiable_region_consistency_loss_and_stop_gradient():
    b, c, z, h, w = 2, 4, 2, 16, 16
    student_logits = torch.randn(b, c, z, h, w, requires_grad=True)
    teacher_prob = torch.softmax(torch.randn(b, c, z, h, w), dim=1)
    teacher_prob.requires_grad_(True)  # To test stop-gradient

    valid = torch.ones(b, z, h, w, dtype=torch.bool)
    meta = _make_meta(patient="p1") + _make_meta(patient="p2")

    rm = RegionManager(RegionConfig(mode="ema", weight=0.5, target_num_regions=4))
    targets = rm.targets(
        image=torch.rand(b, 3, z, h, w),
        geometric_valid=valid,
        metadata=meta,
        probabilities=teacher_prob,
        step=0,
    )

    loss, metrics = rm.loss(student_logits, targets, valid)
    assert loss.requires_grad
    assert float(loss.item()) > 0.0
    assert metrics["region_loss"] > 0.0

    loss.backward()

    # 1. Student logits MUST have nonzero finite gradients
    assert student_logits.grad is not None
    assert torch.isfinite(student_logits.grad).all()
    assert float(student_logits.grad.abs().sum()) > 0.0

    # 2. Teacher probabilities MUST NOT receive gradients (detached targets)
    assert teacher_prob.grad is None


def test_zero_loss_on_uniform_predictions():
    b, c, z, h, w = 1, 4, 2, 16, 16
    student_logits = torch.ones(b, c, z, h, w, requires_grad=True) * 2.0
    valid = torch.ones(b, z, h, w, dtype=torch.bool)
    meta = _make_meta()

    rm = RegionManager(RegionConfig(mode="image", weight=1.0, target_num_regions=4))
    targets = rm.targets(
        image=torch.rand(b, 3, z, h, w),
        geometric_valid=valid,
        metadata=meta,
        step=0,
    )

    loss, metrics = rm.loss(student_logits, targets, valid)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_mode_none_behavior():
    rm = RegionManager(RegionConfig(mode="none"))
    b, c, z, h, w = 1, 3, 2, 8, 8
    img = torch.rand(b, c, z, h, w)
    valid = torch.ones(b, z, h, w, dtype=torch.bool)

    targets = rm.targets(img, valid, None, step=0)
    assert targets.mode == "none"

    logits = torch.randn(b, 4, z, h, w, requires_grad=True)
    loss, metrics = rm.loss(logits, targets, valid)
    assert loss.item() == 0.0
    assert metrics["region_loss"] == 0.0


def test_empty_valid_support():
    rm = RegionManager(RegionConfig(mode="image", target_num_regions=4))
    b, c, z, h, w = 1, 3, 2, 8, 8
    img = torch.rand(b, c, z, h, w)
    valid = torch.zeros(b, z, h, w, dtype=torch.bool)
    meta = _make_meta()

    targets = rm.targets(img, valid, meta, step=0)
    logits = torch.randn(b, 4, z, h, w, requires_grad=True)
    loss, metrics = rm.loss(logits, targets, valid)
    assert loss.item() == 0.0
    assert metrics["active_regions"] == 0.0


def test_volume_trainer_region_manager_integration():
    from self_audit.volume.config import default_volume_config
    from self_audit.volume.engine import VolumeTrainer

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv3d(3, 4, 1)

        def forward(self, x, valid_mask=None):
            return self.conv(x)

    cfg = default_volume_config()
    rm = RegionManager(RegionConfig(mode="image", target_num_regions=4))
    b, c, z, h, w = 1, 3, 2, 8, 8
    img = torch.rand(b, c, z, h, w)
    valid = torch.ones(b, z, h, w, dtype=torch.bool)
    meta = _make_meta()
    rm.targets(img, valid, meta, step=1)

    trainer1 = VolumeTrainer(ToyModel(), cfg, region_manager=rm)
    sd = trainer1.state_dict()
    assert sd["region_state"] is not None
    assert sd["region_state"]["last_step"] == 1

    rm2 = RegionManager(RegionConfig(mode="image"))
    trainer2 = VolumeTrainer(ToyModel(), cfg, region_manager=rm2)
    trainer2.load_state_dict(sd)
    assert rm2._last_step == 1
    assert rm2._cached_targets is not None


def test_r2_boundary_rejection_no_cross_class_interior_pooling():
    """R2 regression: confident class boundary split and rejection.

    Verifies:
    1. Astra R2 witness: 8x8x8 volume, all valid, zeros image, EMA confident class 1 on left (x<4)
       and class 3 on right (x>=4), target_num_regions=1, spatial_weight=1000, boundary_rejection=True,
       spacing [1, 1, 4].
    2. Student matching teacher incurs loss < 1e-4 (resolving original 0.499 loss bug).
    3. Student gradients at interiors do NOT push towards the wrong class.
    4. Active regions with positive weight are homogeneous: zero cross-class interior pooling.
    5. Metrics are accurate: active_regions > 0, valid_region_voxels == 384 (128 boundary voxels excluded).
    6. If min_region_size is too large to split, entire crossing region is rejected (loss 0.0, valid voxels 0).
    """
    x = torch.zeros(1, 1, 8, 8, 8)
    v = torch.ones(1, 8, 8, 8, dtype=torch.bool)
    p = torch.zeros(1, 4, 8, 8, 8)
    p[:, 1, :, :, :4] = 1.0  # Left half class 1
    p[:, 3, :, :, 4:] = 1.0  # Right half class 3

    meta = [
        {
            "patient": "A",
            "frame": 0,
            "z_indices": list(range(8)),
            "spacing_xyz_mm": [1.0, 1.0, 4.0],
            "source_image_sha256": "A" * 64,
            "native_shape_zyx": [8, 8, 8],
            "current_shape_zyx": [8, 8, 8],
            "current_affine_xyz_to_world_mm": np.diag([1.0, 1.0, 4.0, 1.0]).tolist(),
            "reversible_transform": {
                "z_indices": list(range(8)),
                "flip": False,
                "crop": {"top": 0, "left": 0, "height": 8, "width": 8, "pad_lrtb": [0, 0, 0, 0]},
                "current_shape_zyx": [8, 8, 8],
            },
        }
    ]

    m = RegionManager({
        "mode": "ema",
        "target_num_regions": 1,
        "spatial_weight": 1000.0,
        "boundary_rejection": True,
        "weight": 1.0,
        "min_region_size": 8,
    })

    t = m.targets(x, v, meta, probabilities=p, step=0)

    # 1. Matching student loss must be strictly < 1e-4
    student_logits = (p * 0.999 + 0.001 / 4).log().requires_grad_()
    loss, metrics = m.loss(student_logits, t, v)
    assert float(loss.detach()) < 1e-4, f"Loss was {float(loss.detach())}, expected < 1e-4"

    # 2. Check student gradients: no push toward wrong class in interior
    loss.backward()
    grad = student_logits.grad
    assert grad is not None
    # On left interior (x=1): correct is class 1 (grad <= 0), wrong is class 3 (grad >= 0)
    assert float(grad[0, 1, 0, 0, 1]) <= 0.0, "Expected non-positive grad for correct class 1 on left"
    assert float(grad[0, 3, 0, 0, 1]) >= 0.0, "Expected non-negative grad for wrong class 3 on left"
    # On right interior (x=6): correct is class 3 (grad <= 0), wrong is class 1 (grad >= 0)
    assert float(grad[0, 3, 0, 0, 6]) <= 0.0, "Expected non-positive grad for correct class 3 on right"
    assert float(grad[0, 1, 0, 0, 6]) >= 0.0, "Expected non-negative grad for wrong class 1 on right"

    # 3. Check that no region pools both class 1 and class 3 interiors
    active_mask = (t.region_weights[0] > 0.0)
    for r_id in torch.unique(t.region_labels[0]):
        if r_id < 0:
            continue
        r_vox = (t.region_labels[0] == r_id) & active_mask
        if not r_vox.any():
            continue
        # The class argmax of all active voxels in this region must be identical
        classes_in_region = torch.unique(p[0].argmax(dim=0)[r_vox])
        assert len(classes_in_region) == 1, f"Region {r_id} crosses classes: {classes_in_region}"

    # 4. Check accurate metrics: 384 interior voxels valid, boundary voxels excluded
    assert metrics["active_regions"] > 0
    assert metrics["valid_region_voxels"] == 384

    # 5. Fallback check: if min_region_size is too large to split, reject entire region
    m_large = RegionManager({
        "mode": "ema",
        "target_num_regions": 1,
        "spatial_weight": 1000.0,
        "boundary_rejection": True,
        "weight": 1.0,
        "min_region_size": 1000,  # too large for 256 voxels per half
    })
    t_large = m_large.targets(x, v, meta, probabilities=p, step=0)
    loss_large, metrics_large = m_large.loss(student_logits, t_large, v)
    assert float(loss_large.detach()) == 0.0
    assert metrics_large["active_regions"] == 0.0
    assert metrics_large["valid_region_voxels"] == 0.0


def test_reject_missing_detached_ema_source():
    """Verify that mode='ema' strictly requires detached teacher probabilities before cache/build."""
    b, c, z, h, w = 1, 3, 2, 8, 8
    img = torch.rand(b, c, z, h, w)
    valid = torch.ones(b, z, h, w, dtype=torch.bool)
    meta = _make_meta()

    rm = RegionManager(RegionConfig(mode="ema"))

    # 1. Neither probabilities nor features -> fail closed
    with pytest.raises(ValueError, match="requires an actual supported detached EMA source"):
        rm.targets(img, valid, meta, probabilities=None, features=None)

    # 2. Features only in mode='ema' -> explicitly reject unsupported features-only path
    features = torch.rand(b, 16, z, h, w)
    with pytest.raises(ValueError, match="does not support features-only inputs"):
        rm.targets(img, valid, meta, probabilities=None, features=features)

    # 3. Existing mode='image' remains available separately and accepts optional features
    rm_img = RegionManager(RegionConfig(mode="image"))
    t_img1 = rm_img.targets(img, valid, meta, probabilities=None, features=None)
    assert t_img1.mode == "image"
    t_img2 = rm_img.targets(img, valid, meta, probabilities=None, features=features)
    assert t_img2.mode == "image"


def test_lru_resume_capacity_restoration():
    """Astra exact R1 witness: max_cache_size restoration across load_state_dict.

    Setup: EMA refresh_every=5, max_cache_size=2.
    Step 0: cache A, B with class 1.
    State is deepcopied and loaded into a newly initialized default RegionManager() (default max_cache_size=64).
    Step 1: add C.
    Query A at Step 1 with class 3:
    In capacity=2, adding C evicted A. Querying A with class 3 returns fresh class 3 targets.
    Without restoring capacity=2, capacity=64 would keep A and return stale class 1 targets.
    """
    b, c, z, h, w = 1, 3, 2, 8, 8
    valid = torch.ones(b, z, h, w, dtype=torch.bool)

    img_a = torch.zeros(b, c, z, h, w)
    meta_a = _make_meta(patient="patient_A")
    prob_a1 = torch.zeros(b, 4, z, h, w); prob_a1[:, 1] = 1.0  # class 1
    prob_a3 = torch.zeros(b, 4, z, h, w); prob_a3[:, 3] = 1.0  # class 3

    img_b = torch.ones(b, c, z, h, w)
    meta_b = _make_meta(patient="patient_B")
    prob_b = torch.zeros(b, 4, z, h, w); prob_b[:, 1] = 1.0

    img_c = torch.full((b, c, z, h, w), 2.0)
    meta_c = _make_meta(patient="patient_C")
    prob_c = torch.zeros(b, 4, z, h, w); prob_c[:, 1] = 1.0

    m = RegionManager({"mode": "ema", "refresh_every": 5, "max_cache_size": 2})

    # Cache A, B at step 0
    t_a0 = m.targets(img_a, valid, meta_a, probabilities=prob_a1, step=0)
    assert float(t_a0.target_probabilities[:, 1].mean().item()) == 1.0
    t_b0 = m.targets(img_b, valid, meta_b, probabilities=prob_b, step=0)

    # Save state
    state = copy.deepcopy(m.state_dict())

    # Load into default RegionManager (default capacity 64)
    resumed = RegionManager()
    assert resumed.max_cache_size == 64
    resumed.load_state_dict(state)
    assert resumed.max_cache_size == 2, f"Expected restored max_cache_size=2, got {resumed.max_cache_size}"

    # Step 1: Add C into resumed manager (capacity 2 must evict oldest entry A)
    resumed.targets(img_c, valid, meta_c, probabilities=prob_c, step=1)

    # Step 1: Query A with teacher prob class 3
    t_a_resumed = resumed.targets(img_a, valid, meta_a, probabilities=prob_a3, step=1)
    # A was evicted: must compute fresh targets with class 3!
    assert float(t_a_resumed.target_probabilities[:, 3].mean().item()) == 1.0
    assert float(t_a_resumed.target_probabilities[:, 1].mean().item()) == 0.0

    # Uninterrupted manager must also evict A when C is added
    m.targets(img_c, valid, meta_c, probabilities=prob_c, step=1)
    t_a_uninterrupted = m.targets(img_a, valid, meta_a, probabilities=prob_a3, step=1)
    assert float(t_a_uninterrupted.target_probabilities[:, 3].mean().item()) == 1.0


def test_batched_resume_capacity_boundary_preserves_sample_lru_and_partitioning():
    """Astra R3 regression: a combined batch target is not a sample-cache entry.

    A/B are cached at step 0 with teacher class 1, then the state is loaded into
    a default-capacity manager.  At step 1, the uninterrupted and resumed
    managers process [A, C] with teacher class 3.  Restoring ``cached_targets``
    as an extra LRU entry would evict A and change its target to class 3.
    """
    cfg = {
        "mode": "ema",
        "refresh_every": 5,
        "max_cache_size": 2,
        "target_num_regions": 2,
        "min_region_size": 1,
    }
    b, c, z, h, w = 1, 3, 2, 8, 8
    valid = torch.ones(b, z, h, w, dtype=torch.bool)
    image_a = torch.zeros(b, c, z, h, w)
    image_b = torch.ones_like(image_a)
    image_c = torch.full_like(image_a, 2.0)
    meta_a = _make_meta(patient="resume_A")
    meta_b = _make_meta(patient="resume_B")
    meta_c = _make_meta(patient="resume_C")

    teacher_class1 = torch.zeros(b, 4, z, h, w)
    teacher_class1[:, 1] = 1.0
    teacher_class3 = torch.zeros_like(teacher_class1)
    teacher_class3[:, 3] = 1.0

    uninterrupted = RegionManager(cfg)
    step0 = uninterrupted.targets(
        torch.cat([image_a, image_b]),
        valid.repeat(2, 1, 1, 1),
        meta_a + meta_b,
        probabilities=teacher_class1.repeat(2, 1, 1, 1, 1),
        step=0,
    )
    state = copy.deepcopy(uninterrupted.state_dict())
    sample_keys = list(uninterrupted._cache)
    sample_targets = {
        key: copy.deepcopy(uninterrupted._cache[key]) for key in sample_keys
    }
    combined_hash = str(state["cache_hash"])
    assert len(sample_keys) == 2
    assert combined_hash not in sample_keys

    resumed = RegionManager()
    resumed.load_state_dict(state)
    assert resumed.max_cache_size == 2
    assert list(resumed._cache) == sample_keys
    assert len(resumed._cache) == 2
    # The last combined return survives separately, but never consumes sample
    # identity-cache capacity (including when loading into a default manager).
    assert resumed._cache_hash == combined_hash
    assert combined_hash not in resumed._cache
    assert resumed._cached_targets is not None
    assert torch.equal(resumed._cached_targets.region_labels, step0.region_labels)
    assert torch.equal(resumed._cached_targets.region_weights, step0.region_weights)
    assert torch.equal(resumed._cached_targets.target_probabilities, step0.target_probabilities)
    for key in sample_keys:
        assert torch.equal(resumed._cache[key].region_labels, sample_targets[key].region_labels)
        assert torch.equal(resumed._cache[key].region_weights, sample_targets[key].region_weights)
        assert torch.equal(
            resumed._cache[key].target_probabilities,
            sample_targets[key].target_probabilities,
        )

    next_images = torch.cat([image_a, image_c])
    next_meta = meta_a + meta_c
    next_probs = teacher_class3.repeat(2, 1, 1, 1, 1)
    target_uninterrupted = uninterrupted.targets(
        next_images,
        valid.repeat(2, 1, 1, 1),
        next_meta,
        probabilities=next_probs,
        step=1,
    )
    target_resumed = resumed.targets(
        next_images,
        valid.repeat(2, 1, 1, 1),
        next_meta,
        probabilities=next_probs,
        step=1,
    )

    # A remains the cached class-1 target in both paths; the old restore logic
    # evicted it via an extra combined entry and produced a max diff of 1.0.
    assert torch.equal(target_resumed.target_probabilities, target_uninterrupted.target_probabilities)
    assert float(target_uninterrupted.target_probabilities[0, 1].mean()) == 1.0
    assert float(target_uninterrupted.target_probabilities[0, 3].mean()) == 0.0
    assert float((target_resumed.target_probabilities - target_uninterrupted.target_probabilities).abs().max()) == 0.0

    # The same resumed state is invariant to splitting the logical batch into
    # microbatches, preserving the direct/full-batch partition contract.
    resumed_split = RegionManager()
    resumed_split.load_state_dict(state)
    target_a_split = resumed_split.targets(
        image_a, valid, meta_a, probabilities=teacher_class3, step=1
    )
    target_c_split = resumed_split.targets(
        image_c, valid, meta_c, probabilities=teacher_class3, step=1
    )
    assert torch.equal(target_a_split.target_probabilities, target_uninterrupted.target_probabilities[:1])
    assert torch.equal(target_c_split.target_probabilities, target_uninterrupted.target_probabilities[1:])


def test_per_sample_cache_accumulation_and_companion_invariance():
    """Astra accumulation counterexample from review_models_trainer.md:

    Logical batch [A, B] at step 0 with teacher prob class 1.
    Logical batch [A, C] at step 1 with teacher prob class 2, where sample A is identical.
    Under per-sample caching:
    - Full batch B=2 preserves A's cached targets from step 0 (class 1).
    - Chunked/split batch B=1 preserves A's cached targets from step 0 (class 1).
    - Max difference between full and split target probabilities on A is strictly 0.0.
    - Permuting companion [C, A] yields identical target for A.
    """
    cfg = dict(mode="ema", weight=0.1, refresh_every=5, target_num_regions=2, min_region_size=1)
    a = torch.zeros(1, 3, 2, 4, 4); b = a + 0.25; c = a + 0.5
    v = torch.ones(1, 2, 4, 4, dtype=torch.bool)
    p1 = torch.zeros(1, 4, 2, 4, 4); p1[:, 1] = 1.0  # class 1
    p2 = torch.zeros_like(p1); p2[:, 2] = 1.0        # class 2
    ma, mb, mc = [[dict(patient=x, spacing_xyz_mm=[1., 1., 1.])] for x in "ABC"]

    full = RegionManager(cfg)
    split = RegionManager(cfg)

    # Step 0: [A, B]
    full.targets(
        torch.cat([a, b]),
        v.repeat(2, 1, 1, 1),
        ma + mb,
        probabilities=p1.repeat(2, 1, 1, 1, 1),
        step=0,
    )
    for x_i, m_i in [(a, ma), (b, mb)]:
        split.targets(x_i, v, m_i, probabilities=p1, step=0)

    # Step 1: [A, C] with teacher updated to class 2
    yf = full.targets(
        torch.cat([a, c]),
        v.repeat(2, 1, 1, 1),
        ma + mc,
        probabilities=p2.repeat(2, 1, 1, 1, 1),
        step=1,
    )
    ys = split.targets(a, v, ma, probabilities=p2, step=1)

    # Both full (B=2) and split (B=1) must retain cached A (class 1), max diff strictly 0
    assert torch.equal(yf.target_probabilities[0], ys.target_probabilities[0])
    assert float(yf.target_probabilities[0, 1].mean().item()) == 1.0
    assert float(yf.target_probabilities[0, 2].mean().item()) == 0.0
    assert float((yf.target_probabilities[0] - ys.target_probabilities[0]).abs().max().item()) == 0.0

    # Permutation test: [C, A] at step 1
    full_perm = RegionManager(cfg)
    full_perm.load_state_dict(full.state_dict())
    yf_perm = full_perm.targets(
        torch.cat([c, a]),
        v.repeat(2, 1, 1, 1),
        mc + ma,
        probabilities=p2.repeat(2, 1, 1, 1, 1),
        step=1,
    )
    # Slot 1 in yf_perm is sample A; must match slot 0 of yf
    assert torch.equal(yf_perm.target_probabilities[1], yf.target_probabilities[0])
