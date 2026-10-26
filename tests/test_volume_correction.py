"""Tests for native-3D volume correction, hard manual preservation, provenance, and security gates."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

import scripts.volume_correct as volume_correct_module
import self_audit.volume.correction as correction_module
from self_audit.volume.correction import (
    PROVENANCE_MANUAL,
    PROVENANCE_ORIGINAL,
    PROVENANCE_PROPAGATED,
    PROVENANCE_PSEUDO,
    PROVENANCE_UNKNOWN,
    CorrectionResult,
    InteractiveCorrectionProtocol,
    PointPrompt,
    VolumeCorrector,
    authorize_simulation,
    compute_geodesic_distance_map_2d,
    compute_geodesic_distance_map_3d,
    compute_image_guided_affinity,
    interactive_correction_protocol,
    load_prompts,
    refine_volume_probabilities,
    resolve_prompt_to_voxel_zyx,
    save_prompts,
    simulate_clicks_from_ground_truth,
)
from self_audit.volume.fixtures import make_volume_fixture
from self_audit.volume.geometry import (
    UNKNOWN_LABEL,
    load_sidecar,
    sha256_file,
    spacing_from_affine,
    voxel_to_world,
    world_to_voxel,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_point_prompt_creation_and_dict_serialization():
    p = PointPrompt(
        coord=(12.5, 34.0, 5.0),
        coord_system="world_mm",
        target_class=2,
        prompt_type="positive",
        radius_voxels=1.5,
        weight=0.8,
        slice_index=5,
        prompt_id="test_001",
    )
    d = p.to_dict()
    assert d["coord"] == (12.5, 34.0, 5.0)
    assert d["coord_system"] == "world_mm"
    assert d["target_class"] == 2
    assert d["radius_voxels"] == 1.5

    p_restored = PointPrompt.from_dict(d)
    assert p_restored == p


def test_point_prompt_validation():
    with pytest.raises(ValueError, match="Unsupported coord_system"):
        PointPrompt(coord=(1, 2, 3), coord_system="polar")

    with pytest.raises(ValueError, match="target_class must be non-negative"):
        PointPrompt(coord=(1, 2, 3), target_class=-1)

    with pytest.raises(ValueError, match="radius_voxels must be >= 0"):
        PointPrompt(coord=(1, 2, 3), radius_voxels=-0.5)


def test_resolve_prompt_to_voxel_coordinate_systems():
    shape_zyx = (8, 32, 32)
    # 1. native_voxel_xyz: (x=10, y=20, z=3) -> (z=3, y=20, x=10)
    p_xyz = PointPrompt(coord=(10.2, 19.8, 3.1), coord_system="native_voxel_xyz")
    iz, iy, ix = resolve_prompt_to_voxel_zyx(p_xyz, shape_zyx)
    assert (iz, iy, ix) == (3, 20, 10)

    # 2. native_voxel_zyx: (z=4, y=15, x=25)
    p_zyx = PointPrompt(coord=(4.0, 15.0, 25.0), coord_system="native_voxel_zyx")
    iz, iy, ix = resolve_prompt_to_voxel_zyx(p_zyx, shape_zyx)
    assert (iz, iy, ix) == (4, 15, 25)

    # 3. world_mm with affine
    affine = np.eye(4, dtype=float)
    affine[0, 0], affine[1, 1], affine[2, 2] = 2.0, 1.5, 5.0
    affine[:3, 3] = [10.0, 20.0, 50.0]
    # Point at voxel (x=6, y=10, z=2)
    vox_xyz = np.array([6.0, 10.0, 2.0])
    world_xyz = voxel_to_world(affine, vox_xyz)
    p_world = PointPrompt(coord=tuple(world_xyz), coord_system="world_mm")
    iz, iy, ix = resolve_prompt_to_voxel_zyx(p_world, shape_zyx, affine)
    assert (iz, iy, ix) == (2, 10, 6)

    # Out of bounds rejection
    p_oob = PointPrompt(coord=(100.0, 100.0, 100.0), coord_system="native_voxel_xyz")
    with pytest.raises(ValueError, match="out of bounds"):
        resolve_prompt_to_voxel_zyx(p_oob, shape_zyx)


def test_hard_manual_preservation():
    """Manual prompt voxels must have exact label and probability 1.0."""
    zdim, h, w = 4, 16, 16
    img = np.ones((zdim, h, w), dtype=np.float32) * 0.5
    # Prior strongly favors class 0 (background)
    prior_p = np.zeros((4, zdim, h, w), dtype=np.float32)
    prior_p[0] = 0.99
    prior_p[1:] = 0.01 / 3.0

    prompt = PointPrompt(coord=(8.0, 8.0, 2.0), coord_system="native_voxel_xyz", target_class=3)
    corrector = VolumeCorrector(num_classes=4, propagation_mode="manual_only")
    res = corrector.correct(img, [prompt], prior_probabilities=prior_p, synthetic_fixture=True)

    assert res.labels[2, 8, 8] == 3
    assert np.isclose(res.probabilities[3, 2, 8, 8], 1.0)
    assert np.allclose(res.probabilities[:3, 2, 8, 8], 0.0)
    assert res.provenance[2, 8, 8] == PROVENANCE_MANUAL


def test_hard_manual_preservation_with_radius():
    zdim, h, w = 3, 10, 10
    img = np.zeros((zdim, h, w), dtype=np.float32)
    prompt = PointPrompt(
        coord=(5.0, 5.0, 1.0),
        coord_system="native_voxel_xyz",
        target_class=2,
        radius_voxels=1.5,
    )
    corrector = VolumeCorrector(num_classes=4, propagation_mode="manual_only")
    res = corrector.correct(img, [prompt], synthetic_fixture=True)

    # Center voxel
    assert res.labels[1, 5, 5] == 2
    assert res.provenance[1, 5, 5] == PROVENANCE_MANUAL
    # Immediate 6-neighbors inside sphere of radius 1.5
    for dz, dy, dx in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
        assert res.labels[1 + dz, 5 + dy, 5 + dx] == 2
        assert res.provenance[1 + dz, 5 + dy, 5 + dx] == PROVENANCE_MANUAL


def test_geodesic_propagation_respects_intensity_boundaries():
    """Geodesic propagation should spread within similar intensity and stop at sharp edge."""
    zdim, h, w = 3, 16, 16
    img = np.zeros((zdim, h, w), dtype=np.float32)
    img[:, :, :8] = 1.0  # bright left block

    # Click in bright region for class 1
    click_left = PointPrompt(coord=(4.0, 8.0, 1.0), coord_system="native_voxel_xyz", target_class=1)

    corrector = VolumeCorrector(
        num_classes=4,
        propagation_mode="geodesic_3d",
        sigma_mm=5.0,
        gamma_intensity=20.0,
        prompt_weight=0.95,
    )
    res = corrector.correct(img, [click_left], synthetic_fixture=True)

    # Voxels inside the bright region on the same slice should have high class 1 probability
    assert res.probabilities[1, 1, 8, 2] > 0.60
    assert res.labels[1, 8, 2] == 1

    # Voxels across the edge in the dark region should NOT be labeled class 1
    assert res.labels[1, 8, 12] != 1
    assert res.probabilities[1, 1, 8, 12] < 0.30


def test_geodesic_distance_anisotropic_spacing():
    img = np.zeros((5, 10, 10), dtype=np.float32)
    seeds = [(2, 5, 5)]
    # Z spacing is 8.0 mm, XY spacing is 1.0 mm
    dist = compute_geodesic_distance_map_3d(img, seeds, spacing_xyz_mm=(1.0, 1.0, 8.0), gamma_intensity=0.0)

    # In XY plane, 1 step distance = 1.0 mm
    assert np.isclose(dist[2, 5, 6], 1.0)
    assert np.isclose(dist[2, 6, 5], 1.0)
    # In Z direction, 1 step distance = 8.0 mm
    assert np.isclose(dist[3, 5, 5], 8.0)
    assert np.isclose(dist[1, 5, 5], 8.0)


def test_multi_class_competing_prompts():
    zdim, h, w = 3, 20, 20
    img = np.zeros((zdim, h, w), dtype=np.float32)
    p_rv = PointPrompt(coord=(4.0, 10.0, 1.0), target_class=1)  # RV on left
    p_lv = PointPrompt(coord=(16.0, 10.0, 1.0), target_class=3)  # LV on right

    corrector = VolumeCorrector(num_classes=4, propagation_mode="geodesic_3d", sigma_mm=8.0)
    res = corrector.correct(img, [p_rv, p_lv], synthetic_fixture=True)

    assert res.labels[1, 10, 4] == 1
    assert res.labels[1, 10, 16] == 3
    assert res.provenance[1, 10, 4] == PROVENANCE_MANUAL
    assert res.provenance[1, 10, 16] == PROVENANCE_MANUAL


def test_provenance_separation_and_counts():
    zdim, h, w = 2, 8, 8
    img = np.zeros((zdim, h, w), dtype=np.float32)
    prior_labels = np.zeros((zdim, h, w), dtype=np.uint8)
    valid_mask = np.ones((zdim, h, w), dtype=bool)
    valid_mask[0, 0, 0] = False  # one invalid voxel

    prompt = PointPrompt(coord=(4.0, 4.0, 1.0), target_class=2)
    corrector = VolumeCorrector(num_classes=4, propagation_mode="geodesic_3d", sigma_mm=5.0)
    res = corrector.correct(img, [prompt], prior_labels=prior_labels, geometric_valid=valid_mask, synthetic_fixture=True)

    # Verify provenance codes
    assert res.provenance[1, 4, 4] == PROVENANCE_MANUAL
    assert res.provenance[0, 0, 0] == PROVENANCE_UNKNOWN
    # At least some surrounding voxels should be propagated
    assert np.any(res.provenance == PROVENANCE_PROPAGATED)
    # Distant voxels should remain original
    assert np.any(res.provenance == PROVENANCE_ORIGINAL)

    # Check summary counts consistency
    s = res.summary
    total = s["total_voxels"]
    assert total == zdim * h * w
    assert s["manual_voxels"] == 1
    assert s["unknown_voxels"] == 1
    assert s["manual_voxels"] + s["propagated_voxels"] + s["original_voxels"] + s["unknown_voxels"] == total


def test_click_simulation_rejects_u_val_test():
    """Verify strict allowlist: U, VAL, TEST, TRAIN-U, TESTING, TYPO, and unproven TRAIN are rejected."""
    gt = np.zeros((3, 10, 10), dtype=np.uint8)
    pred = np.zeros((3, 10, 10), dtype=np.uint8)

    forbidden_roles = [
        "U", "u", "VAL", "val", "TEST", "test", "UNLABELED", "TEST50", "EVAL",
        "TRAIN-U", "TESTING", "TYPO", "TRAIN",  # Plain TRAIN alone is not proof of L!
    ]
    for role in forbidden_roles:
        with pytest.raises(ValueError, match="forbidden"):
            simulate_clicks_from_ground_truth(gt, pred, dataset_role=role)


def test_click_simulation_finds_error_center_on_train():
    gt = np.zeros((3, 16, 16), dtype=np.uint8)
    pred = np.zeros((3, 16, 16), dtype=np.uint8)
    # Introduce error block for class 2 in slice 1, rows 4..12, cols 4..12
    gt[1, 4:12, 4:12] = 2

    clicks = simulate_clicks_from_ground_truth(
        gt,
        pred,
        dataset_role="TRAIN-L",
        label_budget=8,
        patient_id="patient_test_001",
        synthetic_fixture=True,
        max_clicks=3,
        strategy="error_component_center",
    )
    assert len(clicks) >= 1
    first = clicks[0]
    assert first.target_class == 2
    assert first.slice_index == 1
    iz, iy, ix = resolve_prompt_to_voxel_zyx(first, gt.shape)
    assert iz == 1
    assert 5 <= iy <= 10
    assert 5 <= ix <= 10


def test_replayable_prompt_fixture_roundtrip(tmp_path):
    prompts = [
        PointPrompt(coord=(1.0, 2.0, 3.0), coord_system="native_voxel_xyz", target_class=1),
        PointPrompt(coord=(10.5, 20.2, 5.0), coord_system="world_mm", target_class=2, radius_voxels=2.0),
    ]
    p_path = tmp_path / "prompts.json"
    save_prompts(prompts, p_path)
    loaded = load_prompts(p_path)

    assert len(loaded) == len(prompts)
    for p_orig, p_load in zip(prompts, loaded):
        assert p_orig.coord == p_load.coord
        assert p_orig.coord_system == p_load.coord_system
        assert p_orig.target_class == p_load.target_class
        assert p_orig.radius_voxels == p_load.radius_voxels


def test_nifti_and_json_export(tmp_path):
    zdim, h, w = 3, 12, 12
    img = np.zeros((zdim, h, w), dtype=np.float32)
    prompt = PointPrompt(coord=(6.0, 6.0, 1.0), target_class=1)

    corrector = VolumeCorrector(num_classes=4, propagation_mode="manual_only")
    res = corrector.correct(img, [prompt], patient_id="patient_test_001", affine=np.eye(4), synthetic_fixture=True)

    nii_lab_path = tmp_path / "seg.nii.gz"
    nii_prov_path = tmp_path / "prov.nii.gz"
    json_path = tmp_path / "summary.json"

    res.export_nifti(nii_lab_path, nii_prov_path)
    res.export_json(json_path)

    assert nii_lab_path.is_file()
    assert nii_prov_path.is_file()
    assert json_path.is_file()

    # Verify NIfTI data
    nii_lab = nib.load(str(nii_lab_path))
    data = nii_lab.get_fdata()
    # NIfTI shape is (X, Y, Z) = (12, 12, 3)
    assert data.shape == (w, h, zdim)
    # Check manual preservation in exported NIfTI
    assert data[6, 6, 1] == 1

    # Verify JSON summary
    meta = json.loads(json_path.read_text())
    assert meta["patient_id"] == "patient_test_001"
    assert meta["prompt_count"] == 1
    assert meta["summary"]["manual_voxels"] == 1


def test_cli_execution_end_to_end(tmp_path):
    """Test invoking scripts/volume_correct.py CLI with TRAIN-L budget and patient."""
    img = np.ones((4, 16, 16), dtype=np.float32) * 0.2
    img_path = tmp_path / "img.nii.gz"
    nii_img = nib.Nifti1Image(img.transpose((2, 1, 0)), np.eye(4))
    nii_img.header.set_xyzt_units("mm", "sec")
    nib.save(nii_img, str(img_path))

    gt = np.zeros((4, 16, 16), dtype=np.uint8)
    gt[2, 6:10, 6:10] = 3
    gt_path = tmp_path / "gt.nii.gz"
    nii_gt = nib.Nifti1Image(gt.transpose((2, 1, 0)), np.eye(4))
    nii_gt.header.set_xyzt_units("mm", "sec")
    nib.save(nii_gt, str(gt_path))

    out_lab = tmp_path / "out_lab.nii.gz"
    out_prov = tmp_path / "out_prov.nii.gz"
    out_json = tmp_path / "out_meta.json"
    out_prompts = tmp_path / "sim_prompts.json"

    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "volume_correct.py"),
        "--image", str(img_path),
        "--simulate-from-gt", str(gt_path),
        "--dataset-role", "TRAIN-L",
        "--synthetic-fixture",
        "--label-budget", "8",
        "--patient-id", "patient_001",
        "--max-clicks", "2",
        "--out-labels", str(out_lab),
        "--out-provenance", str(out_prov),
        "--out-json", str(out_json),
        "--out-prompts", str(out_prompts),
        "--mode", "geodesic_3d",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert proc.returncode == 0
    assert out_lab.is_file()
    assert out_prov.is_file()
    assert out_json.is_file()
    assert out_prompts.is_file()

    summary = json.loads(proc.stdout)
    assert summary["status"] == "PASS"
    assert summary["dataset_role"] == "TRAIN-L"
    assert summary["label_budget"] == 8
    assert summary["prompts_applied"] > 0
    assert summary["manual_voxels"] > 0


def test_cli_rejects_val_simulation(tmp_path):
    img = np.ones((3, 10, 10), dtype=np.float32)
    img_path = tmp_path / "img.npy"
    np.save(img_path, img)
    gt = np.zeros((3, 10, 10), dtype=np.uint8)
    gt_path = tmp_path / "gt.npy"
    np.save(gt_path, gt)

    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "volume_correct.py"),
        "--image", str(img_path),
        "--simulate-from-gt", str(gt_path),
        "--dataset-role", "VAL",  # Should trigger failure!
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "Click simulation forbidden on dataset_role='VAL'" in proc.stderr


def test_cli_io_spy_rejects_forbidden_masks_before_any_read(tmp_path, monkeypatch):
    """Astra regression: CLI MUST authorize role, budget, and patient BEFORE touching mask path."""
    opened_paths: list[str] = []
    real_load_volume = volume_correct_module._load_volume

    def spy_load_volume(path, key=None):
        opened_paths.append(str(path))
        return real_load_volume(path, key)

    monkeypatch.setattr(volume_correct_module, "_load_volume", spy_load_volume)

    # Valid dummy image
    img = np.zeros((3, 8, 8), dtype=np.float32)
    img_path = tmp_path / "image.npy"
    np.save(img_path, img)

    forbidden_mask_path = tmp_path / "forbidden_eval_mask.npy"
    # Note: Even if forbidden mask exists on disk, it MUST NEVER BE OPENED
    np.save(forbidden_mask_path, np.zeros((3, 8, 8), dtype=np.uint8))

    unauthorized_invocations = [
        # 1. TEST role
        ["--image", str(img_path), "--simulate-from-gt", str(forbidden_mask_path), "--dataset-role", "TEST"],
        # 2. VAL role
        ["--image", str(img_path), "--simulate-from-gt", str(forbidden_mask_path), "--dataset-role", "VAL"],
        # 3. TRAIN-U role
        ["--image", str(img_path), "--simulate-from-gt", str(forbidden_mask_path), "--dataset-role", "TRAIN-U"],
        # 4. TESTING role
        ["--image", str(img_path), "--simulate-from-gt", str(forbidden_mask_path), "--dataset-role", "TESTING"],
        # 5. TYPO role
        ["--image", str(img_path), "--simulate-from-gt", str(forbidden_mask_path), "--dataset-role", "TYPO"],
        # 6. Default / unproven TRAIN role (DefaultTRAIN alone notproofL)
        ["--image", str(img_path), "--simulate-from-gt", str(forbidden_mask_path), "--dataset-role", "TRAIN"],
        # 7. Forged TRAIN-L without declared budget
        ["--image", str(img_path), "--simulate-from-gt", str(forbidden_mask_path), "--dataset-role", "TRAIN-L", "--patient-id", "p01"],
        # 8. Forged TRAIN-L without patient ID
        ["--image", str(img_path), "--simulate-from-gt", str(forbidden_mask_path), "--dataset-role", "TRAIN-L", "--label-budget", "8"],
    ]

    for argv in unauthorized_invocations:
        opened_paths.clear()
        with pytest.raises(ValueError, match="forbidden|required"):
            volume_correct_module.main(argv)
        # Verify that the mask was NEVER accessed or opened
        assert str(forbidden_mask_path) not in opened_paths, (
            f"SECURITY BREACH: mask {forbidden_mask_path} was accessed before authorization for {argv}"
        )


@pytest.mark.parametrize(
    ("shape", "time_index", "message"),
    [
        ((2, 1, 2, 2), -1, "non-negative"),
        ((2, 1, 2, 2), 99, "out of range"),
        ((1, 2, 2), 1, "3D"),
    ],
)
def test_cli_invalid_time_index_fails_before_prompt_or_gt_access(
    tmp_path, monkeypatch, shape, time_index, message
):
    """Frame validation must fail closed before any prompt/GT input is read."""
    image_path = tmp_path / "image.npz"
    np.savez(image_path, image=np.zeros(shape, dtype=np.float32), affine=np.eye(4))
    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_text("[]")
    gt_path = tmp_path / "gt.npy"
    np.save(gt_path, np.zeros(shape[-3:], dtype=np.uint8))

    loaded_paths: list[str] = []
    real_load_volume = volume_correct_module._load_volume

    def spy_load_volume(path, *args, **kwargs):
        loaded_paths.append(str(path))
        return real_load_volume(path, *args, **kwargs)

    def forbidden_prompt_load(_path):
        raise AssertionError("prompt input was read before invalid frame rejection")

    monkeypatch.setattr(volume_correct_module, "_load_volume", spy_load_volume)
    monkeypatch.setattr(volume_correct_module, "load_prompts", forbidden_prompt_load)

    with pytest.raises(ValueError, match=message):
        volume_correct_module.main(
            [
                "--image",
                str(image_path),
                "--prompts",
                str(prompt_path),
                "--simulate-from-gt",
                str(gt_path),
                "--dataset-role",
                "FIXTURE",
                "--synthetic-fixture",
                "--time-index",
                str(time_index),
                "--mode",
                "manual_only",
            ]
        )

    assert str(prompt_path) not in loaded_paths
    assert str(gt_path) not in loaded_paths


def test_cli_valid_time_index_selects_exact_4d_boundary_frame(tmp_path, monkeypatch):
    """4D frame 0 and final frame are selected exactly, without clamping."""
    frames = np.stack(
        [
            np.full((1, 2, 2), 3.0, dtype=np.float32),
            np.full((1, 2, 2), 7.0, dtype=np.float32),
        ],
        axis=0,
    )
    image_path = tmp_path / "image.npz"
    np.savez(image_path, image=frames, affine=np.eye(4))

    seen_frames: list[np.ndarray] = []
    original_correct = volume_correct_module.VolumeCorrector.correct

    def spy_correct(self, image_zyx, *args, **kwargs):
        seen_frames.append(np.array(image_zyx, copy=True))
        return original_correct(self, image_zyx, *args, **kwargs)

    monkeypatch.setattr(volume_correct_module.VolumeCorrector, "correct", spy_correct)

    for time_index, expected_value in ((0, 3.0), (1, 7.0)):
        assert (
            volume_correct_module.main(
                [
                    "--image",
                    str(image_path),
                    "--time-index",
                    str(time_index),
                    "--mode",
                    "manual_only",
                ]
            )
            == 0
        )
        assert np.all(seen_frames[-1] == expected_value)


def test_manual_only_sparse_unknown_preserves_255():
    """Astra bug regression: manual_only zeros((1,1,8)) prompt(0,0,0), prior_labels=full255.

    Untouched unknown voxels MUST remain 255 with PROVENANCE_UNKNOWN, never turn to background 0 ORIGINAL.
    """
    shape = (1, 1, 8)
    img = np.zeros(shape, dtype=np.float32)
    prior_labels = np.full(shape, UNKNOWN_LABEL, dtype=np.uint8)  # all 255
    prompt = PointPrompt(coord=(0.0, 0.0, 0.0), coord_system="native_voxel_xyz", target_class=1)

    corrector = VolumeCorrector(num_classes=4, propagation_mode="manual_only")
    res = corrector.correct(img, [prompt], prior_labels=prior_labels, synthetic_fixture=True)

    # Prompt voxel (0, 0, 0)
    assert res.labels[0, 0, 0] == 1
    assert res.provenance[0, 0, 0] == PROVENANCE_MANUAL

    # Untouched voxels (0, 0, 1..7) MUST remain 255 and PROVENANCE_UNKNOWN
    for x in range(1, 8):
        assert res.labels[0, 0, x] == UNKNOWN_LABEL, f"Voxel (0, 0, {x}) turned to {res.labels[0, 0, x]}"
        assert res.provenance[0, 0, x] == PROVENANCE_UNKNOWN, f"Voxel (0, 0, {x}) got provenance {res.provenance[0, 0, x]}"

    assert res.summary["manual_voxels"] == 1
    assert res.summary["unknown_voxels"] == 7
    assert res.summary["original_voxels"] == 0
    assert res.summary["propagated_voxels"] == 0
    assert res.summary["pseudo_voxels"] == 0


def test_no_prompts_all_false_geometric_valid():
    """Astra bug regression: no prompts + all-false geometric_valid must return labels 255 and UNKNOWN provenance."""
    shape = (1, 1, 8)
    img = np.zeros(shape, dtype=np.float32)
    geo_valid = np.zeros(shape, dtype=bool)  # all False

    corrector = VolumeCorrector(num_classes=4)
    res = corrector.correct(img, prompts=[], geometric_valid=geo_valid, synthetic_fixture=True)

    # All labels must be 255, all provenance must be PROVENANCE_UNKNOWN
    assert np.all(res.labels == UNKNOWN_LABEL)
    assert np.all(res.provenance == PROVENANCE_UNKNOWN)
    assert res.summary["unknown_voxels"] == 8
    assert res.summary["original_voxels"] == 0
    assert res.summary["pseudo_voxels"] == 0
    assert res.summary["manual_voxels"] == 0
    assert res.summary["propagated_voxels"] == 0


def test_sparse_prior_labels_mixed_known_and_unknown():
    """Verify sparse prior labels: known classes, unknown 255, and manual preservation with propagation."""
    zdim, h, w = 1, 1, 10
    shape = (zdim, h, w)
    img = np.zeros(shape, dtype=np.float32)

    # Mixed prior:
    # x=0..2: class 0 (background)
    # x=3..5: class 255 (unknown)
    # x=6..9: class 2 (myocardium)
    prior_lab = np.zeros(shape, dtype=np.uint8)
    prior_lab[0, 0, 3:6] = UNKNOWN_LABEL
    prior_lab[0, 0, 6:10] = 2

    # Click at x=4 (inside unknown region) to label class 1 (RV)
    prompt = PointPrompt(coord=(4.0, 0.0, 0.0), coord_system="native_voxel_xyz", target_class=1)

    corrector = VolumeCorrector(num_classes=4, propagation_mode="manual_only")
    res = corrector.correct(img, [prompt], prior_labels=prior_lab, synthetic_fixture=True)

    # Manual edit
    assert res.labels[0, 0, 4] == 1
    assert res.provenance[0, 0, 4] == PROVENANCE_MANUAL

    # Untouched unknowns (x=3 and x=5) must remain 255 UNKNOWN
    assert res.labels[0, 0, 3] == UNKNOWN_LABEL
    assert res.provenance[0, 0, 3] == PROVENANCE_UNKNOWN
    assert res.labels[0, 0, 5] == UNKNOWN_LABEL
    assert res.provenance[0, 0, 5] == PROVENANCE_UNKNOWN

    # Untouched knowns must preserve original label and ORIGINAL provenance
    assert res.labels[0, 0, 0] == 0
    assert res.provenance[0, 0, 0] == PROVENANCE_ORIGINAL
    assert res.labels[0, 0, 8] == 2
    assert res.provenance[0, 0, 8] == PROVENANCE_ORIGINAL

    # Total counts must sum exactly
    s = res.summary
    assert s["manual_voxels"] == 1
    assert s["unknown_voxels"] == 2
    assert s["original_voxels"] == 7
    assert s["manual_voxels"] + s["unknown_voxels"] + s["original_voxels"] == 10


def test_interactive_correction_protocol():
    """Verify I01 interactive correction protocol callable with explicit click budget."""
    zdim, h, w = 3, 16, 16
    img = np.zeros((zdim, h, w), dtype=np.float32)
    gt = np.zeros((zdim, h, w), dtype=np.uint8)
    gt[1, 6:10, 6:10] = 2  # target error region

    # 1. Execute protocol functional entry point
    res = interactive_correction_protocol(
        img,
        click_budget=3,
        dataset_role="TRAIN-L",
        label_budget=8,
        patient_id="patient_001",
        synthetic_fixture=True,
        ground_truth_labels=gt,
        num_classes=4,
        propagation_mode="image_guided",
    )
    assert res.click_budget == 3
    assert 0 < res.clicks_used <= 3
    assert len(res.clicks) == res.clicks_used
    assert res.patient_id == "patient_001"
    assert res.dataset_role == "TRAIN-L"
    assert res.metadata["protocol"] == "I01_interactive"
    assert res.correction_result.summary["manual_voxels"] > 0

    # 2. Execute protocol callable class
    protocol = InteractiveCorrectionProtocol(
        click_budget=2,
        dataset_role="TRAIN-L",
        label_budget=8,
        synthetic_fixture=True,
        propagation_mode="geodesic_3d",
    )
    res_callable = protocol(img, patient_id="patient_002", ground_truth_labels=gt)
    assert res_callable.click_budget == 2
    assert 0 < res_callable.clicks_used <= 2
    assert res_callable.patient_id == "patient_002"

    # 3. Protocol rejects unauthorized roles
    with pytest.raises(ValueError, match="forbidden"):
        interactive_correction_protocol(
            img,
            click_budget=3,
            dataset_role="VAL",
            ground_truth_labels=gt,
        )


def test_cli_synthetic_fixture_flag(tmp_path):
    """Test running CLI with --synthetic-fixture and --dataset-role FIXTURE."""
    img = np.ones((3, 12, 12), dtype=np.float32) * 0.3
    img_path = tmp_path / "img.npy"
    np.save(img_path, img)

    gt = np.zeros((3, 12, 12), dtype=np.uint8)
    gt[1, 4:8, 4:8] = 1
    gt_path = tmp_path / "gt.npy"
    np.save(gt_path, gt)

    out_json = tmp_path / "out.json"

    argv = [
        "--image", str(img_path),
        "--simulate-from-gt", str(gt_path),
        "--dataset-role", "FIXTURE",
        "--synthetic-fixture",
        "--max-clicks", "2",
        "--out-json", str(out_json),
    ]
    ret = volume_correct_module.main(argv)
    assert ret == 0
    assert out_json.is_file()
    data = json.loads(out_json.read_text())
    assert data["summary"]["synthetic_fixture"] is True
    assert data["summary"]["dataset_role"] == "FIXTURE"
    assert data["prompt_count"] > 0


def test_image_guided_affinity_and_refine_volume_probabilities():
    zdim, h, w = 3, 10, 10
    img = np.zeros((zdim, h, w), dtype=np.float32)
    img[1, 5, 5] = 1.0  # single bright pixel
    seeds = [(1, 5, 5)]

    aff = compute_image_guided_affinity(img, seeds, sigma_spatial_mm=5.0, gamma_intensity=10.0)
    assert np.isclose(aff[1, 5, 5], 1.0)
    assert aff[1, 5, 6] < aff[1, 5, 5]

    # Test convenience functional wrapper
    prompt = PointPrompt(coord=(5.0, 5.0, 1.0), target_class=2)
    res = refine_volume_probabilities(img, [prompt], num_classes=4, propagation_mode="image_guided", synthetic_fixture=True)
    assert res.labels[1, 5, 5] == 2
    assert res.provenance[1, 5, 5] == PROVENANCE_MANUAL


def test_geodesic_2d_propagation():
    zdim, h, w = 3, 12, 12
    img = np.zeros((zdim, h, w), dtype=np.float32)
    prompt = PointPrompt(coord=(6.0, 6.0, 1.0), target_class=1)

    corrector = VolumeCorrector(num_classes=4, propagation_mode="geodesic_2d", sigma_mm=8.0)
    res = corrector.correct(img, [prompt], synthetic_fixture=True)

    # Slice 1 should have manual click and spread
    assert res.labels[1, 6, 6] == 1
    assert res.provenance[1, 6, 6] == PROVENANCE_MANUAL


def test_pseudo_prior_provenance():
    zdim, h, w = 2, 8, 8
    img = np.zeros((zdim, h, w), dtype=np.float32)
    prior_p = np.zeros((4, zdim, h, w), dtype=np.float32)
    prior_p[0] = 1.0

    prompt = PointPrompt(coord=(4.0, 4.0, 0.0), target_class=3)
    corrector = VolumeCorrector(num_classes=4, propagation_mode="manual_only")
    res = corrector.correct(img, [prompt], prior_probabilities=prior_p, prior_is_pseudo=True, synthetic_fixture=True)

    # Click is manual
    assert res.provenance[0, 4, 4] == PROVENANCE_MANUAL
    # Unedited voxels should be labeled PROVENANCE_PSEUDO
    assert res.provenance[1, 4, 4] == PROVENANCE_PSEUDO
    assert res.summary["pseudo_voxels"] > 0
    assert res.summary["original_voxels"] == 0


def test_click_simulation_ignores_unknown_labels_and_random_strategy():
    gt = np.zeros((3, 12, 12), dtype=np.uint8)
    gt[1, 2:6, 2:6] = 255  # UNKNOWN_LABEL - must be ignored!
    gt[1, 6:10, 6:10] = 2  # Real error region
    pred = np.zeros((3, 12, 12), dtype=np.uint8)

    clicks = simulate_clicks_from_ground_truth(
        gt,
        pred,
        dataset_role="TRAIN-L",
        label_budget=8,
        patient_id="patient_001",
        synthetic_fixture=True,
        max_clicks=4,
        strategy="random_error",
        seed=123,
    )
    assert len(clicks) > 0
    for cl in clicks:
        assert cl.target_class != 255
        assert cl.target_class == 2


def test_cli_with_geometry_sidecar(tmp_path):
    # Create fixture sidecar
    *_, geometry_path, gsha = make_volume_fixture(tmp_path / "fix")
    sidecar = load_sidecar(geometry_path, gsha)
    pid = "synthetic_volume_000"

    img_path = tmp_path / "fix" / f"{pid}.npy"
    out_lab = tmp_path / "out_lab.nii.gz"
    out_prov = tmp_path / "out_prov.nii.gz"
    out_json = tmp_path / "out_meta.json"

    # Create a prompt JSON
    prompt_file = tmp_path / "test_prompt.json"
    save_prompts([PointPrompt(coord=(5.0, 5.0, 2.0), target_class=1)], prompt_file)

    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "volume_correct.py"),
        "--image", str(img_path),
        "--geometry", str(geometry_path),
        "--patient-id", pid,
        "--prompts", str(prompt_file),
        "--out-labels", str(out_lab),
        "--out-provenance", str(out_prov),
        "--out-json", str(out_json),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert proc.returncode == 0
    assert out_lab.is_file()
    assert out_prov.is_file()
    assert out_json.is_file()


def test_iterative_correction_preserves_prior_manual_edits():
    """C2 regression: iterative correction via prior_labels and prior_provenance must preserve previous manual edits."""
    zdim, h, w = 3, 16, 16
    img = np.zeros((zdim, h, w), dtype=np.float32)

    # Round 1: user edits voxel (1, 5, 5) to class 1
    p1 = PointPrompt(coord=(5.0, 5.0, 1.0), target_class=1)
    corrector = VolumeCorrector(num_classes=4, propagation_mode="geodesic_3d", sigma_mm=10.0)
    res1 = corrector.correct(img, [p1], synthetic_fixture=True)

    assert res1.labels[1, 5, 5] == 1
    assert res1.provenance[1, 5, 5] == PROVENANCE_MANUAL

    # Round 2: user edits voxel (1, 10, 10) to class 3, passing prior_labels AND prior_provenance
    # Even if propagation from (1, 10, 10) would reach (1, 5, 5), (1, 5, 5) MUST REMAIN class 1 and PROVENANCE_MANUAL!
    p2 = PointPrompt(coord=(10.0, 10.0, 1.0), target_class=3)
    res2 = corrector.correct(
        img,
        [p2],
        prior_labels=res1.labels,
        prior_provenance=res1.provenance,
        synthetic_fixture=True,
    )

    # Prior edit at (1, 5, 5) is strictly preserved as class 1 and PROVENANCE_MANUAL
    assert res2.labels[1, 5, 5] == 1
    assert res2.provenance[1, 5, 5] == PROVENANCE_MANUAL

    # New edit at (1, 10, 10) is class 3 and PROVENANCE_MANUAL
    assert res2.labels[1, 10, 10] == 3
    assert res2.provenance[1, 10, 10] == PROVENANCE_MANUAL

    # Conflicting edit policy: A later prompt at the exact same voxel supersedes the earlier manual edit
    p_conflict = PointPrompt(coord=(5.0, 5.0, 1.0), target_class=2)
    res3 = corrector.correct(
        img,
        [p_conflict],
        prior_labels=res2.labels,
        prior_provenance=res2.provenance,
        synthetic_fixture=True,
    )
    assert res3.labels[1, 5, 5] == 2
    assert res3.provenance[1, 5, 5] == PROVENANCE_MANUAL


def test_nifti_meter_to_mm_conversion(tmp_path):
    """C3 regression: NIfTI file with declared meter units (.001, .002, .003) must be converted to mm (1.0, 2.0, 3.0)."""
    # Create NIfTI file in meters
    data = np.zeros((10, 8, 4), dtype=np.float32)
    meter_affine = np.diag([0.001, 0.002, 0.003, 1.0])
    nii = nib.Nifti1Image(data, meter_affine)
    nii.header.set_xyzt_units("meter", "sec")
    nii_path = tmp_path / "volume_meters.nii.gz"
    nib.save(nii, str(nii_path))

    # Load with volume_correct._load_volume
    arr, affine = volume_correct_module._load_volume(nii_path)
    assert affine is not None
    # Spacing should be converted to mm: 1.0, 2.0, 3.0
    spacing = tuple(float(x) for x in spacing_from_affine(affine))
    assert np.isclose(spacing[0], 1.0)
    assert np.isclose(spacing[1], 2.0)
    assert np.isclose(spacing[2], 3.0)


def test_nifti_export_declares_mm_units(tmp_path):
    """C3 regression: exported NIfTI must declare mm units in header."""
    res = CorrectionResult(
        probabilities=np.zeros((4, 2, 4, 4), dtype=np.float32),
        labels=np.zeros((2, 4, 4), dtype=np.uint8),
        provenance=np.full((2, 4, 4), PROVENANCE_ORIGINAL, dtype=np.uint8),
        summary={},
        affine=np.diag([1.5, 1.5, 4.0, 1.0]),
    )
    out_nii = tmp_path / "test_out.nii.gz"
    res.export_nifti(out_labels_path=out_nii)

    loaded = nib.load(str(out_nii))
    units = loaded.header.get_xyzt_units()
    assert units[0] == "mm"


def test_prior_and_gt_affine_mismatch_fails_closed(tmp_path):
    """C3 regression: prior affine or GT affine mismatching image affine must fail closed."""
    # Geometry sidecar with affine A and shape (5, 17, 19)
    *_, geometry_path, gsha = make_volume_fixture(tmp_path / "fix", shape_tzyx=(2, 5, 17, 19))
    sidecar = load_sidecar(geometry_path, gsha)
    rec = sidecar["patients"]["synthetic_volume_000"]
    shape = tuple(rec["native_shape_zyx"])  # (5, 17, 19)

    # Use the fixture's existing image so image_sha256 matches sidecar
    img_path = tmp_path / "fix" / "synthetic_volume_000.npy"

    # Prior with conflicting affine B
    mismatched_npz = tmp_path / "mismatched.npz"
    np.savez(mismatched_npz, data=np.zeros(shape, dtype=np.uint8), affine=np.diag([99.0, 99.0, 99.0, 1.0]))

    # Prior labels affine mismatch
    argv_prior = [
        "--image", str(img_path),
        "--geometry", str(geometry_path),
        "--prior-labels", str(mismatched_npz),
        "--patient-id", "synthetic_volume_000",
        "--synthetic-fixture",
    ]
    with pytest.raises(ValueError, match=r"(?s)Prior labels affine.*image affine"):
        volume_correct_module.main(argv_prior)

    # GT affine mismatch
    argv_gt = [
        "--image", str(img_path),
        "--geometry", str(geometry_path),
        "--simulate-from-gt", str(mismatched_npz),
        "--dataset-role", "FIXTURE",
        "--synthetic-fixture",
        "--patient-id", "synthetic_volume_000",
    ]
    with pytest.raises(ValueError, match=r"(?s)Ground truth affine.*image affine"):
        volume_correct_module.main(argv_gt)


def test_sidecar_patient_binding_mismatch_fails_closed(tmp_path):
    """C3 regression: sidecar shape or hash mismatch must fail closed."""
    # Image with shape (3, 8, 8)
    img = np.zeros((3, 8, 8), dtype=np.float32)
    img_path = tmp_path / "img.npy"
    np.save(img_path, img)

    *_, geometry_path, _ = make_volume_fixture(tmp_path / "fix", shape_tzyx=(2, 5, 17, 19))

    # Mismatched shape between image (3, 8, 8) and sidecar (5, 17, 19)
    argv = [
        "--image", str(img_path),
        "--geometry", str(geometry_path),
        "--patient-id", "synthetic_volume_000",
        "--synthetic-fixture",
    ]
    with pytest.raises(ValueError, match="native_shape_zyx.*!= image shape"):
        volume_correct_module.main(argv)


def test_npz_export_creates_valid_npz_file(tmp_path):
    """C4 regression: exporting .npz must create valid .npz file (not .npz.npy)."""
    img = np.ones((2, 8, 8), dtype=np.float32) * 0.5
    img_path = tmp_path / "img.npy"
    np.save(img_path, img)

    out_lab_npz = tmp_path / "labels.npz"
    out_prov_npz = tmp_path / "prov.npz"
    out_prob_npz = tmp_path / "probs.npz"

    argv = [
        "--image", str(img_path),
        "--synthetic-fixture",
        "--out-labels", str(out_lab_npz),
        "--out-provenance", str(out_prov_npz),
        "--out-probabilities", str(out_prob_npz),
    ]
    ret = volume_correct_module.main(argv)
    assert ret == 0

    assert out_lab_npz.is_file()
    assert not Path(str(out_lab_npz) + ".npy").exists()
    lab_data = np.load(out_lab_npz)
    assert "labels" in lab_data

    assert out_prov_npz.is_file()
    prov_data = np.load(out_prov_npz)
    assert "provenance" in prov_data

    assert out_prob_npz.is_file()
    prob_data = np.load(out_prob_npz)
    assert "probabilities" in prob_data


def test_independent_nifti_export_paths(tmp_path):
    """C4 regression: requesting out-provenance alone without out-labels must export cleanly."""
    img = np.zeros((2, 8, 8), dtype=np.float32)
    img_path = tmp_path / "img.npy"
    np.save(img_path, img)

    out_prov_nii = tmp_path / "only_prov.nii.gz"
    argv = [
        "--image", str(img_path),
        "--synthetic-fixture",
        "--out-provenance", str(out_prov_nii),
    ]
    ret = volume_correct_module.main(argv)
    assert ret == 0
    assert out_prov_nii.is_file()


def test_point_prompt_weight_and_negative_type():
    """C4 regression: validate prompt weight and prompt_type support."""
    # Negative weight rejected
    with pytest.raises(ValueError, match="weight must be positive"):
        PointPrompt(coord=(1, 2, 3), weight=-1.0)

    # Unsupported prompt_type rejected
    with pytest.raises(ValueError, match="Unsupported prompt_type"):
        PointPrompt(coord=(1, 2, 3), prompt_type="unsupported_type")

    # Negative prompt suppresses target class
    zdim, h, w = 3, 10, 10
    img = np.zeros((zdim, h, w), dtype=np.float32)
    p_neg = PointPrompt(coord=(5.0, 5.0, 1.0), target_class=2, prompt_type="negative")
    corrector = VolumeCorrector(num_classes=4, propagation_mode="manual_only")
    res = corrector.correct(img, [p_neg], synthetic_fixture=True)
    assert res.labels[1, 5, 5] == 0
    assert np.isclose(res.probabilities[2, 1, 5, 5], 0.0)


def test_no_prompt_cli_does_not_crash(tmp_path):
    """C4 regression: no-prompt CLI must not crash with KeyError class_distribution."""
    img = np.zeros((2, 8, 8), dtype=np.float32)
    img_path = tmp_path / "img.npy"
    np.save(img_path, img)

    out_json = tmp_path / "out.json"
    argv = [
        "--image", str(img_path),
        "--synthetic-fixture",
        "--out-json", str(out_json),
    ]
    proc = subprocess.run([sys.executable, str(_REPO_ROOT / "scripts" / "volume_correct.py")] + argv,
                          capture_output=True, text=True, check=True)
    assert proc.returncode == 0
    out_dict = json.loads(proc.stdout)
    assert out_dict["status"] == "PASS"
    assert "class_distribution" in out_dict


def test_negative_prompt_background_correction_applied_and_persisted():
    """Witness 1 regression:
    prior_labels all 3, manual_only, PointPrompt((2,0,0), target_class=0, prompt_type='negative')
    must produce label 0, PROVENANCE_MANUAL, manual_voxels count 1, slice 2 affected,
    and persist across iterative rounds without untouched restoration overwriting it.
    """
    img = np.zeros((4, 8, 8), dtype=np.float32)
    prior_labels = np.full((4, 8, 8), 3, dtype=np.uint8)
    prompt = PointPrompt(coord=(0.0, 0.0, 2.0), coord_system="native_voxel_xyz", target_class=0, prompt_type="negative")
    corrector = VolumeCorrector(num_classes=4, propagation_mode="manual_only")
    res = corrector.correct(img, [prompt], prior_labels=prior_labels, synthetic_fixture=True)

    # Voxel (2, 0, 0) must be 0 (background) and PROVENANCE_MANUAL (NOT 3 and NOT ORIGINAL)
    assert res.labels[2, 0, 0] == 0, f"Expected 0, got {res.labels[2, 0, 0]}"
    assert res.provenance[2, 0, 0] == PROVENANCE_MANUAL, f"Expected MANUAL (2), got {res.provenance[2, 0, 0]}"
    assert res.summary["manual_voxels"] == 1
    assert 2 in res.summary["affected_slices"]
    # Untouched voxels remain class 3 with ORIGINAL provenance
    assert res.labels[0, 0, 0] == 3
    assert res.provenance[0, 0, 0] == PROVENANCE_ORIGINAL

    # Repeated / iterative manual BG edit:
    # Round 2: add another background click at (1, 1, 1) and preserve earlier manual edit at (2, 0, 0)
    p2 = PointPrompt(coord=(1.0, 1.0, 1.0), coord_system="native_voxel_xyz", target_class=0, prompt_type="negative")
    res2 = corrector.correct(
        img,
        [p2],
        prior_labels=res.labels,
        prior_provenance=res.provenance,
        synthetic_fixture=True,
    )
    assert res2.labels[2, 0, 0] == 0
    assert res2.provenance[2, 0, 0] == PROVENANCE_MANUAL
    assert res2.labels[1, 1, 1] == 0
    assert res2.provenance[1, 1, 1] == PROVENANCE_MANUAL
    assert res2.summary["manual_voxels"] == 2


def test_no_prior_no_prompts_returns_unknown_255():
    """Witness 2 regression: no prior and no prompts must return UNKNOWN 255 for labels and provenance.
    No reference means no original supervision.
    """
    img = np.zeros((2, 4, 4), dtype=np.float32)
    corrector = VolumeCorrector(num_classes=4)
    res = corrector.correct(img, prompts=[], synthetic_fixture=True)

    assert np.all(res.labels == UNKNOWN_LABEL)
    assert np.all(res.provenance == PROVENANCE_UNKNOWN)
    assert res.summary["unknown_voxels"] == 2 * 4 * 4
    assert res.summary["original_voxels"] == 0
    assert res.summary["pseudo_voxels"] == 0
    assert res.summary["manual_voxels"] == 0
    assert res.summary["propagated_voxels"] == 0


def test_noop_replay_preserves_all_prior_provenance_categories():
    """Witness 2 regression: no-op replay with
    prior_labels=[0, 1, 2, 255, 3] and
    prior_provenance=[ORIGINAL 1, PSEUDO 4, PROPAGATED 3, UNKNOWN 255, MANUAL 2]
    must preserve all provenance categories and labels without erasing pseudo/propagated lineage.
    """
    img = np.zeros((1, 1, 5), dtype=np.float32)
    prior_labels = np.array([[[0, 1, 2, UNKNOWN_LABEL, 3]]], dtype=np.uint8)
    prior_prov = np.array([[[PROVENANCE_ORIGINAL, PROVENANCE_PSEUDO, PROVENANCE_PROPAGATED, PROVENANCE_UNKNOWN, PROVENANCE_MANUAL]]], dtype=np.uint8)

    corrector = VolumeCorrector(num_classes=4, propagation_mode="manual_only")
    res = corrector.correct(
        img,
        prompts=[],
        prior_labels=prior_labels,
        prior_provenance=prior_prov,
        synthetic_fixture=True,
    )

    expected_labels = np.array([[[0, 1, 2, UNKNOWN_LABEL, 3]]], dtype=np.uint8)
    expected_prov = np.array([[[PROVENANCE_ORIGINAL, PROVENANCE_PSEUDO, PROVENANCE_PROPAGATED, PROVENANCE_UNKNOWN, PROVENANCE_MANUAL]]], dtype=np.uint8)

    assert np.array_equal(res.labels, expected_labels), f"Labels changed: {res.labels}"
    assert np.array_equal(res.provenance, expected_prov), f"Provenance erased: {res.provenance}"

    assert res.summary["original_voxels"] == 1
    assert res.summary["pseudo_voxels"] == 1
    assert res.summary["propagated_voxels"] == 1
    assert res.summary["unknown_voxels"] == 1
    assert res.summary["manual_voxels"] == 1


def test_physical_geometry_fail_closed_and_validation(tmp_path):
    """Witness 3 regression:
    1. correct() and export_nifti() without affine/spacing fail closed unless synthetic_fixture=True.
    2. Non-finite or singular affine fails closed.
    3. Inconsistent spacing vs affine fails closed.
    4. NIfTI unknown units (0 or 'unknown') rejected on real data.
    5. Recognized micron units converted correctly.
    """
    img = np.zeros((2, 4, 4), dtype=np.float32)
    prompt = PointPrompt(coord=(1.0, 1.0, 0.0), target_class=1)
    corrector = VolumeCorrector(num_classes=4)

    # 1. Real public API call without geometry fails closed
    with pytest.raises(ValueError, match="Missing physical geometry"):
        corrector.correct(img, [prompt], synthetic_fixture=False)

    # 2. Singular affine rejected
    singular_affine = np.zeros((4, 4), dtype=float)
    with pytest.raises(ValueError, match="nonsingular"):
        corrector.correct(img, [prompt], affine=singular_affine)

    # 3. Non-finite affine rejected
    nan_affine = np.eye(4, dtype=float)
    nan_affine[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        corrector.correct(img, [prompt], affine=nan_affine)

    # 4. Inconsistent spacing rejected
    valid_affine = np.diag([2.0, 2.0, 4.0, 1.0])
    with pytest.raises(ValueError, match="Inconsistent geometry"):
        corrector.correct(img, [prompt], affine=valid_affine, spacing_xyz_mm=(1.0, 1.0, 1.0))

    # 5. export_nifti without affine fails closed when synthetic_fixture=False
    res_no_affine = CorrectionResult(
        probabilities=np.zeros((4, 2, 4, 4), dtype=np.float32),
        labels=np.zeros((2, 4, 4), dtype=np.uint8),
        provenance=np.zeros((2, 4, 4), dtype=np.uint8),
        summary={},
        affine=None,
        synthetic_fixture=False,
    )
    with pytest.raises(ValueError, match="export_nifti requires known physical affine"):
        res_no_affine.export_nifti(out_labels_path=tmp_path / "test.nii.gz")

    # 6. NIfTI unknown units rejected for real data in _load_volume
    unknown_units_nii = nib.Nifti1Image(np.zeros((4, 4, 2)), np.eye(4))
    # By default nibabel sets spatial units to unknown (0)
    p_unknown = tmp_path / "unknown_units.nii.gz"
    nib.save(unknown_units_nii, str(p_unknown))
    with pytest.raises(ValueError, match="NIfTI spatial units are unknown"):
        volume_correct_module._load_volume(p_unknown, synthetic_fixture=False)

    # 7. Micron units converted to mm (.001)
    micron_nii = nib.Nifti1Image(np.zeros((4, 4, 2)), np.diag([1000.0, 2000.0, 3000.0, 1.0]))
    micron_nii.header.set_xyzt_units("micron", "sec")
    p_micron = tmp_path / "micron.nii.gz"
    nib.save(micron_nii, str(p_micron))
    _, aff_converted = volume_correct_module._load_volume(p_micron, synthetic_fixture=False)
    assert np.isclose(aff_converted[0, 0], 1.0)
    assert np.isclose(aff_converted[1, 1], 2.0)
    assert np.isclose(aff_converted[2, 2], 3.0)


def test_correctly_aligned_nifti_positive_case(tmp_path):
    """End-to-end correctly aligned NIfTI test with declared mm units and world_mm coordinate prompt."""
    img = np.ones((4, 16, 16), dtype=np.float32) * 0.1
    # Spacing: x=1.5mm, y=1.5mm, z=3.0mm, origin offset (10, 20, 30)
    affine = np.array([
        [1.5, 0.0, 0.0, 10.0],
        [0.0, 1.5, 0.0, 20.0],
        [0.0, 0.0, 3.0, 30.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    # Voxel (z=2, y=8, x=6):
    # world_x = 10.0 + 6 * 1.5 = 19.0
    # world_y = 20.0 + 8 * 1.5 = 32.0
    # world_z = 30.0 + 2 * 3.0 = 36.0
    prompt_world = PointPrompt(
        coord=(19.0, 32.0, 36.0),
        coord_system="world_mm",
        target_class=2,
    )
    corrector = VolumeCorrector(num_classes=4, propagation_mode="manual_only")
    res = corrector.correct(img, [prompt_world], affine=affine)

    assert res.labels[2, 8, 6] == 2
    assert res.provenance[2, 8, 6] == PROVENANCE_MANUAL

    out_lab = tmp_path / "aligned_out.nii.gz"
    res.export_nifti(out_labels_path=out_lab)
    assert out_lab.is_file()

    loaded = nib.load(str(out_lab))
    assert loaded.header.get_xyzt_units()[0] == "mm"
    assert np.allclose(loaded.affine, affine, atol=1e-3)
    fdata = loaded.get_fdata()
    # (X, Y, Z) in NIfTI
    assert fdata[6, 8, 2] == 2


def test_authoritative_train_manifest_binding(tmp_path):
    """Witness 4 regression:
    1. Real TRAIN-L simulation requires explicit --train-manifest and verifies patient membership.
    2. Unlabeled patient in manifest is rejected.
    3. Patient in VAL/TEST is rejected.
    4. Unproven role 'TRAIN' is rejected.
    5. Authorized TRAIN-L patient in manifest succeeds.
    6. All output reports mark simulated / additional supervision; no human annotation time claimed.
    """
    image_file = tmp_path / "p001.nii.gz"
    mask_file = tmp_path / "p001_mask_t000.nii.gz"
    image_file.write_bytes(b"declared image bytes")
    mask_file.write_bytes(b"declared mask bytes")
    manifest_file = tmp_path / "train.json"
    manifest_data = {
        "schema": "self_audit.data.v2",
        "role": "training",
        "synthetic": False,
        "label_budget": 1,
        "labeled": [
            {
                "id": "patient_001",
                "image": image_file.name,
                "image_sha256": sha256_file(image_file),
                "masks": {
                    "0": {"path": mask_file.name, "sha256": sha256_file(mask_file)}
                }
            }
        ],
        "unlabeled": [
            {"id": "patient_u99", "image": "pu99.nii.gz", "image_sha256": "1" * 64}
        ],
        "evaluation_ids": ["patient_val01"],
    }
    manifest_file.write_text(json.dumps(manifest_data, indent=2))
    m_sha = sha256_file(manifest_file)

    # 1. Missing manifest on real TRAIN-L fails closed
    with pytest.raises(ValueError, match="Real TRAIN-L simulation requires authoritative --train-manifest"):
        authorize_simulation(
            dataset_role="TRAIN-L",
            label_budget=8,
            patient_id="patient_001",
            synthetic_fixture=False,
            train_manifest=None,
        )

    # 2. Unproven TRAIN role rejected
    with pytest.raises(ValueError, match="Default 'TRAIN' alone is not proof"):
        authorize_simulation(
            dataset_role="TRAIN",
            label_budget=8,
            patient_id="patient_001",
            synthetic_fixture=False,
            train_manifest=manifest_file,
        )

    # 3. Patient in UNLABELED records rejected
    with pytest.raises(ValueError, match="UNLABELED records"):
        authorize_simulation(
            dataset_role="TRAIN-L",
            label_budget=8,
            patient_id="patient_u99",
            synthetic_fixture=False,
            train_manifest=manifest_file,
            train_manifest_sha256=m_sha,
        )

    # 4. Patient in EVALUATION records rejected
    with pytest.raises(ValueError, match="EVALUATION records"):
        authorize_simulation(
            dataset_role="TRAIN-L",
            label_budget=8,
            patient_id="patient_val01",
            synthetic_fixture=False,
            train_manifest=manifest_file,
            train_manifest_sha256=m_sha,
        )

    # 5. Non-existent patient in manifest rejected
    with pytest.raises(ValueError, match="not in authoritative TRAIN-L labeled records"):
        authorize_simulation(
            dataset_role="TRAIN-L",
            label_budget=8,
            patient_id="patient_nonexistent",
            synthetic_fixture=False,
            train_manifest=manifest_file,
            train_manifest_sha256=m_sha,
        )

    # 6. Authorized patient in labeled records passes
    auth = authorize_simulation(
        dataset_role="TRAIN-L",
        label_budget=1,
        patient_id="patient_001",
        synthetic_fixture=False,
        train_manifest=manifest_file,
        train_manifest_sha256=m_sha,
        frame_index=0,
        mask_path=mask_file,
        image_path=image_file,
    )
    assert auth["authorized"] is True
    assert auth["mode"] == "real_train_l"

    # 7. Mask path mismatch for authorized patient rejected
    with pytest.raises(ValueError, match="Mask path mismatch"):
        authorize_simulation(
            dataset_role="TRAIN-L",
            label_budget=1,
            patient_id="patient_001",
            synthetic_fixture=False,
            train_manifest=manifest_file,
            train_manifest_sha256=m_sha,
            frame_index=0,
            mask_path="wrong_mask_name.nii.gz",
            image_path=image_file,
        )


def test_cli_real_train_manifest_simulation(tmp_path):
    """Verify scripts/volume_correct.py CLI execution on real data binding --train-manifest."""
    affine = np.diag([1.5, 1.5, 2.0, 1.0])
    img = np.ones((3, 10, 10), dtype=np.float32) * 0.2
    img_path = tmp_path / "img.nii.gz"
    nii_img = nib.Nifti1Image(img.transpose((2, 1, 0)), affine)
    nii_img.header.set_xyzt_units("mm", "sec")
    nib.save(nii_img, str(img_path))

    gt = np.zeros((3, 10, 10), dtype=np.uint8)
    gt[1, 3:7, 3:7] = 2
    gt_path = tmp_path / "p001_mask_t000.nii.gz"
    nii_gt = nib.Nifti1Image(gt.transpose((2, 1, 0)), affine)
    nii_gt.header.set_xyzt_units("mm", "sec")
    nib.save(nii_gt, str(gt_path))

    manifest_file = tmp_path / "train.json"
    manifest_data = {
        "schema": "self_audit.data.v2",
        "role": "training",
        "synthetic": False,
        "label_budget": 1,
        "labeled": [
            {
                "id": "patient_001",
                "image": "img.nii.gz",
                "image_sha256": sha256_file(img_path),
                "masks": {
                    "0": {"path": gt_path.name, "sha256": sha256_file(gt_path)}
                }
            }
        ],
        "unlabeled": [],
    }
    manifest_file.write_text(json.dumps(manifest_data, indent=2))
    m_sha = sha256_file(manifest_file)

    out_lab = tmp_path / "out_lab.nii.gz"
    out_prov = tmp_path / "out_prov.nii.gz"
    out_json = tmp_path / "out_summary.json"

    argv = [
        "--image", str(img_path),
        "--simulate-from-gt", str(gt_path),
        "--dataset-role", "TRAIN-L",
        "--train-manifest", str(manifest_file),
        "--train-manifest-sha256", m_sha,
        "--label-budget", "1",
        "--patient-id", "patient_001",
        "--max-clicks", "2",
        "--out-labels", str(out_lab),
        "--out-provenance", str(out_prov),
        "--out-json", str(out_json),
    ]
    ret = volume_correct_module.main(argv)
    assert ret == 0
    assert out_lab.is_file()
    assert out_prov.is_file()
    assert out_json.is_file()

    summary = json.loads(out_json.read_text())
    assert summary["summary"]["protocol"] == "simulated_additional_supervision"
    assert summary["summary"]["supervision_type"] == "additional_supervision"
    assert summary["summary"]["simulated"] is True
    assert summary["summary"]["human_annotation_claimed"] is False
    assert summary["summary"]["human_annotation_time_sec"] is None
    assert summary["summary"]["train_manifest"] == str(manifest_file)


def _strict_real_auth_fixture(tmp_path):
    """Small hash-pinned v2 real-mode fixture for authorization-only tests."""
    image_file = tmp_path / "A_image.npy"
    mask_dir = tmp_path / "L"
    mask_dir.mkdir()
    mask_file = mask_dir / "gt.npy"
    np.save(image_file, np.zeros((2, 3, 4, 5), dtype=np.float32))
    np.save(mask_file, np.zeros((3, 4, 5), dtype=np.uint8))
    manifest_file = tmp_path / "train.json"
    manifest_file.write_text(
        json.dumps(
            {
                "schema": "self_audit.data.v2",
                "role": "training",
                "synthetic": False,
                "label_budget": 1,
                "labeled": [
                    {
                        "id": "A",
                        "image": image_file.name,
                        "image_sha256": sha256_file(image_file),
                        "masks": {
                            "0": {"path": "L/gt.npy", "sha256": sha256_file(mask_file)}
                        },
                    }
                ],
                "unlabeled": [
                    {"id": "U", "image": "U_image.npy", "image_sha256": "1" * 64}
                ],
                "evaluation_ids": ["E"],
            }
        )
    )
    return manifest_file, image_file, mask_file, sha256_file(manifest_file)


def test_real_authorization_rejects_foreign_paths_and_bad_contract_before_mask_io(tmp_path, monkeypatch):
    manifest_file, image_file, mask_file, manifest_sha = _strict_real_auth_fixture(tmp_path)
    foreign_dir = tmp_path / "foreign"
    foreign_dir.mkdir()
    foreign_mask = foreign_dir / "gt.npy"  # same basename, deliberately different full path
    foreign_mask.write_bytes(mask_file.read_bytes())

    original_sha256_file = correction_module.sha256_file
    hashed_paths: list[str] = []

    def spy_sha256_file(path):
        hashed_paths.append(str(path))
        return original_sha256_file(path)

    monkeypatch.setattr(correction_module, "sha256_file", spy_sha256_file)

    base = dict(
        dataset_role="TRAIN-L",
        synthetic_fixture=False,
        patient_id="A",
        train_manifest=manifest_file,
        train_manifest_sha256=manifest_sha,
        frame_index=0,
        mask_path=foreign_mask,
        image_path=image_file,
    )
    with pytest.raises(ValueError, match="Mask path mismatch"):
        authorize_simulation(label_budget=1, **base)
    assert str(foreign_mask) not in hashed_paths

    hashed_paths.clear()
    with pytest.raises(ValueError, match="label_budget"):
        authorize_simulation(label_budget=777, **base)
    assert str(foreign_mask) not in hashed_paths

    hashed_paths.clear()
    with pytest.raises(ValueError, match="no declared mask for frame"):
        authorize_simulation(label_budget=1, frame_index=99, **{k: v for k, v in base.items() if k != "frame_index"})
    assert str(foreign_mask) not in hashed_paths

    # A declared mask with the right path but changed bytes is rejected before
    # a caller can load it as GT.
    mask_file.write_bytes(b"tampered mask bytes")
    hashed_paths.clear()
    with pytest.raises(ValueError, match="Mask SHA-256 mismatch"):
        authorize_simulation(
            dataset_role="TRAIN-L",
            synthetic_fixture=False,
            label_budget=1,
            patient_id="A",
            train_manifest=manifest_file,
            train_manifest_sha256=manifest_sha,
            frame_index=0,
            mask_path=mask_file,
            image_path=image_file,
        )
    assert str(mask_file) in hashed_paths


def test_cli_real_foreign_mask_is_rejected_before_any_volume_read(tmp_path, monkeypatch):
    manifest_file, image_file, mask_file, manifest_sha = _strict_real_auth_fixture(tmp_path)
    foreign_mask = tmp_path / "other" / "gt.npy"
    foreign_mask.parent.mkdir()
    foreign_mask.write_bytes(mask_file.read_bytes())
    opened: list[str] = []
    original_load = volume_correct_module._load_volume

    def spy_load(path, *args, **kwargs):
        opened.append(str(path))
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(volume_correct_module, "_load_volume", spy_load)
    with pytest.raises(ValueError, match="Mask path mismatch"):
        volume_correct_module.main(
            [
                "--image",
                str(image_file),
                "--simulate-from-gt",
                str(foreign_mask),
                "--dataset-role",
                "TRAIN-L",
                "--train-manifest",
                str(manifest_file),
                "--train-manifest-sha256",
                manifest_sha,
                "--label-budget",
                "1",
                "--patient-id",
                "A",
                "--time-index",
                "0",
            ]
        )
    assert str(foreign_mask) not in opened


def test_real_authorization_rejects_generic_train_splits_before_mask_io(tmp_path):
    manifest_file, image_file, _mask_file, _ = _strict_real_auth_fixture(tmp_path)
    generic = {
        "schema": "legacy.train.v1",
        "role": "training",
        "synthetic": False,
        "train": ["A"],
        "val": [],
        "test": [],
    }
    manifest_file.write_text(json.dumps(generic))
    generic_sha = sha256_file(manifest_file)
    with pytest.raises(ValueError, match="schema"):
        authorize_simulation(
            dataset_role="TRAIN-L",
            synthetic_fixture=False,
            label_budget=1,
            patient_id="A",
            train_manifest=manifest_file,
            train_manifest_sha256=generic_sha,
            frame_index=0,
            mask_path=tmp_path / "foreign.npy",
            image_path=image_file,
        )


@pytest.mark.parametrize("mode", ["geodesic_3d", "geodesic_2d", "image_guided"])
def test_invalid_support_intensity_is_invariant_and_barriered(mode):
    """C6: unsupported intensity values cannot affect valid correction output."""
    shape = (1, 5, 9)
    image_a = np.zeros(shape, dtype=np.float32)
    image_a[:, :4, 4:] = 1.0
    image_b = image_a.copy()
    image_b[:, 4, :] = 1000.0  # invalid support only
    geometric_valid = np.ones(shape, dtype=bool)
    geometric_valid[:, 4, :] = False
    prompt = PointPrompt(coord=(1.0, 1.0, 0.0), target_class=1)

    corrector = VolumeCorrector(propagation_mode=mode)
    result_a = corrector.correct(
        image_a,
        [prompt],
        geometric_valid=geometric_valid,
        affine=np.eye(4),
        synthetic_fixture=True,
    )
    result_b = corrector.correct(
        image_b,
        [prompt],
        geometric_valid=geometric_valid,
        affine=np.eye(4),
        synthetic_fixture=True,
    )
    assert np.allclose(result_a.probabilities[:, geometric_valid], result_b.probabilities[:, geometric_valid])
    assert np.array_equal(result_a.labels[geometric_valid], result_b.labels[geometric_valid])
    assert np.all(result_a.labels[~geometric_valid] == UNKNOWN_LABEL)
    assert np.all(result_b.provenance[~geometric_valid] == PROVENANCE_UNKNOWN)


def test_invalid_prompt_and_empty_support_fail_closed():
    image = np.zeros((1, 5, 9), dtype=np.float32)
    valid = np.ones_like(image, dtype=bool)
    valid[:, 4, :] = False
    corrector = VolumeCorrector(propagation_mode="geodesic_3d")
    with pytest.raises(ValueError, match="invalid geometric support"):
        corrector.correct(
            image,
            [PointPrompt(coord=(1.0, 4.0, 0.0), target_class=1)],
            geometric_valid=valid,
            synthetic_fixture=True,
        )

    empty = np.zeros_like(valid)
    result = corrector.correct(image, [], geometric_valid=empty, synthetic_fixture=True)
    assert np.all(result.labels == UNKNOWN_LABEL)
    assert np.all(result.provenance == PROVENANCE_UNKNOWN)
