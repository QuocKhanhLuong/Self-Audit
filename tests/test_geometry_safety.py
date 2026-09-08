"""Regression tests for Wave 2.3: minimum geometry safety and guards."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from scripts.preprocess_acdc import (
    main as preprocess_acdc_main,
    validate_image_mask_geometry,
)
from src.self_audit.data import ACDCDataset
from src.self_audit.data.common import (
    VolumeRecord,
    VolumeSliceDataset,
    build_25d_triplet,
    resize_sample,
)
from src.self_audit.evaluation.metrics import annotation_metrics, surface_metrics
from src.self_audit.evaluation.volume_inference import evaluate_volume_native


def _create_mock_record(
    tmp_path: Path,
    case_id: str,
    shape: tuple[int, int, int] = (4, 200, 300),
    spacing: tuple[float, float, float] | None = (10.0, 1.5, 2.0),
) -> VolumeRecord:
    vol_dir = tmp_path / "volumes"
    mask_dir = tmp_path / "masks"
    vol_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    img = np.random.default_rng(42).standard_normal(shape).astype(np.float32)
    mask = np.zeros(shape, dtype=np.int64)
    mask[:, 10:30, 20:40] = 1

    img_path = vol_dir / f"{case_id}.npy"
    mask_path = mask_dir / f"{case_id}.npy"
    np.save(img_path, img)
    np.save(mask_path, mask)

    return VolumeRecord(
        case_id=case_id,
        patient_id=case_id.split("_")[0],
        image_path=img_path,
        mask_path=mask_path,
        spacing=spacing,
        source_format="npy",
    )


# ---------------------------------------------------------------------------
# 1. Real two-resize integration test: NIfTI preprocessor -> metadata -> ACDCDataset
# ---------------------------------------------------------------------------


def test_two_resize_production_chain_integration(tmp_path: Path) -> None:
    """Production metadata integration test across actual preprocessing -> loader resize chain."""
    raw_dir = tmp_path / "raw"
    p_dir = raw_dir / "patient001"
    p_dir.mkdir(parents=True)
    (p_dir / "Info.cfg").write_text("ED: 1\nES: 1\n", encoding="utf-8")

    # Anisotropic non-square synthetic NIfTI: H=216, W=256, Z=5; zooms: sx=1.4, sy=1.6, sz=10.0
    raw_data = np.zeros((216, 256, 5), dtype=np.float32)
    raw_data[50:150, 60:180, :] = 1.0
    raw_mask = np.zeros((216, 256, 5), dtype=np.int16)
    raw_mask[50:150, 60:180, :] = 2
    affine = np.diag([1.4, 1.6, 10.0, 1.0])

    nib.save(nib.Nifti1Image(raw_data, affine), str(p_dir / "patient001_frame01.nii.gz"))
    nib.save(nib.Nifti1Image(raw_mask, affine), str(p_dir / "patient001_frame01_gt.nii.gz"))

    # Actual Operation 1: Preprocess to target size (224, 224)
    out_dir = tmp_path / "preprocessed"
    with patch("sys.argv", ["preprocess_acdc.py", "--input", str(raw_dir), "--output", str(out_dir), "--size", "224"]):
        preprocess_acdc_main()

    # Verify saved metadata after operation 1
    meta_path = out_dir / "metadata.json"
    assert meta_path.exists()
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    p1_info = meta["volume_info"]["patient001_ED"]
    # In ACDC preprocessing: eff_spacing_y = 1.4 * 216 / 224 = 1.35, eff_spacing_x = 1.6 * 256 / 224 = 1.82857...
    expected_eff1 = (10.0, 1.4 * 216.0 / 224.0, 1.6 * 256.0 / 224.0)
    assert np.allclose(p1_info["effective_spacing"], expected_eff1)

    # Actual Operation 2: Load via ACDCDataset and resize to (256, 128)
    dataset = ACDCDataset(out_dir, image_size=(256, 128))
    sample = dataset[0]

    assert sample["source_shape"] == (5, 224, 224)
    assert sample["network_shape"] == (5, 256, 128)
    assert sample["source_axis_order"] == "ZHW"
    assert sample["network_axis_order"] == "ZHW"
    assert sample["spacing_known"] is True
    assert sample["spacing_units"] == "mm"

    # Compound effective spacing after both actual operations:
    # sh = 1.35 * (224 / 256) = 1.4 * (216 / 256) = 1.18125
    # sw = (1.6 * 256 / 224) * (224 / 128) = 1.6 * (256 / 128) = 3.2
    compound_expected = (10.0, 1.4 * 216.0 / 256.0, 1.6 * 256.0 / 128.0)
    assert np.allclose(sample["effective_spacing"], compound_expected)
    assert np.allclose(sample["spacing"], compound_expected)


# ---------------------------------------------------------------------------
# 2. Pixel/mask parity before and after patch
# ---------------------------------------------------------------------------


def test_dataset_pixel_mask_parity(tmp_path: Path) -> None:
    """Resizing and bookkeeping changes must not alter underlying pixel/mask outputs."""
    record = _create_mock_record(tmp_path, "patient002_ED", shape=(3, 100, 120), spacing=None)
    dataset = VolumeSliceDataset([record], image_size=(64, 64))
    sample = dataset[1]

    raw_vol = np.load(record.image_path)
    raw_mask = np.load(record.mask_path)
    from src.self_audit.data.common import percentile_clip_and_zscore

    norm_vol = percentile_clip_and_zscore(raw_vol)
    trip = build_25d_triplet(norm_vol, 1)
    t_img = torch.from_numpy(trip).float()
    t_mask = torch.from_numpy(raw_mask[1]).long()
    expected_img, expected_mask = resize_sample(t_img, t_mask, (64, 64))

    assert torch.equal(sample["image"], expected_img)
    assert torch.equal(sample["mask"], expected_mask)


# ---------------------------------------------------------------------------
# 3. Unknown spacing stays unknown and gives unit pixel distances
# ---------------------------------------------------------------------------


def test_unknown_spacing_stays_unknown(tmp_path: Path) -> None:
    """When spacing is None, dataset emits pixel units and unit defaults, not fabricated mm."""
    record = _create_mock_record(tmp_path, "patient003_ED", shape=(3, 80, 80), spacing=None)
    dataset = VolumeSliceDataset([record], image_size=(64, 64))
    sample = dataset[0]

    assert sample["spacing_known"] is False
    assert sample["spacing_units"] == "pixel"
    assert sample["spacing"] == (1.0, 1.0, 1.0)
    assert sample["effective_spacing"] == (1.0, 1.0, 1.0)
    assert sample["source_spacing"] == (1.0, 1.0, 1.0)


def test_annotation_metrics_uses_unit_grid_when_spacing_not_known() -> None:
    """Supplied non-unit spacing must not yield mislabeled pixel distances when spacing_known=False."""
    pred = np.zeros((32, 32), dtype=np.int64)
    true = np.zeros((32, 32), dtype=np.int64)
    pred[5:15, 5:15] = 1
    true[8:18, 8:18] = 1

    # Case A: explicit pixel scoring with no spacing
    res_none = annotation_metrics(pred, true, num_classes=2, spacing=None, spacing_known=False)
    assert res_none["distance_space"] == "pixel"
    assert res_none["spacing_known"] is False

    # Case B: non-unit spacing supplied but spacing_known=False (ignored physical spacing)
    res_ignored = annotation_metrics(
        pred, true, num_classes=2, spacing=(10.0, 5.0), spacing_known=False
    )
    assert res_ignored["distance_space"] == "pixel"
    assert res_ignored["spacing_known"] is False

    # Explicit pixel distances must be identical and independent of the ignored non-unit spacing
    assert pytest.approx(res_none["hd95"]) == res_ignored["hd95"]
    assert pytest.approx(res_none["assd"]) == res_ignored["assd"]

    # Case C: physical spacing with spacing_known=True produces scaled physical distances
    res_phys = annotation_metrics(
        pred, true, num_classes=2, spacing=(10.0, 5.0), spacing_known=True
    )
    assert res_phys["distance_space"] == "physical"
    assert res_phys["spacing_known"] is True
    assert res_phys["hd95"] != res_ignored["hd95"]


# ---------------------------------------------------------------------------
# 4. Physical spacing boundary validation (zeros, negative, NaN, Inf, empty masks)
# ---------------------------------------------------------------------------


def test_physical_spacing_boundary_validation_and_empty_masks() -> None:
    """Physical spacing must be positive, finite, and dimension-matched before empty-mask return."""
    empty_mask = np.zeros((16, 16), dtype=np.int64)

    # In surface_metrics: empty masks with invalid spacing must fail closed
    with pytest.raises(ValueError, match="spacing must be positive and finite"):
        surface_metrics(empty_mask, empty_mask, spacing=(0.0, 1.0))

    with pytest.raises(ValueError, match="spacing must be positive and finite"):
        surface_metrics(empty_mask, empty_mask, spacing=(-1.0, 1.0))

    with pytest.raises(ValueError, match="spacing must be positive and finite"):
        surface_metrics(empty_mask, empty_mask, spacing=(float("nan"), 1.0))

    with pytest.raises(ValueError, match="spacing must be positive and finite"):
        surface_metrics(empty_mask, empty_mask, spacing=(float("inf"), 1.0))

    with pytest.raises(ValueError, match="spacing must have 2 values"):
        surface_metrics(empty_mask, empty_mask, spacing=(1.0,))

    # In annotation_metrics: spacing_known=True with invalid spacing must fail closed
    with pytest.raises(ValueError, match="physical spacing must be positive and finite"):
        annotation_metrics(empty_mask, empty_mask, num_classes=2, spacing=(0.0, 1.0), spacing_known=True)

    with pytest.raises(ValueError, match="physical spacing must be positive and finite"):
        annotation_metrics(empty_mask, empty_mask, num_classes=2, spacing=(-1.0, 1.0), spacing_known=True)

    with pytest.raises(ValueError, match="physical spacing must be positive and finite"):
        annotation_metrics(empty_mask, empty_mask, num_classes=2, spacing=(float("nan"), 1.0), spacing_known=True)

    with pytest.raises(ValueError, match="spacing must have 2 values"):
        annotation_metrics(empty_mask, empty_mask, num_classes=2, spacing=(1.0,), spacing_known=True)


# ---------------------------------------------------------------------------
# 5. Actual default DataLoader collation across multiple cases
# ---------------------------------------------------------------------------


def test_actual_dataloader_collates_multiple_cases_with_mixed_geometry(tmp_path: Path) -> None:
    """Default DataLoader collates multiple cases with different source H/W, known and unknown spacing."""
    r1 = _create_mock_record(tmp_path / "c1", "p1_ED", shape=(1, 200, 300), spacing=(10.0, 1.5, 2.0))
    r2 = _create_mock_record(tmp_path / "c2", "p2_ED", shape=(1, 216, 256), spacing=None)
    r3 = _create_mock_record(tmp_path / "c3", "p3_ED", shape=(1, 180, 180), spacing=(8.0, 1.25, 1.25))
    r4 = _create_mock_record(tmp_path / "c4", "p4_ED", shape=(1, 240, 320), spacing=None)

    dataset = VolumeSliceDataset([r1, r2, r3, r4], image_size=(128, 128))
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    batch = next(iter(loader))
    assert batch["image"].shape == (4, 3, 128, 128)
    assert batch["mask"].shape == (4, 128, 128)

    # Boolean spacing_known collated into Tensor([B])
    assert torch.equal(batch["spacing_known"], torch.tensor([True, False, True, False]))
    assert batch["spacing_units"] == ["mm", "pixel", "mm", "pixel"]

    # Source shape collates into list of 3 tensors [Tensor([Z...]), Tensor([H...]), Tensor([W...])]
    assert len(batch["source_shape"]) == 3
    source_h = batch["source_shape"][1]
    assert torch.equal(source_h, torch.tensor([200, 216, 180, 240]))


# ---------------------------------------------------------------------------
# 6. Preprocessing geometry validation (shape and affine)
# ---------------------------------------------------------------------------


def test_validate_image_mask_geometry_mismatches() -> None:
    """Preprocessing validates shape and affine compatibility."""
    affine1 = np.eye(4)
    affine2 = np.diag([2.0, 2.0, 5.0, 1.0])

    data = np.zeros((20, 20, 5), dtype=np.float32)
    img_nii = nib.Nifti1Image(data, affine1)
    mask_matching = nib.Nifti1Image(data.astype(np.int16), affine1)
    mask_diff_shape = nib.Nifti1Image(np.zeros((20, 20, 6), dtype=np.int16), affine1)
    mask_diff_affine = nib.Nifti1Image(data.astype(np.int16), affine2)

    validate_image_mask_geometry(img_nii, mask_matching, "test_case")

    with pytest.raises(ValueError, match="Image/mask shape mismatch"):
        validate_image_mask_geometry(img_nii, mask_diff_shape, "test_case")

    with pytest.raises(ValueError, match="Image/mask affine mismatch"):
        validate_image_mask_geometry(img_nii, mask_diff_affine, "test_case")


# ---------------------------------------------------------------------------
# 7. Partial-pair failure without --no-skip
# ---------------------------------------------------------------------------


def test_preprocess_partial_pair_fails(tmp_path: Path) -> None:
    """Preprocessor fails loudly on partial pair without --no-skip instead of silently overwriting."""
    out_dir = tmp_path / "preprocessed"
    vol_dir = out_dir / "volumes"
    mask_dir = out_dir / "masks"
    vol_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    # Partial pair: volume exists on disk, mask does not
    np.save(vol_dir / "patient001_ED.npy", np.zeros((224, 224, 5), dtype=np.float32))

    raw_dir = tmp_path / "raw"
    patient_dir = raw_dir / "patient001"
    patient_dir.mkdir(parents=True)
    (patient_dir / "Info.cfg").write_text("ED: 1\nES: 1\n", encoding="utf-8")
    raw_arr = np.zeros((216, 256, 5), dtype=np.float32)
    aff = np.diag([1.4, 1.4, 10.0, 1.0])
    nib.save(nib.Nifti1Image(raw_arr, aff), str(patient_dir / "patient001_frame01.nii.gz"))
    nib.save(nib.Nifti1Image(raw_arr.astype(np.int16), aff), str(patient_dir / "patient001_frame01_gt.nii.gz"))

    test_args = ["preprocess_acdc.py", "--input", str(raw_dir), "--output", str(out_dir), "--size", "224"]
    with patch("sys.argv", test_args):
        with pytest.raises(ValueError, match="Partial pair detected"):
            preprocess_acdc_main()


# ---------------------------------------------------------------------------
# 8. Skip path preflight: validates both shapes, rejects malformed json & size changes
# ---------------------------------------------------------------------------


def test_preprocess_skip_validates_both_shapes_and_preserves_subset_metadata(tmp_path: Path) -> None:
    """Skip path validates volume and mask shapes and preserves old metadata for retained arrays."""
    out_dir = tmp_path / "preprocessed"
    vol_dir = out_dir / "volumes"
    mask_dir = out_dir / "masks"
    vol_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    # Create dummy existing outputs for patient001 (ED) and patient002 (ED)
    np.save(vol_dir / "patient001_ED.npy", np.zeros((224, 224, 5), dtype=np.float32))
    np.save(mask_dir / "patient001_ED.npy", np.zeros((224, 224, 5), dtype=np.uint8))
    np.save(vol_dir / "patient002_ED.npy", np.zeros((224, 224, 4), dtype=np.float32))
    np.save(mask_dir / "patient002_ED.npy", np.zeros((224, 224, 4), dtype=np.uint8))

    historical_metadata = {
        "dataset": "ACDC",
        "target_size": [224, 224],
        "volume_info": {
            "patient001_ED": {"num_slices": 5, "legacy_marker": "p1_retained"},
            "patient002_ED": {"num_slices": 4, "legacy_marker": "p2_retained"},
        },
    }
    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps(historical_metadata, indent=2), encoding="utf-8")

    # Create raw directory containing ONLY patient003 (a completely NEW patient)
    raw_dir = tmp_path / "raw"
    p3_dir = raw_dir / "patient003"
    p3_dir.mkdir(parents=True)
    (p3_dir / "Info.cfg").write_text("ED: 1\nES: 1\n", encoding="utf-8")
    raw_arr = np.zeros((216, 256, 5), dtype=np.float32)
    aff = np.diag([1.4, 1.4, 10.0, 1.0])
    nib.save(nib.Nifti1Image(raw_arr, aff), str(p3_dir / "patient003_frame01.nii.gz"))
    nib.save(nib.Nifti1Image(raw_arr.astype(np.int16), aff), str(p3_dir / "patient003_frame01_gt.nii.gz"))

    # Test 1: Changed target size (256 != 224) with existing retained outputs must fail globally BEFORE any writes
    with patch("sys.argv", ["preprocess_acdc.py", "--input", str(raw_dir), "--output", str(out_dir), "--size", "256"]):
        with pytest.raises(ValueError, match="declares target_size .* which differs from requested size"):
            preprocess_acdc_main()

    # Test 2: Malformed existing JSON must fail closed, not reset to {}
    meta_path.write_text("{malformed_json: true", encoding="utf-8")
    with patch("sys.argv", ["preprocess_acdc.py", "--input", str(raw_dir), "--output", str(out_dir), "--size", "224"]):
        with pytest.raises(ValueError, match="is malformed"):
            preprocess_acdc_main()

    # Restore valid metadata
    meta_path.write_text(json.dumps(historical_metadata, indent=2), encoding="utf-8")

    # Test 3: Process subset/new patient003. patient001 and patient002 are retained and MUST NOT be dropped
    with patch("sys.argv", ["preprocess_acdc.py", "--input", str(raw_dir), "--output", str(out_dir), "--size", "224"]):
        preprocess_acdc_main()

    with open(meta_path, encoding="utf-8") as f:
        updated_meta = json.load(f)

    assert "patient001_ED" in updated_meta["volume_info"]
    assert "patient002_ED" in updated_meta["volume_info"]
    assert "patient003_ED" in updated_meta["volume_info"]
    assert updated_meta["volume_info"]["patient002_ED"]["legacy_marker"] == "p2_retained"


# ---------------------------------------------------------------------------
# 9. Native evaluation: strict_physical deferred and bypasses rejected
# ---------------------------------------------------------------------------


def test_evaluate_volume_native_strict_physical_deferred_and_bypasses_rejected() -> None:
    """Strict physical path is deferred, and all malformed/single/inconsistent affine bypasses fail closed."""
    pred = np.zeros((3, 64, 64), dtype=np.int64)
    pred[:, 10:20, 10:20] = 1
    target = np.zeros((3, 64, 64), dtype=np.int64)
    target[:, 12:22, 12:22] = 1
    diag_aff = np.diag([1.5, 1.5, 10.0, 1.0])

    # 1. Missing ground truth raises ValueError
    with pytest.raises(ValueError, match="evaluate_volume_native requires real native ground truth"):
        evaluate_volume_native(pred, None)

    # 2. strict_physical=True raises ValueError (deferred pending real transform chain)
    with pytest.raises(ValueError, match="strict_physical evaluation is DEFERRED"):
        evaluate_volume_native(pred, target, strict_physical=True, spacing=(10.0, 1.5, 1.5))

    # 3. Single affine without paired target_affine raises ValueError
    with pytest.raises(ValueError, match="Single affine supplied without paired target_affine"):
        evaluate_volume_native(pred, target, spacing=(10.0, 1.5, 1.5), affine=diag_aff, target_affine=None)

    with pytest.raises(ValueError, match="Single target_affine supplied without paired affine"):
        evaluate_volume_native(pred, target, spacing=(10.0, 1.5, 1.5), affine=None, target_affine=diag_aff)

    # 4. Malformed 1x1 affine raises ValueError
    with pytest.raises(ValueError, match="affine must be a 4x4 matrix"):
        evaluate_volume_native(pred, target, spacing=(10.0, 1.5, 1.5), affine=[[1]], target_affine=[[1]])

    # 5. Inconsistent spacing against affine column norms raises ValueError
    with pytest.raises(ValueError, match="Supplied spacing .* is inconsistent with affine voxel dimensions"):
        evaluate_volume_native(pred, target, spacing=(5.0, 2.0, 2.0), affine=diag_aff, target_affine=diag_aff)

    # 6. Sheared affine raises ValueError
    sheared_aff = np.array([
        [1.5, 0.5, 0.0, 0.0],
        [0.2, 1.5, 0.0, 0.0],
        [0.0, 0.0, 10.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    with pytest.raises(ValueError, match="Non-axis-aligned or sheared physical grids are not supported"):
        evaluate_volume_native(pred, target, spacing=(10.0, 1.5, 1.5), affine=sheared_aff, target_affine=sheared_aff)

    # 7. Mismatched affine between prediction and target raises ValueError
    diff_aff = np.diag([2.0, 2.0, 10.0, 1.0])
    with pytest.raises(ValueError, match="Image/prediction affine does not match target ground-truth affine"):
        evaluate_volume_native(pred, target, spacing=(10.0, 1.5, 1.5), affine=diag_aff, target_affine=diff_aff)


# ---------------------------------------------------------------------------
# 10. Native evaluation unverified path suppresses native-mm surface metrics
# ---------------------------------------------------------------------------


def test_evaluate_volume_native_unverified_suppresses_native_mm() -> None:
    """Unverified native path computes native Dice but suppresses native-mm surface metrics."""
    pred = np.zeros((3, 64, 64), dtype=np.int64)
    pred[:, 10:20, 10:20] = 1
    target = np.zeros((3, 64, 64), dtype=np.int64)
    target[:, 12:22, 12:22] = 1

    block = evaluate_volume_native(pred, target, spacing=(10.0, 1.5, 1.5))

    # Native Dice is evaluated
    assert block["metric_space"] == "volume_native"
    assert "macro_dice" in block
    assert "per_class_dice" in block

    # Native-mm surface metrics are SUPPRESSED
    assert "hd95_mm" not in block
    assert "assd_mm" not in block
    assert "per_class_hd95_mm" not in block
    assert "per_class_assd_mm" not in block

    # Explicitness
    assert block["native_mm_available"] is False
    assert "Native-mm surface metrics (HD95, ASSD) are suppressed" in block["native_mm_note"]
    assert block["native_reconstruction_verification"] == "DEFERRED"
    assert block["geometry_verification"]["status"] == "DEFERRED"
    assert block["geometry_verification"]["verified_evidence"] is False
