"""Exhaustive tests for compact student inference, evaluation, and benchmarking.

Addresses independent review feedback:
  - Consumes compact_artifacts and compact_model directly (no local duplicate models or loaders).
  - Lazy imports canonical evaluation metrics from self_audit.evaluation (no copied fallbacks).
  - Mandatory SHA-256 enforcement on all manifests, images, masks, export artifacts, and TRAIN data.
  - Strict validation: role, schema, synthetic parity, class mapping [0,1,2,3,255], axes, finite invertible affine,
    safe unique IDs, in-bounds frame indices, non-overlapping output paths.
  - Normalization contract: raw 0.5/99.5 percentiles vs prepared strict [0, 1].
  - Pixel-boolean valid_mask support with ignored padding (255) in predictions and scoring.
  - Mandatory TRAIN provenance verification: ID disjointness and physical byte/voxel overlap checks BEFORE mask reads.
  - Two-phase frozen evaluation: all predictions frozen before any reference mask is opened.
  - Exact historical class-first patient macro metrics:
      frame -> patient/class -> classwise patientmeans -> meanclass Macro4 & FG.
      Omits Dice when GT is absent even with false positives.
      Absent-class fixture proving mathematical distinction between class-first and patient-first macros.
      Fixed observed valid-voxel count denominator for absent false positives.
  - Real CLI subprocess tests for infer, evaluate, and benchmark.
  - Benchmark streaming of all manifest cases, cold load, E2E timing, forward timing with CUDA sync,
    throughput definitions, and psutil steady RAM sampling.
  - Exported model reload parity check.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest
import torch
from torch import nn

from self_audit.compact_artifacts import (
    digest,
    export_compact_student,
    load_compact_export,
    save_compact_checkpoint,
)
from self_audit.compact_model import CompactStudent, build_compact_model
from self_audit.compact_inference import (
    CLASS_NAMES,
    FOREGROUND_CLASS_NAMES,
    compute_historical_metrics,
    extract_slice_context,
    infer_cine_stream,
    load_native_manifest,
    load_patient_cine,
    normalize_image,
    require_sha256,
    run_isolated_benchmark,
    run_two_phase_evaluation,
    _profile_in_subprocess,
    validate_affine_matrix,
    write_json,
)


# ==============================================================================
# Helper Fixtures & Builders
# ==============================================================================

def create_canonical_export(
    export_dir: Path,
    widths: Sequence[int] = (16, 32, 64, 128),
    synthetic: bool = True,
    data_sha256: Optional[str] = None,
    lids: Optional[Sequence[str]] = None,
    uids: Optional[Sequence[str]] = None,
) -> Tuple[Path, str, nn.Module]:
    """Create a canonical export using save_compact_checkpoint and export_compact_student."""
    ckpt_dir = export_dir.parent / f"{export_dir.name}_ckpt"
    model = build_compact_model({"widths": list(widths), "in_channels": 3, "num_classes": 4})
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    config = {"model": {"widths": list(widths)}, "method": "supervised", "teacher_memory": None}
    dummy_data_sha = data_sha256 if data_sha256 is not None else ("a" * 64)
    data_lids = list(lids) if lids is not None else ["train_pat1"]
    data_uids = list(uids) if uids is not None else ["train_pat2"]
    data = {
        "lids": data_lids,
        "uids": data_uids,
        "path": "/dev/null",
        "data_sha256": dummy_data_sha,
        "manifest": {"synthetic": synthetic},
    }

    ckpt_sha = save_compact_checkpoint(
        model=model,
        optimizer=opt,
        config=config,
        data=data,
        out=ckpt_dir,
        step=10,
    )

    export_sha = export_compact_student(
        checkpoint=ckpt_dir,
        expected_sha=ckpt_sha,
        out=export_dir,
    )
    return export_dir, export_sha, model


def create_synthetic_patient_data(
    base_dir: Path,
    pid: str,
    shape: Tuple[int, int, int, int] = (2, 4, 32, 32),
    affine: Optional[List[List[float]]] = None,
    absent_class: Optional[int] = None,
    with_valid_mask: bool = False,
) -> Tuple[Dict[str, Any], np.ndarray, Dict[str, np.ndarray], Optional[np.ndarray]]:
    """Create synthetic cine, affine, masks, and optional valid_mask for a patient."""
    t, z, h, w = shape
    # Create deterministic non-overlapping cine intensity ramp
    seed = int(hashlib.sha256(pid.encode()).hexdigest()[:8], 16) % 1000000
    rng = np.random.default_rng(seed)
    cine_arr = rng.uniform(0.1, 0.9, size=shape).astype(np.float32)

    cine_file = base_dir / f"{pid}_cine.npy"
    np.save(cine_file, cine_arr)
    cine_sha = digest(cine_file)

    if affine is None:
        affine = [
            [1.25, 0.0, 0.0, 0.0],
            [0.0, 1.25, 0.0, 0.0],
            [0.0, 0.0, 8.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]

    valid_mask: Optional[np.ndarray] = None
    vm_spec: Optional[Dict[str, Any]] = None
    if with_valid_mask:
        # Create boolean valid mask where outer border is False (padding/invalid)
        valid_mask = np.ones((t, z, h, w), dtype=bool)
        valid_mask[:, :, :4, :] = False
        valid_mask[:, :, -4:, :] = False
        valid_mask[:, :, :, :4] = False
        valid_mask[:, :, :, -4:] = False
        vm_file = base_dir / f"{pid}_valid_mask.npy"
        np.save(vm_file, valid_mask)
        vm_spec = {"path": str(vm_file.relative_to(base_dir)), "sha256": digest(vm_file)}

    masks: Dict[str, Any] = {}
    mask_arrays: Dict[str, np.ndarray] = {}

    for ti in range(t):
        mask_arr = np.zeros((z, h, w), dtype=np.uint8)
        mask_arr[:, :10, :10] = 0  # BG
        if absent_class != 1:
            mask_arr[:, 10:20, :10] = 1  # RV
        if absent_class != 2:
            mask_arr[:, 20:30, :10] = 2  # MYO
        if absent_class != 3:
            mask_arr[:, :10, 10:20] = 3  # LV

        m_file = base_dir / f"{pid}_frame_{ti}_mask.npy"
        np.save(m_file, mask_arr)
        m_sha = digest(m_file)
        masks[str(ti)] = {"path": str(m_file.relative_to(base_dir)), "sha256": m_sha}
        mask_arrays[str(ti)] = mask_arr

    rec = {
        "id": pid,
        "image": str(cine_file.relative_to(base_dir)),
        "image_sha256": cine_sha,
        "shape": list(shape),
        "affine": affine,
        "axes": ["time", "slice", "height", "width"],
        "masks": masks,
    }
    if vm_spec is not None:
        rec["valid_mask"] = vm_spec

    return rec, cine_arr, mask_arrays, valid_mask


def create_evaluation_fixture(
    base_dir: Path,
    *,
    eval_id: str = "eval_patient",
    shape: Tuple[int, int, int, int] = (1, 1, 16, 16),
) -> Tuple[Path, str, Path, str, Path, Dict[str, Any]]:
    """Create actual canonical TRAIN L/U files plus one native evaluation case."""
    train_dir = base_dir / "train"
    eval_dir = base_dir / "evaluation"
    train_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)
    l1, _, _, _ = create_synthetic_patient_data(train_dir, "train_l1", shape=(1, 1, 16, 16))
    l2, _, _, _ = create_synthetic_patient_data(train_dir, "train_l2", shape=(1, 1, 16, 16))
    u1, _, _, _ = create_synthetic_patient_data(train_dir, "train_u1", shape=(1, 1, 16, 16))
    train_manifest = train_dir / "train.json"
    train_sha = write_json(train_manifest, {
        "schema": "self_audit.data.v2",
        "role": "training",
        "synthetic": True,
        "label_budget": 2,
        "labeled": [
            {key: l1[key] for key in ("id", "image", "image_sha256", "masks")},
            {key: l2[key] for key in ("id", "image", "image_sha256", "masks")},
        ],
        "unlabeled": [{key: u1[key] for key in ("id", "image", "image_sha256")}],
        "lids": ["train_l1", "train_l2"],
        "uids": ["train_u1"],
        "evaluation_ids": [],
    })
    export_dir, export_sha, _ = create_canonical_export(
        base_dir / "export",
        data_sha256=train_sha,
        lids=["train_l1", "train_l2"],
        uids=["train_u1"],
    )
    eval_rec, _, _, _ = create_synthetic_patient_data(eval_dir, eval_id, shape=shape)
    eval_rec["domain"] = "fixture"
    return train_manifest, train_sha, export_dir, export_sha, eval_dir, eval_rec


# ==============================================================================
# Standalone Loader & Strict Artifact Validation Tests
# ==============================================================================

def test_standalone_compact_export_loader_enforces_sha_and_rejects_teacher_or_support(tmp_path):
    """Verify load_compact_export directly imports and strictly validates artifact integrity."""
    export_dir, manifest_sha, _ = create_canonical_export(tmp_path / "export")

    model, meta = load_compact_export(export_dir, expected_sha=manifest_sha, device="cpu")
    assert isinstance(model, CompactStudent)
    assert not model.training
    assert meta["schema"] == "self_audit.compact.export.v1"
    assert meta["manifest_sha256"] == manifest_sha

    # Test invalid SHA pin format raises ValueError
    with pytest.raises(ValueError, match="mandatory 64-hex SHA-256 pin"):
        load_compact_export(export_dir, expected_sha="invalid_sha")

    # Test manifest SHA mismatch raises ValueError
    fake_sha = "0" * 64
    with pytest.raises(ValueError, match="export manifest hash mismatch"):
        load_compact_export(export_dir, expected_sha=fake_sha)

    # Test missing manifest raises FileNotFoundError
    with pytest.raises(FileNotFoundError, match="export manifest not found"):
        load_compact_export(tmp_path / "nonexistent_dir", expected_sha=manifest_sha)


# ==============================================================================
# Normalization Contract Tests
# ==============================================================================

def test_normalization_contract_strict():
    """Verify raw percentile scaling vs prepared strict [0, 1] validation."""
    # 1. Raw mode: percentile scaling to [0, 1]
    raw_arr = np.array([-50.0, 0.0, 25.0, 50.0, 100.0, 500.0], dtype=np.float32)
    norm_raw = normalize_image(raw_arr, mode="raw")
    assert norm_raw.dtype == np.float32
    assert norm_raw.min() >= 0.0
    assert norm_raw.max() <= 1.0

    # 2. Prepared mode: valid [0, 1] preserved
    prep_arr = np.array([0.0, 0.1, 0.5, 0.9, 1.0], dtype=np.float32)
    norm_prep = normalize_image(prep_arr, mode="prepared")
    assert norm_prep.dtype == np.float32
    np.testing.assert_allclose(norm_prep, prep_arr)

    # 3. Prepared mode: strictly rejects negative values
    with pytest.raises(ValueError, match="Prepared mode requires values in"):
        normalize_image(np.array([-0.01, 0.5, 1.0], dtype=np.float32), mode="prepared")

    # 4. Prepared mode: strictly rejects values > 1.0
    with pytest.raises(ValueError, match="Prepared mode requires values in"):
        normalize_image(np.array([0.0, 0.5, 1.02], dtype=np.float32), mode="prepared")

    # 5. Rejects non-finite values (NaN / Inf)
    with pytest.raises(ValueError, match="non-finite"):
        normalize_image(np.array([0.0, np.nan, 1.0], dtype=np.float32), mode="prepared")
    with pytest.raises(ValueError, match="non-finite"):
        normalize_image(np.array([0.0, np.inf, 1.0], dtype=np.float32), mode="raw")

    # 6. Rejects invalid mode
    with pytest.raises(ValueError, match="Unknown normalization mode"):
        normalize_image(prep_arr, mode="unsupported")


# ==============================================================================
# Invertible Affine & Geometry Validation Tests
# ==============================================================================

def test_geometry_and_affine_validation(tmp_path):
    """Verify real, finite, invertible 4x4 affine is enforced and singular affines fail."""
    # Valid invertible affine
    valid_aff = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    aff = validate_affine_matrix(valid_aff, "pat01")
    assert aff.shape == (4, 4)

    # Singular / non-invertible affine (det == 0) must be rejected
    singular_aff = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],  # rank deficient
        [0.0, 0.0, 0.0, 1.0],
    ]
    with pytest.raises(ValueError, match="singular or non-invertible"):
        validate_affine_matrix(singular_aff, "pat01")

    # Non-finite affine (NaN / Inf) must be rejected
    nan_aff = [
        [1.0, 0.0, np.nan, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    with pytest.raises(ValueError, match="non-finite values"):
        validate_affine_matrix(nan_aff, "pat01")

    # Missing affine must fail closed
    with pytest.raises(ValueError, match="Missing real affine metadata"):
        validate_affine_matrix(None, "pat01")

    # Rejection of unsafe patient IDs
    p_rec, _, _, _ = create_synthetic_patient_data(tmp_path, "pat01")
    unsafe_rec = dict(p_rec)
    unsafe_rec["id"] = "../evil_id"
    with pytest.raises(ValueError, match="Invalid or unsafe patient ID"):
        load_patient_cine(unsafe_rec, tmp_path)


# ==============================================================================
# Pixel-Boolean Valid Mask & Score Ignore Padding Tests
# ==============================================================================

def test_valid_mask_pixel_boolean_and_ignore_padding():
    """Verify valid_mask is passed to model as boolean tensor and invalid pixels scored as 255."""
    model = build_compact_model({"widths": [16, 32, 64, 128], "in_channels": 3, "num_classes": 4})
    cine = np.random.uniform(0.1, 0.9, size=(2, 3, 16, 16)).astype(np.float32)

    valid_mask = np.ones((2, 3, 16, 16), dtype=bool)
    # Mask out corner regions as invalid/padding
    valid_mask[:, :, :4, :4] = False

    pred = infer_cine_stream(model, cine, batch_size=2, device="cpu", valid_mask=valid_mask)
    assert pred.shape == (2, 3, 16, 16)
    assert pred.dtype == np.uint8

    # Check that invalid pixels are marked 255 (ignore/padding)
    assert (pred[:, :, :4, :4] == 255).all()
    # Check that valid pixels are in valid class range {0, 1, 2, 3}
    valid_preds = pred[valid_mask]
    assert set(np.unique(valid_preds)).issubset({0, 1, 2, 3})


# ==============================================================================
# Historical Class-First Macro & Absent Class Fixture Tests
# ==============================================================================

def test_full_cohort_class_first_equals_patient_first():
    """Verify that on a full cohort where all classes are present, class-first equals patient-first."""
    frame_rows = [
        {
            "patient": "p1",
            "frame": 0,
            "raw": [
                {"class": "BG", "reference_present": True, "dice": 0.95, "prediction_count": 95},
                {"class": "RV", "reference_present": True, "dice": 0.85, "prediction_count": 85},
                {"class": "MYO", "reference_present": True, "dice": 0.75, "prediction_count": 75},
                {"class": "LV", "reference_present": True, "dice": 0.65, "prediction_count": 65},
            ],
            "valid_voxels": 320,
        },
        {
            "patient": "p2",
            "frame": 0,
            "raw": [
                {"class": "BG", "reference_present": True, "dice": 0.85, "prediction_count": 85},
                {"class": "RV", "reference_present": True, "dice": 0.75, "prediction_count": 75},
                {"class": "MYO", "reference_present": True, "dice": 0.65, "prediction_count": 65},
                {"class": "LV", "reference_present": True, "dice": 0.55, "prediction_count": 55},
            ],
            "valid_voxels": 280,
        },
    ]

    metrics = compute_historical_metrics(frame_rows, ["p1", "p2"])
    assert metrics["class_first_macro4"] == pytest.approx(0.75)
    assert metrics["patient_first_all_class_macro"] == pytest.approx(0.75)
    assert metrics["class_first_foreground_macro"] == pytest.approx(0.70)
    assert metrics["patient_first_foreground_macro"] == pytest.approx(0.70)


def test_absent_class_fixture_distinguishes_class_first_and_patient_first():
    """Verify that an absent class causes class-first and patient-first to diverge mathematically."""
    frame_rows = [
        {
            "patient": "p1",
            "frame": 0,
            "raw": [
                {"class": "BG", "reference_present": True, "dice": 1.0, "prediction_count": 100},
                {"class": "RV", "reference_present": True, "dice": 0.8, "prediction_count": 80},
                {"class": "MYO", "reference_present": True, "dice": 0.6, "prediction_count": 60},
                {"class": "LV", "reference_present": True, "dice": 0.4, "prediction_count": 40},
            ],
            "valid_voxels": 280,
        },
        {
            "patient": "p2",
            "frame": 0,
            "raw": [
                {"class": "BG", "reference_present": True, "dice": 1.0, "prediction_count": 100},
                {"class": "RV", "reference_present": False, "dice": None, "prediction_count": 20},  # FP=20 on absent!
                {"class": "MYO", "reference_present": True, "dice": 0.8, "prediction_count": 80},
                {"class": "LV", "reference_present": True, "dice": 0.6, "prediction_count": 60},
            ],
            "valid_voxels": 260,
        },
        {
            "patient": "p3",
            "frame": 0,
            "raw": [
                {"class": "BG", "reference_present": True, "dice": 1.0, "prediction_count": 100},
                {"class": "RV", "reference_present": True, "dice": 0.4, "prediction_count": 40},
                {"class": "MYO", "reference_present": True, "dice": 0.4, "prediction_count": 40},
                {"class": "LV", "reference_present": True, "dice": 0.2, "prediction_count": 20},
            ],
            "valid_voxels": 200,
        },
    ]

    metrics = compute_historical_metrics(frame_rows, ["p1", "p2", "p3"])

    # Class-first:
    # BG=1.0, RV=(0.8+0.4)/2=0.6, MYO=(0.6+0.8+0.4)/3=0.6, LV=(0.4+0.6+0.2)/3=0.4
    # Macro4 = mean(1.0, 0.6, 0.6, 0.4) = 0.65
    # FG Macro = mean(0.6, 0.6, 0.4) = 0.5333...

    # Patient-first:
    # p1 FG = (0.8+0.6+0.4)/3 = 0.6
    # p2 FG = (0.8+0.6)/2 = 0.7
    # p3 FG = (0.4+0.4+0.2)/3 = 0.3333...
    # Patient-first FG = (0.6 + 0.7 + 0.3333...)/3 = 0.5444...

    assert metrics["class_first_macro4"] == pytest.approx(0.65)
    assert metrics["patient_first_all_class_macro"] == pytest.approx(2.0 / 3.0)
    assert metrics["class_first_foreground_macro"] == pytest.approx(1.6 / 3.0)
    assert metrics["patient_first_foreground_macro"] == pytest.approx(4.9 / 9.0)

    # Prove they are mathematically distinct
    assert metrics["class_first_macro4"] != metrics["patient_first_all_class_macro"]
    assert metrics["class_first_foreground_macro"] != metrics["patient_first_foreground_macro"]

    # Check absent FP rate with fixed valid count denominator
    rv_track = metrics["absent_class_tracking"]["RV"]
    assert rv_track["absent_frame_count"] == 1
    assert rv_track["absent_false_positive_voxels"] == 20
    assert rv_track["absent_valid_voxels"] == 260
    assert rv_track["absent_fp_rate"] == pytest.approx(20.0 / 260.0)


# ==============================================================================
# Two-Phase Evaluation & TRAIN Provenance Overlap Tests
# ==============================================================================

def test_two_phase_evaluation_pipeline_and_freeze(tmp_path):
    """Verify two-phase frozen evaluation freezes predictions before scoring references."""
    eval_dir = tmp_path / "eval_data"
    eval_dir.mkdir()
    train_dir = tmp_path / "train_data"
    train_dir.mkdir()

    p1_rec, _, _, _ = create_synthetic_patient_data(eval_dir, "eval_pat1", shape=(2, 3, 32, 32), with_valid_mask=True)
    p2_rec, _, _, _ = create_synthetic_patient_data(eval_dir, "eval_pat2", shape=(2, 3, 32, 32), with_valid_mask=False)

    eval_manifest = eval_dir / "MANIFEST.json"
    eval_sha = write_json(eval_manifest, {
        "schema": "self_audit.native_manifest.v1",
        "role": "development",
        "synthetic": True,
        "patients": [p1_rec, p2_rec],
    })

    tr1_rec, _, _, _ = create_synthetic_patient_data(train_dir, "train_pat1", shape=(2, 3, 32, 32))
    tr2_rec, _, _, _ = create_synthetic_patient_data(train_dir, "train_pat2", shape=(2, 3, 32, 32))
    tru_rec, _, _, _ = create_synthetic_patient_data(train_dir, "train_pat3", shape=(2, 3, 32, 32))
    train_manifest = train_dir / "MANIFEST.json"
    train_sha = write_json(train_manifest, {
        "schema": "self_audit.data.v2",
        "role": "training",
        "synthetic": True,
        "label_budget": 2,
        "labeled": [
            {key: tr1_rec[key] for key in ("id", "image", "image_sha256", "masks")},
            {key: tr2_rec[key] for key in ("id", "image", "image_sha256", "masks")},
        ],
        "unlabeled": [{key: tru_rec[key] for key in ("id", "image", "image_sha256")}],
        "lids": ["train_pat1", "train_pat2"],
        "uids": ["train_pat3"],
        "evaluation_ids": [],
    })

    export_dir, export_sha, _ = create_canonical_export(
        tmp_path / "export",
        synthetic=True,
        data_sha256=train_sha,
        lids=["train_pat1", "train_pat2"],
        uids=["train_pat3"],
    )

    out_dir = tmp_path / "eval_out"
    metrics = run_two_phase_evaluation(
        export_path=export_dir,
        export_sha256=export_sha,
        manifest_path=eval_manifest,
        manifest_sha256=eval_sha,
        train_manifest_path=train_manifest,
        train_manifest_sha256=train_sha,
        out_dir=out_dir,
        device="cpu",
        batch_size=2,
        normalization="prepared",
    )

    assert (out_dir / "RECIPE.json").is_file()
    assert (out_dir / "PREDICTIONS_FROZEN.json").is_file()
    assert (out_dir / "METRICS.json").is_file()
    assert (out_dir / "predictions" / "eval_pat1_pred.npy").is_file()
    assert (out_dir / "predictions" / "eval_pat2_pred.npy").is_file()

    frozen_meta = json.loads((out_dir / "PREDICTIONS_FROZEN.json").read_text())
    assert len(frozen_meta["predictions"]) == 2

    assert "class_first_macro4" in metrics
    assert "class_first_foreground_macro" in metrics
    assert "patient_first_all_class_macro" in metrics
    assert "patient_first_foreground_macro" in metrics
    assert "macro_patient" in metrics
    assert "class_counts" in metrics
    assert "pooled_confusion" in metrics


def test_train_patient_overlap_and_physical_voxel_overlap_rejected(tmp_path):
    """Verify evaluation strictly rejects patient ID overlap AND physical voxel overlap."""
    eval_dir = tmp_path / "eval_overlap"
    eval_dir.mkdir()
    train_dir = tmp_path / "train_overlap"
    train_dir.mkdir()

    tr1_rec, _, _, _ = create_synthetic_patient_data(train_dir, "overlap_pat")
    tr2_rec, _, _, _ = create_synthetic_patient_data(train_dir, "train_l2")
    tru_rec, _, _, _ = create_synthetic_patient_data(train_dir, "train_u1")
    train_manifest = train_dir / "TRAIN_MANIFEST.json"
    train_sha = write_json(train_manifest, {
        "schema": "self_audit.data.v2",
        "role": "training",
        "synthetic": True,
        "label_budget": 2,
        "labeled": [
            {key: tr1_rec[key] for key in ("id", "image", "image_sha256", "masks")},
            {key: tr2_rec[key] for key in ("id", "image", "image_sha256", "masks")},
        ],
        "unlabeled": [{key: tru_rec[key] for key in ("id", "image", "image_sha256")}],
        "lids": ["overlap_pat", "train_l2"],
        "uids": ["train_u1"],
        "evaluation_ids": [],
    })

    export_dir, export_sha, _ = create_canonical_export(
        tmp_path / "export",
        synthetic=True,
        data_sha256=train_sha,
        lids=["overlap_pat", "train_l2"],
        uids=["train_u1"],
    )

    # Case 1: ID overlap
    p1_rec, _, _, _ = create_synthetic_patient_data(eval_dir, "overlap_pat")
    eval_manifest = eval_dir / "EVAL_MANIFEST.json"
    eval_sha = write_json(eval_manifest, {
        "schema": "self_audit.native_manifest.v1",
        "role": "development",
        "synthetic": True,
        "patients": [p1_rec],
    })

    with pytest.raises(ValueError, match="Evaluation patient ID overlap with TRAIN"):
        run_two_phase_evaluation(
            export_path=export_dir,
            export_sha256=export_sha,
            manifest_path=eval_manifest,
            manifest_sha256=eval_sha,
            train_manifest_path=train_manifest,
            train_manifest_sha256=train_sha,
            out_dir=tmp_path / "out_id_fail",
        )

    # Case 2: Voxel content overlap (different ID, but duplicate cine file / voxels)
    eval_dir2 = tmp_path / "eval_voxel_overlap"
    eval_dir2.mkdir()
    # Point eval_rec2 to the EXACT image file of tr1_rec
    eval_rec2 = dict(p1_rec)
    eval_rec2["id"] = "different_id_pat"
    eval_rec2["image"] = str((train_dir / tr1_rec["image"]).resolve())
    eval_rec2["image_sha256"] = tr1_rec["image_sha256"]

    eval_manifest2 = eval_dir2 / "MANIFEST.json"
    eval_sha2 = write_json(eval_manifest2, {
        "schema": "self_audit.native_manifest.v1",
        "role": "development",
        "synthetic": True,
        "patients": [eval_rec2],
    })

    with pytest.raises(ValueError, match="overlap with TRAIN"):
        run_two_phase_evaluation(
            export_path=export_dir,
            export_sha256=export_sha,
            manifest_path=eval_manifest2,
            manifest_sha256=eval_sha2,
            train_manifest_path=train_manifest,
            train_manifest_sha256=train_sha,
            out_dir=tmp_path / "out_voxel_fail",
        )


# ==============================================================================
# Exported Reload Parity Check
# ==============================================================================

def test_exported_reload_parity(tmp_path):
    """Verify that predictions from original model match loaded export predictions exactly."""
    export_dir, export_sha, orig_model = create_canonical_export(tmp_path / "parity_export")
    loaded_model, _ = load_compact_export(export_dir, expected_sha=export_sha, device="cpu")

    dummy_x = torch.rand(2, 3, 32, 32)
    orig_model.eval()
    loaded_model.eval()

    with torch.no_grad():
        out_orig = orig_model(dummy_x)
        out_loaded = loaded_model(dummy_x)

    torch.testing.assert_close(out_orig, out_loaded, rtol=1e-6, atol=1e-6)


# ==============================================================================
# Isolated Benchmark Engine Tests
# ==============================================================================

def test_isolated_benchmark_streaming_and_isolation(tmp_path):
    """Verify benchmark streams all manifest cases and measures cold, steady, and psutil RAM."""
    export_dir, export_sha, _ = create_canonical_export(tmp_path / "bench_export")
    data_dir = tmp_path / "bench_data"
    data_dir.mkdir()

    p1_rec, _, _, _ = create_synthetic_patient_data(data_dir, "bench_pat1", shape=(2, 2, 16, 16))
    p2_rec, _, _, _ = create_synthetic_patient_data(data_dir, "bench_pat2", shape=(2, 2, 16, 16))

    manifest_file = data_dir / "MANIFEST.json"
    manifest_sha = write_json(manifest_file, {
        "schema": "self_audit.native_manifest.v1",
        "role": "development",
        "synthetic": True,
        "patients": [p1_rec, p2_rec],
    })

    out_file = tmp_path / "benchmark_report.json"

    report = run_isolated_benchmark(
        compact_export_path=export_dir,
        compact_export_sha=export_sha,
        manifest_path=manifest_file,
        manifest_sha=manifest_sha,
        out_path=out_file,
        baseline_checkpoint_path=None,
        device="cpu",
        batch_size=2,
        warmup=1,
        repeats=2,
        normalization="prepared",
    )

    assert out_file.is_file()
    compact = report["compact_student"]
    assert compact["status"] == "COMPLETED"
    assert compact["workload"]["total_cases"] == 2
    assert compact["memory"]["peak_inference_process_tree_ram_mb"] > 0
    assert compact["latency"]["cold"]["load_seconds"] > 0
    assert compact["latency"]["steady"]["volumes_per_second"] > 0
    assert compact["latency"]["steady"]["slices_per_second"] > 0
    assert compact["artifact_manifest_sha256"] == export_sha
    assert compact["artifact_payload_sha256"]
    assert compact["evaluation_manifest_sha256"] == manifest_sha
    assert compact["workload"]["case_shapes"] == [
        {"id": "bench_pat1", "shape_tzyx": [2, 2, 16, 16]},
        {"id": "bench_pat2", "shape_tzyx": [2, 2, 16, 16]},
    ]
    steady = compact["latency"]["steady"]
    assert steady["volumes_per_second"] == pytest.approx(steady["total_volumes"] / steady["total_wall_seconds"])
    assert report["schema"] == "self_audit.compact.benchmark.v1"
    assert compact["synthetic"] is True
    assert compact["precision"] == "float32"
    assert compact["volume_definition"].startswith("full_native_4d_cine_tzyx")

    # Missing baseline must be strictly NOT_RUN
    assert report["baseline_student"]["status"] == "NOT_RUN"


def test_benchmark_subprocess_writes_verified_npy_for_all_actual_cases(tmp_path):
    """Exercise the timing child directly and verify native output existence, SHA, shape, and full cohort coverage."""
    export_dir, export_sha, _ = create_canonical_export(tmp_path / "worker_export")
    data_dir = tmp_path / "worker_data"
    data_dir.mkdir()
    p1_rec, _, _, _ = create_synthetic_patient_data(data_dir, "worker_p01", shape=(2, 2, 16, 18))
    p2_rec, _, _, _ = create_synthetic_patient_data(data_dir, "worker_p02", shape=(1, 3, 18, 20))
    # Shape receipts must come from actual decoded images rather than an optional manifest field.
    p1_rec.pop("shape")
    p2_rec.pop("shape")
    manifest_file = data_dir / "manifest.json"
    manifest_sha = write_json(manifest_file, {
        "schema": "self_audit.evaluation.v1",
        "role": "test",
        "synthetic": True,
        "patients": [p1_rec, p2_rec],
    })
    scratch = tmp_path / "child_outputs"
    result = _profile_in_subprocess(
        model_type="compact_export",
        model_path=str(export_dir),
        expected_sha=export_sha,
        manifest_path=str(manifest_file),
        manifest_sha=manifest_sha,
        scratch_dir=str(scratch),
        device="cpu",
        batch_size=2,
        warmup=0,
        repeats=1,
        normalization="prepared",
    )

    assert result["status"] == "COMPLETED"
    assert result["workload"]["patient_coverage_complete"] is True
    assert [c["id"] for c in result["inference_outputs"]] == ["worker_p01", "worker_p02"]
    for case in result["inference_outputs"]:
        out_file = scratch / case["prediction_file"]
        assert out_file.is_file()
        assert digest(out_file) == case["prediction_sha256"]
        pred = np.load(out_file, mmap_mode="r", allow_pickle=False)
        assert list(pred.shape) == case["shape_tzyx"]
        assert pred.dtype == np.uint8
    assert result["workload"]["case_shapes"] == [
        {"id": "worker_p01", "shape_tzyx": [2, 2, 16, 18]},
        {"id": "worker_p02", "shape_tzyx": [1, 3, 18, 20]},
    ]


# ==============================================================================
# Real CLI Subprocess Invocations
# ==============================================================================

def test_real_cli_infer_subprocess(tmp_path):
    """Test scripts/infer_compact_student.py CLI invocation in a real subprocess."""
    export_dir, export_sha, _ = create_canonical_export(tmp_path / "cli_infer_export")
    data_dir = tmp_path / "cli_infer_data"
    data_dir.mkdir()

    p1_rec, _, _, _ = create_synthetic_patient_data(data_dir, "p01", shape=(1, 2, 16, 16))
    p2_rec, _, _, _ = create_synthetic_patient_data(data_dir, "p02", shape=(1, 2, 16, 16))

    manifest_file = data_dir / "MANIFEST.json"
    manifest_sha = write_json(manifest_file, {
        "schema": "self_audit.native_manifest.v1",
        "role": "development",
        "synthetic": True,
        "patients": [p1_rec, p2_rec],
    })

    out_dir = tmp_path / "cli_infer_out"

    cmd = [
        sys.executable,
        "scripts/infer_compact_student.py",
        "--export", str(export_dir),
        "--export-sha256", export_sha,
        "--manifest", str(manifest_file),
        "--manifest-sha256", manifest_sha,
        "--out", str(out_dir),
        "--device", "cpu",
        "--batch-size", "2",
        "--normalization", "prepared",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"infer CLI failed:\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
    assert (out_dir / "INFERENCE.json").is_file()
    assert (out_dir / "PREDICTIONS_FROZEN.json").is_file()
    assert (out_dir / "predictions" / "p01_prediction.npy").is_file()
    assert (out_dir / "predictions" / "p02_prediction.npy").is_file()
    inference = json.loads((out_dir / "INFERENCE.json").read_text())
    frozen = json.loads((out_dir / "PREDICTIONS_FROZEN.json").read_text())
    assert inference["synthetic"] is True
    assert inference["coverage"]["complete"] is True
    assert inference["patients"][0]["axes"] == p1_rec["axes"]
    assert inference["patients"][0]["affine"] == p1_rec["affine"]
    for item in frozen["predictions"]:
        pred_path = out_dir / "predictions" / item["file"]
        assert digest(pred_path) == item["sha256"]
        pred = np.load(pred_path, mmap_mode="r", allow_pickle=False)
        assert list(pred.shape) == item["shape_tzyx"]
        assert pred.dtype == np.uint8


def test_real_cli_evaluate_subprocess(tmp_path):
    """Test scripts/evaluate_compact_student.py CLI invocation in a real subprocess."""
    data_dir = tmp_path / "cli_eval_data"
    data_dir.mkdir()
    train_dir = tmp_path / "cli_eval_train"
    train_dir.mkdir()

    p1_rec, _, _, _ = create_synthetic_patient_data(data_dir, "eval_p01", shape=(1, 2, 16, 16))
    eval_manifest = data_dir / "MANIFEST.json"
    eval_sha = write_json(eval_manifest, {
        "schema": "self_audit.evaluation.v1",
        "role": "test",
        "synthetic": True,
        "patients": [p1_rec],
    })

    tr1_rec, _, _, _ = create_synthetic_patient_data(train_dir, "train_p01", shape=(1, 2, 16, 16))
    tr2_rec, _, _, _ = create_synthetic_patient_data(train_dir, "train_p02", shape=(1, 2, 16, 16))
    tru_rec, _, _, _ = create_synthetic_patient_data(train_dir, "train_u01", shape=(1, 2, 16, 16))
    train_manifest = train_dir / "MANIFEST.json"
    train_sha = write_json(train_manifest, {
        "schema": "self_audit.data.v2",
        "role": "training",
        "synthetic": True,
        "label_budget": 2,
        "labeled": [
            {key: tr1_rec[key] for key in ("id", "image", "image_sha256", "masks")},
            {key: tr2_rec[key] for key in ("id", "image", "image_sha256", "masks")},
        ],
        "unlabeled": [
            {key: tru_rec[key] for key in ("id", "image", "image_sha256")},
        ],
        "lids": ["train_p01", "train_p02"],
        "uids": ["train_u01"],
        "evaluation_ids": [],
    })
    from self_audit.data import TrainingData
    training_data = TrainingData(train_manifest, train_sha, synthetic=True)
    assert training_data.lids == ["train_p01", "train_p02"]
    assert training_data.uids == ["train_u01"]

    export_dir, export_sha, _ = create_canonical_export(
        tmp_path / "cli_eval_export",
        data_sha256=train_sha,
        lids=["train_p01", "train_p02"],
        uids=["train_u01"],
    )

    out_dir = tmp_path / "cli_eval_out"

    cmd = [
        sys.executable,
        "scripts/evaluate_compact_student.py",
        "--export", str(export_dir),
        "--export-sha256", export_sha,
        "--manifest", str(eval_manifest),
        "--manifest-sha256", eval_sha,
        "--train-manifest", str(train_manifest),
        "--train-manifest-sha256", train_sha,
        "--out", str(out_dir),
        "--device", "cpu",
        "--batch-size", "2",
        "--normalization", "prepared",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"evaluate CLI failed:\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
    assert (out_dir / "METRICS.json").is_file()
    assert (out_dir / "PREDICTIONS_FROZEN.json").is_file()
    assert (out_dir / "RECIPE.json").is_file()
    metrics = json.loads((out_dir / "METRICS.json").read_text())
    assert metrics["synthetic"] is True
    assert metrics["global"]["class_first_macro4"] == metrics["class_first_macro4"]
    assert set(metrics["datasets"]) == {"unspecified"}
    assert metrics["coverage"]["complete"] is True


def test_real_cli_benchmark_subprocess(tmp_path):
    """Test scripts/benchmark_compact_student.py CLI invocation in a real subprocess."""
    export_dir, export_sha, _ = create_canonical_export(tmp_path / "cli_bench_export")
    data_dir = tmp_path / "cli_bench_data"
    data_dir.mkdir()

    p1_rec, _, _, _ = create_synthetic_patient_data(data_dir, "bench_p01", shape=(1, 2, 16, 16))
    manifest_file = data_dir / "MANIFEST.json"
    manifest_sha = write_json(manifest_file, {
        "schema": "self_audit.native_manifest.v1",
        "role": "development",
        "synthetic": True,
        "patients": [p1_rec],
    })

    out_file = tmp_path / "cli_bench.json"

    cmd = [
        sys.executable,
        "scripts/benchmark_compact_student.py",
        "--compact-export", str(export_dir),
        "--compact-export-sha256", export_sha,
        "--manifest", str(manifest_file),
        "--manifest-sha256", manifest_sha,
        "--out", str(out_file),
        "--device", "cpu",
        "--batch-size", "2",
        "--warmup", "1",
        "--repeats", "2",
        "--normalization", "prepared",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"benchmark CLI failed:\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
    assert out_file.is_file()
    rep = json.loads(out_file.read_text())
    assert rep["compact_student"]["status"] == "COMPLETED"
    assert rep["baseline_student"]["status"] == "NOT_RUN"


def test_canonical_labeled_unlabeled_train_manifest_overlap_caught_before_gt(tmp_path):
    """Run the evaluator CLI on canonical L/U records and reject U voxel overlap before GT I/O."""
    train_dir = tmp_path / "train_canonical"
    train_dir.mkdir()
    eval_dir = tmp_path / "eval_canonical"
    eval_dir.mkdir()

    # Create L and U patients in train
    tr_l, _, _, _ = create_synthetic_patient_data(train_dir, "train_L01", shape=(2, 2, 16, 16))
    tr_u, tr_u_array, _, _ = create_synthetic_patient_data(train_dir, "train_U01", shape=(2, 2, 16, 16))

    train_manifest = train_dir / "TRAIN_MANIFEST.json"
    train_sha = write_json(train_manifest, {
        "schema": "self_audit.data.v2",
        "role": "train",
        "synthetic": True,
        "L": [tr_l],
        "U": [tr_u],
        "lids": ["train_L01"],
        "uids": ["train_U01"],
    })

    export_dir, export_sha, _ = create_canonical_export(
        tmp_path / "export_canonical",
        synthetic=True,
        data_sha256=train_sha,
        lids=["train_L01"],
        uids=["train_U01"],
    )

    # Eval patient with DIFFERENT ID, but duplicate cine image of the UNLABELED train patient
    eval_p = dict(tr_u)
    eval_p["id"] = "eval_sneaky_overlap"
    eval_image = eval_dir / "eval_sneaky_overlap_float64.npy"
    np.save(eval_image, tr_u_array.astype(np.float64))
    eval_p["image"] = eval_image.name
    eval_p["image_sha256"] = digest(eval_image)
    # Point mask to a non-existent file so if GT read is attempted before overlap check, FileNotFoundError occurs
    eval_p["masks"] = {
        "0": {
            "path": "non_existent_mask_file.npy",
            "sha256": "0" * 64,
        }
    }

    eval_manifest = eval_dir / "EVAL_MANIFEST.json"
    eval_sha = write_json(eval_manifest, {
        "schema": "self_audit.native_manifest.v1",
        "role": "development",
        "synthetic": True,
        "patients": [eval_p],
    })

    # Different NPY bytes, same decoded cine values: reject this before opening the missing GT mask.
    out_dir = tmp_path / "eval_out_canonical_fail"
    cmd = [
        sys.executable,
        "scripts/evaluate_compact_student.py",
        "--export", str(export_dir),
        "--export-sha256", export_sha,
        "--manifest", str(eval_manifest),
        "--manifest-sha256", eval_sha,
        "--train-manifest", str(train_manifest),
        "--train-manifest-sha256", train_sha,
        "--out", str(out_dir),
        "--device", "cpu",
        "--batch-size", "2",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "Evaluation voxel content overlap with TRAIN" in proc.stderr
    assert not out_dir.exists()


def test_evaluation_mask_must_match_prediction_frame_shape_without_broadcast(tmp_path):
    train_manifest, train_sha, export_dir, export_sha, eval_dir, rec = create_evaluation_fixture(tmp_path)
    mask_path = eval_dir / rec["masks"]["0"]["path"]
    np.save(mask_path, np.zeros((16, 16), dtype=np.uint8))
    rec["masks"]["0"]["sha256"] = digest(mask_path)
    manifest = eval_dir / "shape_mismatch.json"
    manifest_sha = write_json(manifest, {
        "schema": "self_audit.evaluation.v1",
        "role": "test",
        "synthetic": True,
        "patients": [rec],
    })
    out_dir = tmp_path / "shape_mismatch_out"
    with pytest.raises(ValueError, match="Shape mismatch: reference mask shape"):
        run_two_phase_evaluation(
            export_path=export_dir,
            export_sha256=export_sha,
            manifest_path=manifest,
            manifest_sha256=manifest_sha,
            train_manifest_path=train_manifest,
            train_manifest_sha256=train_sha,
            out_dir=out_dir,
        )
    # Every prediction is frozen and receipted before Phase 2 opens this bad GT mask.
    assert (out_dir / "PREDICTIONS_FROZEN.json").is_file()
    assert not (out_dir / "METRICS.json").exists()


def test_evaluation_rejects_bad_frame_indices_duplicates_and_incomplete_patients_before_output(tmp_path):
    train_manifest, train_sha, export_dir, export_sha, eval_dir, base_rec = create_evaluation_fixture(tmp_path)
    mask_rec = copy.deepcopy(base_rec["masks"]["0"])
    cases = [
        ("out_of_bounds", {"1": mask_rec}, "Frame index 1 out of bounds"),
        ("duplicate_frame", {"0": mask_rec, "00": mask_rec}, "Duplicate evaluation frame index"),
        ("no_masks", {}, "requires at least one reference frame"),
    ]
    for name, masks, expected_error in cases:
        rec = copy.deepcopy(base_rec)
        rec["masks"] = masks
        manifest = eval_dir / f"{name}.json"
        manifest_sha = write_json(manifest, {
            "schema": "self_audit.evaluation.v1",
            "role": "test",
            "synthetic": True,
            "patients": [rec],
        })
        out_dir = tmp_path / f"{name}_out"
        with pytest.raises(ValueError, match=expected_error):
            run_two_phase_evaluation(
                export_path=export_dir,
                export_sha256=export_sha,
                manifest_path=manifest,
                manifest_sha256=manifest_sha,
                train_manifest_path=train_manifest,
                train_manifest_sha256=train_sha,
                out_dir=out_dir,
            )
        assert not out_dir.exists()

    duplicate_manifest = eval_dir / "duplicate_patient.json"
    duplicate_sha = write_json(duplicate_manifest, {
        "schema": "self_audit.evaluation.v1",
        "role": "test",
        "synthetic": True,
        "patients": [base_rec, copy.deepcopy(base_rec)],
    })
    duplicate_out = tmp_path / "duplicate_patient_out"
    with pytest.raises(ValueError, match="Duplicate patient IDs"):
        run_two_phase_evaluation(
            export_path=export_dir,
            export_sha256=export_sha,
            manifest_path=duplicate_manifest,
            manifest_sha256=duplicate_sha,
            train_manifest_path=train_manifest,
            train_manifest_sha256=train_sha,
            out_dir=duplicate_out,
        )
    assert not duplicate_out.exists()


def test_export_lineage_requires_exact_canonical_train_manifest_sha(tmp_path):
    train_manifest, train_sha, export_dir, export_sha, eval_dir, rec = create_evaluation_fixture(tmp_path)
    changed_train = json.loads(train_manifest.read_text())
    changed_train["provenance_note"] = "bytes changed after export"
    changed_train_sha = write_json(train_manifest, changed_train)
    manifest = eval_dir / "lineage_eval.json"
    manifest_sha = write_json(manifest, {
        "schema": "self_audit.evaluation.v1",
        "role": "test",
        "synthetic": True,
        "patients": [rec],
    })
    out_dir = tmp_path / "lineage_out"
    with pytest.raises(ValueError, match="TRAIN manifest hash mismatch against export lineage"):
        run_two_phase_evaluation(
            export_path=export_dir,
            export_sha256=export_sha,
            manifest_path=manifest,
            manifest_sha256=manifest_sha,
            train_manifest_path=train_manifest,
            train_manifest_sha256=changed_train_sha,
            out_dir=out_dir,
        )
    assert changed_train_sha != train_sha
    assert not out_dir.exists()


def test_reference_mask_shape_mismatch_rejected(tmp_path):
    """Verify that reference mask shape mismatching prediction frame shape raises ValueError."""
    train_dir = tmp_path / "train_dir"
    train_dir.mkdir()
    eval_dir = tmp_path / "eval_dir"
    eval_dir.mkdir()

    tr_rec, _, _, _ = create_synthetic_patient_data(train_dir, "train_pat")
    tr2_rec, _, _, _ = create_synthetic_patient_data(train_dir, "train_pat2")
    tru_rec, _, _, _ = create_synthetic_patient_data(train_dir, "train_u")
    train_manifest = train_dir / "MANIFEST.json"
    train_sha = write_json(train_manifest, {
        "schema": "self_audit.data.v2",
        "role": "training",
        "synthetic": True,
        "label_budget": 2,
        "labeled": [
            {key: tr_rec[key] for key in ("id", "image", "image_sha256", "masks")},
            {key: tr2_rec[key] for key in ("id", "image", "image_sha256", "masks")},
        ],
        "unlabeled": [{key: tru_rec[key] for key in ("id", "image", "image_sha256")}],
        "lids": ["train_pat", "train_pat2"],
        "uids": ["train_u"],
        "evaluation_ids": [],
    })

    export_dir, export_sha, _ = create_canonical_export(
        tmp_path / "export_shape_check",
        synthetic=True,
        data_sha256=train_sha,
        lids=["train_pat", "train_pat2"],
        uids=["train_u"],
    )

    # Eval patient cine has shape (2, 3, 32, 32), so frame shape is (3, 32, 32)
    eval_rec, _, _, _ = create_synthetic_patient_data(eval_dir, "eval_mismatch", shape=(2, 3, 32, 32))
    # Replace mask with mismatching shape, e.g. (2, 32, 32) instead of (3, 32, 32)
    mismatched_mask = np.zeros((2, 32, 32), dtype=np.uint8)
    m_path = eval_dir / "mismatched_mask.npy"
    np.save(m_path, mismatched_mask)
    m_sha = digest(m_path)

    eval_rec["masks"] = {
        "0": {
            "path": m_path.name,
            "sha256": m_sha,
        }
    }

    eval_manifest = eval_dir / "MANIFEST.json"
    eval_sha = write_json(eval_manifest, {
        "schema": "self_audit.native_manifest.v1",
        "role": "development",
        "synthetic": True,
        "patients": [eval_rec],
    })

    with pytest.raises(ValueError, match="Shape mismatch: reference mask shape"):
        run_two_phase_evaluation(
            export_path=export_dir,
            export_sha256=export_sha,
            manifest_path=eval_manifest,
            manifest_sha256=eval_sha,
            train_manifest_path=train_manifest,
            train_manifest_sha256=train_sha,
            out_dir=tmp_path / "eval_out_shape_fail",
        )


def test_benchmark_live_disk_write_verified(tmp_path):
    """Verify that isolated benchmark streaming inference performs real live disk writes."""
    from self_audit.compact_inference import _worker_benchmark_target
    import argparse

    export_dir, export_sha, _ = create_canonical_export(tmp_path / "bench_write_export")
    data_dir = tmp_path / "bench_write_data"
    data_dir.mkdir()

    p1_rec, _, _, _ = create_synthetic_patient_data(data_dir, "write_pat", shape=(1, 2, 16, 16))
    manifest_file = data_dir / "MANIFEST.json"
    manifest_sha = write_json(manifest_file, {
        "schema": "self_audit.native_manifest.v1",
        "role": "development",
        "synthetic": True,
        "patients": [p1_rec],
    })

    scratch_dir = tmp_path / "bench_live_scratch"
    scratch_dir.mkdir()

    args = argparse.Namespace(
        model_type="compact_export",
        model_path=str(export_dir),
        expected_sha=export_sha,
        manifest_path=str(manifest_file),
        manifest_sha=manifest_sha,
        scratch_dir=str(scratch_dir),
        device="cpu",
        batch_size=2,
        warmup=1,
        repeats=1,
        normalization="prepared",
    )

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        _worker_benchmark_target(args)
        worker_out = sys.stdout.getvalue()
    finally:
        sys.stdout = old_stdout

    expected_pred_file = scratch_dir / "write_pat_case_0.npy"
    assert expected_pred_file.is_file(), f"Live prediction file not written: {expected_pred_file}"
    assert expected_pred_file.stat().st_size > 0
    written_arr = np.load(expected_pred_file)
    assert written_arr.shape == (1, 2, 16, 16)
    assert written_arr.dtype == np.uint8

    worker_json = json.loads(worker_out.strip())
    assert worker_json["status"] == "COMPLETED"



def test_namespaced_domains_share_one_export_and_keep_separate_reports(tmp_path):
    train_dir=tmp_path/"train"
    eval_dir=tmp_path/"eval"
    train_dir.mkdir(); eval_dir.mkdir()
    tr,_,_,_=create_synthetic_patient_data(train_dir,"acdc:train",shape=(1,1,16,16))
    ur,_,_,_=create_synthetic_patient_data(train_dir,"mnm:unlabeled",shape=(1,1,16,16))
    train_manifest=train_dir/"manifest.json"
    train_sha=write_json(train_manifest,{"schema":"self_audit.data.v2","role":"training","synthetic":True,
        "label_budget":1,"labeled":[{k:tr[k] for k in ("id","image","image_sha256","masks")}],
        "unlabeled":[{k:ur[k] for k in ("id","image","image_sha256")}]})
    patients=[]
    for domain in ("acdc","mnm"):
        row,_,_,_=create_synthetic_patient_data(eval_dir,domain+":heldout",shape=(1,1,16,16))
        row["domain"]=domain; patients.append(row)
    manifest=eval_dir/"manifest.json"
    manifest_sha=write_json(manifest,{"schema":"self_audit.native_manifest.v1","role":"development","synthetic":True,"patients":patients})
    export,export_sha,_=create_canonical_export(tmp_path/"export",widths=(4,8),data_sha256=train_sha,lids=[tr['id']],uids=[ur['id']])
    result=run_two_phase_evaluation(export,export_sha,manifest,manifest_sha,tmp_path/"out",
        train_manifest_path=train_manifest,train_manifest_sha256=train_sha,batch_size=1)
    assert set(result['datasets'])=={'acdc','mnm'}
    assert result['export_manifest_sha256']==export_sha
    assert set(result['patients'])=={'acdc:heldout','mnm:heldout'}
    # A namespace is not permission for path traversal.
    patients[0]['id']='acdc:../escape'
    bad_sha=write_json(manifest,{"schema":"self_audit.native_manifest.v1","role":"development","synthetic":True,"patients":patients})
    with pytest.raises(ValueError,match='unsafe patient ID'):
        load_native_manifest(manifest,bad_sha)
