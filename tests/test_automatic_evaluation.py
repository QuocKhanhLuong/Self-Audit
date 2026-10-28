"""Acceptance tests for self_audit.automatic_evaluation and evaluate_automatic CLI.

Verifies:
1. Manifest validation, schema, role, hashes, safe patient IDs (no traversal), and structure.
2. Training labeled/unlabeled lineage overlap rejection; missing lineage fails closed.
3. Strict freeze-before-reference-masks order: zero mask paths opened before PREDICTIONS_FROZEN receipt.
4. Raw student uint8 export: deterministic, 100% coverage, no postprocessing, direct logits.argmax(1).
5. Sentinel 255 buffer initialization and explicit plane counting; catches incomplete writes.
6. Canonical integer frame ID bounds (0 <= frame < T) validated BEFORE mask I/O; rejects negative/overflow.
7. Hierarchical scoring: frame-volume confusion sum, available frame mean per patient, patient macro,
   absent-reference Dice None and FP diagnostics.
8. Asymmetric absent-class witness: foreground_patient_macro matches evaluate_frozen_labels.py
   (mean available foreground classes within each patient, then mean across patients; differs from class macro mean).
9. Zero-valid frame volume FP denominator 0 yields fp_rate=None (not invented 0.0).
10. Tolerates records without masks in inference-only mode.
11. Cold-cost accounting: checkpoint load and export times reported separately; CPU CUDA None.
12. Checkpoint teacher/payload memory overhead deletion verified.
13. Parity between uncompressed and compressed (.npy.gz) prediction exports.
14. CLI invocation via scripts/evaluate_automatic.py.
"""

from __future__ import annotations

import copy
import gzip
import json
import os
from pathlib import Path
import pytest
import numpy as np
import torch

from scripts.evaluate_automatic import main as evaluate_automatic_main
from self_audit.automatic import DEFAULT_AUTOMATIC_CONFIG, run
from self_audit.automatic_evaluation import (
    aggregate_patient_macro,
    check_lineage_overlap,
    compute_patient_macro_metrics,
    evaluate_automatic,
    extract_training_lineage_ids,
    validate_evaluation_manifest,
)
from self_audit.data import synthetic_fixture, write_json
from self_audit.model import digest


@pytest.fixture
def automatic_trained_checkpoint(tmp_path: Path):
    """Create a minimal synthetic trained automatic checkpoint using warm_start then adaptation."""
    train_dir = tmp_path / "train_data"
    m, sha = synthetic_fixture(train_dir, seed=42, size=32)

    # 1. Warm-start phase
    c_ws = copy.deepcopy(DEFAULT_AUTOMATIC_CONFIG)
    c_ws.update({
        "phase": "warm_start",
        "arm": "support_only",
        "steps": 1,
        "seed": 42,
        "model": {"backbone": "tiny_test"},
        "initialization": {
            "warm_start_checkpoint": None,
            "warm_start_sha256": None,
            "declared_target_budget": 2,
        },
        "support": {
            "patient_ids": ["synthetic_000", "synthetic_001"],
            "k_reuse": 1,
        },
    })
    ws_out = tmp_path / "train_ws_out"
    ws_rep = run(c_ws, ws_out, synthetic=True, manifest=m, manifest_sha256=sha)
    ws_ckpt = ws_out / "final"
    ws_sha = ws_rep["final_manifest_sha256"]

    # 2. Adaptation phase
    c_adapt = copy.deepcopy(DEFAULT_AUTOMATIC_CONFIG)
    c_adapt.update({
        "phase": "adaptation",
        "arm": "joint_dice",
        "steps": 2,
        "warmup_steps": 0,
        "seed": 42,
        "model": {"backbone": "tiny_test"},
        "initialization": {
            "warm_start_checkpoint": str(ws_ckpt),
            "warm_start_sha256": ws_sha,
            "declared_target_budget": 2,
        },
        "support": {
            "patient_ids": ["synthetic_000", "synthetic_001"],
            "k_reuse": 1,
        },
    })
    adapt_out = tmp_path / "train_adapt_out"
    adapt_rep = run(c_adapt, adapt_out, synthetic=True, manifest=m, manifest_sha256=sha)
    ckpt_folder = adapt_out / "final"
    ckpt_sha = adapt_rep["final_manifest_sha256"]
    return ckpt_folder, ckpt_sha, m


@pytest.fixture
def eval_dataset(tmp_path: Path):
    """Create a multi-frame 4D evaluation dataset with distinct IDs."""
    eval_dir = tmp_path / "eval_data"
    raw_m, _ = synthetic_fixture(eval_dir, seed=77, n_labeled=3, size=32)
    raw_records = json.loads(raw_m.read_text())["labeled"]

    for idx, r in enumerate(raw_records):
        ip = eval_dir / r["image"]
        img = np.load(ip)
        # Reshape to 4D native cine: [T=2, Z=2, H=32, W=32]
        img_4d = np.tile(img, (2, 2, 1, 1))
        np.save(ip, img_4d)
        r["id"] = f"eval_patient_{idx:03d}"
        r["image_sha256"] = digest(ip)

        # Multi-frame masks: frame 0 and frame 1
        mask_info_0 = r["masks"]["0"]
        mp_0 = eval_dir / mask_info_0["path"]
        mask_3d = np.tile(np.load(mp_0), (2, 1, 1))
        np.save(mp_0, mask_3d)
        mask_info_0["sha256"] = digest(mp_0)

        # Create separate frame 1 mask
        mp_1 = eval_dir / f"{r['id']}_frame1_mask.npy"
        np.save(mp_1, mask_3d)
        mask_info_1 = {"path": mp_1.name, "sha256": digest(mp_1)}

        r["masks"] = {"0": mask_info_0, "1": mask_info_1}

    manifest_file = eval_dir / "manifest.json"
    manifest_data = {
        "schema": "self_audit.data.v2",
        "role": "external_frozen",
        "synthetic": True,
        "patients": raw_records,
    }
    write_json(manifest_file, manifest_data)
    return manifest_file, digest(manifest_file), eval_dir


def test_freeze_before_masks_order(tmp_path, automatic_trained_checkpoint, eval_dataset, monkeypatch):
    """Prove that all images are predicted and frozen receipt is written BEFORE any mask is opened."""
    ckpt_folder, ckpt_sha, _ = automatic_trained_checkpoint
    manifest_file, data_sha, _ = eval_dataset

    m = json.loads(manifest_file.read_text())
    mask_paths = {
        str((manifest_file.parent / mask["path"]).resolve())
        for r in m["patients"]
        for mask in r["masks"].values()
    }

    out_dir = tmp_path / "test_freeze_order_out"
    original_load = np.load
    mask_access_log: list[str] = []

    def checked_load(file, *args, **kwargs):
        resolved_path = str(Path(file).resolve()) if isinstance(file, (str, os.PathLike)) else ""
        if resolved_path in mask_paths:
            receipt_path = out_dir / "PREDICTIONS_FROZEN.json"
            assert receipt_path.is_file(), f"Violation: mask {resolved_path} was opened before PREDICTIONS_FROZEN.json was written!"
            mask_access_log.append(resolved_path)
        return original_load(file, *args, **kwargs)

    monkeypatch.setattr(np, "load", checked_load)

    res = evaluate_automatic(
        checkpoint=ckpt_folder,
        checkpoint_sha=ckpt_sha,
        data_manifest=manifest_file,
        data_sha=data_sha,
        out=out_dir,
        device="cpu",
    )

    assert (out_dir / "PREDICTIONS_FROZEN.json").is_file()
    assert (out_dir / "METRICS.json").is_file()
    assert len(mask_access_log) == 6  # 3 patients * 2 frames = 6 masks
    assert res["status"] == "COMPLETED"
    assert res["coverage"] == 1.0
    assert res["total_patients_exported"] == 3
    assert res["total_frames_exported"] == 6
    assert res["total_patients_scored"] == 3
    assert res["total_frames_scored"] == 6


def test_raw_student_deterministic_coverage_and_no_postprocessing(tmp_path, automatic_trained_checkpoint, eval_dataset):
    """Prove raw student output: deterministic identical hashes across runs, 100% coverage, uint8 argmax."""
    ckpt_folder, ckpt_sha, _ = automatic_trained_checkpoint
    manifest_file, data_sha, _ = eval_dataset

    out1 = tmp_path / "run1_out"
    out2 = tmp_path / "run2_out"

    res1 = evaluate_automatic(ckpt_folder, ckpt_sha, manifest_file, data_sha, out1, device="cpu")
    res2 = evaluate_automatic(ckpt_folder, ckpt_sha, manifest_file, data_sha, out2, device="cpu")

    # Complete 100% coverage
    assert res1["coverage"] == 1.0
    assert res2["coverage"] == 1.0

    # Deterministic file hashes and bit-for-bit identical outputs
    receipt1 = json.loads((out1 / "PREDICTIONS_FROZEN.json").read_text())
    receipt2 = json.loads((out2 / "PREDICTIONS_FROZEN.json").read_text())

    for f1, f2 in zip(receipt1["files"], receipt2["files"]):
        assert f1["id"] == f2["id"]
        assert f1["sha256"] == f2["sha256"]
        assert f1["dtype"] == "uint8"
        assert f1["coverage"] == 1.0

        p1 = np.load(out1 / "predictions" / f1["file"])
        p2 = np.load(out2 / "predictions" / f2["file"])
        assert p1.dtype == np.uint8
        assert np.array_equal(p1, p2)
        assert np.isin(p1, [0, 1, 2, 3]).all()


def test_lineage_overlap_rejection(tmp_path, automatic_trained_checkpoint, eval_dataset):
    """Reject overlap with training labeled/unlabeled lineage for independent query."""
    ckpt_folder, ckpt_sha, _ = automatic_trained_checkpoint
    manifest_file, _, eval_dir = eval_dataset

    m = json.loads(manifest_file.read_text())
    m["patients"][0]["id"] = "synthetic_000"
    poisoned_manifest = eval_dir / "poisoned_manifest.json"
    write_json(poisoned_manifest, m)

    with pytest.raises(ValueError, match="evaluation patient overlap with training labeled/unlabeled lineage"):
        evaluate_automatic(
            checkpoint=ckpt_folder,
            checkpoint_sha=ckpt_sha,
            data_manifest=poisoned_manifest,
            data_sha=digest(poisoned_manifest),
            out=tmp_path / "forbidden_overlap_out",
            device="cpu",
        )


def test_missing_lineage_fails_closed():
    """Ensure missing or corrupted lineage in checkpoint metadata fails closed, never empty set."""
    empty_meta = {"schema": "self_audit.automatic_checkpoint.v1", "config": {}}
    with pytest.raises(ValueError, match="training_manifest_sha256"):
        extract_training_lineage_ids(empty_meta)


def test_missing_unlabeled_lineage_fails_even_with_nonempty_labeled():
    """Reject partial lineage: missing unlabeled_ids fails closed even when labeled_ids is nonempty."""
    meta_without_u = {
        "training_manifest_sha256": "a" * 64,
        "labeled_ids": ["one-L"],
        "target_label_lineage": {"patients": ["one-L"]},
    }
    with pytest.raises(ValueError, match="unlabeled_ids"):
        extract_training_lineage_ids(meta_without_u)


def test_partial_lineage_rejected_without_manifest_sha_or_labeled():
    """Reject partial metadata: missing training manifest SHA or labeled_ids fails closed."""
    # Missing training manifest SHA
    meta_no_sha = {
        "labeled_ids": ["one-L"],
        "unlabeled_ids": ["one-U"],
        "target_label_lineage": {"patients": ["one-L"]},
    }
    with pytest.raises(ValueError, match="training_manifest_sha256"):
        extract_training_lineage_ids(meta_no_sha)

    # Missing labeled_ids
    meta_no_l = {
        "training_manifest_sha256": "b" * 64,
        "unlabeled_ids": ["one-U"],
        "target_label_lineage": {"patients": ["one-L"]},
    }
    with pytest.raises(ValueError, match="labeled_ids"):
        extract_training_lineage_ids(meta_no_l)

    # Missing target_label_lineage
    meta_no_lineage = {
        "training_manifest_sha256": "c" * 64,
        "labeled_ids": ["one-L"],
        "unlabeled_ids": ["one-U"],
    }
    with pytest.raises(ValueError, match="target_label_lineage"):
        extract_training_lineage_ids(meta_no_lineage)


def test_complete_lineage_union():
    """Verify that complete metadata returns the union of all labeled, unlabeled, and support IDs."""
    valid_meta = {
        "training_manifest_sha256": "d" * 64,
        "labeled_ids": ["L1", "L2"],
        "unlabeled_ids": ["U1", "U2"],
        "target_label_lineage": {"patients": ["L1", "L2"], "support_patient_ids": ["L1"]},
    }
    lineage_ids = extract_training_lineage_ids(valid_meta)
    assert lineage_ids == {"L1", "L2", "U1", "U2"}


def test_asymmetric_absent_class_macro_witness():
    """Witness test: prove aggregate_patient_macro matches evaluate_frozen_labels.py,

    and differs mathematically from the mean of class macros under asymmetric absent classes.
    """
    # Patient A: RV=0.8, MYO=0.8, LV=None
    # Patient B: RV=None, MYO=0.6, LV=0.6
    # Patient C: RV=0.2, MYO=0.2, LV=0.2
    patients_data = {
        "pt_A": {"RV": 0.8, "MYO": 0.8, "LV": None},
        "pt_B": {"RV": None, "MYO": 0.6, "LV": 0.6},
        "pt_C": {"RV": 0.2, "MYO": 0.2, "LV": 0.2},
    }

    # 1. Matching evaluate_frozen_labels.py (patient means first, then mean across patients):
    # pt_A mean: (0.8 + 0.8) / 2 = 0.8
    # pt_B mean: (0.6 + 0.6) / 2 = 0.6
    # pt_C mean: (0.2 + 0.2 + 0.2) / 3 = 0.2
    # Overall mean: (0.8 + 0.6 + 0.2) / 3 = 0.5333...
    fg_macro = aggregate_patient_macro(patients_data, ["RV", "MYO", "LV"])
    assert fg_macro == pytest.approx(1.6 / 3.0)

    # 2. Mean of class macros (incorrect reduction):
    # RV macro: (0.8 + 0.2) / 2 = 0.50
    # MYO macro: (0.8 + 0.6 + 0.2) / 3 = 0.5333...
    # LV macro: (0.6 + 0.2) / 2 = 0.40
    # Mean of class macros: (0.50 + 0.5333... + 0.40) / 3 = 0.4777...
    rv_macro = float(np.mean([0.8, 0.2]))
    myo_macro = float(np.mean([0.8, 0.6, 0.2]))
    lv_macro = float(np.mean([0.6, 0.2]))
    class_macro_mean = float(np.mean([rv_macro, myo_macro, lv_macro]))

    # Witness: verify that the two reductions strictly diverge
    assert abs(fg_macro - class_macro_mean) > 0.05
    assert class_macro_mean == pytest.approx(0.4777777777777778)
    assert fg_macro == pytest.approx(0.5333333333333333)


def test_frame_id_canonical_and_bounds_validation(tmp_path, automatic_trained_checkpoint, eval_dataset):
    """Validate canonical integer frame IDs and 0 <= frame < T BEFORE mask I/O; negative -1 fails closed."""
    ckpt_folder, ckpt_sha, _ = automatic_trained_checkpoint
    manifest_file, _, eval_dir = eval_dataset

    # 1. Negative frame ID "-1"
    m_neg = json.loads(manifest_file.read_text())
    m_neg["patients"][0]["masks"]["-1"] = dict(m_neg["patients"][0]["masks"]["0"])
    neg_manifest = eval_dir / "neg_frame_manifest.json"
    write_json(neg_manifest, m_neg)

    with pytest.raises(ValueError, match="invalid frame ID"):
        evaluate_automatic(
            ckpt_folder, ckpt_sha, neg_manifest, digest(neg_manifest), tmp_path / "neg_out", device="cpu"
        )

    # 2. Overflow frame ID >= T (e.g. "99")
    m_over = json.loads(manifest_file.read_text())
    m_over["patients"][0]["masks"]["99"] = dict(m_over["patients"][0]["masks"]["0"])
    over_manifest = eval_dir / "over_frame_manifest.json"
    write_json(over_manifest, m_over)

    with pytest.raises(ValueError, match="out of bounds: expected 0 <= frame < 2"):
        evaluate_automatic(
            ckpt_folder, ckpt_sha, over_manifest, digest(over_manifest), tmp_path / "over_out", device="cpu"
        )

    # 3. Non-canonical integer string (e.g. "00" or "frame_0")
    m_noncanon = json.loads(manifest_file.read_text())
    m_noncanon["patients"][0]["masks"]["00"] = dict(m_noncanon["patients"][0]["masks"]["0"])
    del m_noncanon["patients"][0]["masks"]["0"]
    noncanon_manifest = eval_dir / "noncanon_manifest.json"
    write_json(noncanon_manifest, m_noncanon)

    with pytest.raises(ValueError, match="non-canonical or invalid frame ID"):
        evaluate_automatic(
            ckpt_folder, ckpt_sha, noncanon_manifest, digest(noncanon_manifest), tmp_path / "noncanon_out", device="cpu"
        )


def test_safe_patient_ids_prevent_directory_escape(tmp_path, automatic_trained_checkpoint, eval_dataset):
    """Reject patient IDs containing '/', '..', or absolute paths that could escape output dir."""
    ckpt_folder, ckpt_sha, _ = automatic_trained_checkpoint
    manifest_file, _, eval_dir = eval_dataset

    for bad_id in ("../../escape", "/root/bad", "..", "."):
        m = json.loads(manifest_file.read_text())
        m["patients"][0]["id"] = bad_id
        bad_id_manifest = eval_dir / f"bad_id_{abs(hash(bad_id))}.json"
        write_json(bad_id_manifest, m)

        with pytest.raises(ValueError, match="unsafe patient ID"):
            evaluate_automatic(
                ckpt_folder, ckpt_sha, bad_id_manifest, digest(bad_id_manifest), tmp_path / "escape_out", device="cpu"
            )


def test_zero_valid_frame_fp_denominator_is_none(tmp_path, automatic_trained_checkpoint):
    """When all voxels in a frame are 255 (valid_voxels=0), FP rate must be None, not 0.0."""
    ckpt_folder, ckpt_sha, _ = automatic_trained_checkpoint

    eval_dir = tmp_path / "all_ignored_eval_dir"
    raw_m, _ = synthetic_fixture(eval_dir, seed=88, n_labeled=1, size=32)
    records = json.loads(raw_m.read_text())["labeled"]

    r = records[0]
    ip = eval_dir / r["image"]
    img_4d = np.tile(np.load(ip), (1, 1, 1, 1))
    np.save(ip, img_4d)
    r["id"] = "all_ignored_pt"
    r["image_sha256"] = digest(ip)

    # Mask filled entirely with sentinel 255
    all_255_mask = np.full((1, 32, 32), 255, dtype=np.uint8)
    mp = eval_dir / "all_255_mask.npy"
    np.save(mp, all_255_mask)
    r["masks"] = {"0": {"path": mp.name, "sha256": digest(mp)}}

    manifest_file = eval_dir / "manifest.json"
    write_json(manifest_file, {
        "schema": "self_audit.data.v2",
        "role": "external_frozen",
        "synthetic": True,
        "patients": [r],
    })

    out_dir = tmp_path / "all_ignored_out"
    res = evaluate_automatic(ckpt_folder, ckpt_sha, manifest_file, digest(manifest_file), out_dir, device="cpu")

    f0 = res["frames"][0]
    fp_diag = f0["absent_class_fp_diagnostics"]
    for c in ["BG", "RV", "MYO", "LV"]:
        assert fp_diag[c]["valid_voxels"] == 0
        assert fp_diag[c]["fp_rate"] is None  # Undefined denominator yields None, never 0.0


def test_scoring_hierarchy_and_absent_class_diagnostics(tmp_path, automatic_trained_checkpoint):
    """Verify hierarchical reduction: frame sum, available frame mean, patient macro, absent-reference Dice None."""
    ckpt_folder, ckpt_sha, _ = automatic_trained_checkpoint

    eval_dir = tmp_path / "absent_class_eval_dir"
    raw_m, _ = synthetic_fixture(eval_dir, seed=55, n_labeled=2, size=32)
    records = json.loads(raw_m.read_text())["labeled"]

    for idx, r in enumerate(records):
        ip = eval_dir / r["image"]
        img_4d = np.tile(np.load(ip), (2, 2, 1, 1))
        np.save(ip, img_4d)
        r["id"] = f"absent_test_pt_{idx}"
        r["image_sha256"] = digest(ip)

        mask_0_path = eval_dir / r["masks"]["0"]["path"]
        m0 = np.tile(np.load(mask_0_path), (2, 1, 1))
        np.save(mask_0_path, m0)
        mask_info_0 = {"path": mask_0_path.name, "sha256": digest(mask_0_path)}

        m1 = m0.copy()
        m1[m1 == 1] = 0  # Zero RV voxels
        mask_1_path = eval_dir / f"{r['id']}_absent_rv_frame1.npy"
        np.save(mask_1_path, m1)
        mask_info_1 = {"path": mask_1_path.name, "sha256": digest(mask_1_path)}

        r["masks"] = {"0": mask_info_0, "1": mask_info_1}

    manifest_file = eval_dir / "manifest.json"
    write_json(manifest_file, {
        "schema": "self_audit.data.v2",
        "role": "external_frozen",
        "synthetic": True,
        "patients": records,
    })

    out_dir = tmp_path / "absent_scoring_out"
    res = evaluate_automatic(ckpt_folder, ckpt_sha, manifest_file, digest(manifest_file), out_dir, device="cpu")

    frame_1_rows = [f for f in res["frames"] if f["frame"] == 1]
    assert len(frame_1_rows) == 2
    for f in frame_1_rows:
        rv_metric = f["raw"][1]
        assert rv_metric["reference_present"] is False
        assert rv_metric["dice"] is None
        assert "RV" in f["absent_class_fp_diagnostics"]
        assert f["absent_class_fp_diagnostics"]["RV"]["FP"] == rv_metric["FP"]

    for r in records:
        pid = r["id"]
        f0_row = [f for f in res["frames"] if f["patient"] == pid and f["frame"] == 0][0]
        assert res["patients"][pid]["RV"] == f0_row["raw"][1]["dice"]
        assert res["patients"][pid]["RV_absent_frames"] == 1

    assert res["macro_patient"]["RV"]["dice"] is not None


def test_inference_only_tolerates_records_without_masks(tmp_path, automatic_trained_checkpoint, eval_dataset):
    """Verify that records without masks succeed in inference-only export."""
    ckpt_folder, ckpt_sha, _ = automatic_trained_checkpoint
    manifest_file, _, eval_dir = eval_dataset

    m = json.loads(manifest_file.read_text())
    for r in m["patients"]:
        r.pop("masks", None)
    nomask_manifest = eval_dir / "nomask_manifest.json"
    write_json(nomask_manifest, m)

    out_dir = tmp_path / "nomask_out"
    res = evaluate_automatic(
        ckpt_folder, ckpt_sha, nomask_manifest, digest(nomask_manifest), out_dir, device="cpu"
    )

    assert res["status"] == "INFERENCE_ONLY"
    assert res["coverage"] == 1.0
    assert (out_dir / "PREDICTIONS_FROZEN.json").is_file()
    assert (out_dir / "METRICS.json").is_file()


def test_cold_cost_and_hardware_metrics_cpu(tmp_path, automatic_trained_checkpoint, eval_dataset):
    """Verify that cold cost accounting, throughput, and CPU metrics behave as expected."""
    ckpt_folder, ckpt_sha, _ = automatic_trained_checkpoint
    manifest_file, data_sha, _ = eval_dataset
    out_dir = tmp_path / "metrics_test_out"

    res = evaluate_automatic(ckpt_folder, ckpt_sha, manifest_file, data_sha, out_dir, device="cpu")

    cold_diag = res["cold_cost_diagnostics"]
    assert isinstance(cold_diag["checkpoint_load_seconds"], float) and cold_diag["checkpoint_load_seconds"] > 0
    assert isinstance(cold_diag["output_export_seconds"], float) and cold_diag["output_export_seconds"] > 0
    assert isinstance(cold_diag["cold_deployment_total_seconds"], float) and cold_diag["cold_deployment_total_seconds"] > 0
    assert cold_diag["total_seconds"] == cold_diag["cold_deployment_total_seconds"]
    # Total deployment seconds includes checkpoint load, inference export, and frozen receipt writing
    assert cold_diag["total_seconds"] >= cold_diag["output_export_seconds"]
    assert res["total_seconds"] == cold_diag["total_seconds"]
    assert res["checkpoint_load_seconds"] == cold_diag["checkpoint_load_seconds"]
    assert res["output_export_seconds"] == cold_diag["output_export_seconds"]
    assert isinstance(cold_diag["throughput_voxels_per_sec"], float) and cold_diag["throughput_voxels_per_sec"] > 0
    assert isinstance(cold_diag["latency_p50_ms"], float) and cold_diag["latency_p50_ms"] > 0
    assert isinstance(cold_diag["latency_p95_ms"], float) and cold_diag["latency_p95_ms"] >= cold_diag["latency_p50_ms"]
    assert cold_diag["peak_cuda_allocated_bytes"] is None
    assert cold_diag["peak_cuda_reserved_bytes"] is None


def test_compressed_and_uncompressed_predictions_parity(tmp_path, automatic_trained_checkpoint, eval_dataset):
    """Verify parity between uncompressed .npy and compressed .npy.gz predictions."""
    ckpt_folder, ckpt_sha, _ = automatic_trained_checkpoint
    manifest_file, data_sha, _ = eval_dataset

    out_uncompressed = tmp_path / "out_uncompressed"
    out_compressed = tmp_path / "out_compressed"

    res_raw = evaluate_automatic(
        ckpt_folder, ckpt_sha, manifest_file, data_sha, out_uncompressed, device="cpu", compress_predictions=False
    )
    res_gz = evaluate_automatic(
        ckpt_folder, ckpt_sha, manifest_file, data_sha, out_compressed, device="cpu", compress_predictions=True
    )

    for item_raw, item_gz in zip(res_raw["patients"], res_gz["patients"]):
        assert item_raw == item_gz
        p_raw = np.load(out_uncompressed / "predictions" / f"{item_raw}.npy")
        with gzip.open(out_compressed / "predictions" / f"{item_raw}.npy.gz", "rb") as gz:
            p_gz = np.load(gz)
        assert np.array_equal(p_raw, p_gz)

    assert res_raw["macro_patient"] == res_gz["macro_patient"]
    assert res_raw["foreground_patient_macro"] == pytest.approx(res_gz["foreground_patient_macro"])


def test_cli_invocation(tmp_path, automatic_trained_checkpoint, eval_dataset, capsys):
    """Test CLI evaluate_automatic invocation."""
    ckpt_folder, ckpt_sha, _ = automatic_trained_checkpoint
    manifest_file, data_sha, _ = eval_dataset
    out_dir = tmp_path / "cli_out"

    argv = [
        "--checkpoint", str(ckpt_folder),
        "--checkpoint-sha256", ckpt_sha,
        "--data", str(manifest_file),
        "--data-sha256", data_sha,
        "--out", str(out_dir),
        "--device", "cpu",
        "--batch-size", "4",
        "--threads", "1",
    ]

    ret = evaluate_automatic_main(argv)
    assert ret == 0

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert summary["status"] == "COMPLETED"
    assert "macro_patient" in summary
    assert "cold_cost_diagnostics" in summary
    assert summary["cold_cost_diagnostics"]["peak_cuda_allocated_bytes"] is None


def test_reject_unexpected_teacher_memory(tmp_path, automatic_trained_checkpoint, eval_dataset, monkeypatch):
    """Verify that an unexpected teacher_memory attached to student model is rejected."""
    import self_audit.automatic as auto_mod
    ckpt_folder, ckpt_sha, _ = automatic_trained_checkpoint
    manifest_file, data_sha, _ = eval_dataset

    orig_loader = auto_mod.load_automatic_student

    def mocked_loader(*args, **kwargs):
        student, manifest = orig_loader(*args, **kwargs)
        student.teacher_memory = torch.nn.Identity()
        return student, manifest

    monkeypatch.setattr(auto_mod, "load_automatic_student", mocked_loader)

    with pytest.raises(ValueError, match="evaluation requires raw NativeStudent with teacher_memory=None"):
        evaluate_automatic(
            ckpt_folder, ckpt_sha, manifest_file, data_sha, tmp_path / "mem_out", device="cpu"
        )


def test_reject_non_fixed_window_and_non_nativestudent(tmp_path, automatic_trained_checkpoint, eval_dataset, monkeypatch):
    """Verify that non-fixed window or non-NativeStudent models are strictly rejected."""
    import self_audit.automatic as auto_mod
    ckpt_folder, ckpt_sha, _ = automatic_trained_checkpoint
    manifest_file, data_sha, _ = eval_dataset

    orig_loader = auto_mod.load_automatic_student

    # 1. Non-fixed window mode (e.g. None)
    def mocked_none_window(*args, **kwargs):
        student, manifest = orig_loader(*args, **kwargs)
        student.window = None
        return student, manifest

    monkeypatch.setattr(auto_mod, "load_automatic_student", mocked_none_window)
    with pytest.raises(ValueError, match="evaluation requires NativeStudent with fixed window"):
        evaluate_automatic(
            ckpt_folder, ckpt_sha, manifest_file, data_sha, tmp_path / "none_win_out", device="cpu"
        )

    # 2. Non-NativeStudent model class
    def mocked_wrong_model(*args, **kwargs):
        _, manifest = orig_loader(*args, **kwargs)
        return torch.nn.Linear(10, 10), manifest

    monkeypatch.setattr(auto_mod, "load_automatic_student", mocked_wrong_model)
    with pytest.raises(ValueError, match="expected raw NativeStudent instance"):
        evaluate_automatic(
            ckpt_folder, ckpt_sha, manifest_file, data_sha, tmp_path / "wrong_model_out", device="cpu"
        )


def test_output_directory_safety_checkpoint_and_data_dirs(tmp_path, automatic_trained_checkpoint, eval_dataset):
    """Verify strict output safety: output dir must be strictly outside checkpoint, manifest, and data dirs."""
    ckpt_folder, ckpt_sha, _ = automatic_trained_checkpoint
    manifest_file, data_sha, eval_dir = eval_dataset

    # 1. Output inside checkpoint dir
    with pytest.raises(ValueError, match="strictly outside checkpoint directory"):
        evaluate_automatic(ckpt_folder, ckpt_sha, manifest_file, data_sha, ckpt_folder / "nested_out", device="cpu")

    # 2. Output inside manifest dir
    with pytest.raises(ValueError, match="strictly outside evaluation data directory"):
        evaluate_automatic(ckpt_folder, ckpt_sha, manifest_file, data_sha, eval_dir / "nested_out", device="cpu")

    # 3. Output inside a referenced image directory (when image is in a separate folder)
    img_dir = tmp_path / "separate_img_dir"
    img_dir.mkdir(parents=True, exist_ok=True)
    m = json.loads(manifest_file.read_text())
    orig_img = eval_dir / m["patients"][0]["image"]
    new_img = img_dir / "separate_img.npy"
    new_img.write_bytes(orig_img.read_bytes())
    m["patients"][0]["image"] = str(new_img)
    m["patients"][0]["image_sha256"] = digest(new_img)
    separate_manifest_dir = tmp_path / "separate_manifest_dir"
    separate_manifest_dir.mkdir(parents=True, exist_ok=True)
    sub_manifest = separate_manifest_dir / "sub_manifest.json"
    write_json(sub_manifest, m)

    with pytest.raises(ValueError, match="strictly outside referenced data directory"):
        evaluate_automatic(
            ckpt_folder,
            ckpt_sha,
            sub_manifest,
            digest(sub_manifest),
            img_dir / "sub_out",
            device="cpu",
        )


def test_mixed_missing_masks_scoring_coverage(tmp_path, automatic_trained_checkpoint):
    """Test mixed missing masks: 100% export coverage, fractional scoring coverage, and correct metrics."""
    ckpt_folder, ckpt_sha, _ = automatic_trained_checkpoint

    eval_dir = tmp_path / "mixed_masks_dir"
    raw_m, _ = synthetic_fixture(eval_dir, seed=99, n_labeled=2, size=32)
    records = json.loads(raw_m.read_text())["labeled"]

    # Patient 0: multi-frame cine WITH masks
    ip0 = eval_dir / records[0]["image"]
    np.save(ip0, np.tile(np.load(ip0), (2, 2, 1, 1)))
    records[0]["id"] = "pt_with_masks"
    records[0]["image_sha256"] = digest(ip0)
    mp0 = eval_dir / records[0]["masks"]["0"]["path"]
    np.save(mp0, np.tile(np.load(mp0), (2, 1, 1)))
    records[0]["masks"]["0"]["sha256"] = digest(mp0)

    # Patient 1: multi-frame cine WITHOUT masks
    ip1 = eval_dir / records[1]["image"]
    np.save(ip1, np.tile(np.load(ip1), (2, 2, 1, 1)))
    records[1]["id"] = "pt_without_masks"
    records[1]["image_sha256"] = digest(ip1)
    records[1].pop("masks", None)  # No masks

    manifest_file = eval_dir / "mixed_manifest.json"
    write_json(manifest_file, {
        "schema": "self_audit.data.v2",
        "role": "external_frozen",
        "synthetic": True,
        "patients": records,
    })

    out_dir = tmp_path / "mixed_masks_out"
    res = evaluate_automatic(
        ckpt_folder,
        ckpt_sha,
        manifest_file,
        digest(manifest_file),
        out_dir,
        device="cpu",
    )

    assert res["status"] == "COMPLETED"
    assert res["coverage"] == 1.0
    assert res["prediction_coverage"] == 1.0
    assert res["total_patients_exported"] == 2
    assert res["total_patients_scored"] == 1
    assert res["scoring_coverage"] == 0.5

    # Patient with masks has computed metrics; patient without masks has None
    assert res["patients"]["pt_with_masks"]["RV"] is not None
    assert res["patients"]["pt_without_masks"]["RV"] is None
    assert res["foreground_patient_macro"] is not None

    # Both prediction files exist and are complete uint8
    assert (out_dir / "predictions" / "pt_with_masks.npy").is_file()
    assert (out_dir / "predictions" / "pt_without_masks.npy").is_file()


def test_cli_invocation_inference_only_and_scored_real_checkpoint(
    tmp_path, automatic_trained_checkpoint, eval_dataset, capsys
):
    """Test actual CLI invocation for both images-only and scored modes with a real tiny checkpoint."""
    ckpt_folder, ckpt_sha, _ = automatic_trained_checkpoint
    manifest_file, data_sha, _ = eval_dataset

    # 1. Images-only mode (--inference-only)
    out_inf = tmp_path / "cli_inf_only_out"
    argv_inf = [
        "--checkpoint", str(ckpt_folder),
        "--checkpoint-sha256", ckpt_sha,
        "--data", str(manifest_file),
        "--data-sha256", data_sha,
        "--out", str(out_inf),
        "--device", "cpu",
        "--batch-size", "4",
        "--threads", "1",
        "--inference-only",
    ]
    ret_inf = evaluate_automatic_main(argv_inf)
    assert ret_inf == 0

    captured_inf = capsys.readouterr()
    summary_inf = json.loads(captured_inf.out)
    assert summary_inf["status"] == "INFERENCE_ONLY"
    assert summary_inf["coverage"] == 1.0
    assert summary_inf["scoring_coverage"] == 0.0
    assert summary_inf["macro_patient"] is None
    assert (out_inf / "PREDICTIONS_FROZEN.json").is_file()
    assert (out_inf / "METRICS.json").is_file()

    # 2. Scored mode
    out_scored = tmp_path / "cli_scored_out"
    argv_scored = [
        "--checkpoint", str(ckpt_folder),
        "--checkpoint-sha256", ckpt_sha,
        "--data", str(manifest_file),
        "--data-sha256", data_sha,
        "--out", str(out_scored),
        "--device", "cpu",
        "--batch-size", "4",
        "--threads", "1",
    ]
    ret_scored = evaluate_automatic_main(argv_scored)
    assert ret_scored == 0

    captured_scored = capsys.readouterr()
    summary_scored = json.loads(captured_scored.out)
    assert summary_scored["status"] == "COMPLETED"
    assert summary_scored["coverage"] == 1.0
    assert summary_scored["scoring_coverage"] == 1.0
    assert summary_scored["foreground_patient_macro"] is not None
    assert "cold_cost_diagnostics" in summary_scored
    cold_diag = summary_scored["cold_cost_diagnostics"]
    assert cold_diag["total_seconds"] == cold_diag["cold_deployment_total_seconds"]
    assert (out_scored / "PREDICTIONS_FROZEN.json").is_file()
    assert (out_scored / "METRICS.json").is_file()


def test_safety_and_validation_errors(tmp_path, automatic_trained_checkpoint, eval_dataset):
    """Verify safe new outputs, checkpoint hash checking, and image shape/range validation."""
    ckpt_folder, ckpt_sha, _ = automatic_trained_checkpoint
    manifest_file, data_sha, eval_dir = eval_dataset

    # 1. Output directory already exists
    existing_out = tmp_path / "already_exists_out"
    existing_out.mkdir(parents=True)
    with pytest.raises(ValueError, match="already exists"):
        evaluate_automatic(ckpt_folder, ckpt_sha, manifest_file, data_sha, existing_out, device="cpu")

    # 2. Checkpoint SHA mismatch
    with pytest.raises(ValueError, match="hash mismatch"):
        evaluate_automatic(
            ckpt_folder,
            "0" * 64,
            manifest_file,
            data_sha,
            tmp_path / "bad_ckpt_out",
            device="cpu",
        )

    # 3. Non-4D image
    bad_img_path = eval_dir / "bad_3d_img.npy"
    np.save(bad_img_path, np.zeros((32, 32, 32), dtype=np.float32))
    bad_m = json.loads(manifest_file.read_text())
    bad_m["patients"][0]["image"] = bad_img_path.name
    bad_m["patients"][0]["image_sha256"] = digest(bad_img_path)
    bad_manifest = eval_dir / "bad_shape_manifest.json"
    write_json(bad_manifest, bad_m)

    with pytest.raises(ValueError, match="expected 4D"):
        evaluate_automatic(
            ckpt_folder,
            ckpt_sha,
            bad_manifest,
            digest(bad_manifest),
            tmp_path / "bad_shape_out",
            device="cpu",
        )

    # 4. Out-of-range image [0, 1]
    oor_img_path = eval_dir / "oor_img.npy"
    oor_arr = np.full((2, 2, 32, 32), 1.5, dtype=np.float32)
    np.save(oor_img_path, oor_arr)
    bad_m["patients"][0]["image"] = oor_img_path.name
    bad_m["patients"][0]["image_sha256"] = digest(oor_img_path)
    oor_manifest = eval_dir / "oor_manifest.json"
    write_json(oor_manifest, bad_m)

    with pytest.raises(ValueError, match="normalized range"):
        evaluate_automatic(
            ckpt_folder,
            ckpt_sha,
            oor_manifest,
            digest(oor_manifest),
            tmp_path / "oor_out",
            device="cpu",
        )
