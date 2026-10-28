"""Tests for shared student evaluation wrapper and freeze-before-ref protocol."""

import copy
import json
from pathlib import Path
import re
import numpy as np
import pytest
import torch

from self_audit.data import synthetic_fixture, write_json
from self_audit.model import digest
from self_audit.engine import train
from tests.test_pipeline import recipe
from scripts.evaluate_shared_student import evaluate_shared_student, safe_confusion_matrix, main as eval_main


@pytest.fixture
def trained_student_checkpoint(tmp_path):
    """Create a trained student checkpoint on synthetic data using canonical engine."""
    train_dir = tmp_path / "train_data"
    m_path, m_sha = synthetic_fixture(train_dir, seed=42, n_labeled=3, n_unlabeled=2, size=32)
    run_dir = tmp_path / "run"
    rec = recipe()
    rec["steps"] = 4
    rec["warmup_steps"] = 2
    report = train(rec, m_path, m_sha, run_dir, device="cpu", synthetic=True)
    final_dir = run_dir / "final"
    return final_dir, report["final_manifest_sha256"], m_path, m_sha


def make_eval_domain(folder: Path, domain_name: str, seed: int = 100, n_patients: int = 2, size: int = 32):
    """Create an isolated evaluation domain manifest and data arrays."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    patients = []
    yy, xx = np.mgrid[:size, :size]

    for j in range(n_patients):
        pid = f"{domain_name}_eval_{j:03d}"
        cy, cx = size * (.5 + rng.uniform(-.05, .05)), size * (.5 + rng.uniform(-.05, .05))
        radius = np.sqrt((yy - cy)**2 + (xx - cx)**2)
        y = np.zeros((size, size), np.uint8)
        y[radius < size * .27] = 2
        y[radius < size * .18] = 3
        y[(yy - cy)**2 + (xx - (cx - size * .29))**2 < (size * .12)**2] = 1

        img = np.clip(np.choose(y, [.1, .8, .45, .9]) + rng.normal(0, .04, y.shape), 0, 1).astype(np.float32)[None, None]
        ip = folder / f"{pid}.npy"
        np.save(ip, img)
        mp = folder / f"{pid}_mask.npy"
        np.save(mp, y[None])

        patients.append({
            "id": pid,
            "image": ip.name,
            "image_sha256": digest(ip),
            "masks": {"0": {"path": mp.name, "sha256": digest(mp)}},
        })

    m_path = folder / "eval_manifest.json"
    write_json(m_path, {
        "schema": "self_audit.data.v2",
        "role": "development",
        "domain": domain_name,
        "synthetic": True,
        "patients": patients,
    })
    return m_path, digest(m_path)


def test_unequal_manifest_hashes_and_two_domain_forward(trained_student_checkpoint, tmp_path):
    """Root review: distinct manifest hashes, actual native forward both domains, and metric hashes binding."""
    ckpt_dir, ckpt_sha, train_m, train_m_sha = trained_student_checkpoint

    # Two separate evaluation domains (ACDC and MnM) with distinct data and unequal manifest hashes
    acdc_m, acdc_sha = make_eval_domain(tmp_path / "acdc_eval", "acdc", seed=101, n_patients=2)
    mnm_m, mnm_sha = make_eval_domain(tmp_path / "mnm_eval", "mnm", seed=202, n_patients=2)
    assert acdc_sha != mnm_sha

    out_dir = tmp_path / "eval_shared"
    domains = {
        "acdc": (acdc_m, acdc_sha),
        "mnm": (mnm_m, mnm_sha),
    }

    res = evaluate_shared_student(
        checkpoint_path=ckpt_dir,
        checkpoint_sha256=ckpt_sha,
        domains_data=domains,
        out_dir=out_dir,
        train_manifest_path=train_m,
        train_manifest_sha256=train_m_sha,
        device="cpu",
        threshold=0.5,
    )

    # 1. RECIPE.json correctly binds distinct manifest hash per domain
    recipe_file = out_dir / "RECIPE.json"
    assert recipe_file.is_file()
    rec_data = json.loads(recipe_file.read_text())
    assert rec_data["domains"]["acdc"]["manifest_sha256"] == acdc_sha
    assert rec_data["domains"]["mnm"]["manifest_sha256"] == mnm_sha
    assert rec_data["domains"]["acdc"]["manifest_sha256"] != rec_data["domains"]["mnm"]["manifest_sha256"]

    # 2. PREDICTIONS_FROZEN.json contains checkpoint_manifest_sha256 and recipe_sha256
    frozen_receipt = out_dir / "PREDICTIONS_FROZEN.json"
    assert frozen_receipt.is_file()
    frozen_data = json.loads(frozen_receipt.read_text())
    assert frozen_data["checkpoint_manifest_sha256"] == ckpt_sha
    assert frozen_data["recipe_sha256"] == digest(recipe_file)
    assert "acdc" in frozen_data["domains"]
    assert "mnm" in frozen_data["domains"]

    # Check prediction files on disk: uint8 TZYX format
    for dom in ["acdc", "mnm"]:
        for p_entry in frozen_data["domains"][dom]:
            pred_p = out_dir / "predictions" / p_entry["file"]
            assert pred_p.is_file()
            arr = np.load(pred_p)
            assert arr.dtype == np.uint8
            assert arr.ndim == 4  # (T, Z, H, W)

    # 3. METRICS.json validation: bound hashes, synthetic flag, per-domain manifest sha
    assert res["schema"] == "self_audit.shared_evaluation_metrics.v2"
    assert res["recipe_sha256"] == digest(recipe_file)
    assert res["frozen_predictions_sha256"] == digest(frozen_receipt)
    assert res["checkpoint_manifest_sha256"] == ckpt_sha
    assert res["synthetic"] is True

    for dom in ["acdc", "mnm"]:
        d_res = res["domains"][dom]
        assert d_res["manifest_sha256"] == (acdc_sha if dom == "acdc" else mnm_sha)
        assert d_res["patient_count"]["expected"] == 2
        assert d_res["patient_count"]["scored"] == 2
        assert d_res["patient_count"]["with_defined_fg"] == 2
        assert d_res["frame_coverage"] == 1.0
        assert d_res["prediction_coverage"] == 1.0
        assert d_res["valid_voxels"] > 0
        assert "development_gate" in d_res
        assert d_res["development_gate"]["status"] in {"GO", "NO_GO"}
        assert d_res["development_gate"]["complete_coverage"] is True
        assert isinstance(d_res["development_gate"]["all_classes_predicted"], bool)

        # Per-patient valid prediction counts and coverage
        for pid, p_info in d_res["patient_metrics"].items():
            assert p_info["valid_voxels"] > 0
            assert p_info["prediction_coverage"] == 1.0
            assert sum(p_info["class_prediction_counts"].values()) == p_info["valid_voxels"]


def test_freeze_before_ref_spy_all_prediction_files(trained_student_checkpoint, tmp_path, monkeypatch):
    """Root review: spy verifies ALL prediction files freeze before ANY reference mask is opened."""
    ckpt_dir, ckpt_sha, train_m, train_m_sha = trained_student_checkpoint
    acdc_m, acdc_sha = make_eval_domain(tmp_path / "acdc_eval", "acdc", seed=101, n_patients=2)
    mnm_m, mnm_sha = make_eval_domain(tmp_path / "mnm_eval", "mnm", seed=202, n_patients=2)
    domains = {"acdc": (acdc_m, acdc_sha), "mnm": (mnm_m, mnm_sha)}
    out_dir = tmp_path / "eval_spy"

    orig_load = np.load
    mask_loads = []

    def spy_load(file, *args, **kwargs):
        path_str = str(file)
        if "_mask.npy" in path_str:
            # Mask loading!
            # 1. PREDICTIONS_FROZEN.json MUST already exist on disk
            frozen_receipt = out_dir / "PREDICTIONS_FROZEN.json"
            assert frozen_receipt.is_file(), "Reference mask opened BEFORE PREDICTIONS_FROZEN.json was written!"

            # 2. ALL prediction files for ALL domains must already exist on disk
            for dom, pids in [("acdc", ["acdc_eval_000", "acdc_eval_001"]), ("mnm", ["mnm_eval_000", "mnm_eval_001"])]:
                for pid in pids:
                    pfile = out_dir / "predictions" / dom / f"{pid}.npy"
                    assert pfile.is_file(), f"Prediction file {pfile} missing when reading reference mask!"

            mask_loads.append(path_str)
        return orig_load(file, *args, **kwargs)

    monkeypatch.setattr(np, "load", spy_load)

    evaluate_shared_student(
        checkpoint_path=ckpt_dir,
        checkpoint_sha256=ckpt_sha,
        domains_data=domains,
        out_dir=out_dir,
        train_manifest_path=train_m,
        train_manifest_sha256=train_m_sha,
        device="cpu",
    )

    assert len(mask_loads) == 4, "Expected 4 mask loads across the two domains"


def test_reject_duplicate_case_aliases_and_patient_prefix_conflicts(trained_student_checkpoint, tmp_path):
    """Root review: reject duplicate case aliases and patient prefix conflicts."""
    ckpt_dir, ckpt_sha, train_m, train_m_sha = trained_student_checkpoint
    acdc_m, acdc_sha = make_eval_domain(tmp_path / "acdc_eval", "acdc", seed=101)

    # 1. Duplicate case aliases in domains_data
    with pytest.raises(ValueError, match="Duplicate case alias"):
        evaluate_shared_student(
            checkpoint_path=ckpt_dir,
            checkpoint_sha256=ckpt_sha,
            domains_data={
                "acdc": (acdc_m, acdc_sha),
                "ACDC": (acdc_m, acdc_sha),
            },
            out_dir=tmp_path / "out_dup_alias",
            train_manifest_path=train_m,
            train_manifest_sha256=train_m_sha,
        )

    # 2. Patient prefix conflict: domain acdc with mnm:pid
    conflict_dir = tmp_path / "prefix_conflict_eval"
    conflict_dir.mkdir(parents=True)
    img = np.zeros((1, 1, 32, 32), dtype=np.float32)
    mask = np.zeros((1, 32, 32), dtype=np.uint8)
    ip = conflict_dir / "img.npy"; np.save(ip, img)
    mp = conflict_dir / "mask.npy"; np.save(mp, mask)

    conflict_m = conflict_dir / "manifest.json"
    write_json(conflict_m, {
        "schema": "self_audit.data.v2",
        "role": "development",
        "domain": "acdc",
        "synthetic": True,
        "patients": [{
            "id": "mnm:patient001",  # Prefix conflict with domain acdc!
            "image": ip.name,
            "image_sha256": digest(ip),
            "masks": {"0": {"path": mp.name, "sha256": digest(mp)}},
        }]
    })

    with pytest.raises(ValueError, match="conflicting domain prefix"):
        evaluate_shared_student(
            checkpoint_path=ckpt_dir,
            checkpoint_sha256=ckpt_sha,
            domains_data={"acdc": (conflict_m, digest(conflict_m))},
            out_dir=tmp_path / "out_prefix_conflict",
            train_manifest_path=train_m,
            train_manifest_sha256=train_m_sha,
        )


def test_reject_missing_train_image_fails_closed(trained_student_checkpoint, tmp_path):
    """Root review: TRAIN image missing or unreadable fails closed for image identity proof."""
    ckpt_dir, ckpt_sha, train_m, train_m_sha = trained_student_checkpoint
    acdc_m, acdc_sha = make_eval_domain(tmp_path / "acdc_eval", "acdc")

    # Tamper with TRAIN manifest by deleting one of the referenced train image files
    tm_obj = json.loads(Path(train_m).read_text())
    first_img_rel = tm_obj["labeled"][0]["image"]
    first_img_path = Path(train_m).parent / first_img_rel
    assert first_img_path.is_file()
    first_img_path.unlink()  # Image file deleted!

    with pytest.raises(FileNotFoundError, match="TRAIN image not found"):
        evaluate_shared_student(
            checkpoint_path=ckpt_dir,
            checkpoint_sha256=ckpt_sha,
            domains_data={"acdc": (acdc_m, acdc_sha)},
            out_dir=tmp_path / "out_missing_tr_img",
            train_manifest_path=train_m,
            train_manifest_sha256=train_m_sha,
        )


def test_protect_output_against_all_resolved_image_and_mask_parents(trained_student_checkpoint, tmp_path):
    """Root review: protect out against ALL referenced resolved image/mask parents before any writes."""
    ckpt_dir, ckpt_sha, train_m, train_m_sha = trained_student_checkpoint
    acdc_m, acdc_sha = make_eval_domain(tmp_path / "acdc_eval", "acdc")

    # Attempt to write output into the eval domain image/mask folder
    with pytest.raises(ValueError, match="conflicts with referenced input/parent path"):
        evaluate_shared_student(
            checkpoint_path=ckpt_dir,
            checkpoint_sha256=ckpt_sha,
            domains_data={"acdc": (acdc_m, acdc_sha)},
            out_dir=tmp_path / "acdc_eval" / "sub_pred",
            train_manifest_path=train_m,
            train_manifest_sha256=train_m_sha,
        )


def test_safe_confusion_matrix_rejects_fractional_labels():
    """Root review: mask dtype integer/shape valid classes; reject fractional labels."""
    pred = np.zeros((1, 32, 32), dtype=np.uint8)

    # 1. Fractional / float mask array
    y_float = np.ones((1, 32, 32), dtype=np.float32) * 1.5
    with pytest.raises(ValueError, match="Ground truth mask must have integer dtype"):
        safe_confusion_matrix(y_float, pred)

    # 2. Integer mask with out-of-range class
    y_invalid_class = np.zeros((1, 32, 32), dtype=np.uint8)
    y_invalid_class[0, 0, 0] = 4  # Class 4 is not in {0, 1, 2, 3, 255}
    with pytest.raises(ValueError, match="Mask contains invalid class values"):
        safe_confusion_matrix(y_invalid_class, pred)

    # 3. Shape mismatch
    y_mismatch = np.zeros((1, 16, 16), dtype=np.uint8)
    with pytest.raises(ValueError, match="Shape mismatch"):
        safe_confusion_matrix(y_mismatch, pred)


def test_reject_unnamespaced_train_overlap(trained_student_checkpoint, tmp_path):
    """Root review: reject unnamespaced vs namespaced TRAIN overlap."""
    ckpt_dir, ckpt_sha, train_m, train_m_sha = trained_student_checkpoint
    train_obj = json.loads(Path(train_m).read_text())
    overlap_pid = train_obj["labeled"][0]["id"]

    eval_dir = tmp_path / "unnamespaced_overlap_eval"
    eval_dir.mkdir(parents=True)
    img = np.zeros((1, 1, 32, 32), dtype=np.float32)
    mask = np.zeros((1, 32, 32), dtype=np.uint8)
    ip = eval_dir / "img.npy"; np.save(ip, img)
    mp = eval_dir / "mask.npy"; np.save(mp, mask)

    bad_m = eval_dir / "manifest.json"
    write_json(bad_m, {
        "schema": "self_audit.data.v2",
        "role": "development",
        "domain": "acdc",
        "synthetic": True,
        "patients": [{
            "id": f"acdc:{overlap_pid}",
            "image": ip.name,
            "image_sha256": digest(ip),
            "masks": {"0": {"path": mp.name, "sha256": digest(mp)}},
        }]
    })

    domains = {"acdc": (bad_m, digest(bad_m))}
    with pytest.raises(ValueError, match="matches unnamespaced TRAIN patient|patient ID overlap"):
        evaluate_shared_student(
            checkpoint_path=ckpt_dir,
            checkpoint_sha256=ckpt_sha,
            domains_data=domains,
            out_dir=tmp_path / "out_unnamespaced",
            train_manifest_path=train_m,
            train_manifest_sha256=train_m_sha,
        )


def test_reject_train_image_content_duplicate(trained_student_checkpoint, tmp_path):
    """Root review: reject exact image content duplicate (file SHA and voxel SHA)."""
    ckpt_dir, ckpt_sha, train_m, train_m_sha = trained_student_checkpoint
    train_obj = json.loads(Path(train_m).read_text())
    existing_img_rel = train_obj["labeled"][0]["image"]
    existing_img_sha = train_obj["labeled"][0]["image_sha256"]
    existing_img_p = Path(train_m).parent / existing_img_rel

    eval_dir = tmp_path / "dup_image_eval"
    eval_dir.mkdir(parents=True)
    mask = np.zeros((1, 32, 32), dtype=np.uint8)
    mp = eval_dir / "mask.npy"; np.save(mp, mask)

    bad_m = eval_dir / "manifest.json"
    write_json(bad_m, {
        "schema": "self_audit.data.v2",
        "role": "development",
        "domain": "acdc",
        "synthetic": True,
        "patients": [{
            "id": "completely_new_id_999",
            "image": str(existing_img_p),
            "image_sha256": existing_img_sha,
            "masks": {"0": {"path": mp.name, "sha256": digest(mp)}},
        }]
    })

    domains = {"acdc": (bad_m, digest(bad_m))}
    with pytest.raises(ValueError, match="image content overlap with TRAIN"):
        evaluate_shared_student(
            checkpoint_path=ckpt_dir,
            checkpoint_sha256=ckpt_sha,
            domains_data=domains,
            out_dir=tmp_path / "out_dup_img",
            train_manifest_path=train_m,
            train_manifest_sha256=train_m_sha,
        )


def test_absent_classes_and_all_ignored(trained_student_checkpoint, tmp_path):
    """Root review: absent class gives None Dice, fixed-denominator absent FP rate, ignore 255 has zero mass."""
    ckpt_dir, ckpt_sha, train_m, train_m_sha = trained_student_checkpoint

    eval_dir = tmp_path / "absent_classes_eval"
    eval_dir.mkdir(parents=True)

    size = 32
    # Patient 1: Has BG, MYO, LV, but NO RV (class 1 absent)
    img1 = np.ones((1, 1, size, size), dtype=np.float32) * 0.5
    mask1 = np.zeros((1, size, size), dtype=np.uint8)
    mask1[0, 10:20, 10:20] = 2  # MYO
    mask1[0, 12:18, 12:18] = 3  # LV
    ip1 = eval_dir / "p1.npy"; np.save(ip1, img1)
    mp1 = eval_dir / "m1.npy"; np.save(mp1, mask1)

    # Patient 2: All ignored (all 255)
    img2 = np.ones((1, 1, size, size), dtype=np.float32) * 0.5
    mask2 = np.full((1, size, size), 255, dtype=np.uint8)
    ip2 = eval_dir / "p2.npy"; np.save(ip2, img2)
    mp2 = eval_dir / "m2.npy"; np.save(mp2, mask2)

    m_path = eval_dir / "manifest.json"
    write_json(m_path, {
        "schema": "self_audit.data.v2",
        "role": "development",
        "domain": "acdc",
        "synthetic": True,
        "patients": [
            {
                "id": "absent_rv_patient",
                "image": ip1.name,
                "image_sha256": digest(ip1),
                "masks": {"0": {"path": mp1.name, "sha256": digest(mp1)}},
            },
            {
                "id": "all_ignored_patient",
                "image": ip2.name,
                "image_sha256": digest(ip2),
                "masks": {"0": {"path": mp2.name, "sha256": digest(mp2)}},
            }
        ]
    })

    domains = {"acdc": (m_path, digest(m_path))}
    res = evaluate_shared_student(
        checkpoint_path=ckpt_dir,
        checkpoint_sha256=ckpt_sha,
        domains_data=domains,
        out_dir=tmp_path / "out_absent",
        train_manifest_path=train_m,
        train_manifest_sha256=train_m_sha,
    )

    acdc_res = res["domains"]["acdc"]
    p1_metrics = acdc_res["patient_metrics"]["absent_rv_patient"]
    p2_metrics = acdc_res["patient_metrics"]["all_ignored_patient"]

    # In patient 1: RV Dice is None
    assert p1_metrics["RV"] is None
    avail_fg = [p1_metrics[c] for c in ["MYO", "LV"] if p1_metrics[c] is not None]
    assert p1_metrics["foreground_mean"] == pytest.approx(float(np.mean(avail_fg)))

    # In patient 2: All 255 ignored, so foreground_mean is None (not 0.0)
    assert p2_metrics["RV"] is None
    assert p2_metrics["MYO"] is None
    assert p2_metrics["LV"] is None
    assert p2_metrics["foreground_mean"] is None

    # Development gate status is NO_GO
    assert acdc_res["development_gate"]["status"] in {"NO_GO", "UNDEFINED_NO_GO"}


def test_reject_negative_or_out_of_bounds_frame(trained_student_checkpoint, tmp_path):
    """Root review: negative or out of bounds frame rejected before opening mask arrays."""
    ckpt_dir, ckpt_sha, train_m, train_m_sha = trained_student_checkpoint

    eval_dir = tmp_path / "neg_frame_eval"
    eval_dir.mkdir(parents=True)
    img = np.zeros((1, 1, 32, 32), dtype=np.float32)
    mask = np.zeros((1, 32, 32), dtype=np.uint8)
    ip = eval_dir / "img.npy"; np.save(ip, img)
    mp = eval_dir / "mask.npy"; np.save(mp, mask)

    bad_m = eval_dir / "manifest.json"
    write_json(bad_m, {
        "schema": "self_audit.data.v2",
        "role": "development",
        "domain": "acdc",
        "synthetic": True,
        "patients": [{
            "id": "neg_frame_patient",
            "image": ip.name,
            "image_sha256": digest(ip),
            "masks": {"-1": {"path": mp.name, "sha256": digest(mp)}},
        }]
    })

    domains = {"acdc": (bad_m, digest(bad_m))}
    with pytest.raises(ValueError, match="out of bounds"):
        evaluate_shared_student(
            checkpoint_path=ckpt_dir,
            checkpoint_sha256=ckpt_sha,
            domains_data=domains,
            out_dir=tmp_path / "out_neg_frame",
            train_manifest_path=train_m,
            train_manifest_sha256=train_m_sha,
        )


def test_reject_missing_prediction_or_tampering(trained_student_checkpoint, tmp_path, monkeypatch):
    """Root review: missing or tampered prediction file fails closed in Phase 2."""
    from scripts import evaluate_shared_student as ess_mod
    ckpt_dir, ckpt_sha, train_m, train_m_sha = trained_student_checkpoint
    acdc_m, acdc_sha = make_eval_domain(tmp_path / "acdc_eval", "acdc", seed=101, n_patients=2)
    domains = {"acdc": (acdc_m, acdc_sha)}

    # Subcase A: Tampered prediction file
    out_dir_tamper = tmp_path / "eval_tamper"
    orig_write_json = ess_mod.write_json

    def tamper_after_frozen(path, data):
        orig_write_json(path, data)
        p = Path(path)
        if p.name == "PREDICTIONS_FROZEN.json":
            target_pred = out_dir_tamper / "predictions" / "acdc" / "acdc_eval_000.npy"
            assert target_pred.is_file()
            target_pred.write_bytes(target_pred.read_bytes() + b"tampered_corrupt_bytes")

    monkeypatch.setattr(ess_mod, "write_json", tamper_after_frozen)
    with pytest.raises(ValueError, match="Frozen prediction file modified"):
        evaluate_shared_student(
            checkpoint_path=ckpt_dir,
            checkpoint_sha256=ckpt_sha,
            domains_data=domains,
            out_dir=out_dir_tamper,
            train_manifest_path=train_m,
            train_manifest_sha256=train_m_sha,
        )

    # Subcase B: Missing prediction file (deleted after Phase 1)
    out_dir_delete = tmp_path / "eval_delete"
    def delete_after_frozen(path, data):
        orig_write_json(path, data)
        p = Path(path)
        if p.name == "PREDICTIONS_FROZEN.json":
            target_pred = out_dir_delete / "predictions" / "acdc" / "acdc_eval_000.npy"
            assert target_pred.is_file()
            target_pred.unlink()

    monkeypatch.setattr(ess_mod, "write_json", delete_after_frozen)
    with pytest.raises(FileNotFoundError, match="Missing frozen prediction file"):
        evaluate_shared_student(
            checkpoint_path=ckpt_dir,
            checkpoint_sha256=ckpt_sha,
            domains_data=domains,
            out_dir=out_dir_delete,
            train_manifest_path=train_m,
            train_manifest_sha256=train_m_sha,
        )


def test_reject_different_checkpoint_attempt(trained_student_checkpoint, tmp_path):
    """Root review: reject mismatched checkpoint data_sha256 vs train manifest."""
    ckpt_dir, ckpt_sha, train_m, train_m_sha = trained_student_checkpoint
    acdc_m, acdc_sha = make_eval_domain(tmp_path / "acdc_eval", "acdc")
    domains = {"acdc": (acdc_m, acdc_sha)}

    other_train_dir = tmp_path / "other_train"
    other_m_p, other_m_sha = synthetic_fixture(other_train_dir, seed=999, n_labeled=3, n_unlabeled=2, size=32)

    with pytest.raises(ValueError, match="does not match expected train_manifest_sha256"):
        evaluate_shared_student(
            checkpoint_path=ckpt_dir,
            checkpoint_sha256=ckpt_sha,
            domains_data=domains,
            out_dir=tmp_path / "out_diff_ckpt",
            train_manifest_path=other_m_p,
            train_manifest_sha256=other_m_sha,
        )


def test_cli_reject_partial_pairs_and_missing_inputs(trained_student_checkpoint, tmp_path, monkeypatch):
    """Root review: CLI rejects partial arg pairs and missing required inputs."""
    ckpt_dir, ckpt_sha, train_m, train_m_sha = trained_student_checkpoint
    acdc_m, acdc_sha = make_eval_domain(tmp_path / "acdc_eval", "acdc")

    # Partial pair --data-acdc without --data-acdc-sha256
    monkeypatch.setattr("sys.argv", [
        "evaluate_shared_student.py",
        "--checkpoint", str(ckpt_dir),
        "--checkpoint-sha256", ckpt_sha,
        "--train-manifest", str(train_m),
        "--train-manifest-sha256", train_m_sha,
        "--data-acdc", str(acdc_m),
        "--out", str(tmp_path / "cli_out1"),
    ])
    with pytest.raises(SystemExit):
        eval_main()

    # Missing mandatory --train-manifest
    monkeypatch.setattr("sys.argv", [
        "evaluate_shared_student.py",
        "--checkpoint", str(ckpt_dir),
        "--checkpoint-sha256", ckpt_sha,
        "--data-acdc", str(acdc_m),
        "--data-acdc-sha256", acdc_sha,
        "--out", str(tmp_path / "cli_out2"),
    ])
    with pytest.raises(SystemExit):
        eval_main()
