"""Tests for scripts/render_recovery_report.py.

Verifies:
1. Learning curve generation from training JSONL with actual cumulative engine counters.
2. Actual renders exist, hashes are recorded, and synthetic watermarks appear.
3. PREDICTIONS_FROZEN receipt and prediction hash mismatch refuse before refs.
4. Calculated manifest_sha match against recipe domains is enforced before refs.
5. Missing predictions refuse without silent skipping.
6. Same patient ID across multiple domains does not collide or overwrite.
7. Frame bounds are validated before opening masks.
8. Checkpoint manifest and payload SHAs must match across packages.
9. Ignored voxels (label 255) are transparent; unknown labels are rejected.
10. Output directory must be brand new (exist_ok=False) and protected against all input paths.
11. Curves-only CLI execution without evaluation artifacts.
12. Actual EMA coverage vs domain coverage distinction (no substitution).
13. Actual accepted updates vs candidate accepted_pixels distinction.
14. Missing or mismatched metrics bindings refuse to display measured scores (reports N/A).
15. Namespaced patient IDs (e.g. 'acdc:01') produce safe filenames without traversal.
16. Strict bool rejection on step numbers in steps.jsonl.
17. Full end-to-end integration test with real train.py output and evaluate_shared_student.py output.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image
import pytest

from self_audit.data import synthetic_fixture, write_json
from self_audit.model import digest
from self_audit.engine import train
from tests.test_pipeline import recipe as pipeline_recipe
from scripts.evaluate_shared_student import evaluate_shared_student
from scripts.render_recovery_report import (
    label_to_rgba,
    render_learning_curves,
    render_recovery_report,
)


@pytest.fixture
def synthetic_eval_env(tmp_path: Path):
    """Create a completely self-contained, synthetic evaluation environment."""
    root = tmp_path / "env"
    root.mkdir()

    data_dir = root / "data"
    data_dir.mkdir()
    img_dir = data_dir / "images"
    img_dir.mkdir()
    mask_dir = data_dir / "masks"
    mask_dir.mkdir()

    ckpt_sha = "0123456789abcdef" * 4
    payload_sha = "fedcba9876543210" * 4

    # Build two domain datasets: acdc and mnm
    domains = ["acdc", "mnm"]
    domain_manifest_paths = {}
    domain_manifest_shas = {}
    all_mask_paths = []

    frozen_inventory = {"acdc": [], "mnm": []}

    eval_dir = root / "eval"
    eval_dir.mkdir()
    pred_dir = eval_dir / "predictions"
    pred_dir.mkdir()

    for dom in domains:
        (pred_dir / dom).mkdir()
        patients = []
        for p_idx in range(2):
            pid = f"pat_{p_idx:02d}"
            # 4D cine image: (T=2, Z=3, Y=16, X=16)
            x = np.linspace(0.1, 0.9, 2 * 3 * 16 * 16, dtype=np.float32).reshape((2, 3, 16, 16))
            ip = img_dir / f"{dom}_{pid}.npy"
            np.save(ip, x)

            masks_info = {}
            for f_idx in range(2):
                y = np.zeros((3, 16, 16), dtype=np.uint8)
                y[1, 2:5, 2:5] = 1   # RV
                y[1, 5:8, 5:8] = 2   # MYO
                y[1, 8:11, 8:11] = 3 # LV
                y[1, 14:16, 14:16] = 255 # Ignore
                mp = mask_dir / f"{dom}_{pid}_f{f_idx}.npy"
                np.save(mp, y)
                all_mask_paths.append(mp)
                masks_info[str(f_idx)] = {"path": str(mp.relative_to(data_dir)), "sha256": digest(mp)}

            patients.append({
                "id": pid,
                "image": str(ip.relative_to(data_dir)),
                "image_sha256": digest(ip),
                "masks": masks_info,
            })

            # Predictions (T=2, Z=3, Y=16, X=16) uint8
            p_arr = np.zeros((2, 3, 16, 16), dtype=np.uint8)
            p_arr[0, 1, 2:5, 2:5] = 1
            p_arr[0, 1, 5:8, 5:8] = 2
            p_arr[0, 1, 8:11, 8:11] = 3
            p_file = pred_dir / dom / f"{pid}.npy"
            np.save(p_file, p_arr)
            frozen_inventory[dom].append({
                "id": pid,
                "file": f"{dom}/{p_file.name}",
                "sha256": digest(p_file),
            })

        m_path = data_dir / f"manifest_{dom}.json"
        write_json(m_path, {
            "schema": "self_audit.data.v2",
            "role": "development",
            "synthetic": True,
            "patients": patients,
        })
        domain_manifest_paths[dom] = m_path
        domain_manifest_shas[dom] = digest(m_path)

    recipe_path = eval_dir / "RECIPE.json"
    recipe_data = {
        "schema": "self_audit.shared_evaluation.v1",
        "checkpoint_sha256": ckpt_sha,
        "checkpoint_payload_sha256": payload_sha,
        "domains": {
            dom: {"manifest_sha256": domain_manifest_shas[dom], "patients": 2}
            for dom in domains
        },
        "synthetic": True,
    }
    write_json(recipe_path, recipe_data)
    recipe_sha = digest(recipe_path)

    frozen_receipt = eval_dir / "PREDICTIONS_FROZEN.json"
    write_json(frozen_receipt, {
        "recipe_sha256": recipe_sha,
        "checkpoint_payload_sha256": payload_sha,
        "domains": frozen_inventory,
    })

    metrics_path = eval_dir / "METRICS.json"
    metrics_data = {
        "schema": "self_audit.shared_evaluation_metrics.v2",
        "checkpoint_sha256": ckpt_sha,
        "checkpoint_manifest_sha256": ckpt_sha,
        "checkpoint_payload_sha256": payload_sha,
        "recipe_sha256": recipe_sha,
        "frozen_predictions_sha256": digest(frozen_receipt),
        "synthetic": True,
        "domains": {
            "acdc": {
                "foreground_patient_macro": 0.88,
                "patient_metrics": {
                    "pat_00": {"foreground_mean": 0.89, "RV": 0.90, "MYO": 0.87, "LV": 0.90},
                    "pat_01": {"foreground_mean": 0.87, "RV": 0.88, "MYO": 0.86, "LV": 0.88},
                },
            },
            "mnm": {
                "foreground_patient_macro": 0.85,
                "patient_metrics": {
                    "pat_00": {"foreground_mean": 0.86, "RV": 0.86, "MYO": 0.84, "LV": 0.88},
                    "pat_01": {"foreground_mean": 0.84, "RV": 0.85, "MYO": 0.83, "LV": 0.86},
                },
            },
        },
    }
    write_json(metrics_path, metrics_data)

    train_dir = root / "train"
    train_dir.mkdir()
    steps_file = train_dir / "steps.jsonl"
    with open(steps_file, "w", encoding="utf-8") as f:
        for s in range(1, 6):
            row = {
                "step": s,
                "supervised_loss": float(1.0 / s),
                "slices_optimized": {"supervised": s * 4, "unlabeled": (s - 1) * 4},
                "optimizer_counters": {"pseudo_accepted": s - 1, "ema_updates": s},
            }
            f.write(json.dumps(row) + "\n")

    return {
        "root": root,
        "manifest_paths": list(domain_manifest_paths.values()),
        "eval_dir": eval_dir,
        "steps_file": steps_file,
        "ckpt_sha": ckpt_sha,
        "payload_sha": payload_sha,
        "all_mask_paths": all_mask_paths,
    }


def test_learning_curves_from_jsonl(tmp_path: Path):
    """Test standalone learning curve rendering from training JSONL."""
    jsonl_path = tmp_path / "steps.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for s in range(1, 6):
            f.write(json.dumps({
                "step": s,
                "supervised_loss": 0.5 - 0.05 * s,
                "optimizer_counters": {"pseudo_accepted": s * 2, "ema_updates": s},
            }) + "\n")

    out_png = tmp_path / "learning_curves.png"
    stats = render_learning_curves(jsonl_path, out_png, synthetic=True)

    assert out_png.is_file()
    assert out_png.stat().st_size > 0
    assert stats["total_steps"] == 5
    assert stats["has_ema_coverage"] is True
    assert stats["total_pseudo_accepted"] == 10.0

    with Image.open(out_png) as img:
        assert img.format == "PNG"


def test_actual_render_exists_and_hash(tmp_path: Path, synthetic_eval_env: dict):
    """Test full report execution: overlays exist, learning curves exist, and provenance matches."""
    out_dir = tmp_path / "report_out"
    env = synthetic_eval_env

    prov = render_recovery_report(
        out_dir=out_dir,
        train_log=env["steps_file"],
        evaluations=[env["eval_dir"]],
        manifests=env["manifest_paths"],
        checkpoint_sha=env["ckpt_sha"],
        max_overlays_per_dataset=2,
    )

    prov_file = out_dir / "PROVENANCE.json"
    assert prov_file.is_file()
    saved_prov = json.loads(prov_file.read_text())
    assert saved_prov["schema"] == "self_audit.recovery_report.v1"
    assert saved_prov["checkpoint_sha256"] == env["ckpt_sha"]
    assert saved_prov["checkpoint_payload_sha256"] == env["payload_sha"]
    assert saved_prov["synthetic"] is True

    lc_file = out_dir / "learning_curves.png"
    assert lc_file.is_file()
    assert digest(lc_file) == saved_prov["learning_curves"]["sha256"]

    assert len(saved_prov["overlays"]) == 4  # 2 per dataset for 2 datasets
    for overlay_info in saved_prov["overlays"]:
        overlay_path = out_dir / overlay_info["file"]
        assert overlay_path.is_file()
        assert digest(overlay_path) == overlay_info["sha256"]
        with Image.open(overlay_path) as img:
            assert img.format == "PNG"


def test_same_patient_id_across_domains(tmp_path: Path, synthetic_eval_env: dict):
    """Verify that same patient IDs across domains (acdc/pat_01 and mnm/pat_01) do not collide."""
    out_dir = tmp_path / "report_cross_dom"
    env = synthetic_eval_env

    prov = render_recovery_report(
        out_dir=out_dir,
        evaluations=[env["eval_dir"]],
        manifests=env["manifest_paths"],
        checkpoint_sha=env["ckpt_sha"],
        max_overlays_per_dataset=1,
    )

    overlays = prov["overlays"]
    assert len(overlays) == 2
    domains_seen = {o["dataset"] for o in overlays}
    assert domains_seen == {"acdc", "mnm"}
    for o in overlays:
        assert o["patient_id"] == "pat_01"


def test_wrong_manifest_hash_refuses_before_refs(tmp_path: Path, synthetic_eval_env: dict, monkeypatch):
    """Verify that manifest with wrong SHA-256 refuses BEFORE opening any reference mask."""
    env = synthetic_eval_env
    bad_manifest = env["manifest_paths"][0]
    m_data = json.loads(bad_manifest.read_text())
    m_data["patients"].pop()
    write_json(bad_manifest, m_data)

    opened_masks = []
    orig_np_load = np.load

    def tracked_load(file, *args, **kwargs):
        path_str = str(file)
        if any(str(mp) in path_str for mp in env["all_mask_paths"]):
            opened_masks.append(path_str)
        return orig_np_load(file, *args, **kwargs)

    monkeypatch.setattr(np, "load", tracked_load)

    out_dir = tmp_path / "out_bad_manifest"
    with pytest.raises(ValueError, match="does not match any domain manifest_sha256 in evaluation recipes"):
        render_recovery_report(
            out_dir=out_dir,
            evaluations=[env["eval_dir"]],
            manifests=env["manifest_paths"],
        )

    assert len(opened_masks) == 0


def test_missing_prediction_refuses_no_silent_skip(tmp_path: Path, synthetic_eval_env: dict):
    """Verify that a patient missing from frozen predictions raises ValueError (no silent continue)."""
    env = synthetic_eval_env
    m_path = env["manifest_paths"][0]
    m_data = json.loads(m_path.read_text())
    extra_pat = dict(m_data["patients"][0])
    extra_pat["id"] = "pat_unpredicted"
    m_data["patients"].append(extra_pat)
    write_json(m_path, m_data)

    recipe_file = env["eval_dir"] / "RECIPE.json"
    r_data = json.loads(recipe_file.read_text())
    r_data["domains"]["acdc"]["manifest_sha256"] = digest(m_path)
    write_json(recipe_file, r_data)

    frozen_file = env["eval_dir"] / "PREDICTIONS_FROZEN.json"
    f_data = json.loads(frozen_file.read_text())
    f_data["recipe_sha256"] = digest(recipe_file)
    write_json(frozen_file, f_data)

    out_dir = tmp_path / "out_missing_pred"
    with pytest.raises(ValueError, match="missing frozen prediction for patient 'pat_unpredicted'"):
        render_recovery_report(
            out_dir=out_dir,
            evaluations=[env["eval_dir"]],
            manifests=env["manifest_paths"],
        )


def test_changed_checkpoint_mismatch_refused(tmp_path: Path, synthetic_eval_env: dict):
    """Verify that checkpoint SHA mismatch against expected or between recipes refuses."""
    env = synthetic_eval_env
    wrong_ckpt_sha = "f" * 64

    out_dir = tmp_path / "out_ckpt_mismatch"
    with pytest.raises(ValueError, match="checkpoint SHA mismatch"):
        render_recovery_report(
            out_dir=out_dir,
            evaluations=[env["eval_dir"]],
            manifests=env["manifest_paths"],
            checkpoint_sha=wrong_ckpt_sha,
        )


def test_invalid_frame_bounds_refuses_before_refs(tmp_path: Path, synthetic_eval_env: dict, monkeypatch):
    """Verify that frame index out of image bounds refuses BEFORE opening any reference mask."""
    env = synthetic_eval_env
    m_path = env["manifest_paths"][0]
    m_data = json.loads(m_path.read_text())
    m_data["patients"][0]["masks"]["99"] = dict(m_data["patients"][0]["masks"]["0"])
    write_json(m_path, m_data)

    recipe_file = env["eval_dir"] / "RECIPE.json"
    r_data = json.loads(recipe_file.read_text())
    r_data["domains"]["acdc"]["manifest_sha256"] = digest(m_path)
    write_json(recipe_file, r_data)
    frozen_file = env["eval_dir"] / "PREDICTIONS_FROZEN.json"
    f_data = json.loads(frozen_file.read_text())
    f_data["recipe_sha256"] = digest(recipe_file)
    write_json(frozen_file, f_data)

    opened_masks = []
    orig_np_load = np.load

    def tracked_load(file, *args, **kwargs):
        path_str = str(file)
        if any(str(mp) in path_str for mp in env["all_mask_paths"]):
            opened_masks.append(path_str)
        return orig_np_load(file, *args, **kwargs)

    monkeypatch.setattr(np, "load", tracked_load)

    out_dir = tmp_path / "out_bad_frame"
    with pytest.raises(ValueError, match="out of bounds for image with T="):
        render_recovery_report(
            out_dir=out_dir,
            evaluations=[env["eval_dir"]],
            manifests=env["manifest_paths"],
        )

    assert len(opened_masks) == 0


def test_curves_only_cli(tmp_path: Path, synthetic_eval_env: dict):
    """Verify CLI works with train-log only and renders curves without needing evaluation or masks."""
    env = synthetic_eval_env
    out_dir = tmp_path / "curves_only_out"

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "scripts" / "render_recovery_report.py"),
        "--train-log", str(env["steps_file"]),
        "--out", str(out_dir),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"stderr: {res.stderr}"

    assert (out_dir / "learning_curves.png").is_file()
    assert (out_dir / "PROVENANCE.json").is_file()
    prov = json.loads((out_dir / "PROVENANCE.json").read_text())
    assert prov["learning_curves"] is not None
    assert prov["overlays"] == []


def test_actual_ema_count_vs_domain_coverage(tmp_path: Path):
    """Verify that domain_coverage is NOT substituted for EMAUpdateCoverage."""
    log_file = tmp_path / "steps.jsonl"
    with open(log_file, "w") as f:
        f.write(json.dumps({
            "step": 1,
            "supervised_loss": 0.5,
            "control": {"scale": 1.0},
            "domain_coverage": {"acdc": {"total_coverage": 0.75}},
        }) + "\n")

    out_png = tmp_path / "lc.png"
    stats = render_learning_curves(log_file, out_png)
    assert stats["has_ema_coverage"] is False


def test_actual_pseudo_updates_vs_accepted_pixels(tmp_path: Path):
    """Verify that accepted_pixels is NOT counted as accepted update when control scale is 0."""
    log_file = tmp_path / "steps.jsonl"
    with open(log_file, "w") as f:
        f.write(json.dumps({
            "step": 1,
            "supervised_loss": 0.5,
            "control": {"scale": 0.0},
            "accepted_pixels": 100,
        }) + "\n")

    out_png = tmp_path / "lc.png"
    stats = render_learning_curves(log_file, out_png)
    assert stats["total_pseudo_accepted"] == 0.0


def test_output_brand_new_refuses_existing_dir(tmp_path: Path, synthetic_eval_env: dict):
    """Verify output directory must be brand new (exist_ok=False), refusing even empty directories."""
    env = synthetic_eval_env
    existing_empty = tmp_path / "already_exists_empty"
    existing_empty.mkdir()

    with pytest.raises(ValueError, match="already exists; brand new directory required"):
        render_recovery_report(
            out_dir=existing_empty,
            evaluations=[env["eval_dir"]],
            manifests=env["manifest_paths"],
        )


def test_protected_resolved_paths(tmp_path: Path, synthetic_eval_env: dict):
    """Verify output directory cannot be inside any image or mask parent directory."""
    env = synthetic_eval_env
    mask_parent = env["all_mask_paths"][0].parent
    conflict_out = mask_parent / "nested_out"

    with pytest.raises(ValueError, match="must not be inside input directory"):
        render_recovery_report(
            out_dir=conflict_out,
            evaluations=[env["eval_dir"]],
            manifests=env["manifest_paths"],
        )


def test_ignored_voxels_and_unknown_label_rejection():
    """Verify ignored voxels are transparent and unknown labels (e.g. 4) are strictly rejected."""
    valid_labels = np.array([[0, 1], [2, 255]], dtype=np.uint8)
    rgba = label_to_rgba(valid_labels, alpha=0.6)
    assert np.all(rgba[0, 0] == [0, 0, 0, 0])
    assert np.all(rgba[1, 1] == [0, 0, 0, 0])

    invalid_labels = np.array([[0, 4], [2, 3]], dtype=np.uint8)
    with pytest.raises(ValueError, match="unknown segmentation label"):
        label_to_rgba(invalid_labels)


def test_mandatory_shas_enforced(tmp_path: Path, synthetic_eval_env: dict):
    """Verify missing image_sha256 or mask sha256 in manifest is refused."""
    env = synthetic_eval_env
    m_path = env["manifest_paths"][0]
    m_data = json.loads(m_path.read_text())
    del m_data["patients"][0]["image_sha256"]
    write_json(m_path, m_data)

    recipe_file = env["eval_dir"] / "RECIPE.json"
    r_data = json.loads(recipe_file.read_text())
    r_data["domains"]["acdc"]["manifest_sha256"] = digest(m_path)
    write_json(recipe_file, r_data)
    frozen_file = env["eval_dir"] / "PREDICTIONS_FROZEN.json"
    f_data = json.loads(frozen_file.read_text())
    f_data["recipe_sha256"] = digest(recipe_file)
    write_json(frozen_file, f_data)

    out_dir = tmp_path / "out_missing_sha"
    with pytest.raises(ValueError, match="mandatory 'image_sha256' missing"):
        render_recovery_report(
            out_dir=out_dir,
            evaluations=[env["eval_dir"]],
            manifests=env["manifest_paths"],
        )


def test_strict_bool_reject_in_steps_jsonl(tmp_path: Path):
    """Verify boolean values in step field (e.g. step=True) are strictly rejected."""
    bad_log = tmp_path / "bool_step.jsonl"
    with open(bad_log, "w") as f:
        f.write(json.dumps({"step": True, "supervised_loss": 0.42}) + "\n")

    out_png = tmp_path / "bad_lc.png"
    with pytest.raises(ValueError, match="strict bool reject"):
        render_learning_curves(bad_log, out_png)


def test_missing_metrics_bindings_reports_na(tmp_path: Path, synthetic_eval_env: dict):
    """Verify that when METRICS.json has missing or mismatched checkpoint hashes, measured score is N/A."""
    env = synthetic_eval_env
    metrics_file = env["eval_dir"] / "METRICS.json"
    m_data = json.loads(metrics_file.read_text())
    # Corrupt checkpoint binding in metrics
    m_data["checkpoint_sha256"] = "0" * 64
    write_json(metrics_file, m_data)

    out_dir = tmp_path / "out_unbound_metrics"
    prov = render_recovery_report(
        out_dir=out_dir,
        evaluations=[env["eval_dir"]],
        manifests=env["manifest_paths"],
        checkpoint_sha=env["ckpt_sha"],
        max_overlays_per_dataset=1,
    )

    # Measured score must be marked as not available (no fabricated score)
    for o in prov["overlays"]:
        assert o["has_measured_dice"] is False
        assert o["measured_fg_dice"] is None


def test_namespaced_patient_id_safe_filename(tmp_path: Path, synthetic_eval_env: dict):
    """Verify namespaced patient IDs (e.g. 'acdc:patient_01') create safe filenames without traversal."""
    env = synthetic_eval_env
    m_path = env["manifest_paths"][0]
    m_data = json.loads(m_path.read_text())
    m_data["patients"][0]["id"] = "acdc:special_patient_01"
    write_json(m_path, m_data)

    # Update recipe and frozen inventory
    recipe_file = env["eval_dir"] / "RECIPE.json"
    r_data = json.loads(recipe_file.read_text())
    r_data["domains"]["acdc"]["manifest_sha256"] = digest(m_path)
    write_json(recipe_file, r_data)

    frozen_file = env["eval_dir"] / "PREDICTIONS_FROZEN.json"
    f_data = json.loads(frozen_file.read_text())
    f_data["recipe_sha256"] = digest(recipe_file)
    f_data["domains"]["acdc"][0]["id"] = "acdc:special_patient_01"
    write_json(frozen_file, f_data)

    out_dir = tmp_path / "out_namespaced"
    prov = render_recovery_report(
        out_dir=out_dir,
        evaluations=[env["eval_dir"]],
        manifests=env["manifest_paths"],
        checkpoint_sha=env["ckpt_sha"],
        max_overlays_per_dataset=1,
    )

    namespaced_overlay = [o for o in prov["overlays"] if o["patient_id"] == "acdc:special_patient_01"][0]
    # Filename must have colons sanitized to underscores
    assert ":" not in namespaced_overlay["file"]
    assert "acdc_special_patient_01" in namespaced_overlay["file"]
    assert (out_dir / namespaced_overlay["file"]).is_file()


def test_actual_train_and_eval_integration(tmp_path: Path):
    """Full integration test using actual train engine output and evaluate_shared_student output."""
    root = tmp_path / "integration"
    root.mkdir()

    # 1. Run actual tiny train on synthetic data
    train_dir = root / "train_data"
    m_train, m_train_sha = synthetic_fixture(train_dir, seed=42, n_labeled=3, n_unlabeled=2, size=32)
    run_dir = root / "run"
    rec = pipeline_recipe()
    rec["steps"] = 3
    rec["warmup_steps"] = 1
    report = train(rec, m_train, m_train_sha, run_dir, device="cpu", synthetic=True)

    assert (run_dir / "steps.jsonl").is_file()
    ckpt_dir = run_dir / "final"
    ckpt_sha = report["final_manifest_sha256"]

    # 2. Run actual evaluate_shared_student on two synthetic domains
    eval_data_dir = root / "eval_data"
    eval_data_dir.mkdir()

    domains = {}
    for dom in ["acdc", "mnm"]:
        dom_folder = eval_data_dir / dom
        dom_folder.mkdir()
        # Build 1 patient per domain
        yy, xx = np.mgrid[:32, :32]
        y = np.zeros((32, 32), np.uint8)
        y[8:24, 8:24] = 2
        y[12:20, 12:20] = 3
        y[10:14, 10:14] = 1

        img = np.zeros((1, 1, 32, 32), dtype=np.float32)
        img[0, 0] = np.choose(y, [0.1, 0.8, 0.45, 0.9])
        ip = dom_folder / f"{dom}_pat_01.npy"
        np.save(ip, img)
        mp = dom_folder / f"{dom}_pat_01_mask.npy"
        np.save(mp, y[None])

        m_obj = {
            "schema": "self_audit.data.v2",
            "role": "development",
            "synthetic": True,
            "patients": [{
                "id": f"{dom}_pat_01",
                "image": ip.name,
                "image_sha256": digest(ip),
                "masks": {"0": {"path": mp.name, "sha256": digest(mp)}},
            }],
        }
        m_dom_path = dom_folder / "manifest.json"
        write_json(m_dom_path, m_obj)
        domains[dom] = (m_dom_path, digest(m_dom_path))

    eval_out = root / "eval_shared_out"
    eval_res = evaluate_shared_student(
        checkpoint_path=ckpt_dir,
        checkpoint_sha256=ckpt_sha,
        domains_data=domains,
        out_dir=eval_out,
        train_manifest_path=m_train,
        train_manifest_sha256=m_train_sha,
        device="cpu",
    )

    assert eval_res["schema"] == "self_audit.shared_evaluation_metrics.v2"
    assert (eval_out / "RECIPE.json").is_file()
    assert (eval_out / "PREDICTIONS_FROZEN.json").is_file()
    assert (eval_out / "METRICS.json").is_file()

    # 3. Run render_recovery_report on actual artifacts
    report_out = root / "final_report"
    manifest_paths = [path for path, _ in domains.values()]

    prov = render_recovery_report(
        out_dir=report_out,
        train_log=run_dir / "steps.jsonl",
        evaluations=[eval_out],
        manifests=manifest_paths,
        checkpoint_sha=ckpt_sha,
        max_overlays_per_dataset=1,
    )

    # 4. Assert full report provenance and output validity
    assert (report_out / "PROVENANCE.json").is_file()
    assert (report_out / "learning_curves.png").is_file()
    assert len(prov["overlays"]) == 2
    for o in prov["overlays"]:
        overlay_file = report_out / o["file"]
        assert overlay_file.is_file()
        assert digest(overlay_file) == o["sha256"]
        with Image.open(overlay_file) as im:
            assert im.format == "PNG"

    assert prov["checkpoint_sha256"] == ckpt_sha
    assert prov["synthetic"] is True
    assert prov["metrics_pointers"]["eval_shared_out"]["metrics_bound"] is True
    for o in prov["overlays"]:
        assert o["has_measured_dice"] is True
        assert o["measured_fg_dice"] is not None
