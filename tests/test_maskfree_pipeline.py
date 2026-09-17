"""Bounded native-grid software integration; no real data or GPU evidence."""
from pathlib import Path
import io
import json
import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest
import torch

from self_audit_maskfree.config import MaskfreeConfig
from self_audit_maskfree.trainer import MaskfreeTrainer
from self_audit_maskfree.export import validate_freeze
from self_audit_maskfree.progress import TerminalProgress


@pytest.mark.parametrize("dataset", ["acdc", "mnms"])
def test_native_multiframe_pipeline_freezes_all_methods_before_verification(tmp_path, dataset):
    torch.set_num_threads(1)
    data = tmp_path / "images"
    data.mkdir()
    rng = np.random.default_rng(123)
    yy, xx = np.mgrid[-1:1:24j, -1:1:24j]
    radius = np.sqrt(xx ** 2 + yy ** 2)
    plane = 50 + 35 * (radius < .55) + 25 * (radius < .32)
    for patient in range(4):
        volume = np.stack([plane + rng.normal(0, 2, plane.shape) for _ in range(4)], axis=-1)
        volume = volume.reshape(24, 24, 2, 2).astype(np.float32)
        image = nib.Nifti1Image(volume, np.diag([1.5, 1.5, 8., 1.]))
        image.header.set_xyzt_units("mm", "sec")
        nib.save(image, data / f"patient{patient:03d}_4d.nii.gz")
    config = MaskfreeConfig(dataset=dataset, data_root=str(data), output_dir=str(tmp_path),
        total_epochs=1, warmup_epochs=0, batch_size=2, image_size=24,
        width=4, feature_dim=4, device="cpu", allow_cpu=True, amp=False,
        wandb_mode="disabled", run_id="native-integration")
    trainer = MaskfreeTrainer(config)
    with TerminalProgress(stream=io.StringIO()) as progress:
        progress.attach(trainer.paths.reports / "progress.jsonl")
        report = trainer.run()
    assert report["status"] == "partial", report
    assert not report["full_150_complete"]
    assert report["finalization"]["available"], report["finalization"]
    assert report["finalization"]["units"] == 16
    assert report["finalization"]["verification_rows"] == 16
    events = list(map(json.loads, (trainer.paths.reports / "progress.jsonl").read_text().splitlines()))
    aggregates = [r for r in events if r["event"] == "image_only.metric"]
    assert aggregates and all("available" in r["metric"] and "count" in r["metric"] for r in aggregates)
    assert any("student_audited" in r["metric"]["name"] for r in aggregates)
    frozen = json.loads(trainer.paths.freeze_manifest.read_text())
    validate_freeze(frozen)
    volumes = [row for row in frozen["predictions"] if row["kind"] == "volume"]
    assert len({row["volume_id"] for row in volumes}) == 8
    assert all(row["complete_volume"] for row in volumes)
    assert all(row["native_export_available"] for row in volumes)
    verification = json.loads((trainer.paths.reports / "verification.json").read_text())["rows"]
    assert len({row["unit_id"] for row in verification}) == 16
    assert all({"student_no_audit", "student_audited"} <= set(row["method_to_candidate"])
               for row in verification)
    for row in verification:
        ids = {candidate["candidate_id"] for candidate in row["candidates"]}
        assert set(row["method_to_candidate"].values()) <= ids
        assert row["repeated_annotation_stability"]["available"]
        assert set(row["repeated_annotation_stability"]["methods"]) == {
            "E1_cuts_inspired_control", "E5_evidence_plus_challenge",
            "student_no_audit", "student_audited"}
    # A continuation of finalized software artifacts must preserve frozen bytes.
    before = trainer.paths.freeze_manifest.read_bytes()
    trainer.paths.failure_report.write_text(json.dumps({"stage": "injected stale finalization failure"}))
    resumed = MaskfreeTrainer(config.replace(resume=str(trainer.paths.last_checkpoint)))
    resumed_report = resumed.run()
    assert resumed_report["finalization"]["available"], resumed_report["finalization"]
    assert trainer.paths.freeze_manifest.read_bytes() == before
    assert not trainer.paths.failure_report.exists()
    assert list(trainer.paths.reports.glob("recovered_failure_*.json"))
    validate_freeze(frozen)

    # Synthetic reference masks are created only after both prediction sets
    # freeze. Exercise the isolated native evaluator, including a wrong-grid
    # rejection; no reference quantity can feed this completed software run.
    reference_root = tmp_path / "synthetic_reference"
    reference_root.mkdir()
    reference = np.zeros((24, 24, 2), dtype=np.uint8)
    reference[5:19, 5:19] = 2
    reference[8:16, 8:16] = 3
    reference[3:8, 10:16] = 1
    cases = []
    for i, entry in enumerate(v for v in volumes if v["prediction_name"] == "E5_evidence_plus_challenge"):
        path = reference_root / f"phantom_{i}.nii.gz"
        image = nib.Nifti1Image(reference, np.diag([1.5, 1.5, 8., 1.]))
        image.header.set_xyzt_units("mm", "sec")
        nib.save(image, path)
        cases.append({"patient_id": entry["patient_id"], "volume_id": entry["volume_id"],
                      "mask_path": str(path), "geometry_valid": True, "spacing": [1.5, 1.5, 8.]})
    reference_config = reference_root / "config.json"
    reference_config.write_text(json.dumps({"dataset": dataset, "split": "all", "epoch": 0,
        "protocol": "spatial_predictive", "reference_label_map": {"0": "BG", "1": "RV", "2": "MYO", "3": "LV"},
        "cases": cases}))
    output = reference_root / "evaluation"
    command = [sys.executable, "scripts/evaluate_maskfree_reference.py",
               "--frozen-manifest", str(trainer.paths.freeze_manifest),
               "--reference-config", str(reference_config), "--output", str(output)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    reference_report = json.loads((output / "reference_metrics.json").read_text())
    assert reference_report["available"]
    assert reference_report["overlap"]["E5_evidence_plus_challenge"]["volumes_counted"] == 8
    assert any(row["available"] and row["unit"] == "mm" for row in reference_report["rows"]
               if row["name"].endswith(".hd95"))
    first_reference = Path(cases[0]["mask_path"])
    wrong_affine = np.diag([1.5, 1.5, 8., 1.])
    wrong_affine[0, 3] = 3.0
    image = nib.Nifti1Image(reference, wrong_affine)
    image.header.set_xyzt_units("mm", "sec")
    nib.save(image, first_reference)
    rejected = subprocess.run(command, capture_output=True, text=True, timeout=45)
    assert rejected.returncode != 0
    assert "affines differ" in rejected.stderr
