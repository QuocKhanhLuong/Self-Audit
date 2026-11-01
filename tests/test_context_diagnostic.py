from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import run_context_diagnostic as diag


def test_pair_rejects_reproduced_teacher_and_ema_state():
    teacher = {"payload_sha256": diag.TEACHER_PAYLOAD}
    student = {"lineage": {
        "data_sha256": diag.TRAIN_MANIFEST, "label_budget": 8, "lids": ["p1"],
        "teacher_identity": {
            "payload_sha256": diag.TEACHER_PAYLOAD, "manifest_sha256": diag.TEACHER_MANIFEST,
            "inference_state": "student", "train_manifest_sha256": diag.TRAIN_MANIFEST,
            "labeled_ids": ["p1"],
        },
    }}
    diag.validate_pair(teacher, student)
    with pytest.raises(ValueError, match="not distilled"):
        diag.validate_pair({"payload_sha256": "f" * 64}, student)
    other = deepcopy(student)
    other["lineage"]["teacher_identity"]["inference_state"] = "ema"
    with pytest.raises(ValueError, match="raw student"):
        diag.validate_pair(teacher, other)


def frozen_fixture(tmp_path):
    gt = np.array([[[0, 1], [2, 3]]], dtype=np.uint8)
    gt_path = tmp_path / "gt.npy"
    np.save(gt_path, gt)
    manifest = {"patients": [{"id": "patientA", "image_sha256": "a" * 64,
                              "masks": {"0": {"path": str(gt_path), "sha256": diag._sha256(gt_path)}}}]}
    manifest_path = tmp_path / "manifest.json"
    diag.write_json(manifest_path, manifest)
    config = {"device": "cpu", "threads": 1,
              "val_manifest": {"path": str(manifest_path), "sha256": diag._sha256(manifest_path)}}
    out = tmp_path / "out"
    out.mkdir()
    import torch
    torch.set_num_threads(1)
    protocol = diag.protocol_snapshot("cpu", 1)
    diag.write_json(out / "PROTOCOL.json", protocol)
    for role in ("teacher", "student"):
        config[role] = {"manifest_sha256": "b" * 64, "payload_sha256": "c" * 64, "runtime_source_sha256": "d" * 64}
        folder = out / role
        folder.mkdir()
        records = []
        for lam in (0., .5, 1.):
            path = folder / f"{lam}.npy"
            np.save(path, gt)
            records.append({"patient": "patientA", "frame": 0, "lambda": lam,
                            "path": str(path), "sha256": diag._sha256(path),
                            "shape_zyx": list(gt.shape), "image_sha256": "a" * 64})
        diag.write_json(folder / "PREDICTIONS_FROZEN.json", {
            "schema": "self_audit.context_diagnostic.predictions.v1", "role": role,
            "val_manifest_sha256": config["val_manifest"]["sha256"],
            "model_manifest_sha256": "b" * 64, "model_payload_sha256": "c" * 64,
            "runtime_source_sha256": "d" * 64, "records": records,
            "protocol": protocol, "protocol_sha256": diag.protocol_sha256(protocol),
            "parity": {"mismatches": 0, "slices": 1},
        })
    return config, out, gt_path


def test_incomplete_inventory_stops_before_any_gt_read(tmp_path, monkeypatch):
    config, out, gt_path = frozen_fixture(tmp_path)
    receipt_path = out / "student/PREDICTIONS_FROZEN.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["records"].pop()
    diag.write_json(receipt_path, receipt)
    original_sha = diag._sha256
    def no_gt_hash(path):
        assert Path(path) != gt_path, "GT opened before complete global inventory"
        return original_sha(path)
    monkeypatch.setattr(diag, "_sha256", no_gt_hash)
    with pytest.raises(ValueError, match="incomplete"):
        diag.score_frozen(config, out, "e" * 64)
    assert not (out / "PREDICTIONS_FROZEN.json").exists()


def test_scoring_freezes_before_gt_and_reproduction_mismatch_is_explicit(tmp_path, monkeypatch):
    config, out, gt_path = frozen_fixture(tmp_path)
    original_sha = diag._sha256
    def check_freeze(path):
        if Path(path) == gt_path:
            assert (out / "PREDICTIONS_FROZEN.json").is_file()
        return original_sha(path)
    monkeypatch.setattr(diag, "_sha256", check_freeze)
    result = diag.score_frozen(config, out, "e" * 64)
    assert result["status"] == "REPRODUCTION_MISMATCH_DO_NOT_INTERPRET"
    assert result["lambda1_wide_reproduction"]["actual"] == 1.
    assert result["summary"]["patient_count"] == 1
    assert (out / "frame_transitions.csv").is_file()
    assert (out / "patient_transitions.csv").is_file()
    with pytest.raises(ValueError, match="already exists"):
        diag.score_frozen(config, out, "e" * 64)


def test_modified_prediction_fails_before_gt(tmp_path, monkeypatch):
    config, out, gt_path = frozen_fixture(tmp_path)
    (out / "teacher/0.0.npy").write_bytes(b"tampered")
    original_sha = diag._sha256
    def no_gt(path):
        assert Path(path) != gt_path
        return original_sha(path)
    monkeypatch.setattr(diag, "_sha256", no_gt)
    with pytest.raises(ValueError, match="hash mismatch"):
        diag.score_frozen(config, out, "e" * 64)


def test_changed_protocol_blocks_scoring(tmp_path):
    config, out, _ = frozen_fixture(tmp_path)
    protocol = json.loads((out / "PROTOCOL.json").read_text())
    protocol["tampered"] = True
    diag.write_json(out / "PROTOCOL.json", protocol)
    with pytest.raises(ValueError, match="code/environment changed"):
        diag.score_frozen(config, out, "e" * 64)
    assert not (out / "PREDICTIONS_FROZEN.json").exists()
