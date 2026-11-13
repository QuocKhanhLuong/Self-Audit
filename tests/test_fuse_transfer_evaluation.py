"""Software-only coverage for the locked Track B four-mode TEST transaction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from self_audit.candidate_fuse.config import default_transfer_config
from self_audit.candidate_fuse.evaluation import choose_deployment, evaluate_exports
from self_audit.candidate_fuse.export import export_model
from self_audit.candidate_fuse.student import FuseStudent
from self_audit.candidate_fuse.teacher import build_teacher
from self_audit.candidate_fuse.mnms_data import write_synthetic_mnms_manifests


class _Frame:
    patient_id = "mnms-test-1"
    frame_index = 0
    geometry = {"shape": (8, 16, 16), "spacing_zyx": (1.0, 1.0, 1.0),
                "affine_zyx": np.diag([1.0, 1.0, 1.0, 1.0]).tolist(),
                "context": "full_volume"}
    image = torch.zeros((1, 1, 8, 16, 16))
    target = torch.zeros((8, 16, 16), dtype=torch.long)
    target[:, :8, :8] = 1
    target[:, 8:, :8] = 2
    target[:, :, 8:] = 3


class _Data:
    def __init__(self):
        self.reads = 0

    def labeled_keys(self):
        return [("mnms-test-1", 0)]

    def load(self, key, *, labeled):
        assert labeled
        self.reads += 1
        return _Frame()


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _fixture(tmp_path):
    config = default_transfer_config(synthetic=True)
    test_path = Path(write_synthetic_mnms_manifests(tmp_path / "mnms-fixtures")["test"]["path"])
    test_sha = hashlib.sha256(test_path.read_bytes()).hexdigest()
    config["data"].update({
        "train_manifest": str(tmp_path / "train.json"),
        "train_manifest_sha256": _hash("train"),
        "val_manifest": str(tmp_path / "val.json"),
        "val_manifest_sha256": _hash("val"),
        "test_manifest": str(test_path.resolve()),
        "test_manifest_sha256": test_sha,
    })
    teacher, _ = build_teacher(config, synthetic=True)
    student = FuseStudent(config, teacher_channels=teacher.feature_channels,
                          teacher_phases=teacher.feature_phases,
                          include_fusion=True, include_mimic=True).eval()
    source = config["experiment"]["source_teacher"]
    paths = {mode: tmp_path / "exports" / f"{mode}.pt"
             for mode in ("teacher_zero_shot", "teacher", "student", "fusion")}
    receipts = {
        "teacher_zero_shot": export_model(
            None, teacher, config, paths["teacher_zero_shot"], mode="teacher",
            metadata={"source_teacher": source}),
        "teacher": export_model(None, teacher, config, paths["teacher"], mode="teacher"),
        "student": export_model(student, None, config, paths["student"], mode="student"),
        "fusion": export_model(student, teacher, config, paths["fusion"], mode="fusion"),
    }
    return config, test_path, test_sha, paths, receipts


def test_one_test_ledger_records_all_four_modes_and_patient_metrics(tmp_path, monkeypatch):
    from self_audit.candidate_fuse import data as fuse_data

    _, test_path, test_sha, _, receipts = _fixture(tmp_path)
    data = _Data()
    monkeypatch.setattr(fuse_data, "open_fuse_data", lambda *a, **k: data)
    ledger = evaluate_exports(receipts, test_path, tmp_path / "once", "cpu", "test", test_sha)

    assert ledger["status"] == "COMMITTED"
    assert set(ledger["metrics"]) == {"teacher_zero_shot", "teacher", "student", "fusion"}
    assert all(result["patient_count"] == 1
               and "mnms-test-1" in result["per_patient"]
               for result in ledger["metrics"].values())
    assert all(set(result["per_class"]) == {"RV", "MYO", "LV"}
               for result in ledger["metrics"].values())
    assert set(ledger["artifacts"]) == {"teacher_zero_shot", "teacher", "student", "fusion"}
    assert json.loads((tmp_path / "once" / "TEST_TRANSACTION.json").read_text())["status"] == "COMMITTED"
    with pytest.raises(ValueError, match="already exists"):
        evaluate_exports(receipts, test_path, tmp_path / "once", "cpu", "test", test_sha)


def test_tampered_export_is_rejected_before_target_data_reader(tmp_path, monkeypatch):
    from self_audit.candidate_fuse import data as fuse_data

    _, test_path, test_sha, paths, receipts = _fixture(tmp_path)
    paths["student"].write_bytes(paths["student"].read_bytes() + b"tamper")
    monkeypatch.setattr(
        fuse_data, "open_fuse_data",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("target masks/data reader opened")),
    )
    with pytest.raises(ValueError, match="student artifact SHA-256 mismatch"):
        evaluate_exports(receipts, test_path, tmp_path / "once", "cpu", "test", test_sha)
    assert not (tmp_path / "once" / "TEST_TRANSACTION.json").exists()


def test_export_config_identity_mismatch_is_rejected_before_target_data_reader(tmp_path, monkeypatch):
    from self_audit.candidate_fuse import data as fuse_data

    _, test_path, test_sha, paths, receipts = _fixture(tmp_path)
    path = paths["teacher_zero_shot"]
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["config"]["data"]["test_manifest_sha256"] = _hash("different-target-test")
    torch.save(payload, path)
    artifact_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar_path = path.with_suffix(path.suffix + ".manifest.json")
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["artifact_sha256"] = artifact_sha
    sidecar["artifact_bytes"] = path.stat().st_size
    sidecar_path.write_text(json.dumps(sidecar, sort_keys=True) + "\n")
    receipts["teacher_zero_shot"]["sha256"] = artifact_sha
    monkeypatch.setattr(
        fuse_data, "open_fuse_data",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("target masks/data reader opened")),
    )
    with pytest.raises(ValueError, match="same complete v3 config"):
        evaluate_exports(receipts, test_path, tmp_path / "once", "cpu", "test", test_sha)
    assert not (tmp_path / "once" / "TEST_TRANSACTION.json").exists()


def test_zero_shot_metrics_do_not_enter_deployment_choice():
    comparisons = {
        "teacher_zero_shot": {"fg_dice": 1.0},
        "teacher": {"fg_dice": 0.1},
        "student": {"fg_dice": 0.49},
        "fusion": {"fg_dice": 0.50},
    }
    assert choose_deployment(comparisons)["mode"] == "student"


def test_failed_test_transaction_is_locked_against_rerun(tmp_path, monkeypatch):
    from self_audit.candidate_fuse import data as fuse_data
    from self_audit.candidate_fuse import evaluation as fuse_evaluation

    _, test_path, test_sha, _, receipts = _fixture(tmp_path)
    data = _Data()
    monkeypatch.setattr(fuse_data, "open_fuse_data", lambda *a, **k: data)
    monkeypatch.setattr(fuse_evaluation, "evaluate_modes",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("controlled failure")))
    with pytest.raises(RuntimeError, match="controlled failure"):
        evaluate_exports(receipts, test_path, tmp_path / "once", "cpu", "test", test_sha)
    ledger_path = tmp_path / "once" / "TEST_TRANSACTION.json"
    assert json.loads(ledger_path.read_text())["status"] == "FAILED_LOCKED"
    with pytest.raises(ValueError, match="already exists"):
        evaluate_exports(receipts, test_path, tmp_path / "once", "cpu", "test", test_sha)
    assert data.reads == 0


def test_corrupt_test_archive_fails_preflight_without_ledger_or_mask_read(tmp_path, monkeypatch):
    from self_audit.candidate_fuse import data as fuse_data

    _, test_path, test_sha, _, receipts = _fixture(tmp_path)
    manifest = json.loads(test_path.read_text())
    archive = tmp_path / "corrupt.zip"
    from self_audit.candidate_fuse.mnms_data import _synthetic_zip_source
    original_refs = [(record["image"], [*record["masks"].values()]) for record in manifest["patients"]]
    with __import__("zipfile").ZipFile(archive, "w") as zf:
        for i, (record, mask_refs) in enumerate(zip(manifest["patients"], [x[1] for x in original_refs])):
            zf.writestr(record["image"]["member"], b"bad-image" + bytes([i]))
            for j, source in enumerate(mask_refs):
                zf.writestr(source["member"], b"bad-mask" + bytes([i, j]))
    for record in manifest["patients"]:
        record["image"]["path"] = str(archive)
        record["image"]["container_sha256"] = "0" * 64
        for source in record["masks"].values():
            source["path"] = str(archive)
            source["container_sha256"] = "0" * 64
    with __import__("zipfile").ZipFile(archive) as zf:
        for record in manifest["patients"]:
            for source in [record["image"], *record["masks"].values()]:
                info = zf.getinfo(source["member"])
                source["member_size"] = info.file_size
                source["crc32"] = f"{info.CRC:08x}"
    for record in json.loads(Path(manifest["source_inventory"]["path"]).read_text())["records"]:
        if record["split"] == "test":
            record["source_identity"]["source_sha256"] = "0" * 64
    inventory_path = Path(manifest["source_inventory"]["path"])
    inventory = json.loads(inventory_path.read_text())
    inventory["source_identities_sha256"] = "0" * 64
    inventory_path.write_text(json.dumps(inventory))
    manifest["source_inventory_sha256"] = "0" * 64
    # Keep the locked L8/source identity cross-link consistent with the fixture.
    l8_path = Path(manifest["l8_lock"]["path"])
    l8_lock = json.loads(l8_path.read_text())
    l8_lock["source_identities_sha256"] = "0" * 64
    l8_path.write_text(json.dumps(l8_lock))
    manifest["l8_lock"]["sha256"] = hashlib.sha256(l8_path.read_bytes()).hexdigest()
    manifest["source_inventory"]["sha256"] = hashlib.sha256(inventory_path.read_bytes()).hexdigest()
    geometry_path = Path(manifest["geometry_receipt"]["path"])
    geometry = json.loads(geometry_path.read_text())
    geometry["source_identities_sha256"] = "0" * 64
    geometry_path.write_text(json.dumps(geometry))
    manifest["geometry_receipt"]["sha256"] = hashlib.sha256(geometry_path.read_bytes()).hexdigest()
    test_path.write_text(json.dumps(manifest))
    test_sha = hashlib.sha256(test_path.read_bytes()).hexdigest()
    # The embedded export config pins the prior manifest bytes, so re-pin its
    # test identity consistently while keeping the deliberately false archive SHA.
    from self_audit.candidate_fuse.config import config_sha256
    for receipt in receipts.values():
        payload = torch.load(receipt["path"], map_location="cpu", weights_only=True)
        payload["config"]["data"]["test_manifest_sha256"] = test_sha
        config_hash = config_sha256(payload["config"])
        torch.save(payload, receipt["path"])
        receipt["sha256"] = hashlib.sha256(Path(receipt["path"]).read_bytes()).hexdigest()
        sidecar_path = Path(receipt["path"]).with_suffix(Path(receipt["path"]).suffix + ".manifest.json")
        sidecar = json.loads(sidecar_path.read_text())
        sidecar["artifact_sha256"] = receipt["sha256"]
        sidecar["artifact_bytes"] = Path(receipt["path"]).stat().st_size
        sidecar["metadata"]["config_sha256"] = config_hash
        sidecar["metadata"]["data_manifest_sha256"]["test_manifest_sha256"] = test_sha
        sidecar_path.write_text(json.dumps(sidecar, sort_keys=True) + "\n")
    data_opens = []
    monkeypatch.setattr(fuse_data, "open_fuse_data", lambda *a, **k: data_opens.append(True))
    with pytest.raises(ValueError):
        evaluate_exports(receipts, test_path, tmp_path / "once", "cpu", "test", test_sha)
    assert data_opens == []
    assert not (tmp_path / "once" / "TEST_TRANSACTION.json").exists()
