"""Strict receipt-to-manifest checks for PG's explicit materialization mode."""

import copy
import hashlib
import json

import pytest

import self_audit.candidate_pg.data as pg_data
from self_audit.candidate_pg.config import default_config, validate_config


L8 = ["patient007", "patient010", "patient011", "patient024", "patient026", "patient041", "patient077", "patient099"]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return _sha(path)


def _fixture(tmp_path, monkeypatch):
    historical = {
        "train": hashlib.sha256(b"old-train").hexdigest(),
        "development": hashlib.sha256(b"old-val").hexdigest(),
        "test": hashlib.sha256(b"old-test").hexdigest(),
    }
    monkeypatch.setattr(pg_data, "TRAIN_MANIFEST_SHA256", historical["train"])
    monkeypatch.setattr(pg_data, "VAL_MANIFEST_SHA256", historical["development"])
    monkeypatch.setattr(pg_data, "TEST_MANIFEST_SHA256", historical["test"])
    monkeypatch.setattr(pg_data, "K8_SOURCE_SHA256", hashlib.sha256(b"historical-k8").hexdigest())
    monkeypatch.setattr(pg_data, "SPLIT_LOCK_SHA256", hashlib.sha256(b"split-lock").hexdigest())

    repo = pg_data._repo_root()
    locked_split = pg_data._locked_split_identity()
    split = {"train": locked_split["train"], "val": locked_split["development"], "test": locked_split["test"]}
    manifest_split = {"train": split["train"], "development": split["val"], "test": split["test"]}
    assert [len(split[key]) for key in ("train", "val", "test")] == [80, 18, 50]

    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    raw_image = raw_root / "source-cine.nii"
    raw_image.write_bytes(b"raw-cine-identity")
    raw_mask = raw_root / "source-mask.nii"
    raw_mask.write_bytes(b"raw-mask-identity")
    files_receipt_path = tmp_path / "ACDC_FILES.json"
    files_receipt_sha = _write_json(files_receipt_path, [
        {"relative_path": p.relative_to(raw_root).as_posix(), "sha256": _sha(p), "md5": _md5(p), "bytes": p.stat().st_size}
        for p in (raw_image, raw_mask)
    ])
    source_receipt_path = tmp_path / "ACDC_SOURCE.json"
    source_receipt_sha = _write_json(source_receipt_path, [
        {"Path": p.relative_to(raw_root).as_posix(), "Hashes": {"md5": _md5(p), "sha256": _sha(p)}, "Size": p.stat().st_size}
        for p in (raw_image, raw_mask)
    ])
    download_receipt_path = tmp_path / "ACDC_DOWNLOAD.json"
    download_receipt_sha = _write_json(download_receipt_path, {
        "status": "DOWNLOADED_VERIFIED", "dataset": "ACDC",
        "source_inventory_sha256": source_receipt_sha, "files_receipt": str(files_receipt_path),
    })
    original_mask_inventory_path = tmp_path / "ORIGINAL_MASK_INVENTORY.json"
    original_mask_inventory_sha = _write_json(original_mask_inventory_path, {
        "original_mask_sha256": {raw_mask.name: _sha(raw_mask)},
    })
    monkeypatch.setattr(pg_data, "_ORIGINAL_MASK_INVENTORY_SHA256", original_mask_inventory_sha)
    raw_inventory = {
        "files_receipt": {"path": str(files_receipt_path), "sha256": files_receipt_sha},
        "source_receipt": {"path": str(source_receipt_path), "sha256": source_receipt_sha},
        "download_receipt": {"path": str(download_receipt_path), "sha256": download_receipt_sha},
    }

    generated_mask = tmp_path / "generated-mask.npy"
    generated_mask.write_bytes(b"generated-mask-array")
    generated_mask_sha = _sha(generated_mask)

    lock = {
        "schema": "self_audit.data.v2",
        "role": "training",
        "synthetic": False,
        "dataset": "acdc",
        "label_seed": 42,
        "label_budget": 8,
        "source_lock_sha256": pg_data.K8_SOURCE_SHA256,
        "source_lock_patient_ids": L8,
        "labeled": [
            {"id": pid, "masks": {"1": {"path": str(generated_mask), "sha256": generated_mask_sha}, "3": {"path": str(generated_mask), "sha256": generated_mask_sha}}}
            for pid in L8
        ],
    }
    lock_path = tmp_path / "source-lock.json"
    lock_sha = _write_json(lock_path, lock)

    evidence_refs = {
        "val_receipt": {"path": str(repo / "reports/fuse-semi-20261113/track-A/PATIENT_METRICS.json"), "sha256": pg_data._SPLIT_EVIDENCE["reports/fuse-semi-20261113/track-A/PATIENT_METRICS.json"]},
        "test_receipt": {"path": str(repo / "reports/fuse-semi-20261113/track-A/TEST_ONCE/TEST_TRANSACTION.json"), "sha256": pg_data._SPLIT_EVIDENCE["reports/fuse-semi-20261113/track-A/TEST_ONCE/TEST_TRANSACTION.json"]},
        "l8_receipt": {"path": str(repo / "reports/fuse-acdc-k8-aug-20261114/PHASE0_L8.json"), "sha256": pg_data._SPLIT_EVIDENCE["reports/fuse-acdc-k8-aug-20261114/PHASE0_L8.json"]},
        "exclusions": {"path": str(repo / "reports/automatic_pilot_pro4000/FINAL_REPORT.md"), "sha256": pg_data._SPLIT_EVIDENCE["reports/automatic_pilot_pro4000/FINAL_REPORT.md"]},
        "original_mask_inventory": {"path": str(original_mask_inventory_path), "sha256": original_mask_inventory_sha},
    }
    committed_receipts = {
        "source_k8_sha256": pg_data.K8_SOURCE_SHA256,
        "val_receipt": evidence_refs["val_receipt"],
        "test_receipt": evidence_refs["test_receipt"],
        "l8_receipt": evidence_refs["l8_receipt"],
        "duplicate_image_exclusion_receipt": evidence_refs["exclusions"],
        "original_mask_inventory": evidence_refs["original_mask_inventory"],
    }
    manifests = {}
    geometry_audit = {}
    geometry_policy = "acdc_pixdim_native_v1"
    for role, key in (("train", "train"), ("val", "development"), ("test", "test")):
        records = []
        geometry_by_patient = {}
        source_images = {}
        patient_role_ids = manifest_split[key]
        for pid in patient_role_ids:
            source_images[pid] = {"source_path": str(raw_image), "source_sha256": _sha(raw_image)}
            audit_row = {"effective_spacing_zyx_mm": [1.0, 1.0, 1.0], "effective_affine_zyx": [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]}
            geometry_audit[pid] = audit_row
            geometry_by_patient[pid] = {
                "shape": [2, 4, 4], "spacing_zyx": [1.0, 1.0, 1.0],
                "affine_zyx": [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]],
                "context": "full_volume",
                "provenance": {"pixdim_policy": geometry_policy, "source_geometry_audit": audit_row},
            }
            rec = {"id": pid}
            if key == "train" and pid in L8:
                rec["phases"] = {"ED": 1, "ES": 3}
                rec["masks"] = {
                    "1": {"path": "mask-ed.npy", "sha256": generated_mask_sha, "source_path": str(raw_mask), "source_sha256": _sha(raw_mask)},
                    "3": {"path": "mask-es.npy", "sha256": generated_mask_sha, "source_path": str(raw_mask), "source_sha256": _sha(raw_mask)},
                }
            records.append(rec)
        manifest = {
            "schema": "self_audit.candidate_fuse.data.v2" if key == "train" else "self_audit.candidate_fuse.data.v1",
            "role": key,
            "split": copy.deepcopy(manifest_split),
            "geometry": {"by_patient": geometry_by_patient},
            "relocation_lineage": {
                "source_k8_lock_sha256": pg_data.K8_SOURCE_SHA256,
                "historical_source_k8_sha256": pg_data.K8_SOURCE_SHA256,
                "derived_k8_lock_sha256": lock_sha,
                "source_manifest_sha256": None,
                "raw_root": str(raw_root),
                "committed_receipts": committed_receipts,
                "source_images": source_images,
                "raw_identity_verified": True,
            },
        }
        if key == "train":
            manifest.update({
                "labeled_contract": {"patient_ids": L8, "source_manifest_sha256": lock_sha},
                "labeled": [r for r in records if r["id"] in L8],
                "unlabeled": records,
            })
        else:
            manifest["patients"] = records
        out = tmp_path / f"{role}.json"
        _write_json(out, manifest)
        manifests[role] = {"path": str(out), "sha256": _sha(out)}

    receipt = {
        "schema": pg_data._MATERIALIZATION_SCHEMA,
        "status": "READY",
        "historical_source_k8_sha256": pg_data.K8_SOURCE_SHA256,
        "split_lock_sha256": pg_data.SPLIT_LOCK_SHA256,
        "raw_source": str(raw_root),
        "raw_inventory": raw_inventory,
        "split_ids": split,
        "labeled_ids": L8,
        "source_lock": {"path": str(lock_path), "sha256": lock_sha},
        "derived_k8_lock": {"path": str(lock_path), "sha256": lock_sha},
        "locked_k8_source": {"path": None, "sha256": pg_data.K8_SOURCE_SHA256},
        "historical_source_lock": {"path": None, "sha256": pg_data.K8_SOURCE_SHA256},
        "evidence_pins": evidence_refs,
        "geometry_policy": geometry_policy,
        "geometry_audit": geometry_audit,
        "normalization": "global cine percentiles 0.5/99.5, clip [0,1], transpose XYZT to TZYX float32; raw NIfTI bytes untouched",
        "manifests": manifests,
        "artifacts": [
            {"path": str(lock_path), "sha256": lock_sha},
            {"path": str(files_receipt_path), "sha256": files_receipt_sha},
            {"path": str(source_receipt_path), "sha256": source_receipt_sha},
            {"path": str(download_receipt_path), "sha256": download_receipt_sha},
            *[copy.deepcopy(ref) for ref in evidence_refs.values()],
            *[copy.deepcopy(ref) for ref in manifests.values()],
            *[copy.deepcopy(ref) for ref in manifests.values()],
        ],
    }
    receipt_path = tmp_path / "PREPARATION_RECEIPT.json"
    receipt_sha = _write_json(receipt_path, receipt)
    refs = {role: copy.deepcopy(ref) for role, ref in manifests.items()}
    config = {
        "runtime": {"synthetic": False},
        "data": {
            "manifests": refs,
            "materialization": {"receipt": str(receipt_path), "sha256": receipt_sha},
        },
    }
    return config, receipt, receipt_path, manifests


def _refresh(tmp_path, config, receipt, receipt_path, manifests, role, manifest):
    manifest_path = tmp_path / f"{role}.json"
    _write_json(manifest_path, manifest)
    digest = _sha(manifest_path)
    manifests[role] = {"path": str(manifest_path), "sha256": digest}
    config["data"]["manifests"][role] = copy.deepcopy(manifests[role])
    receipt["manifests"][role] = copy.deepcopy(manifests[role])
    for artifact in receipt["artifacts"]:
        if artifact["path"] == str(manifest_path):
            artifact["sha256"] = digest
    config["data"]["materialization"]["sha256"] = _write_json(receipt_path, receipt)


def test_materialized_train_val_test_require_receipt_split_l8_and_raw_lineage(tmp_path, monkeypatch):
    config, _, _, manifests = _fixture(tmp_path, monkeypatch)
    for role in ("train", "val", "test"):
        path = tmp_path / f"{role}.json"
        pg_data._validate_materialization(config, role if role != "val" else "val", path, manifests[role]["sha256"])


def test_relocation_rejects_changed_split_even_when_manifest_hash_is_repinned(tmp_path, monkeypatch):
    config, receipt, receipt_path, manifests = _fixture(tmp_path, monkeypatch)
    path = tmp_path / "train.json"
    manifest = json.loads(path.read_text())
    manifest["split"]["train"][0] = "patient150"
    _refresh(tmp_path, config, receipt, receipt_path, manifests, "train", manifest)
    with pytest.raises(ValueError, match="split differs"):
        pg_data._validate_materialization(config, "train", path, manifests["train"]["sha256"])


def test_relocation_rejects_changed_l8_even_when_manifest_hash_is_repinned(tmp_path, monkeypatch):
    config, receipt, receipt_path, manifests = _fixture(tmp_path, monkeypatch)
    path = tmp_path / "train.json"
    manifest = json.loads(path.read_text())
    manifest["labeled_contract"]["patient_ids"][0] = "patient008"
    _refresh(tmp_path, config, receipt, receipt_path, manifests, "train", manifest)
    with pytest.raises(ValueError, match="L8 identities"):
        pg_data._validate_materialization(config, "train", path, manifests["train"]["sha256"])


def test_relocation_rejects_unchecked_or_changed_lineage(tmp_path, monkeypatch):
    config, receipt, receipt_path, manifests = _fixture(tmp_path, monkeypatch)
    path = tmp_path / "val.json"
    manifest = json.loads(path.read_text())
    manifest["relocation_lineage"]["source_k8_lock_sha256"] = "0" * 64
    _refresh(tmp_path, config, receipt, receipt_path, manifests, "val", manifest)
    with pytest.raises(ValueError, match="historical K8 source identity"):
        pg_data._validate_materialization(config, "val", path, manifests["val"]["sha256"])


def test_config_materialization_is_explicit_and_real_only():
    config = default_config()
    config["data"]["materialization"] = {"receipt": "/data/PREPARATION_RECEIPT.json", "sha256": "a" * 64}
    assert validate_config(config)["data"]["materialization"] == config["data"]["materialization"]

    config["data"]["materialization"]["allow_unchecked_hashes"] = True
    with pytest.raises(ValueError, match="materialization lineage fields"):
        validate_config(config)

    synthetic = default_config(synthetic=True)
    synthetic["data"]["materialization"] = {"receipt": "/tmp/receipt.json", "sha256": "a" * 64}
    with pytest.raises(ValueError, match="real-data only"):
        validate_config(synthetic)
