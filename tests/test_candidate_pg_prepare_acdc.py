from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pg_prepare_acdc.py"
SPEC = importlib.util.spec_from_file_location("pg_prepare_acdc", SCRIPT)
assert SPEC and SPEC.loader
pg = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pg)
VAL_IDS = ["patient004", "patient005", "patient012", "patient014", "patient015", "patient018", "patient028", "patient029", "patient030", "patient032", "patient036", "patient065", "patient070", "patient082", "patient087", "patient089", "patient095", "patient098"]
L8_IDS = ["patient007", "patient010", "patient011", "patient024", "patient026", "patient041", "patient077", "patient099"]

def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()

def _write_nii(path: Path, arr: np.ndarray) -> None:
    image = nib.Nifti1Image(arr, np.diag([1.2, 0.8, 2.5, 1.0]))
    image.header.set_xyzt_units("mm")
    nib.save(image, path)

def _fixture(tmp_path: Path):
    raw = tmp_path / "raw"
    (raw / "training").mkdir(parents=True)
    (raw / "testing").mkdir()
    training_ids = [f"patient{i:03}" for i in range(1, 101)]
    testing_ids = [f"patient{i:03}" for i in range(101, 151)]
    labeled = set(L8_IDS)
    val = set(VAL_IDS)
    all_mask_hashes = {}
    for ids, folder_name in ((training_ids, "training"), (testing_ids, "testing")):
        for pid in ids:
            folder = raw / folder_name / pid
            folder.mkdir()
            _write_nii(folder / f"{pid}_4d.nii", np.arange(6 * 5 * 3 * 4, dtype=np.float32).reshape(6, 5, 3, 4))
            (folder / "Info.cfg").write_text("ED: 1\nES: 3\n", encoding="utf-8")
            if (folder_name == "training" and (pid in labeled or pid in val)) or folder_name == "testing":
                for frame in (0, 2):
                    mask_path = folder / f"{pid}_frame{frame + 1:02d}_gt.nii"
                    mask = np.zeros((6, 5, 3), dtype=np.uint8)
                    mask[1:3, 1:4, 1:2] = 2
                    _write_nii(mask_path, mask)
                    all_mask_hashes[mask_path.name] = _sha(mask_path)
    val_receipt = tmp_path / "val.json"
    val_receipt.write_text(json.dumps({"splits": {"VAL18": {"models": {"fusion": {pid: {} for pid in VAL_IDS}}}}}))
    test_receipt = tmp_path / "test.json"
    test_ids = [f"patient{i:03}" for i in range(101, 151)]
    test_receipt.write_text(json.dumps({"metrics": {"fusion": {"per_patient": {pid: {} for pid in test_ids}}}}))
    l8_receipt = tmp_path / "l8.json"
    l8_receipt.write_text(json.dumps({"metrics": {"L8_TRAIN_UNAUGMENTED": {"fusion": {"per_patient": {pid: {} for pid in L8_IDS}}}}}))
    exclusion = tmp_path / "exclusion.md"
    exclusion.write_text("Reviewed duplicate-image exclusions: patient055 and patient076.\n")
    mask_inventory = tmp_path / "mask_inventory.json"
    mask_inventory.write_text(json.dumps({"original_mask_sha256": all_mask_hashes}))
    source_rows, verified_rows = [], []
    for path in sorted(p for p in raw.rglob("*") if p.is_file() and p.suffix == ".nii"):
        rel = path.relative_to(raw).as_posix()
        digest = _sha(path)
        source_rows.append({"Path": rel, "Size": path.stat().st_size, "Hashes": {"md5": hashlib.md5(path.read_bytes()).hexdigest()}})
        verified_rows.append({"relative_path": rel, "bytes": path.stat().st_size, "sha256": digest})
    raw_files = tmp_path / "ACDC_FILES.json"
    source_files = tmp_path / "ACDC_SOURCE.json"
    raw_files.write_text(json.dumps(verified_rows))
    source_files.write_text(json.dumps(source_rows))
    download = tmp_path / "ACDC_DOWNLOAD.json"
    download.write_text(json.dumps({"schema": "self_audit.dataset_download.v1", "dataset": "ACDC", "status": "DOWNLOADED_VERIFIED", "files_receipt": str(raw_files), "source_inventory_sha256": _sha(source_files)}))
    return raw, val_receipt, test_receipt, l8_receipt, mask_inventory, exclusion, raw_files, source_files, download, training_ids, test_ids

def test_committed_locked_patient_identity_and_exclusions_are_exact():
    assert pg.L8_IDS == L8_IDS
    assert pg.EXCLUDED_DUPLICATE_IDS == ["patient055", "patient076"]

def test_raw_only_materialization_opens_through_fusedata_and_preserves_roles(tmp_path, monkeypatch):
    (raw, val, test, l8, mask_inventory, exclusion, raw_files, source_files, download, training_ids, test_ids) = _fixture(tmp_path)
    calls = []
    original_mask_path = pg._mask_path
    def recording_mask_path(folder, pid, frame):
        calls.append(pid)
        return original_mask_path(folder, pid, frame)
    monkeypatch.setattr(pg, "_mask_path", recording_mask_path)
    historical_lock = tmp_path / "historical-k8.json"
    source_lock_obj = {"schema": "self_audit.data.v2", "role": "training", "synthetic": False,
                       "dataset": "acdc", "label_seed": 42, "label_budget": 8,
                       "labeled": [{"id": pid, "image": f"{pid}.npy", "image_sha256": "a" * 64,
                                    "masks": {"0": {"path": f"{pid}-0.npy", "sha256": "b" * 64},
                                              "2": {"path": f"{pid}-2.npy", "sha256": "c" * 64}}} for pid in L8_IDS]}
    historical_lock.write_text(json.dumps(source_lock_obj))
    historical_lock_bytes = historical_lock.read_bytes()
    out = tmp_path / "prepared"
    receipt = pg.prepare(None, None, None, raw, out, val_receipt=val, test_receipt=test,
                         l8_receipt=l8, mask_inventory=mask_inventory, exclusion_receipt=exclusion,
                         raw_inventory=raw_files, source_inventory=source_files, locked_k8=historical_lock,
                         download_receipt=download, expected_k8_sha=_sha(historical_lock))
    assert receipt["schema"] == "self_audit.candidate_pg.materialization.v1"
    assert receipt["status"] == "READY"
    assert set(receipt["source_manifest_hashes"]) == {"train", "val", "test"}
    assert {a["role"] for a in receipt["artifacts"]} == {"train", "val", "test", "source_lock"}
    assert receipt["counts"] == {"train": 80, "val": 18, "test": 50, "labeled_train_patients": 8, "labeled_train_masks": 16, "train_non_l8_mask_reads": 0}
    train_manifest = Path(receipt["manifests"]["train"]["path"])
    val_manifest = Path(receipt["manifests"]["val"]["path"])
    test_manifest = Path(receipt["manifests"]["test"]["path"])
    train_obj = json.loads(train_manifest.read_text())
    val_obj = json.loads(val_manifest.read_text())
    test_obj = json.loads(test_manifest.read_text())
    train_ids = [f"patient{i:03}" for i in range(1, 101) if f"patient{i:03}" not in set(VAL_IDS) | {"patient055", "patient076"}]
    assert train_obj["split"]["train"] == train_ids
    assert [r["id"] for r in train_obj["labeled"]] == L8_IDS
    assert train_obj["relocation_lineage"]["historical_source_k8_sha256"] == _sha(historical_lock)
    assert [r["id"] for r in train_obj["unlabeled"]] == train_ids
    assert val_obj["schema"] == test_obj["schema"] == "self_audit.candidate_fuse.data.v1"
    assert val_obj["role"] == "development" and test_obj["role"] == "test"
    assert [r["id"] for r in val_obj["patients"]] == VAL_IDS
    assert [r["id"] for r in test_obj["patients"]] == test_ids
    assert set(calls) == set(L8_IDS) | set(VAL_IDS) | set(test_ids)
    assert all(obj["synthetic"] is False for obj in (train_obj, val_obj, test_obj))
    # Convert only this temporary test fixture to explicit synthetic identity
    # before passing it through the real loader, avoiding a false real-data claim.
    derived_path = Path(receipt["source_lock"]["path"])
    derived_sha_from_receipt = _sha(derived_path)
    for obj in (train_obj, val_obj, test_obj):
        assert obj["relocation_lineage"]["derived_k8_lock_sha256"] == derived_sha_from_receipt
    assert train_obj["relocation_lineage"]["source_images"][train_ids[0]]["source_sha256"]
    assert receipt["source_lock"]["sha256"] == derived_sha_from_receipt
    derived_obj = json.loads(derived_path.read_text())
    derived_obj["synthetic"] = True
    derived_path.write_text(json.dumps(derived_obj, indent=2, sort_keys=True) + "\n")
    train_obj["synthetic"] = val_obj["synthetic"] = test_obj["synthetic"] = True
    train_obj["labeled_contract"]["source_manifest_sha256"] = _sha(derived_path)
    train_obj["relocation_lineage"]["derived_k8_lock_sha256"] = _sha(derived_path)
    train_manifest.write_text(json.dumps(train_obj, indent=2, sort_keys=True) + "\n")
    val_manifest.write_text(json.dumps(val_obj, indent=2, sort_keys=True) + "\n")
    test_manifest.write_text(json.dumps(test_obj, indent=2, sort_keys=True) + "\n")
    config = {"data": {"split_lock": "configs/candidate_fuse/semi_k8.json", "source_k8_sha256": pg.K8_SOURCE_SHA256,
                       "u_partner_policy": "different_patient", "u_sampling": "patient_balanced", "native_geometry": True,
                       "manifests": {role: {"path": str(path), "sha256": _sha(path)} for role, path in (("train", train_manifest), ("val", val_manifest), ("test", test_manifest))}},
              "runtime": {"synthetic": True}}
    from self_audit.candidate_pg import data as pg_data
    loaded = pg_data.open_pg_data(config)
    assert loaded.train.patients == tuple(train_ids)
    assert loaded.val.patients == tuple(VAL_IDS)
    assert loaded.test.patients == tuple(test_ids)
    assert len(loaded.train.labeled_keys()) == 16
    assert not (set(loaded.train.labeled_keys()) & set(loaded.train.unlabeled_keys()))
    assert receipt["historical_source_lock"]["sha256"] == _sha(historical_lock)
    assert receipt["source_lock"]["sha256"] == derived_sha_from_receipt
    assert Path(receipt["source_lock"]["path"]) != historical_lock
    assert historical_lock.read_bytes() == historical_lock_bytes

def test_fractional_l8_mask_fails_before_output(tmp_path, monkeypatch):
    raw, val, test, l8, mask_inventory, exclusion, raw_files, source_files, download, *_ = _fixture(tmp_path)
    pid = L8_IDS[0]
    mask_path = raw / "training" / pid / f"{pid}_frame01_gt.nii"
    array = np.asarray(nib.load(str(mask_path)).dataobj).astype(np.float32)
    array.flat[0] = 1.5
    _write_nii(mask_path, array)
    out = tmp_path / "prepared"
    # Refresh only fixture provenance to pin the intentionally fractional bytes.
    rows = json.loads(raw_files.read_text())
    for row in rows:
        if row["relative_path"] == f"training/{pid}/{pid}_frame01_gt.nii":
            row["sha256"] = _sha(mask_path); row["bytes"] = mask_path.stat().st_size
    raw_files.write_text(json.dumps(rows))
    rows = json.loads(source_files.read_text())
    for row in rows:
        if row["Path"] == f"training/{pid}/{pid}_frame01_gt.nii":
            row["Size"] = mask_path.stat().st_size
            row["Hashes"]["md5"] = hashlib.md5(mask_path.read_bytes()).hexdigest()
    source_files.write_text(json.dumps(rows))
    download_obj = json.loads(download.read_text())
    download_obj["source_inventory_sha256"] = _sha(source_files)
    download.write_text(json.dumps(download_obj))
    with pytest.raises(ValueError, match="fractional or nonfinite"):
        pg.prepare(None, None, None, raw, out, val_receipt=val, test_receipt=test, l8_receipt=l8,
                   mask_inventory=mask_inventory, exclusion_receipt=exclusion, raw_inventory=raw_files,
                   source_inventory=source_files, download_receipt=download)
    assert not out.exists()

def test_exact_suffix_rejects_interrupted_file(tmp_path):
    folder = tmp_path / "patient001"; folder.mkdir()
    (folder / "patient001_4d.nii.interrupted-123").write_bytes(b"x")
    with pytest.raises(ValueError, match="exactly one raw CINE"):
        pg._cine_file(folder, "patient001")
    (folder / "patient001_frame01_gt.nii.interrupted-123").write_bytes(b"x")
    with pytest.raises(ValueError, match="one ED/ES mask"):
        pg._mask_path(folder, "patient001", 0)
