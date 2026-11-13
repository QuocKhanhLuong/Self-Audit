from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from self_audit.data import label_subset
from scripts.fuse_lock_mnms_l8 import lock_l8


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(tmp_path: Path, *, invalid_label: bool = False):
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    rows = []
    ids = [f"patient{i:03d}" for i in range(345)]
    invalid_pid = label_subset([f"patient{i:03d}" for i in range(10)], 8, 42)[0]
    for index, pid in enumerate(ids):
        if index < 175:
            split = "train"
            labeled = index < 10
            partition = "Training/Labeled" if labeled else "Training/Unlabeled"
        elif index < 209:
            split, partition, labeled = "val", "Validation", False
        else:
            split, partition, labeled = "test", "Testing", False
        image_rel = (Path(partition) / pid / f"{pid}_sa.nii.gz" if split == "train" else
                     Path("Validation") / f"{pid}_sa.nii.gz" if split == "val" else
                     Path("Testing") / f"{pid}_sa.nii.gz")
        image_path = raw_root / image_rel
        image_path.parent.mkdir(parents=True, exist_ok=True)
        affine = np.diag([1.2, 1.3, 8.0, 1.0])
        image = nib.Nifti1Image(np.zeros((4, 5, 6, 3), dtype=np.float32), affine)
        image.header.set_xyzt_units("unknown", "sec")
        nib.save(image, image_path)
        mask_ed = mask_es = ""
        ed_index, es_index = "0", "1"
        if split == "train" and not labeled:
            ed_index = es_index = ""
        if labeled:
            mask_rel = Path("Training") / "Labeled" / pid / f"{pid}_sa_gt.nii.gz"
            mask_path = raw_root / mask_rel
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            values = np.zeros((4, 5, 6, 3), dtype=np.uint8)
            for frame in (0, 1):
                values[0, 0, 0, frame] = 1  # source LV -> canonical LV=3
                values[0, 0, 1, frame] = 2  # source MYO -> canonical MYO=2
                values[0, 0, 2, frame] = 3  # source RV -> canonical RV=1
            if invalid_label and pid == invalid_pid:
                values[0, 0, 3, 0] = 4
            mask = nib.Nifti1Image(values, affine)
            mask.header.set_xyzt_units("unknown", "sec")
            nib.save(mask, mask_path)
            mask_ed = mask_es = mask_rel.as_posix()
        rows.append({"patient_id": pid, "split": split, "image": image_rel.as_posix(),
                     "ed_frame": ed_index, "es_frame": es_index, "source_partition": partition,
                     "mask_ed": mask_ed, "mask_es": mask_es})

    index_path = tmp_path / "index.csv"
    index_rows = rows[:209]
    with index_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(index_rows)
    index_sha = _sha(index_path)

    geometry_rows = []
    snapshot_rows = []
    identities = []
    header_hex = "00" * 348
    header_sha = hashlib.sha256(bytes.fromhex(header_hex)).hexdigest()
    for row in rows:
        pid = row["patient_id"]
        shape = [6, 5, 4]
        affine_zyx = np.diag([8.0, 1.3, 1.2, 1.0]).tolist()
        geometry_rows.append({"patient_id": pid, "split": row["split"], "status": "GEOMETRY_READY",
                              "geometry": {"shape": shape, "spacing_zyx": [8.0, 1.3, 1.2],
                                           "affine_zyx": affine_zyx, "context": "full_volume"}})
        identity = {"patient_id": pid, "split": row["split"], "path": row["image"],
                    "source_sha256": _sha(raw_root / row["image"]), "header_sha256": header_sha}
        identities.append(identity)
        snapshot_rows.append({"patient_id": pid, "split": row["split"], "path": row["image"],
                              "shape": [4, 5, 6, 3], "header_hex": header_hex,
                              "header_sha256": header_sha, "source_identity": identity,
                              "metadata": ({"ED": None, "ES": None, "phase_status": "UNANNOTATED"}
                                           if row["source_partition"] == "Training/Unlabeled"
                                           else {"ED": 0, "ES": 1, "phase_status": "ANNOTATED"})})
    id_payload = json.dumps(identities, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    source_ids_sha = hashlib.sha256(id_payload).hexdigest()
    snapshot = {"schema": "self_audit.fuse_mnms_open345_snapshot.v1", "dataset_release": "original_2020_open345",
                "artifact_sha256": {"index": index_sha, "metadata": "c" * 64, "test_archive": "d" * 64},
                "source_identities": identities, "source_identities_sha256": source_ids_sha,
                "records": snapshot_rows}
    report = tmp_path / "geometry-report"
    report.mkdir()
    snapshot_path = report / "source_snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, sort_keys=True), encoding="utf-8")
    geometry_manifest = {"schema": "self_audit.fuse_mnms_geometry_receipt.v1",
                         "dataset_release": "original_2020_open345", "source_identities_sha256": source_ids_sha,
                         "records": geometry_rows}
    manifest_path = report / "geometry_manifest.json"
    manifest_path.write_text(json.dumps(geometry_manifest, sort_keys=True), encoding="utf-8")
    patients_path = report / "per_patient.json"
    patients_path.write_text(json.dumps(geometry_rows, sort_keys=True), encoding="utf-8")
    summary = {"schema": "self_audit.fuse_mnms_open345_adjudication.v1", "status": "GEOMETRY_READY",
               "dataset_release": "original_2020_open345", "artifact_sha256": {"index": index_sha},
               "snapshot_sha256": _sha(snapshot_path), "source_identities_sha256": source_ids_sha,
               "geometry_manifest_sha256": _sha(manifest_path), "per_patient_sha256": _sha(patients_path)}
    (report / "SUMMARY.json").write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")
    return index_path, raw_root, report, rows


def test_l8_selection_is_exact_repeatable_source_locked_and_reads_only_selected_masks(tmp_path, monkeypatch):
    index, raw_root, report, rows = _write_fixture(tmp_path)
    original_load = nib.load
    opened = []

    def spy_load(filename, *args, **kwargs):
        candidate = Path(filename).resolve()
        if candidate.name.endswith("_sa_gt.nii.gz"):
            assert (tmp_path / "l8-output" / "L8_LOCK.json").is_file()
        opened.append(candidate)
        return original_load(filename, *args, **kwargs)

    monkeypatch.setattr(nib, "load", spy_load)
    out = tmp_path / "l8-output"
    result = lock_l8(index, raw_root, report, out)
    eligible = [row["patient_id"] for row in rows[:10]]
    expected = label_subset(eligible, 8, 42)
    assert result["lock"]["selected_ids"] == expected
    assert result["lock"]["selected_ids"] == sorted(result["lock"]["selected_ids"])
    assert result["lock"]["selected_ids_sha256"] == hashlib.sha256(
        json.dumps(expected, separators=(",", ":")).encode()
    ).hexdigest()
    assert result["lock"]["selector_sha256_by_patient"] == {
        pid: hashlib.sha256(f"42:{pid}".encode()).hexdigest() for pid in expected
    }
    assert result["lock"]["seed"] == 42 and result["lock"]["k"] == 8
    assert result["lock"]["patient_ids_by_split"]["test"] == [f"patient{i:03d}" for i in range(209, 345)]
    opened_masks = {path for path in opened if path.name.endswith("_sa_gt.nii.gz")}
    assert opened_masks == {raw_root / "Training" / "Labeled" / pid / f"{pid}_sa_gt.nii.gz" for pid in expected}
    assert result["receipt"]["mask_reads_outside_l8"] == 0
    assert result["receipt"]["prepared_manifests"] == "NOT_RUN"
    assert (out / "L8_LOCK.json").is_file()
    assert (out / "MASK_MAPPING_RECEIPT.json").is_file()
    second = lock_l8(index, raw_root, report, tmp_path / "l8-output-repeat")
    assert second["lock"]["selected_ids"] == result["lock"]["selected_ids"]
    assert second["lock"]["selected_ids_sha256"] == result["lock"]["selected_ids_sha256"]


def test_mapping_swaps_rv_lv_and_preserves_myo_background(tmp_path):
    index, raw_root, report, _rows = _write_fixture(tmp_path)
    result = lock_l8(index, raw_root, report, tmp_path / "out")
    mask = result["receipt"]["records"][0]["masks"][0]
    assert mask["source_class_counts"] == {"0": 117, "1": 1, "2": 1, "3": 1}
    assert mask["canonical_class_counts"] == {"0": 117, "1": 1, "2": 1, "3": 1}
    assert mask["spatial_shape_matches_image"] and mask["raw_affine_matches_image"]


def test_invalid_label_rejected_after_lock_without_reading_any_non_l8_mask(tmp_path):
    index, raw_root, report, _rows = _write_fixture(tmp_path, invalid_label=True)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="labels outside"):
        lock_l8(index, raw_root, report, out)
    lock = json.loads((out / "L8_LOCK.json").read_text())
    assert len(lock["selected_ids"]) == 8
    assert not (out / "MASK_MAPPING_RECEIPT.json").exists()


def test_wrong_patient_mask_path_is_rejected_before_hash_or_nibabel_load(tmp_path, monkeypatch):
    index, raw_root, report, _rows = _write_fixture(tmp_path)
    chosen = label_subset([f"patient{i:03d}" for i in range(10)], 8, 42)[0]
    with index.open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
        fields = list(records[0])
    for record in records:
        if record["patient_id"] == chosen:
            record["mask_ed"] = record["mask_es"] = (
                f"Training/Labeled/patient999/patient999_sa_gt.nii.gz"
            )
    with index.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    snapshot_path = report / "source_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text())
    snapshot["artifact_sha256"]["index"] = _sha(index)
    snapshot_path.write_text(json.dumps(snapshot, sort_keys=True), encoding="utf-8")
    summary_path = report / "SUMMARY.json"
    summary = json.loads(summary_path.read_text())
    summary["artifact_sha256"]["index"] = _sha(index)
    summary["snapshot_sha256"] = _sha(snapshot_path)
    summary_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")

    original_load = nib.load
    loaded_masks = []

    def spy_load(filename, *args, **kwargs):
        if Path(filename).name.endswith("_sa_gt.nii.gz"):
            loaded_masks.append(Path(filename))
        return original_load(filename, *args, **kwargs)

    monkeypatch.setattr(nib, "load", spy_load)
    out = tmp_path / "wrong-mask-out"
    with pytest.raises(ValueError, match="mask path must be bound"):
        lock_l8(index, raw_root, report, out)
    assert (out / "L8_LOCK.json").is_file()
    assert loaded_masks == []


def test_in_root_symlink_to_non_l8_mask_is_rejected_before_hash_or_load(tmp_path, monkeypatch):
    index, raw_root, report, _rows = _write_fixture(tmp_path)
    selected_ids = label_subset([f"patient{i:03d}" for i in range(10)], 8, 42)
    selected = selected_ids[0]
    nonselected = next(f"patient{i:03d}" for i in range(10) if f"patient{i:03d}" not in selected_ids)
    selected_mask = raw_root / "Training" / "Labeled" / selected / f"{selected}_sa_gt.nii.gz"
    other_mask = raw_root / "Training" / "Labeled" / nonselected / f"{nonselected}_sa_gt.nii.gz"
    selected_mask.unlink()
    selected_mask.symlink_to(other_mask)

    original_load = nib.load
    loaded_masks = []

    def spy_load(filename, *args, **kwargs):
        if Path(filename).name.endswith("_sa_gt.nii.gz"):
            loaded_masks.append(Path(filename))
        return original_load(filename, *args, **kwargs)

    monkeypatch.setattr(nib, "load", spy_load)
    out = tmp_path / "symlink-mask-out"
    with pytest.raises(ValueError, match="patient-bound source path"):
        lock_l8(index, raw_root, report, out)
    assert (out / "L8_LOCK.json").is_file()
    assert loaded_masks == []


def test_rehashed_geometry_manifest_tamper_disagrees_with_per_patient_before_masks(tmp_path, monkeypatch):
    index, raw_root, report, _rows = _write_fixture(tmp_path)
    manifest_path = report / "geometry_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["records"][0]["geometry"]["spacing_zyx"][0] += 0.0001
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    summary_path = report / "SUMMARY.json"
    summary = json.loads(summary_path.read_text())
    summary["geometry_manifest_sha256"] = _sha(manifest_path)
    summary_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")

    original_load = nib.load
    loaded_masks = []

    def spy_load(filename, *args, **kwargs):
        if Path(filename).name.endswith("_sa_gt.nii.gz"):
            loaded_masks.append(Path(filename))
        return original_load(filename, *args, **kwargs)

    monkeypatch.setattr(nib, "load", spy_load)
    with pytest.raises(ValueError, match="per-patient canonical geometry differs"):
        lock_l8(index, raw_root, report, tmp_path / "geometry-tamper-out")
    assert loaded_masks == []


def test_output_never_overwrites_and_index_duplicate_is_rejected(tmp_path):
    index, raw_root, report, _rows = _write_fixture(tmp_path)
    out = tmp_path / "out"
    lock_l8(index, raw_root, report, out)
    with pytest.raises(FileExistsError):
        lock_l8(index, raw_root, report, out)
    text = index.read_text()
    lines = text.splitlines()
    index.write_text("\n".join(lines + [lines[1]]) + "\n")
    with pytest.raises(ValueError, match="duplicate patient"):
        lock_l8(index, raw_root, report, tmp_path / "duplicate-out")
