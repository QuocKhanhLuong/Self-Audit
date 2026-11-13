from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import zipfile

import nibabel as nib
import numpy as np
import pytest

from scripts import fuse_adjudicate_mnms as release


def _nifti_bytes(shape=(4, 5, 6, 3)):
    header = nib.Nifti1Header()
    header.set_data_shape(shape)
    header.set_data_dtype(np.float32)
    header.set_xyzt_units("unknown", "sec")
    header.set_sform(np.diag([1.0, 1.25, 2.0, 1.0]), code=1)
    header["pixdim"][1:4] = [1.0, 1.25, 2.0]
    return bytes(header.binaryblock) + b"\0\0\0\0" + b"voxel payload must not be touched"


def _gzip_bytes(payload):
    return gzip.compress(payload)


def _write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)


def _release_fixture(tmp_path, *, duplicate_zip=False):
    raw = tmp_path / "raw"
    raw.mkdir()
    index_rows, metadata_rows = [], []
    names = []
    for i in range(209):
        pid = f"P{i:03d}"
        split = "train" if i < 175 else "val"
        if split == "train" and i < 150:
            partition = "Training/Labeled"
            rel = f"Training/Labeled/{pid}/{pid}_sa.nii.gz"
            ed, es = "0", "1"
            source_ed, source_es = "0", "1"
        elif split == "train":
            partition = "Training/Unlabeled"
            rel = f"Training/Unlabeled/{pid}/{pid}_sa.nii.gz"
            ed, es = "", ""
            source_ed, source_es = "0", "0"
        elif split == "val":
            partition = "Validation"
            rel = f"Validation/{pid}/{pid}_sa.nii.gz"
            ed, es = "0", "1"
            source_ed, source_es = "0", "1"
        index_rows.append({"patient_id": pid, "split": split, "image": rel,
                           "ed_frame": ed, "es_frame": es, "source_partition": partition})
        metadata_rows.append({"External code": pid, "ED": source_ed, "ES": source_es, "VendorName": "vname",
                              "Vendor": "vendor", "Centre": "centre"})
        if split != "test":
            path = raw / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(_gzip_bytes(_nifti_bytes()))
    for i in range(209, 345):
        pid = f"P{i:03d}"
        names.append(f"Testing/{pid}/{pid}_sa.nii.gz")
        metadata_rows.append({"External code": pid, "ED": "0", "ES": "1", "VendorName": "vname",
                              "Vendor": "vendor", "Centre": "centre"})
    index = tmp_path / "index.csv"
    metadata = tmp_path / "metadata.csv"
    _write_csv(index, ["patient_id", "split", "image", "ed_frame", "es_frame", "source_partition"], index_rows)
    _write_csv(metadata, ["External code", "ED", "ES", "VendorName", "Vendor", "Centre"], metadata_rows)
    archive_path = tmp_path / "test.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.writestr(name, _gzip_bytes(_nifti_bytes()))
            archive.writestr(name.replace("_sa.nii.gz", "_sa_gt.nii.gz"), b"sealed target annotation member")
    if duplicate_zip:
        with pytest.warns(UserWarning, match="Duplicate name"):
            with zipfile.ZipFile(archive_path, "a", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(names[0], _gzip_bytes(_nifti_bytes()))
    return index, raw, archive_path, metadata


def test_snapshot_reads_test_zip_header_without_decoding_voxel_data(tmp_path, monkeypatch):
    index, raw, archive, metadata = _release_fixture(tmp_path)
    original_open = zipfile.ZipFile.open

    class GuardedMember:
        def __init__(self, wrapped): self.wrapped = wrapped
        def read(self, size=-1):
            # gzip may request small compressed-byte chunks from this wrapper.
            return self.wrapped.read(size)
        def __enter__(self): self.wrapped.__enter__(); return self
        def __exit__(self, *args): return self.wrapped.__exit__(*args)

    def guarded_open(self, name, *args, **kwargs):
        member_name = name.filename if isinstance(name, zipfile.ZipInfo) else str(name)
        assert not member_name.lower().endswith("_sa_gt.nii.gz"), f"opened segmentation member: {member_name}"
        return GuardedMember(original_open(self, name, *args, **kwargs))

    monkeypatch.setattr(zipfile.ZipFile, "open", guarded_open)
    original_gzip_file = gzip.GzipFile

    class GuardedGzip:
        def __init__(self, *args, **kwargs): self.wrapped = original_gzip_file(*args, **kwargs)
        def read(self, size=-1):
            assert size == 348, f"attempted to decode NIfTI payload with request size={size}"
            return self.wrapped.read(size)
        def __enter__(self): return self
        def __exit__(self, *args): self.wrapped.close()

    monkeypatch.setattr(gzip, "GzipFile", GuardedGzip)
    out = tmp_path / "snapshot.json"
    result = release.snapshot(index, raw, archive, metadata, out)
    assert result["split_counts"] == {"train": 175, "val": 34, "test": 136}
    assert len(result["source_identities"]) == 345
    unlabeled = [r for r in result["records"] if r["metadata"]["phase_status"] == "UNANNOTATED"]
    assert len(unlabeled) == 25
    assert all(r["metadata"]["source_ED"] == r["metadata"]["source_ES"] == "0" for r in unlabeled)
    assert all(r["metadata"]["ED"] is None and r["metadata"]["ES"] is None for r in unlabeled)
    test_row = next(r for r in result["records"] if r["split"] == "test")
    assert test_row["source_identity"]["member"] == test_row["path"]
    assert len(bytes.fromhex(test_row["header_hex"])) == 348
    assert out.is_file()
    with pytest.raises(FileExistsError):
        release.snapshot(index, raw, archive, metadata, out)


def test_snapshot_rejects_duplicate_zip_member_names(tmp_path):
    index, raw, archive, metadata = _release_fixture(tmp_path, duplicate_zip=True)
    with pytest.raises(ValueError, match="duplicate member"):
        release.snapshot(index, raw, archive, metadata, tmp_path / "snapshot.json")


def test_snapshot_rejects_traversal_and_duplicate_index_ids(tmp_path):
    index, raw, archive, metadata = _release_fixture(tmp_path)
    with index.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream)); fields = list(rows[0])
    rows[0]["image"] = "../escape.nii.gz"
    _write_csv(index, fields, rows)
    with pytest.raises(ValueError, match="unsafe relative path"):
        release.snapshot(index, raw, archive, metadata, tmp_path / "bad.json")
    rows[0]["image"] = "Training/Labeled/P000/P000_sa.nii.gz"
    rows[1]["patient_id"] = rows[0]["patient_id"]
    _write_csv(index, fields, rows)
    with pytest.raises(ValueError, match="duplicate index patient"):
        release.snapshot(index, raw, archive, metadata, tmp_path / "bad2.json")


def _policy_for(snapshot, evidence):
    if not evidence.exists():
        evidence.write_bytes(b"locked evidence")
    return {"schema": "self_audit.mnms_geometry_policy.v1", "name": "mnms_open345_mm_polar_v1",
            "dataset_release": "original_2020_open345", "units_evidence_url": release.UNITS_EVIDENCE_URL,
            "units_evidence_sha256": release.UNITS_EVIDENCE_SHA256,
            "source_identities_sha256": snapshot["source_identities_sha256"], "reason": "locked test policy"}


def test_adjudication_canonicalizes_all_records_and_writes_receipts(tmp_path):
    index, raw, archive, metadata = _release_fixture(tmp_path)
    snapshot_path = tmp_path / "snapshot.json"
    snap = release.snapshot(index, raw, archive, metadata, snapshot_path)
    evidence = tmp_path / "paper.pdf"
    evidence.write_bytes(b"fixture bytes with the locked evidence identity mocked below")
    from self_audit.candidate_fuse import geometry as geometry_module
    real_evidence_sha = release.UNITS_EVIDENCE_SHA256
    real_geometry_sha = geometry_module.UNITS_EVIDENCE_SHA256
    release.UNITS_EVIDENCE_SHA256 = hashlib.sha256(evidence.read_bytes()).hexdigest()
    geometry_module.UNITS_EVIDENCE_SHA256 = release.UNITS_EVIDENCE_SHA256
    policy_path = tmp_path / "policy.json"
    policy = _policy_for(snap, evidence)
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    try:
        summary = release.adjudicate(snapshot_path, policy_path, evidence, tmp_path / "adjudication")
    finally:
        release.UNITS_EVIDENCE_SHA256 = real_evidence_sha
        geometry_module.UNITS_EVIDENCE_SHA256 = real_geometry_sha
    assert summary["status"] == "GEOMETRY_READY"
    assert summary["geometry_ready_counts"] == {"train": 175, "val": 34, "test": 136}
    assert summary["raw_unknown_unit_count"] == 345
    assert summary["target_L8_lock"] == "NOT_RUN"
    receipt = json.loads((tmp_path / "adjudication" / "geometry_manifest.json").read_text())
    per_patient_bytes = (tmp_path / "adjudication" / "per_patient.json").read_bytes()
    geometry_manifest_bytes = (tmp_path / "adjudication" / "geometry_manifest.json").read_bytes()
    summary_json = json.loads((tmp_path / "adjudication" / "SUMMARY.json").read_text())
    assert receipt["schema"] == "self_audit.fuse_mnms_geometry_receipt.v1"
    assert len(receipt["records"]) == 345
    assert summary_json["per_patient_sha256"] == hashlib.sha256(per_patient_bytes).hexdigest()
    assert summary_json["geometry_manifest_sha256"] == hashlib.sha256(geometry_manifest_bytes).hexdigest()
    assert (tmp_path / "adjudication" / "per_patient.csv").is_file()


@pytest.mark.parametrize("tamper", ["identity_digest", "header_hex", "identity_sha"])
def test_adjudication_rejects_snapshot_provenance_tampering(tmp_path, tamper):
    index, raw, archive, metadata = _release_fixture(tmp_path)
    snapshot_path = tmp_path / "snapshot.json"
    snap = release.snapshot(index, raw, archive, metadata, snapshot_path)
    if tamper == "identity_digest":
        snap["source_identities_sha256"] = "0" * 64
    elif tamper == "header_hex":
        snap["records"][0]["header_hex"] = "00" * 348
    else:
        snap["records"][0]["source_identity"]["source_sha256"] = "0" * 64
    snapshot_path.write_text(json.dumps(snap), encoding="utf-8")
    evidence = tmp_path / "paper.pdf"; evidence.write_bytes(b"locked evidence")
    policy_path = tmp_path / "policy.json"
    policy = _policy_for(snap, evidence)
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(ValueError, match="source_identities_sha256|identity record tampered|header integrity"):
        release.adjudicate(snapshot_path, policy_path, evidence, tmp_path / "adjudication")


def test_adjudication_checks_policy_identity_and_units_pdf_bytes(tmp_path):
    index, raw, archive, metadata = _release_fixture(tmp_path)
    snapshot_path = tmp_path / "snapshot.json"
    snap = release.snapshot(index, raw, archive, metadata, snapshot_path)
    evidence = tmp_path / "paper.pdf"; evidence.write_bytes(b"locked evidence")
    policy_path = tmp_path / "policy.json"
    policy = _policy_for(snap, evidence); policy["units_evidence_sha256"] = "0" * 64
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(ValueError, match="units evidence SHA256"):
        release.adjudicate(snapshot_path, policy_path, evidence, tmp_path / "adjudication")


def test_all_train_geometry_failures_make_overall_summary_blocked(tmp_path):
    index, raw, archive, metadata = _release_fixture(tmp_path)
    with index.open(newline="", encoding="utf-8") as stream:
        train_rows = [row for row in csv.DictReader(stream) if row["split"] == "train"]
    for row in train_rows:
        image_path = raw / row["image"]
        payload = gzip.decompress(image_path.read_bytes())
        header = nib.Nifti1Header.from_fileobj(io.BytesIO(payload[:348]))
        affine = np.asarray(header.get_sform(), dtype=float)
        affine[0, 1] = 0.2
        header.set_sform(affine, code=1)
        header["pixdim"][1:4] = [1.0, 1.25, 2.0]
        image_path.write_bytes(_gzip_bytes(bytes(header.binaryblock) + payload[348:]))

    snapshot_path = tmp_path / "snapshot.json"
    snap = release.snapshot(index, raw, archive, metadata, snapshot_path)
    evidence = tmp_path / "paper.pdf"
    evidence.write_bytes(b"fixture bytes with the locked evidence identity mocked below")
    from self_audit.candidate_fuse import geometry as geometry_module
    real_evidence_sha = release.UNITS_EVIDENCE_SHA256
    real_geometry_sha = geometry_module.UNITS_EVIDENCE_SHA256
    fake_sha = hashlib.sha256(evidence.read_bytes()).hexdigest()
    release.UNITS_EVIDENCE_SHA256 = fake_sha
    geometry_module.UNITS_EVIDENCE_SHA256 = fake_sha
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(_policy_for(snap, evidence)), encoding="utf-8")
    try:
        summary = release.adjudicate(snapshot_path, policy_path, evidence, tmp_path / "adjudication")
    finally:
        release.UNITS_EVIDENCE_SHA256 = real_evidence_sha
        geometry_module.UNITS_EVIDENCE_SHA256 = real_geometry_sha
    assert summary["geometry_ready_counts"]["train"] == 0
    assert summary["geometry_ready_counts"]["val"] == 34
    assert summary["geometry_ready_counts"]["test"] == 136
    assert summary["status"] == "BLOCKED"
    assert any(blocker.startswith("TRAIN_geometry_failures=") for blocker in summary["blockers"])
