from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import nibabel as nib
import numpy as np
import pytest

from self_audit.candidate_fuse.data import FuseData, FuseSampler, open_fuse_data, write_synthetic_manifests
from self_audit.candidate_fuse.mnms_data import write_synthetic_mnms_manifests
from scripts import fuse_prepare_mnms


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _nifti(shape=(5, 6, 3, 4), *, mask=False):
    if mask:
        data = np.zeros(shape, dtype=np.uint8)
        data[1:3, 1:4, 1:, :] = 1
        data[2:4, 2:4, 1:, :] = 2
        return nib.Nifti1Image(data, np.diag([1., 1.5, 2., 1.]))
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    return nib.Nifti1Image(data, np.diag([1., 1.5, 2., 1.]))


def _write_nii(path, *, mask=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(_nifti(mask=mask), path)
    return path


def _receipt_files(root, ids_by_split):
    geometry_path = root / "geometry.json"
    lock_path = root / "lock.json"
    mapping_path = root / "mapping.json"
    rows = []
    for role, ids in ids_by_split.items():
        for pid in ids:
            rows.append({"patient_id": pid, "path": f"{role}/{pid}_sa.nii.gz", "split": role,
                         "status": "GEOMETRY_READY",
                         "geometry": {"shape": [3, 6, 5], "spacing_zyx": [2., 1.5, 1.],
                                      "affine_zyx": np.diag([2., 1.5, 1., 1.]).tolist(),
                                      "context": "full_volume"}})
    geometry_path.write_text(json.dumps({"schema": "self_audit.fuse_mnms_geometry_receipt.v1", "records": rows}))
    l8 = sorted(ids_by_split["train"][:8])
    lock_path.write_text(json.dumps({"schema": "self_audit.fuse_mnms_target_l8_lock.v1", "seed": 42,
                                     "selected_ids": l8, "selected_ids_sha256": hashlib.sha256(
                                         json.dumps(l8, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest(),
                                     "mask_allowlist": l8, "mask_reads_outside_l8": 0}))
    mapping_rows = []
    for pid in l8:
        mapping_rows.append({"patient_id": pid, "masks": [
            {"frame": f, "mask_path": f"train/{pid}_sa_gt.nii.gz", "mask_sha256": "a" * 64}
            for f in (0, 2)]})
    mapping_path.write_text(json.dumps({"schema": "self_audit.fuse_mnms_l8_mask_mapping_receipt.v1",
                                       "dataset_release": "original_2020_open345", "selected_ids": l8,
                                       "l8_lock_sha256": _sha(lock_path), "mask_reads_outside_l8": 0,
                                       "source_to_canonical_class_map": {"0": 0, "1": 3, "2": 2, "3": 1},
                                       "records": mapping_rows}))
    return geometry_path, lock_path, mapping_path


def _zip_source(zip_path, entries):
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member, path in entries.items():
            archive.write(path, member)
    return zip_path


def _compressed_hash(path, member):
    # Mirrors the v3 stored member SHA: bytes between ZIP local header and data descriptor.
    import struct
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(member)
        offset, size = info.header_offset, info.compress_size
    with Path(path).open("rb") as stream:
        stream.seek(offset)
        head = stream.read(30)
        name_len, extra_len = struct.unpack_from("<HH", head, 26)
        stream.seek(name_len + extra_len, 1)
        return hashlib.sha256(stream.read(size)).hexdigest()


def _source(path, member):
    path = Path(path).resolve()
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(member)
    return {"path": str(path), "member": member, "container_sha256": _sha(path),
            "member_sha256": _compressed_hash(path, member), "member_size": info.file_size,
            "crc32": f"{info.CRC:08x}"}


def _manifest(root, *, role="train", alter=None):
    return Path(write_synthetic_mnms_manifests(Path(root) / "fixtures")[role]["path"])


def test_v3_l8_frames_mapping_and_lazy_test_payload(tmp_path, monkeypatch):
    manifest_path = _manifest(tmp_path)
    data = open_fuse_data(manifest_path, _sha(manifest_path), role="train", synthetic=True)
    assert len(data.labeled_keys()) == 16
    assert len(data.unlabeled_keys()) == 8 * 2 + 2 * 4
    assert not any(pid.startswith("mnmsU") for pid, _ in data.labeled_keys())
    assert all("masks" not in record for record in data.manifest["unlabeled"])
    frame = data.load(("mnmsL00", 0), labeled=True)
    assert frame.image.shape == (1, 1, 3, 8, 9)
    assert set(frame.target.unique().tolist()).issubset({0, 1, 2, 3})
    assert 3 in frame.target.unique().tolist()  # source label 1 -> canonical LV
    assert ("mnmsL00", 1) in data.unlabeled_keys()
    reads = []
    read_member = data._read_member
    def spy_member(source):
        if "_gt.nii.gz" in (source.get("member") or ""):
            reads.append(source["member"])
        return read_member(source)
    monkeypatch.setattr(data, "_read_member", spy_member)
    data.load(("mnmsU00", 0), labeled=False)
    assert reads == []

    test_manifest = _manifest(tmp_path / "evaluation", role="test")
    import nibabel
    original_load = nibabel.Nifti1Image.from_bytes
    calls = []
    def spy(payload):
        calls.append(len(payload))
        return original_load(payload)
    monkeypatch.setattr(nibabel.Nifti1Image, "from_bytes", spy)
    test_data = open_fuse_data(test_manifest, role="test", synthetic=True)
    assert calls == []
    assert test_data.patients == ("mnmsT00", "mnmsT01")
    assert tuple(test_data.geometry_for("mnmsT00")["shape"]) == (3, 8, 9)
    assert test_data.frame_count_for("mnmsT00") == 4
    assert calls == []
    test_ref = test_data.manifest["patients"][0]["image"]
    assert test_ref["member_sha256"] is None
    test_data.load(("mnmsT00", 0), labeled=True)
    assert len(calls) == 2


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_v3_l8_accepts_integral_float_mask_labels(tmp_path, dtype):
    manifest = _manifest(tmp_path)
    obj = json.loads(manifest.read_text())
    record = next(r for r in obj["labeled"] if r["id"] == "mnmsL00")
    source = record["masks"]["0"]
    path = Path(source["path"])
    nii = nib.load(path)
    data = np.asarray(nii.dataobj).astype(dtype)
    nib.save(nib.Nifti1Image(data, nii.affine), path)
    raw_bytes = path.read_bytes()
    import zlib
    digest = hashlib.sha256(raw_bytes).hexdigest()
    for mask_source in record["masks"].values():
        mask_source.update(container_sha256=digest, member_sha256=digest, member_size=len(raw_bytes),
                           crc32=f"{zlib.crc32(raw_bytes) & 0xFFFFFFFF:08x}")
    mapping_path = Path(obj["mask_mapping_receipt"]["path"])
    mapping = json.loads(mapping_path.read_text())
    mapped = next(r for r in mapping["records"] if r["patient_id"] == "mnmsL00")
    for mapped_mask in mapped["masks"]:
        mapped_mask["mask_sha256"] = digest
    mapping_path.write_text(json.dumps(mapping))
    obj["mask_mapping_receipt"]["sha256"] = _sha(mapping_path)
    manifest.write_text(json.dumps(obj))

    frame = open_fuse_data(manifest, role="train", synthetic=True).load(("mnmsL00", 0), labeled=True)
    # Verify all source labels retain the canonical map 0->0, 1->3, 2->2, 3->1.
    raw = np.asarray(nib.load(path).dataobj)[..., 0].transpose(2, 1, 0)
    expected = np.take(np.asarray([0, 3, 2, 1]), raw.astype(np.uint8))
    np.testing.assert_array_equal(frame.target[0].numpy(), expected)


@pytest.mark.parametrize("bad_value", [1.5, np.nan, np.inf, -1.0, 4.0])
def test_v3_l8_rejects_invalid_float_mask_labels(tmp_path, bad_value):
    manifest = _manifest(tmp_path)
    obj = json.loads(manifest.read_text())
    source = next(r for r in obj["labeled"] if r["id"] == "mnmsL00")["masks"]["0"]
    path = Path(source["path"])
    nii = nib.load(path)
    data = np.asarray(nii.dataobj).astype(np.float32)
    data[0, 0, 0, 0] = bad_value
    nib.save(nib.Nifti1Image(data, nii.affine), path)
    raw_bytes = path.read_bytes()
    import zlib
    digest = hashlib.sha256(raw_bytes).hexdigest()
    record = next(r for r in obj["labeled"] if r["id"] == "mnmsL00")
    for mask_source in record["masks"].values():
        mask_source.update(container_sha256=digest, member_sha256=digest, member_size=len(raw_bytes),
                           crc32=f"{zlib.crc32(raw_bytes) & 0xFFFFFFFF:08x}")
    mapping_path = Path(obj["mask_mapping_receipt"]["path"])
    mapping = json.loads(mapping_path.read_text())
    mapping_row = next(r for r in mapping["records"] if r["patient_id"] == "mnmsL00")
    for mapped_mask in mapping_row["masks"]:
        mapped_mask["mask_sha256"] = digest
    mapping_path.write_text(json.dumps(mapping))
    obj["mask_mapping_receipt"]["sha256"] = _sha(mapping_path)
    manifest.write_text(json.dumps(obj))
    data_reader = open_fuse_data(manifest, role="train", synthetic=True)
    with pytest.raises(ValueError, match="non-finite|fractional|outside"):
        data_reader.load(("mnmsL00", 0), labeled=True)


def test_synthetic_writer_factory_sampler_and_sixteen_l_keys(tmp_path):
    outputs = write_synthetic_mnms_manifests(tmp_path / "synthetic-mnms", seed=9)
    assert set(outputs) == {"train", "development", "test"}
    assert all(set(value) == {"path", "sha256"} for value in outputs.values())
    data = open_fuse_data(outputs["train"]["path"], outputs["train"]["sha256"], role="train", synthetic=True)
    with pytest.raises(ValueError, match="synthetic/patient identity"):
        open_fuse_data(outputs["train"]["path"], role="train", synthetic=False)
    sampler = FuseSampler(data, seed=42)
    assert sampler.teacher_epoch_size == 16
    assert len(data.labeled_keys()) == 16
    assert sampler.joint_epoch_size == 10
    first = sampler.next_labeled()
    assert first in data.labeled_keys()
    l_key, u_key = sampler.next_pair()
    assert l_key in data.labeled_keys()
    assert u_key in data.unlabeled_keys()


def test_mnms_cache_limit_environment_is_bounded_and_applied_by_factory(tmp_path, monkeypatch):
    from self_audit.candidate_fuse.mnms_data import _DEFAULT_CACHE_BYTES, _MAX_CACHE_BYTES

    manifest = _manifest(tmp_path)
    monkeypatch.delenv("FUSE_MNMS_CACHE_BYTES", raising=False)
    assert open_fuse_data(manifest, role="train", synthetic=True)._cache_limit == _DEFAULT_CACHE_BYTES
    monkeypatch.setenv("FUSE_MNMS_CACHE_BYTES", "0")
    assert open_fuse_data(manifest, role="train", synthetic=True)._cache_limit == 0
    monkeypatch.setenv("FUSE_MNMS_CACHE_BYTES", str(_MAX_CACHE_BYTES))
    assert open_fuse_data(manifest, role="train", synthetic=True)._cache_limit == _MAX_CACHE_BYTES
    for invalid in ("-1", str(_MAX_CACHE_BYTES + 1), "1.5", "not-bytes"):
        monkeypatch.setenv("FUSE_MNMS_CACHE_BYTES", invalid)
        with pytest.raises(ValueError, match="FUSE_MNMS_CACHE_BYTES"):
            open_fuse_data(manifest, role="train", synthetic=True)


def test_v3_receipt_cross_links_and_full_split_are_locked(tmp_path):
    manifest = _manifest(tmp_path / "geometry-lock")
    obj = json.loads(manifest.read_text())
    geometry_path = Path(obj["geometry_receipt"]["path"])
    geometry = json.loads(geometry_path.read_text())
    geometry["records"][0]["geometry"]["shape"][0] = 4
    geometry_path.write_text(json.dumps(geometry))
    obj["geometry_receipt"]["sha256"] = _sha(geometry_path)
    manifest.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="geometry receipt SHA"):
        open_fuse_data(manifest, role="train", synthetic=True)

    manifest = _manifest(tmp_path / "source-lock")
    obj = json.loads(manifest.read_text())
    obj["source_inventory_sha256"] = "0" * 64
    manifest.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="source inventory identity digest"):
        open_fuse_data(manifest, role="train", synthetic=True)

    manifest = _manifest(tmp_path / "split-lock", role="development")
    obj = json.loads(manifest.read_text())
    obj["split"]["development"][0] = "mnmsFAKE"
    obj["patients"][0]["id"] = "mnmsFAKE"
    manifest.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="split membership differs from pinned L8 lock"):
        open_fuse_data(manifest, role="development", synthetic=True)


def test_real_geometry_rows_without_paths_use_pinned_source_inventory(tmp_path):
    """Real receipt schema separates patient geometry from source identity."""
    repo = Path(__file__).resolve().parents[1]
    manifest = _manifest(tmp_path / "manifest")
    obj = json.loads(manifest.read_text())
    geometry_path = tmp_path / "geometry-no-path.json"
    geometry_rows = json.loads(Path(obj["geometry_receipt"]["path"]).read_text())["records"]
    assert set(geometry_rows[0]) == {"patient_id", "path", "split", "status", "geometry"}
    geometry_path.write_text(json.dumps({
        "schema": "self_audit.fuse_mnms_geometry_receipt.v1",
        "source_identities_sha256": obj["source_inventory_sha256"],
        "records": [
            {key: value for key, value in r.items() if key != "path"}
            for r in geometry_rows
        ],
    }))
    obj["geometry_receipt"] = {"path": str(geometry_path), "sha256": _sha(geometry_path)}
    # The real report confirms that the row schema is metadata-only and that
    # source path and hash live in its independently pinned inventory.
    geometry_report = json.loads((repo / "reports/fuse-mnms-geometry-20261113/adjudicated/geometry_manifest.json").read_text())
    a0_geometry = next(r for r in geometry_report["records"] if r["patient_id"] == "A0S9V9")
    assert set(a0_geometry) == {"patient_id", "split", "status", "geometry"}
    snapshot = json.loads((repo / "reports/fuse-mnms-geometry-20261113/adjudicated/source_snapshot.json").read_text())
    a0_source = next(r for r in snapshot["records"] if r["patient_id"] == "A0S9V9")
    assert a0_source["source_identity"]["path"] == "Training/Labeled/A0S9V9/A0S9V9_sa.nii.gz"
    assert a0_source["source_identity"]["source_sha256"] == (
        "b6ba3e48a1358a7de9fe7fc1ed2e382c113f0280a85a234e800387e87d20fc27"
    )

    inventory_rows = []
    for record in obj["unlabeled"]:
        src = record["image"]
        inventory_rows.append({"patient_id": record["id"], "split": "train", "source_identity": {
            "member": src["member"], "source_sha256": src["container_sha256"], "crc32": src["crc32"]}})
    for role, records in (("development", json.loads(_manifest(tmp_path / "dev", role="development").read_text())["patients"]),
                          ("test", json.loads(_manifest(tmp_path / "test", role="test").read_text())["patients"])):
        for record in records:
            src = record["image"]
            inventory_rows.append({"patient_id": record["id"], "split": "val" if role == "development" else "test",
                                   "source_identity": {"member": src["member"],
                                                       "source_sha256": src["container_sha256"],
                                                       "crc32": src["crc32"]}})
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps({"schema": "self_audit.fuse_mnms_open345_snapshot.v1",
                                         "source_identities_sha256": obj["source_inventory_sha256"],
                                         "records": inventory_rows}))
    obj["source_inventory"] = {"path": str(inventory_path), "sha256": _sha(inventory_path)}
    manifest.write_text(json.dumps(obj))
    lock_path = Path(obj["l8_lock"]["path"])
    lock = json.loads(lock_path.read_text())
    lock["source_identities_sha256"] = obj["source_inventory_sha256"]
    lock["geometry_manifest_sha256"] = obj["geometry_receipt"]["sha256"]
    lock_path.write_text(json.dumps(lock))
    obj["l8_lock"]["sha256"] = _sha(lock_path)
    manifest.write_text(json.dumps(obj))
    assert open_fuse_data(manifest, role="train", synthetic=True).patients

    bad = json.loads(manifest.read_text())
    bad["unlabeled"][0]["image"]["member"] = "wrong/member.nii.gz"
    manifest.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="source identity differs from pinned inventory"):
        open_fuse_data(manifest, role="train", synthetic=True)

    bad = json.loads(manifest.read_text())
    bad["unlabeled"][0]["image"]["container_sha256"] = "0" * 64
    manifest.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="source identity differs from pinned inventory"):
        open_fuse_data(manifest, role="train", synthetic=True)


def test_pathless_geometry_requires_source_inventory(tmp_path):
    manifest = _manifest(tmp_path)
    obj = json.loads(manifest.read_text())
    geometry_path = Path(obj["geometry_receipt"]["path"])
    geometry = json.loads(geometry_path.read_text())
    geometry["records"] = [{k: v for k, v in row.items() if k != "path"}
                            for row in geometry["records"]]
    geometry_path.write_text(json.dumps(geometry))
    obj["geometry_receipt"]["sha256"] = _sha(geometry_path)
    lock_path = Path(obj["l8_lock"]["path"])
    lock = json.loads(lock_path.read_text())
    lock["geometry_manifest_sha256"] = obj["geometry_receipt"]["sha256"]
    lock_path.write_text(json.dumps(lock))
    obj["l8_lock"]["sha256"] = _sha(lock_path)
    obj["source_inventory"] = None
    manifest.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="pathless geometry row requires a pinned source inventory"):
        open_fuse_data(manifest, role="train", synthetic=True)


def test_v3_rejects_classmap_crc_member_sha_and_geometry_mismatch(tmp_path):
    manifest = _manifest(tmp_path / "base")
    obj = json.loads(manifest.read_text())
    obj["class_mapping"] = {"0": 0, "1": 1, "2": 2, "3": 3}
    manifest.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="class map"):
        open_fuse_data(manifest, role="train", synthetic=True)

    manifest = _manifest(tmp_path / "crc", role="test")
    obj = json.loads(manifest.read_text())
    obj["patients"][0]["image"]["crc32"] = "00000000"
    manifest.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="CRC"):
        open_fuse_data(manifest, role="test", synthetic=True)

    manifest = _manifest(tmp_path / "sha")
    obj = json.loads(manifest.read_text())
    obj["unlabeled"][0]["image"]["member_sha256"] = "0" * 64
    manifest.write_text(json.dumps(obj))
    data = open_fuse_data(manifest, role="train", synthetic=True)
    with pytest.raises(ValueError, match="member SHA"):
        data.load(("mnmsL00", 1), labeled=False)

    manifest = _manifest(tmp_path / "geometry")
    obj = json.loads(manifest.read_text())
    obj["unlabeled"][0]["frame_count"] = 2
    manifest.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="frame_count"):
        open_fuse_data(manifest, role="train", synthetic=True)


def test_legacy_factory_v1_v2(tmp_path):
    v1 = write_synthetic_manifests(tmp_path / "v1")
    assert isinstance(open_fuse_data(v1["train"]["path"], v1["train"]["sha256"], role="train", synthetic=True), FuseData)

    # Build the minimal v2 lineage contract from the unchanged v1 fixture.
    src_path = tmp_path / "source.json"
    source_records = []
    obj = json.loads(Path(v1["train"]["path"]).read_text())
    for r in obj["labeled"]:
        source_records.append({"id": r["id"], "image": r["image"], "image_sha256": r["image_sha256"],
                               "masks": {k: {"path": m["path"], "sha256": m["sha256"]} for k, m in r["masks"].items()}})
    source = {"schema": "self_audit.data.v2", "role": "training", "synthetic": True, "dataset": "synthetic",
              "label_budget": len(source_records), "labeled": source_records}
    src_path.write_text(json.dumps(source))
    v2 = dict(obj)
    v2["schema"] = "self_audit.candidate_fuse.data.v2"
    ids = sorted(r["id"] for r in v2["labeled"])
    v2["dataset"] = "synthetic"
    v2["labeled_contract"] = {"patient_ids": ids, "patient_ids_sha256": hashlib.sha256(
        json.dumps(ids, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest(),
        "source_manifest": str(src_path), "source_manifest_sha256": _sha(src_path)}
    v2_path = tmp_path / "v2.json"
    v2_path.write_text(json.dumps(v2))
    assert isinstance(open_fuse_data(v2_path, role="train", synthetic=True), FuseData)


def test_metadata_builder_does_not_decode_test_voxels(tmp_path, monkeypatch):
    import shutil
    repo = Path(__file__).resolve().parents[1]
    source_root = repo / "reports/fuse-mnms-geometry-20261113"
    copied_root = tmp_path / "geometry-receipts"
    shutil.copytree(source_root, copied_root)
    adjudicated = copied_root / "adjudicated"
    lock_path = copied_root / "l8/L8_LOCK.json"
    mapping_path = copied_root / "l8/MASK_MAPPING_RECEIPT.json"
    snapshot = json.loads((adjudicated / "source_snapshot.json").read_text())
    mapping = json.loads(mapping_path.read_text())
    roots = {role: tmp_path / role for role in ("train", "development", "test")}
    for root in roots.values():
        root.mkdir()
    test_zip = tmp_path / "test-source.zip"
    for row in snapshot["records"]:
        role = {"train": "train", "val": "development", "test": "test"}[row["split"]]
        relative = row["source_identity"].get("member", row["path"])
        path = roots[role] / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        if role != "train":
            mask_path = roots[role] / Path(relative).parent / Path(relative).name.replace("_sa.nii.gz", "_sa_gt.nii.gz")
            mask_path.touch()
    for row in mapping["records"]:
        pid = row["patient_id"]
        raw = next(x for x in snapshot["records"] if x["patient_id"] == pid)
        image_relative = Path(raw["path"])
        mask_relative = image_relative.parent / Path(row["masks"][0]["mask_path"]).name
        mask_path = roots["train"] / mask_relative
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.touch()
    test_files = {str(path.relative_to(roots["test"])): path for path in roots["test"].rglob("*.nii.gz")}
    _zip_source(test_zip, test_files)

    identity_by_path = {}
    for row in snapshot["records"]:
        identity_by_path[row["path"]] = row["source_identity"]["source_sha256"]
    for row in mapping["records"]:
        identity_by_path[row["masks"][0]["mask_path"]] = row["masks"][0]["mask_sha256"]

    actual_source = fuse_prepare_mnms._source
    def fake_source(path, member, *, expected_container_sha=None):
        path = Path(path).resolve()
        if path == test_zip.resolve():
            return actual_source(path, member, expected_container_sha=expected_container_sha)
        match = next((key for key in identity_by_path if Path(key).name == Path(member).name), None)
        digest = identity_by_path.get(str(member), identity_by_path.get(match, "1" * 64))
        return {"path": str(path), "member": member, "container_sha256": digest,
                "member_sha256": digest, "member_size": 128, "crc32": "12345678"}

    monkeypatch.setattr(fuse_prepare_mnms, "_source", fake_source)
    original_path_open = Path.open
    def no_test_payload_read(path, *args, **kwargs):
        if path.resolve() == test_zip.resolve():
            raise AssertionError("TEST ZIP payload must not be opened during preparation")
        return original_path_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", no_test_payload_read)
    actual_sha = fuse_prepare_mnms._sha
    monkeypatch.setattr(fuse_prepare_mnms, "_sha", lambda path: "0" * 64 if Path(path).resolve() == test_zip.resolve() else actual_sha(path))
    summary_path = adjudicated / "SUMMARY.json"
    summary = json.loads(summary_path.read_text())
    summary.setdefault("artifact_sha256", {})["test_archive"] = "0" * 64
    summary_path.write_text(json.dumps(summary))
    snapshot_path = adjudicated / "source_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text())
    for row in snapshot["records"]:
        if row["split"] == "test":
            row["source_identity"]["source_sha256"] = "0" * 64
    snapshot_path.write_text(json.dumps(snapshot))
    summary["snapshot_sha256"] = _sha(snapshot_path)
    summary_path.write_text(json.dumps(summary))
    original_zip_read = zipfile.ZipFile.read
    def no_test_zip_read(archive, name, *args, **kwargs):
        if Path(archive.filename).resolve() == test_zip.resolve():
            raise AssertionError("TEST ZIP member payload must not be read during preparation")
        return original_zip_read(archive, name, *args, **kwargs)
    monkeypatch.setattr(zipfile.ZipFile, "read", no_test_zip_read)
    monkeypatch.setattr(nib.Nifti1Image, "from_bytes", lambda *_: (_ for _ in ()).throw(AssertionError("voxel decode")))
    out_dir = tmp_path / "prepared"
    result = fuse_prepare_mnms.prepare_manifests(
        adjudicated=adjudicated, l8_lock=lock_path, mask_mapping_receipt=mapping_path,
        train_source=roots["train"], development_source=roots["development"],
        test_source=test_zip, out_dir=out_dir)
    assert set(result) == {"train", "development", "test"}
    test_manifest = json.loads(Path(result["test"]["path"]).read_text())
    assert len(test_manifest["patients"]) == 136
    assert all("masks" in record for record in test_manifest["patients"])
    assert all(record["image"]["member_sha256"] is None for record in test_manifest["patients"])
    assert all(mask["member_sha256"] is None for record in test_manifest["patients"] for mask in record["masks"].values())


def test_builder_requires_pinned_test_archive_sha_before_manifest_creation(tmp_path, monkeypatch):
    import shutil
    repo = Path(__file__).resolve().parents[1]
    source_adjudicated = repo / "reports/fuse-mnms-geometry-20261113/adjudicated"
    adjudicated = tmp_path / "adjudicated"
    shutil.copytree(source_adjudicated, adjudicated)
    summary_path = adjudicated / "SUMMARY.json"
    summary = json.loads(summary_path.read_text())
    artifacts = summary.setdefault("artifact_sha256", {})
    artifacts.pop("test_archive", None)
    summary_path.write_text(json.dumps(summary))
    for name, key in (("geometry_manifest.json", "geometry_manifest_sha256"),
                      ("source_snapshot.json", "snapshot_sha256"),
                      ("per_patient.json", "per_patient_sha256")):
        summary[key] = _sha(adjudicated / name)
    summary_path.write_text(json.dumps(summary))
    roots = {role: tmp_path / role for role in ("train", "development")}
    for root in roots.values():
        root.mkdir()
    test_zip = tmp_path / "test.zip"
    with zipfile.ZipFile(test_zip, "w"):
        pass
    with pytest.raises(ValueError, match="test_archive SHA256 pin"):
        fuse_prepare_mnms.prepare_manifests(
            adjudicated=adjudicated,
            l8_lock=repo / "reports/fuse-mnms-geometry-20261113/l8/L8_LOCK.json",
            train_source=roots["train"], development_source=roots["development"],
            test_source=test_zip, out_dir=tmp_path / "prepared-missing-pin")
