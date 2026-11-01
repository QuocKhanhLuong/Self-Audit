import csv
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from prepare_acdc import normalize_cine
from scripts.prepare_server_evaluation import prepare_server_evaluation


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save_nifti(path: Path, data: np.ndarray, affine: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), str(path))


def _split(path: Path, *, train: list[str], val: list[str], test: list[str], dataset: str | None = None) -> str:
    payload = {"train": train, "val": val, "test": test}
    if dataset is not None:
        payload["dataset"] = dataset
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return _sha(path)


def test_normalize_cine_is_byte_identical_to_historical_prepare_expression():
    raw_xyzt = np.random.default_rng(17).normal(size=(7, 5, 3, 4)).astype(np.float64)
    raw_xyzt[0, 0, 0, 0] = -100.0
    raw_xyzt[-1, -1, -1, -1] = 100.0
    lo, hi = np.percentile(raw_xyzt, [.5, 99.5])
    historical = np.clip((raw_xyzt - lo) / max(float(hi - lo), 1e-6), 0, 1).transpose(3, 2, 1, 0).astype(np.float32)
    extracted = normalize_cine(raw_xyzt)
    assert extracted.dtype == np.float32
    assert extracted.shape == (4, 3, 5, 7)
    assert extracted.tobytes() == historical.tobytes()


def _make_acdc_case(tmp_path: Path, *, broken_frame: bool = False) -> tuple[Path, Path, str, np.ndarray, np.ndarray]:
    raw_root = tmp_path / "acdc_test50"
    raw_root.mkdir()
    split_path = tmp_path / "acdc_split.json"
    split_sha = _split(split_path, train=["patient001"], val=["patient004"], test=["patient101"])

    raw_xyzt = np.random.default_rng(6).normal(size=(5, 4, 3, 2)).astype(np.float32)
    cine_affine = np.array([[1.2, 0, 0, 9], [0, 1.3, 0, 8], [0, 0, 4.5, 7], [0, 0, 0, 1]], dtype=np.float64)
    placeholder_affine = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
    folder = raw_root / "patient101"
    _save_nifti(folder / "patient101_4d.nii.gz", raw_xyzt, cine_affine)
    frame = raw_xyzt[..., 0].copy()
    if broken_frame:
        frame[0, 0, 0] += 1
    _save_nifti(folder / "patient101_frame01.nii.gz", frame, placeholder_affine)
    labels = np.zeros((5, 4, 3), dtype=np.uint8)
    labels[1, 1, 1] = 3
    _save_nifti(folder / "patient101_frame01_gt.nii.gz", labels, placeholder_affine)

    # An invalid TRAIN payload is present to prove this role adapter never opens it.
    train = raw_root / "patient001"
    _save_nifti(train / "patient001_4d.nii.gz", np.zeros((2, 2, 2, 2), dtype=np.float32), np.eye(4))
    _save_nifti(train / "patient001_frame01.nii.gz", np.zeros((2, 2, 2), dtype=np.float32), np.eye(4))
    _save_nifti(train / "patient001_frame01_gt.nii.gz", np.full((2, 2, 2), 9, dtype=np.uint8), np.eye(4))
    return raw_root, split_path, split_sha, raw_xyzt, cine_affine


def test_acdc_test_role_keeps_cine_affine_records_placeholder_and_never_reads_train(tmp_path, monkeypatch):
    raw_root, split_path, split_sha, raw_xyzt, cine_affine = _make_acdc_case(tmp_path)
    opened: list[str] = []
    load = nib.load

    def tracked_load(path, *args, **kwargs):
        opened.append(str(path))
        return load(path, *args, **kwargs)

    monkeypatch.setattr(nib, "load", tracked_load)
    out = tmp_path / "prepared"
    result = prepare_server_evaluation(
        dataset="acdc", raw_root=raw_root, split_json=split_path, split_sha256=split_sha,
        role="test", out_dir=out, synthetic=True,
    )
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    patient = manifest["patients"][0]
    receipt = json.loads(Path(result["receipt_path"]).read_text())

    assert manifest["schema"] == "self_audit.data.v2"
    assert manifest["role"] == "test"
    assert manifest["cohort_ids"] == ["patient101"]
    assert patient["axes"] == ["T", "Z", "Y", "X"]
    assert np.allclose(np.asarray(patient["affine"]), cine_affine, atol=1e-6, rtol=0)
    assert patient["geometry_source_image"]["sha256"]
    assert patient["frame_geometry_bridge"][0]["frame_affine_matches_cine"] is False
    assert np.load(out / patient["image"]).shape == (2, 3, 4, 5)
    assert np.array_equal(np.load(out / patient["image"]), normalize_cine(raw_xyzt))
    from self_audit.compact_inference import load_native_manifest, load_patient_cine
    loaded_manifest, _ = load_native_manifest(result["manifest_path"], result["manifest_sha256"], expected_roles=["test"])
    loaded_cine, _, _ = load_patient_cine(loaded_manifest["patients"][0], out, normalization="prepared")
    assert np.array_equal(loaded_cine, normalize_cine(raw_xyzt))
    assert all("patient001" not in path for path in opened)
    assert receipt["access"]["train_reads"] == 0
    assert receipt["access"]["train_payload_files_opened"] == 0
    assert receipt["access"]["train_mask_files_discovered"] == 0
    assert receipt["access"]["unlabeled_mask_reads"] == 0
    assert receipt["access"]["unlabeled_mask_files_discovered"] == 0


def test_acdc_frame_voxel_bridge_failure_leaves_no_output(tmp_path):
    raw_root, split_path, split_sha, _, _ = _make_acdc_case(tmp_path, broken_frame=True)
    out = tmp_path / "bad-bridge-output"
    with pytest.raises(ValueError, match="exact voxel bridge"):
        prepare_server_evaluation(
            dataset="acdc", raw_root=raw_root, split_json=split_path, split_sha256=split_sha,
            role="test", out_dir=out, synthetic=True,
        )
    assert not out.exists()
    assert not list(tmp_path.glob(".bad-bridge-output.staging-*"))


def test_bad_split_hash_fails_before_opening_nifti(tmp_path, monkeypatch):
    raw_root, split_path, _, _, _ = _make_acdc_case(tmp_path)
    opened: list[str] = []
    load = nib.load

    def tracked_load(path, *args, **kwargs):
        opened.append(str(path))
        return load(path, *args, **kwargs)

    monkeypatch.setattr(nib, "load", tracked_load)
    with pytest.raises(ValueError, match="split JSON SHA-256 mismatch"):
        prepare_server_evaluation(
            dataset="acdc", raw_root=raw_root, split_json=split_path, split_sha256="0" * 64,
            role="test", out_dir=tmp_path / "bad-hash-output", synthetic=True,
        )
    assert opened == []


def test_mnm_uses_explicit_split_metadata_quarantine_and_canonical_map_once(tmp_path, monkeypatch):
    raw_root = tmp_path / "mnm"
    raw_root.mkdir()
    split_path = tmp_path / "mnm_split.json"
    split_sha = _split(split_path, train=["TRAIN000"], val=["VAL000"], test=["TEST000", "A7G0P5"], dataset="M&Ms")
    csv_path = tmp_path / "211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["External code", "ED", "ES"])
        writer.writeheader()
        writer.writerows([
            {"External code": "TRAIN000", "ED": "0", "ES": "2"},
            {"External code": "VAL000", "ED": "0", "ES": "2"},
            {"External code": "TEST000", "ED": "1", "ES": "3"},
            {"External code": "A7G0P5", "ED": "1", "ES": "3"},
        ])
    csv_sha = _sha(csv_path)
    raw_xyzt = np.random.default_rng(22).normal(size=(4, 3, 2, 4)).astype(np.float32)
    affine = np.array([[1.4, 0, 0, 2], [0, 1.4, 0, 3], [0, 0, 8, 4], [0, 0, 0, 1]], dtype=np.float64)
    image_path = raw_root / "TEST000_sa.nii.gz"
    mask_path = raw_root / "TEST000_sa_gt.nii.gz"
    _save_nifti(image_path, raw_xyzt, affine)
    raw_mask = np.zeros(raw_xyzt.shape, dtype=np.float32)
    source_classes = np.resize(np.array([0, 1, 2, 3, 255], dtype=np.float32), raw_mask[..., 1].size).reshape(raw_mask[..., 1].shape)
    raw_mask[..., 1] = source_classes
    raw_mask[..., 3] = source_classes
    _save_nifti(mask_path, raw_mask, affine)

    # Invalid TRAIN labels must remain unopened.
    train_image = np.zeros((2, 2, 2, 3), dtype=np.float32)
    _save_nifti(raw_root / "TRAIN000_sa.nii.gz", train_image, affine)
    _save_nifti(raw_root / "TRAIN000_sa_gt.nii.gz", np.full(train_image.shape, 9, dtype=np.uint8), affine)

    opened: list[str] = []
    load = nib.load

    def tracked_load(path, *args, **kwargs):
        opened.append(str(path))
        return load(path, *args, **kwargs)

    monkeypatch.setattr(nib, "load", tracked_load)
    out = tmp_path / "mnm-prepared"
    result = prepare_server_evaluation(
        dataset="mnm", raw_root=raw_root, split_json=split_path, split_sha256=split_sha,
        role="test", out_dir=out, metadata_csv=csv_path, metadata_csv_sha256=csv_sha, synthetic=True,
    )
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    patient = manifest["patients"][0]
    receipt = json.loads(Path(result["receipt_path"]).read_text())

    assert manifest["cohort_ids"] == ["TEST000"]
    assert manifest["quarantined_ids_excluded"] == ["A7G0P5"]
    assert patient["metadata_frames_zero_based"] == {"ed": 1, "es": 3}
    assert patient["masks"].keys() == {"1", "3"}
    expected = np.where(source_classes == 1, 3, np.where(source_classes == 3, 1, source_classes)).astype(np.uint8)
    prepared = np.load(out / patient["masks"]["1"]["path"])
    assert np.array_equal(prepared, expected.transpose(2, 1, 0))
    assert receipt["preprocessing"]["mask_mapping"]["mapping_applied_once"] == {"0": 0, "1": 3, "2": 2, "3": 1, "255": 255}
    assert all("TRAIN000" not in path for path in opened)
    assert receipt["access"]["train_reads"] == 0
    assert receipt["access"]["train_payload_files_opened"] == 0
    assert receipt["access"]["train_mask_files_discovered"] == 0


def test_mnm_metadata_hash_is_fail_closed_before_raw_payload_open(tmp_path, monkeypatch):
    raw_root = tmp_path / "mnm"
    raw_root.mkdir()
    split_path = tmp_path / "mnm_split.json"
    split_sha = _split(split_path, train=["TRAIN000"], val=["VAL000"], test=["TEST000"])
    csv_path = tmp_path / "metadata.csv"
    csv_path.write_text("External code,ED,ES\nTEST000,0,1\n", encoding="utf-8")
    opened: list[str] = []

    def tracked_load(path, *args, **kwargs):
        opened.append(str(path))
        raise AssertionError("NIfTI payload must not open after a bad metadata hash")

    monkeypatch.setattr(nib, "load", tracked_load)
    with pytest.raises(ValueError, match="M&Ms metadata CSV SHA-256 mismatch"):
        prepare_server_evaluation(
            dataset="mnm", raw_root=raw_root, split_json=split_path, split_sha256=split_sha,
            role="test", out_dir=tmp_path / "bad-csv-output", metadata_csv=csv_path,
            metadata_csv_sha256="0" * 64, synthetic=True,
        )
    assert opened == []
