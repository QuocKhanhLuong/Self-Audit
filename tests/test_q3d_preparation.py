from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
from torch import nn

from self_audit.candidate_a3d.data import CandidateA3DData
from self_audit.candidate_q3d.data import Q3DData
from self_audit.candidate_q3d.diagnostics import _phase_frame
from self_audit.candidate_q3d.engine import _development_fg_dice
from self_audit.candidate_q3d.evaluation import _validated_manifest, evaluate_manifest
from self_audit.data import CLASS_MAPPING, digest
from prepare_acdc import _read_acdc_phases


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CLASS_MAP = '{"0":"background","1":"LV","2":"MYO","3":"RV"}'


class _FixtureStudent(nn.Module):
    def forward(self, image: torch.Tensor, geometry=None) -> dict:
        shape = tuple(int(value) for value in image.shape[-3:])
        assert tuple(geometry["shape"]) == shape
        logits = torch.zeros((1, 4, *shape), dtype=torch.float32, device=image.device)
        logits[:, 1, :, :, : shape[2] // 2] = 1.0
        logits[:, 3, :, :, shape[2] // 2 :] = 1.0
        return {"logits": logits}


def _mask(shape: tuple[int, int, int], offset: int = 0) -> np.ndarray:
    result = np.zeros(shape, dtype=np.uint8)
    result[1:3, 1:4, 1:4] = 1
    result[2:4, 3:5, 3:5] = 2
    result[1:4, 5:7, 1:3] = 3
    if offset:
        result = np.roll(result, offset, axis=1)
    return result


def _save_acdc_patient(root: Path, patient: str, value: int, *, phases: bool) -> None:
    folder = root / patient
    folder.mkdir(parents=True)
    x, y, z, t = 8, 8, 4, 2
    raw = (np.arange(x * y * z * t, dtype=np.int16).reshape(x, y, z, t) + value * 1000)
    raw[0, 0, 0, 0] += value * 100
    affine = np.diag((1.0, 1.0, 5.0, 1.0))
    image = nib.Nifti1Image(raw, affine)
    image.header.set_xyzt_units("mm")
    nib.save(image, folder / f"{patient}_4d.nii.gz")
    for frame in range(t):
        frame_image = nib.Nifti1Image(raw[..., frame], affine)
        frame_image.header.set_xyzt_units("mm")
        nib.save(frame_image, folder / f"{patient}_frame{frame + 1:02d}.nii.gz")
        label_image = nib.Nifti1Image(_mask((x, y, z), frame).astype(np.uint8), affine)
        label_image.header.set_xyzt_units("mm")
        nib.save(label_image, folder / f"{patient}_frame{frame + 1:02d}_gt.nii.gz")
    if phases:
        (folder / "Info.cfg").write_text("ED: 1\nES: 2\nGroup: NOR\n", encoding="utf-8")


def _run_acdc(root: Path, out: Path) -> dict:
    split = root.parent / "split.json"
    split.write_text(json.dumps({
        "train": ["patient001", "patient002", "patient005"],
        "val": ["patient003"],
        "test": ["patient004"],
    }), encoding="utf-8")
    for index, patient in enumerate(("patient001", "patient002", "patient003", "patient004", "patient005"), start=1):
        _save_acdc_patient(root, patient, index, phases=patient in {"patient003", "patient004"})
    result = subprocess.run([
        sys.executable, str(ROOT / "prepare_acdc.py"), "--root", str(root),
        "--split", str(split), "--out", str(out), "--labels", "2", "--seed", "42",
        "--q3d-evaluation", "--spatial-unit", "mm",
    ], check=False, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads((out / "train.json").read_text(encoding="utf-8"))


def _save_mnms_patient(root: Path, patient: str, value: int) -> tuple[str, str, str]:
    x, y, z, t = 8, 8, 4, 2
    raw = (np.arange(x * y * z * t, dtype=np.int16).reshape(x, y, z, t) + value * 1000)
    raw[0, 0, 0, 0] += value * 100
    affine = np.diag((1.0, 1.0, 5.0, 1.0))
    image_path = root / f"{patient}_cine.nii.gz"
    image = nib.Nifti1Image(raw, affine)
    image.header.set_xyzt_units("mm")
    nib.save(image, image_path)
    masks = []
    for frame in range(t):
        # Original M&Ms source IDs: 1=LV, 2=MYO, 3=RV.
        source = _mask((x, y, z), frame)
        source = np.choose(source, [0, 3, 2, 1]).astype(np.uint8)
        mask_path = root / f"{patient}_mask_{frame}.nii.gz"
        mask_image = nib.Nifti1Image(source, affine)
        mask_image.header.set_xyzt_units("mm")
        nib.save(mask_image, mask_path)
        masks.append(mask_path.name)
    return image_path.name, masks[0], masks[1]


def _run_mnms(root: Path, out: Path) -> dict:
    rows = []
    for index, (patient, split) in enumerate((
        ("train001", "train"), ("train002", "train"), ("train003", "train"),
        ("dev001", "val"), ("test001", "test"),
    ), start=1):
        image, ed_mask, es_mask = _save_mnms_patient(root, patient, index)
        rows.append({
            "patient_id": patient, "split": split, "image": image,
            "ed_frame": "0", "es_frame": "1", "mask_ed": ed_mask,
            "mask_es": es_mask, "axis_order": "XYZT", "mask_axis_order": "XYZ",
            "vendor": "fixture-vendor",
        })
    index_path = root.parent / "index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = subprocess.run([
        sys.executable, str(ROOT / "prepare_mnms.py"), "--index", str(index_path),
        "--root", str(root), "--out", str(out), "--labels", "2", "--seed", "42",
        "--source-class-map", SOURCE_CLASS_MAP, "--frame-base", "0",
        "--dataset-release", "original_2020", "--q3d-evaluation",
    ], check=False, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads((out / "train.json").read_text(encoding="utf-8"))


def _assert_q3d_routes(out: Path, train_manifest: dict, dataset: str) -> None:
    train_path = out / "train.json"
    train_sha = digest(train_path)
    training = CandidateA3DData(train_path, train_sha, synthetic=False)
    routed = Q3DData(training, protocol="A")
    assert len(routed.labeled_patients) == 2
    assert len(routed.unlabeled_patients) == 1
    assert routed.student_data.unlabeled[0].keys() == {"id", "image", "image_sha256"}
    assert train_manifest["class_mapping"] == CLASS_MAPPING
    assert train_manifest["unlabeled"]

    train_ids = set(routed.train_patients)
    for role in ("development", "test"):
        manifest_path = out / f"{role}.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["role"] == role
        assert manifest["dataset"] == dataset
        assert manifest["class_mapping"] == CLASS_MAPPING
        record = manifest["patients"][0]
        assert record["phases"] == {"ED": 0, "ES": 1}
        assert record["masks"]["0"]["phase"] == "ED"
        assert record["masks"]["1"]["phase"] == "ES"
        assert record["image_sha256"] == digest(out / record["image"])
        assert all(entry["sha256"] == digest(out / entry["path"])
                   for entry in record["masks"].values())
        assert record["id"] not in train_ids

        if role == "development":
            _, checked, _, _, _ = _validated_manifest(
                manifest_path, digest(manifest_path), train_patient_ids=train_ids, expected_synthetic=False,
            )
            assert _phase_frame(checked["patients"][0], "ED") == 0
            assert _phase_frame(checked["patients"][0], "ES") == 1
            assert record["masks"]["0"]["phase"] == "ED"
            assert record["masks"]["1"]["phase"] == "ES"

        metrics = evaluate_manifest(
            _FixtureStudent(), manifest_path, digest(manifest_path),
            out.parent / f"{out.name}-{dataset}-{role}-metrics",
            split=role, train_patient_ids=train_ids, expected_synthetic=False,
        )
        clinical = metrics["per_patient"][record["id"]]["clinical"]
        assert clinical["status"] == "OK"
        assert clinical["errors"]["LV_EDV_abs_error_ml"] >= 0

        if role == "development":
            selection = _development_fg_dice(
                _FixtureStudent(), {"path": str(manifest_path)},
                sha256=digest(manifest_path), train_ids=tuple(sorted(train_ids)),
                synthetic=False, output_kind="student", device=torch.device("cpu"),
            )
            assert selection["patient_count"] == metrics["patients"] == 1
            assert selection["patients"][record["id"]] == pytest.approx(
                metrics["per_patient"][record["id"]]["fg_dice"], abs=1e-7,
            )
            assert selection["score"] == pytest.approx(
                metrics["patient_macro"]["fg_dice"], abs=1e-7,
            )


def test_acdc_cli_prepares_q3d_dev_test_and_preserves_image_only_u(tmp_path: Path) -> None:
    root = tmp_path / "acdc-source"
    root.mkdir()
    out = tmp_path / "acdc-cache"
    train_manifest = _run_acdc(root, out)
    assert train_manifest["evaluation_ids"] == ["patient003", "patient004"]
    assert all("phases" not in record and "masks" not in record
               for record in train_manifest["unlabeled"])
    prep = json.loads((out / "PREPARATION.json").read_text(encoding="utf-8"))
    assert prep["unlabeled_mask_reads"] == 0
    assert prep["test_manifest_sha256"] == digest(out / "test.json")
    _assert_q3d_routes(out, train_manifest, "acdc")


def test_acdc_phase_parser_rejects_missing_duplicate_and_out_of_range_frames(tmp_path: Path) -> None:
    info = tmp_path / "Info.cfg"
    info.write_text("ED: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="distinct ED and ES"):
        _read_acdc_phases(info, 2)
    info.write_text("ED: 1\nED: 1\nES: 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate ED"):
        _read_acdc_phases(info, 2)
    info.write_text("ED: 1\nES: 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside cine"):
        _read_acdc_phases(info, 2)


def test_original_mnms_cli_prepares_q3d_dev_test_and_preserves_class_map(tmp_path: Path) -> None:
    root = tmp_path / "mnms-source"
    root.mkdir()
    out = tmp_path / "mnms-cache"
    train_manifest = _run_mnms(root, out)
    assert train_manifest["evaluation_ids"] == ["dev001", "test001"]
    assert all("phases" not in record and "masks" not in record
               for record in train_manifest["unlabeled"])
    prep = json.loads((out / "PREPARATION.json").read_text(encoding="utf-8"))
    assert prep["unlabeled_mask_reads"] == 0
    assert prep["source_class_map"] == {"0": "background", "1": "lv", "2": "myo", "3": "rv"}
    assert train_manifest["class_mapping"] == CLASS_MAPPING
    assert prep["test_manifest_sha256"] == digest(out / "test.json")
    _assert_q3d_routes(out, train_manifest, "mnms")
