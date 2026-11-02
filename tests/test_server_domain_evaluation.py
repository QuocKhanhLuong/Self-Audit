"""Manufactured fixtures for shared ACDC+M&Ms compact domain evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from self_audit.compact_artifacts import digest, export_compact_student, save_compact_checkpoint
from self_audit.compact_model import build_compact_model
from self_audit.compact_inference import write_json
from self_audit.server_evaluation import evaluate_shared_domains


SHAPE = (1, 1, 16, 16)
WIDTHS = [16, 32, 64, 128]


def _write_image(folder: Path, name: str, value: float) -> tuple[str, str]:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    array = np.full(SHAPE, value, dtype=np.float32)
    np.save(path, array)
    return str(path.relative_to(folder)), digest(path)


def _build_shared_train(root: Path):
    train_dir = root / "train"
    train_dir.mkdir(parents=True)
    labeled = []
    unlabeled = []
    lids: list[str] = []
    uids: list[str] = []
    domain_map = {}
    domain_patients = {"acdc": [], "mnm": []}
    counter = 1
    for domain in ("acdc", "mnm"):
        for index in range(4):
            patient_id = f"{domain}:train{index}"
            image_rel, image_sha = _write_image(train_dir, f"{domain}_train{index}.npy", counter / 20.0)
            mask_path = train_dir / f"{domain}_train{index}_mask.npy"
            np.save(mask_path, np.zeros((1, 16, 16), dtype=np.uint8))
            labeled.append({
                "id": patient_id,
                "image": image_rel,
                "image_sha256": image_sha,
                "masks": {"0": {"path": mask_path.name, "sha256": digest(mask_path)}},
            })
            lids.append(patient_id)
            domain_map[patient_id] = domain
            domain_patients[domain].append(patient_id)
            counter += 1

        patient_id = f"{domain}:unlabeled0"
        image_rel, image_sha = _write_image(train_dir, f"{domain}_unlabeled.npy", counter / 20.0)
        unlabeled.append({"id": patient_id, "image": image_rel, "image_sha256": image_sha})
        uids.append(patient_id)
        domain_map[patient_id] = domain
        domain_patients[domain].append(patient_id)
        counter += 1

    train_path = train_dir / "train.json"
    train_sha = write_json(train_path, {
        "schema": "self_audit.data.v2",
        "role": "training",
        "synthetic": True,
        "shared": True,
        "domains": ["acdc", "mnm"],
        "label_budget": 8,
        "domain_budgets": {"acdc": 4, "mnm": 4},
        "labeled": labeled,
        "unlabeled": unlabeled,
        "lids": lids,
        "uids": uids,
        "domain_map": domain_map,
        "domain_patients": domain_patients,
        "evaluation_ids": [],
    })

    torch.manual_seed(42)
    model = build_compact_model({"widths": WIDTHS, "in_channels": 3, "num_classes": 4})
    # A stable all-background prediction makes the absent-class behavior easy to audit.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    checkpoint_dir = root / "checkpoint"
    data = SimpleNamespace(
        lids=lids,
        uids=uids,
        path=str(train_path),
        data_sha256=train_sha,
        manifest={"synthetic": True, "label_budget": 8},
    )
    checkpoint_sha = save_compact_checkpoint(
        model=model,
        optimizer=None,
        config={"model": {"widths": WIDTHS}, "method": "supervised", "teacher_memory": None},
        data=data,
        out=checkpoint_dir,
        step=10,
    )
    export_dir = root / "compact_export"
    export_sha = export_compact_student(checkpoint_dir, expected_sha=checkpoint_sha, out=export_dir)
    return train_path, train_sha, lids, uids, export_dir, export_sha


def _write_development_manifest(
    root: Path,
    *,
    domain: str,
    ids: list[str],
    all_background: bool,
) -> tuple[Path, str]:
    manifest_dir = root / domain
    manifest_dir.mkdir(parents=True, exist_ok=True)
    patients = []
    for index, patient_id in enumerate(ids):
        image_path = manifest_dir / f"case_{index}.npy"
        image = np.full(SHAPE, 0.6 + index * 0.05 + (0.12 if domain == "mnm" else 0.0), dtype=np.float32)
        np.save(image_path, image)
        if all_background:
            mask = np.zeros((1, 16, 16), dtype=np.uint8)
        else:
            mask = (np.arange(16 * 16, dtype=np.uint8).reshape(1, 16, 16) % 4).astype(np.uint8)
        mask_path = manifest_dir / f"case_{index}_mask.npy"
        np.save(mask_path, mask)
        affine = np.diag([1.25, 1.5, 7.5, 1.0])
        affine[0, 3] = float(index + (10 if domain == "mnm" else 0))
        patients.append({
            "id": patient_id,
            "domain": domain,
            "image": image_path.name,
            "image_sha256": digest(image_path),
            "axes": ["T", "Z", "Y", "X"],
            "shape_tzyx": list(SHAPE),
            "affine": affine.tolist(),
            "masks": {"0": {"path": mask_path.name, "sha256": digest(mask_path)}},
        })
    manifest_path = manifest_dir / "development.json"
    manifest_sha = write_json(manifest_path, {
        "schema": "self_audit.native_manifest.v1",
        "role": "development",
        "synthetic": True,
        "domain": domain,
        "dataset": domain,
        "patients": patients,
    })
    return manifest_path, manifest_sha


def _write_index(root: Path, *, acdc_ids: list[str], mnm_ids: list[str]) -> tuple[Path, str]:
    maps_dir = root / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    acdc_path, acdc_sha = _write_development_manifest(
        root / "development", domain="acdc", ids=acdc_ids, all_background=True
    )
    mnm_path, mnm_sha = _write_development_manifest(
        root / "development", domain="mnm", ids=mnm_ids, all_background=False
    )
    index_path = maps_dir / "EVALUATION_MAPS.json"
    index_sha = write_json(index_path, {
        "schema": "self_audit.server_evaluation_maps.v1",
        "maps": {
            "acdc_val18": {"domain": "acdc", "manifest": str(acdc_path), "sha256": acdc_sha},
            "mnm_val32": {"domain": "mnm", "manifest": str(mnm_path), "sha256": mnm_sha},
        },
    })
    return index_path, index_sha


def _make_fixture(root: Path, *, acdc_ids: list[str] | None = None):
    train_path, train_sha, lids, uids, export_dir, export_sha = _build_shared_train(root / "source")
    index_path, index_sha = _write_index(
        root / "source",
        acdc_ids=acdc_ids or ["val_acdc0"],
        # One source ID is already namespaced; it must not become mnm:mnm:already.
        mnm_ids=["mnm:already", "val_mnm1"],
    )
    return train_path, train_sha, export_dir, export_sha, index_path, index_sha, lids, uids


def test_shared_domain_evaluation_averages_dataset_macro4_and_preserves_identity_geometry(tmp_path):
    train_path, train_sha, export_dir, export_sha, index_path, index_sha, _, _ = _make_fixture(tmp_path)
    result = evaluate_shared_domains(
        export_path=export_dir,
        export_sha256=export_sha,
        train_manifest_path=train_path,
        train_manifest_sha256=train_sha,
        evaluation_maps_path=index_path,
        evaluation_maps_sha256=index_sha,
        out_dir=tmp_path / "run" / "validation" / "domain_eval",
        device="cpu",
        batch_size=1,
        normalization="prepared",
    )

    assert result["selection_metric"] == "equal_domain_class_first_macro4"
    assert result["selection_rule"].startswith("selection_score = arithmetic mean")
    assert result["domain_macro4"]["acdc"] == pytest.approx(1.0)
    assert result["domain_macro4"]["mnm"] == pytest.approx(0.1)
    assert result["selection_score"] == pytest.approx(0.55)
    assert result["domain_results"]["acdc"]["patient_coverage"]["expected_patient_count"] == 1
    assert result["domain_results"]["mnm"]["patient_coverage"]["expected_patient_count"] == 2
    assert set(result["domain_results"]["mnm"]["macro_patient"]) == {"BG", "RV", "MYO", "LV"}
    assert result["domain_results"]["mnm"]["macro_patient"]["RV"]["recall"] == pytest.approx(0.0)
    assert result["domain_results"]["mnm"]["macro_patient"]["RV"]["precision"] is None

    metrics = json.loads(Path(result["historical_metrics_path"]).read_text())
    assert result["selection_score"] != pytest.approx(metrics["class_first_macro4"])
    assert result["export_manifest_sha256"] == export_sha
    assert metrics["export_manifest_sha256"] == export_sha
    frozen = json.loads(Path(result["prediction_freeze"]["path"]).read_text())
    assert frozen["export_manifest_sha256"] == export_sha
    assert result["prediction_freeze"]["precedes_scoring"] is True

    composite = json.loads(Path(result["namespaced_manifest_path"]).read_text())
    ids = [row["id"] for row in composite["patients"]]
    assert ids == ["acdc:val_acdc0", "mnm:already", "mnm:val_mnm1"]
    prediction_by_id = {row["id"]: row for row in frozen["predictions"]}
    assert set(prediction_by_id) == set(ids)
    for row in composite["patients"]:
        prediction = prediction_by_id[row["id"]]
        assert prediction["axes"] == ["T", "Z", "Y", "X"]
        assert prediction["shape_tzyx"] == list(SHAPE)
        np.testing.assert_array_equal(np.asarray(prediction["affine"]), np.asarray(row["affine"]))

    selection_lock = json.loads(Path(result["selection_lock_path"]).read_text())
    assert selection_lock["locked_before_scoring"] is True
    assert selection_lock["rule"] == result["selection_rule"]
    assert selection_lock["export_manifest_sha256"] == export_sha


def test_shared_domain_evaluation_uses_canonical_namespaced_train_overlap_check(tmp_path):
    train_path, train_sha, export_dir, export_sha, _, _, _, _ = _make_fixture(
        tmp_path,
        acdc_ids=["acdc:train0"],
    )
    index_path, index_sha = _write_index(
        tmp_path / "overlap_source",
        acdc_ids=["acdc:train0"],
        mnm_ids=["mnm:val_mnm0"],
    )
    out_dir = tmp_path / "run" / "validation" / "domain_eval"
    with pytest.raises(ValueError, match="Evaluation patient ID overlap with TRAIN"):
        evaluate_shared_domains(
            export_path=export_dir,
            export_sha256=export_sha,
            train_manifest_path=train_path,
            train_manifest_sha256=train_sha,
            evaluation_maps_path=index_path,
            evaluation_maps_sha256=index_sha,
            out_dir=out_dir,
            device="cpu",
            batch_size=1,
            normalization="prepared",
        )
    assert not out_dir.exists()


def test_shared_domain_evaluation_rejects_acdc_l8_train_as_joint_l8(tmp_path):
    train_path, train_sha, _, _, _, _, _, _ = _make_fixture(tmp_path)
    train = json.loads(train_path.read_text())
    train["domain_budgets"] = {"acdc": 8, "mnm": 0}
    train["label_budget"] = 8
    wrong_sha = write_json(train_path, train)
    # The shared identity is rejected before evaluation maps or export are opened.
    with pytest.raises(ValueError, match=r"exactly ACDC4 \+ M&Ms4"):
        evaluate_shared_domains(
            export_path=tmp_path / "missing-export",
            export_sha256="0" * 64,
            train_manifest_path=train_path,
            train_manifest_sha256=wrong_sha,
            evaluation_maps_path=tmp_path / "missing-index",
            evaluation_maps_sha256="0" * 64,
            out_dir=tmp_path / "run" / "validation" / "domain_eval",
        )
