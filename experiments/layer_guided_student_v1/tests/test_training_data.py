"""Manufactured tests for the opt-in frozen Astra training-bank adapter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from training_v1.data import (
    FEATURE_DIM,
    POLICY,
    REVIEWED_CARDINAL_PRODUCER_HASHES,
    FrozenTrainingBank,
    prepare_targets,
)

from self_audit_guidance.io import freeze, write_json


def _patient_arrays(patient_index: int):
    image = np.arange(2 * 2 * 17 * 19, dtype=np.float32).reshape(2, 2, 17, 19)
    image = image + patient_index * 0.25
    features = np.arange(2 * 2 * 3 * 4 * FEATURE_DIM, dtype=np.float32).reshape(
        2, 2, 3, 4, FEATURE_DIM
    )
    features = (features + patient_index).astype(np.float16)
    q = np.full((2, 2, 3, 4, 4), 0.05, dtype=np.float32)
    q[..., 1] = 0.75 if patient_index == 0 else 0.10
    q[..., 2] = 0.15 if patient_index == 0 else 0.80
    q[..., 3] = 0.05
    # Keep a valid simplex for the second patient's class-2-dominant target.
    q /= q.sum(axis=-1, keepdims=True)
    r0 = q.copy()
    reliability = np.full((2, 2, 3, 4), 0.5, dtype=np.float32)
    reliability[..., 0, 0] = 0.0
    labels = np.full((2, 2, 3, 4), 255, dtype=np.uint8)
    labels[..., 0, 0] = 1 if patient_index == 0 else 2
    return image, features, q, r0, reliability, labels


def _make_bank(tmp_path: Path, *, locked_splits=None, patients=None, mutate=None):
    patients = list(patients or ("patient_a", "patient_b", "patient_c"))
    splits = locked_splits or {
        "train": patients,
        "val": ["patient_val"],
        "test": ["patient_test"],
    }
    image_records = {}
    reports = {}
    for index, pid in enumerate(patients):
        arrays = _patient_arrays(index)
        np.savez(
            tmp_path / f"{pid}.npz",
            image=arrays[0],
            features=arrays[1],
            q=arrays[2],
            r0=arrays[3],
            reliability=arrays[4],
            labels=arrays[5],
        )
        class_pixels = [int((arrays[5] == class_id).sum()) for class_id in range(4)]
        reports[pid] = {"class_pixels": class_pixels}
        image_records[pid] = {
            "image_sha256": f"{index + 1:x}".zfill(64),
            "shape_xyzt": [19, 17, 2, 2],
            "affine": np.eye(4).tolist(),
            "spatial_unit": "unknown",
        }
    write_json(tmp_path / "config.json", {"feature_dim": FEATURE_DIM, "grid": 48})
    write_json(tmp_path / "split.json", splits)
    write_json(tmp_path / "REPORT.json", {"status": "SUPPORT_NOT_ACCURACY"})
    source_hashes = dict(REVIEWED_CARDINAL_PRODUCER_HASHES)
    manifest = {
        "role": "guidance_bank",
        "split": "train",
        "patients": patients,
        "locked_splits": splits,
        "config": {"feature_dim": FEATURE_DIM, "grid": 48},
        "teacher": {
            "frozen": True,
            "checkpoint_sha256": "4" * 64,
            "segmentation_finetune": False,
            "provider": "manufactured-test",
            "source_commit_expected": "a" * 40,
        },
        "source_sha256": source_hashes,
        "image_records": image_records,
        "patient_reports": reports,
    }
    frozen = freeze(tmp_path, manifest)
    if mutate is not None:
        mutate(tmp_path)
    return tmp_path, frozen, manifest


def test_init_pins_cardinal_provenance_and_patient_support(tmp_path):
    root, frozen, _ = _make_bank(tmp_path)
    bank = FrozenTrainingBank(root, expected_manifest_sha256=_sha(root))

    assert bank.sha256 == _sha(root)
    assert bank.patients == ["patient_a", "patient_b", "patient_c"]
    # BG and class 1 have support in all three patients; class 2 only has
    # support in two patients; class 3 is unsupported and is not admitted.
    assert bank.class_patients == [0, 1, 2, 0]
    assert bank.allowed_classes == [False, False, True, False]
    assert (
        bank.manifest["source_sha256"]["core.py"]
        == REVIEWED_CARDINAL_PRODUCER_HASHES["core.py"]
    )
    assert bank.reverify()["role"] == "guidance_bank"
    assert (
        bank.orientation_contract["basis"]
        == "historical_schema_pinned_producer_without_policy_field"
    )
    assert frozen["schema"] == "self_audit.stable_guidance.v1"


def _sha(root: Path) -> str:
    import hashlib

    return hashlib.sha256((root / "FROZEN.json").read_bytes()).hexdigest()


def test_load_patient_normalizes_and_registers_native_teacher_geometry(tmp_path):
    root, _, _ = _make_bank(tmp_path)
    bank = FrozenTrainingBank(root, _sha(root))
    patient = bank.load_patient("patient_a")

    assert patient["policy"] == POLICY
    assert patient["image"].shape == (2, 2, 17, 19)
    assert patient["features"].shape == (2, 2, 3, 4, 64)
    assert patient["normalized_cine"].shape == (2, 2, 17, 19)
    assert (
        0 <= patient["normalized_cine"].min() <= patient["normalized_cine"].max() <= 1
    )
    assert patient["teacher_features"].shape == (2, 2, 64, 3, 4)
    assert patient["geometry"].native_hw == (17, 19)
    assert patient["geometry"].teacher_hw == (3, 4)
    assert patient["geometry"].registration.source_manifest_sha256 == bank.sha256
    assert patient["normalization"]["scope"].startswith("one_image_all_frames")
    # The source labels remain evidence for the audit; the reader does not
    # silently turn UNKNOWN into BG.
    assert 255 in np.unique(patient["labels"])


def test_batch_uses_historical_reliability_and_allowed_argmax_without_labels(tmp_path):
    root, _, _ = _make_bank(tmp_path)
    bank = FrozenTrainingBank(root, _sha(root))
    patient = bank.load_patient("patient_a")
    batch = bank.batch(patient, [(0, 0), (1, 1)], device="cpu")

    assert batch["images"].shape == (2, 3, 17, 19)
    assert batch["teacher_features"].shape == (2, 64, 3, 4)
    assert batch["q"].shape == (2, 4, 3, 4)
    # patient_a's Q argmax is class 1, which lacks >=2 patient support in this
    # fixture, so its historical reliability is masked at the policy layer.
    assert torch.equal(batch["q_argmax"], torch.ones_like(batch["q_argmax"]))
    assert torch.count_nonzero(batch["weights"]) == 0
    assert batch["semantic_active"] is False
    assert batch["named_mass_status"] == "NOT_CONDITIONAL_Z"

    patient_c = bank.load_patient("patient_c")
    batch_c = bank.batch(patient_c, [(0, 0)], device="cpu")
    assert torch.all(batch_c["q_argmax"] == 2)
    assert torch.count_nonzero(batch_c["weights"]) > 0
    assert batch_c["semantic_active"] is True


def test_prepare_targets_transports_mass_support_and_feature_coverage(tmp_path):
    root, _, _ = _make_bank(tmp_path)
    bank = FrozenTrainingBank(root, _sha(root))
    patient = bank.load_patient("patient_c")
    batch = bank.batch(patient, [(0, 0), (1, 1)], device="cpu")
    prepared = prepare_targets(batch)

    assert prepared["policy"] == POLICY
    assert prepared["named_mass_status"] == "NOT_CONDITIONAL_Z"
    assert prepared["target"].shape == (2, 4, 8, 8)
    assert prepared["weights"].shape == (2, 8, 8)
    assert prepared["valid"].any()
    assert torch.isfinite(prepared["target"]).all()
    torch.testing.assert_close(
        prepared["target"].sum(dim=1)[prepared["valid"]],
        torch.ones_like(prepared["weights"])[prepared["valid"]],
    )
    # Partially covered right/bottom cells carry fractional positive support;
    # padded cells carry none.  They are not converted into background.
    assert ((prepared["weights"] > 0) & (prepared["weights"] < 0.5)).any()
    assert (prepared["weights"] == 0).any()
    invalid_coarse = (~prepared["valid"])[:, None].expand_as(prepared["target"])
    assert torch.all(prepared["target"][invalid_coarse] == 0.25)

    assert prepared["feature_target"].shape == (2, 64, 2, 2)
    assert prepared["feature_valid"].shape == (2, 2, 2)
    assert torch.all(prepared["feature_valid"][:, 0, 0])
    assert not prepared["feature_valid"][:, 1:, :].any()
    invalid_feature = (~prepared["feature_valid"])[:, None].expand_as(
        prepared["feature_target"]
    )
    assert torch.all(prepared["feature_target"][invalid_feature] == 0)
    assert torch.isfinite(prepared["feature_target"]).all()


def test_prepare_targets_zero_support_is_safe_without_minimum_cutoff(tmp_path):
    root, _, _ = _make_bank(tmp_path)
    bank = FrozenTrainingBank(root, _sha(root))
    patient = bank.load_patient("patient_c")
    batch = bank.batch(patient, [(0, 0)], device="cpu")
    batch["weights"].zero_()
    prepared = prepare_targets(batch)

    assert not prepared["valid"].any()
    assert not prepared["weights"].any()
    assert torch.equal(
        prepared["target"],
        torch.full_like(prepared["target"], 0.25),
    )


@pytest.mark.parametrize(
    "bad_splits",
    [
        {"train": ["patient_a", "patient_a"], "val": ["patient_val"], "test": []},
        {"train": ["patient_a"], "val": ["patient_a"], "test": ["patient_test"]},
        {"train": ["../patient_a"], "val": ["patient_val"], "test": ["patient_test"]},
    ],
)
def test_init_rejects_duplicate_leaking_or_unsafe_splits(tmp_path, bad_splits):
    with pytest.raises(ValueError, match="patient|TRAIN"):
        _make_bank(tmp_path, locked_splits=bad_splits, patients=["patient_a"])
        FrozenTrainingBank(tmp_path, _sha(tmp_path))


def test_patient_payload_mutation_is_rejected_and_final_reverify_catches_manifest_change(
    tmp_path,
):
    root, _, _ = _make_bank(tmp_path)
    bank = FrozenTrainingBank(root, _sha(root))
    payload = root / "patient_a.npz"
    payload.write_bytes(payload.read_bytes() + b"mutation")
    with pytest.raises(ValueError, match="payload hash mismatch"):
        bank.load_patient("patient_a")

    # Use a fresh fixture for the manifest mutation because the payload above
    # is intentionally unrecoverable within this test.
    root2 = tmp_path / "second"
    root2.mkdir()
    root2, _, _ = _make_bank(root2)
    bank2 = FrozenTrainingBank(root2, _sha(root2))
    frozen = root2 / "FROZEN.json"
    manifest = json.loads(frozen.read_text())
    manifest["status"] = "mutated-after-init"
    frozen.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest changed"):
        bank2.reverify()


def test_producer_and_teacher_contracts_are_required(tmp_path):
    root, _, _ = _make_bank(tmp_path)
    manifest_path = root / "FROZEN.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_sha256"]["core.py"] = "0" * 64
    # FROZEN is immutable by contract; construct a new frozen fixture instead
    # of rewriting it in place.
    root2 = tmp_path / "bad"
    root2.mkdir()
    _make_bank(root2)
    bad_manifest_path = root2 / "FROZEN.json"
    bad = json.loads(bad_manifest_path.read_text())
    bad["source_sha256"]["core.py"] = "0" * 64
    bad_manifest_path.unlink()
    bad_manifest_path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError):
        FrozenTrainingBank(root2, _sha(root2))


def test_published_bank_provenance_is_compatible_without_invented_fields():
    repo = Path(__file__).resolve().parents[3]
    manifest = json.loads(
        (
            repo
            / "reports/stable_guidance_main513ddda_b12_20261016/completed_run/run/bank/FROZEN.json"
        ).read_text()
    )
    assert "orientation_policy" not in manifest
    assert "source" not in manifest["teacher"]
    FrozenTrainingBank._validate_provenance(manifest)
    manifest["teacher"]["provider"] = ""
    with pytest.raises(ValueError, match="teacher needs"):
        FrozenTrainingBank._validate_provenance(manifest)


def test_conflicting_orientation_is_rejected(tmp_path):
    _, _, manifest = _make_bank(tmp_path)
    manifest["orientation_policy"] = "world_left_v2"
    with pytest.raises(ValueError, match="cardinal_v1"):
        FrozenTrainingBank._validate_provenance(manifest)
