"""Wave 2.1 focused regression tests for authoritative split resolution and safety."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

try:
    from self_audit.data.acdc import (
        ACDCDataset,
        EffectiveSplits,
        discover_acdc_records,
        resolve_acdc_records,
        resolve_effective_acdc_splits,
    )
    from self_audit.data.common import (
        VolumeRecord,
        compute_split_signature,
        read_split_manifest,
        validate_patient_split,
    )
    from self_audit.training._utils import (
        build_data_loader,
        build_patient_dataset,
        validate_dataset_splits,
    )
except ImportError:
    from src.self_audit.data.acdc import (
        ACDCDataset,
        EffectiveSplits,
        discover_acdc_records,
        resolve_acdc_records,
        resolve_effective_acdc_splits,
    )
    from src.self_audit.data.common import (
        VolumeRecord,
        compute_split_signature,
        read_split_manifest,
        validate_patient_split,
    )
    from src.self_audit.training._utils import (
        build_data_loader,
        build_patient_dataset,
        validate_dataset_splits,
    )


def _make_npy_pair(root: Path, case_id: str, split_dir: str | None = None) -> tuple[Path, Path]:
    target_dir = root if split_dir is None else root / split_dir
    (target_dir / "volumes").mkdir(parents=True, exist_ok=True)
    (target_dir / "masks").mkdir(parents=True, exist_ok=True)
    vol_path = target_dir / "volumes" / f"{case_id}.npy"
    mask_path = target_dir / "masks" / f"{case_id}.npy"
    volume = np.arange(4 * 8 * 8, dtype=np.float32).reshape(4, 8, 8)
    mask = np.zeros_like(volume, dtype=np.int64)
    mask[1:3, 2:6, 2:6] = 1
    np.save(vol_path, volume)
    np.save(mask_path, mask)
    return vol_path, mask_path


# ==============================================================================
# 1. Controlled ED-train / ES-validation same patient fails before model training
# ==============================================================================


def test_controlled_ed_train_es_val_leakage_manifest_fails_before_training(tmp_path: Path) -> None:
    _make_npy_pair(tmp_path, "patient001_ED")
    _make_npy_pair(tmp_path, "patient001_ES")
    _make_npy_pair(tmp_path, "patient002_ED")

    manifest = tmp_path / "leak_manifest.json"
    manifest.write_text(
        json.dumps({
            "train_cases": ["patient001_ED"],
            "val_cases": ["patient001_ES", "patient002_ED"],
        }),
        encoding="utf-8",
    )

    config = {
        "dataset": "acdc",
        "data_root": str(tmp_path),
        "split_manifest": str(manifest),
        "depth_axis": 0,
    }
    with pytest.raises(ValueError, match="Patient leakage"):
        validate_dataset_splits(config)


def test_controlled_ed_train_es_val_leakage_tags_fail_before_training(tmp_path: Path) -> None:
    _make_npy_pair(tmp_path, "patient001_ED", split_dir="training")
    _make_npy_pair(tmp_path, "patient001_ES", split_dir="validation")
    _make_npy_pair(tmp_path, "patient002_ED", split_dir="validation")

    config = {
        "dataset": "acdc",
        "data_root": str(tmp_path),
        "depth_axis": 0,
    }
    with pytest.raises(ValueError, match="Patient leakage"):
        validate_dataset_splits(config)


def test_controlled_ed_train_es_val_astra_probe_tag_vs_manifest_conflict(tmp_path: Path) -> None:
    """Astra's probe: directory tags have ED in training and ES in validation.

    Manifest attempts to place both in train. The conflict on ES must fail loudly.
    """
    _make_npy_pair(tmp_path, "patient001_ED", split_dir="training")
    _make_npy_pair(tmp_path, "patient001_ES", split_dir="validation")
    _make_npy_pair(tmp_path, "patient002_ED", split_dir="validation")

    manifest = tmp_path / "probe_manifest.json"
    manifest.write_text(
        json.dumps({
            "train_cases": ["patient001_ED", "patient001_ES"],
            "val_cases": ["patient002_ED"],
        }),
        encoding="utf-8",
    )

    config = {
        "dataset": "acdc",
        "data_root": str(tmp_path),
        "split_manifest": str(manifest),
        "depth_axis": 0,
    }
    with pytest.raises(ValueError, match="Manifest-vs-tag conflict"):
        validate_dataset_splits(config)


# ==============================================================================
# 2. Validator effective records == actual ACDCDataset/DataLoader records
# ==============================================================================


def test_validator_and_loader_parity_with_manifest(tmp_path: Path) -> None:
    for cid in ("patient001_ED", "patient001_ES", "patient002_ED", "patient003_ED"):
        _make_npy_pair(tmp_path, cid)

    manifest = tmp_path / "valid_manifest.json"
    manifest.write_text(
        json.dumps({
            "train_cases": ["patient001_ED", "patient001_ES"],
            "val_cases": ["patient002_ED"],
            "test_cases": ["patient003_ED"],
        }),
        encoding="utf-8",
    )

    config = {
        "dataset": "acdc",
        "data_root": str(tmp_path),
        "split_manifest": str(manifest),
        "depth_axis": 0,
        "image_size": 8,
        "batch_size": 2,
        "num_workers": 0,
    }

    val_res = validate_dataset_splits(config)
    assert val_res["validated"] is True
    assert val_res["strategy"] == "manifest"
    assert val_res["test_available"] is True

    # Loop all configured splits including test, using build_patient_dataset and build_data_loader
    for split in ("train", "val", "test"):
        is_train = (split == "train")
        ds = build_patient_dataset(config, split=split, train=is_train)
        loader = build_data_loader(ds, config, device=torch.device("cpu"), train=is_train)

        expected_ids = val_res["effective_identities"][split]
        assert [r.case_id for r in ds.records] == expected_ids
        assert ds.split_signature == val_res["split_signature"]

        expected_records = val_res["effective_records"][split]
        assert len(ds.records) == len(expected_records)
        for rec, exp in zip(ds.records, expected_records):
            assert rec.case_id == exp["case_id"]
            assert str(rec.image_path) == exp["image_path"]
            assert str(rec.mask_path) == exp["mask_path"]

        # Verify every batch case (not batch[0] only) strictly belongs to this split
        seen_cases: list[str] = []
        for batch in loader:
            batch_cases = batch["case_id"]
            if isinstance(batch_cases, (list, tuple)):
                for cid in batch_cases:
                    assert cid in expected_ids
                    seen_cases.append(cid)
            else:
                assert str(batch_cases) in expected_ids
                seen_cases.append(str(batch_cases))
        assert set(seen_cases) == set(expected_ids)


def test_validator_and_loader_parity_tag_only(tmp_path: Path) -> None:
    _make_npy_pair(tmp_path, "patient001_ED", split_dir="training")
    _make_npy_pair(tmp_path, "patient001_ES", split_dir="training")
    _make_npy_pair(tmp_path, "patient002_ED", split_dir="validation")

    config = {
        "dataset": "acdc",
        "data_root": str(tmp_path),
        "depth_axis": 0,
    }

    val_res = validate_dataset_splits(config)
    assert val_res["strategy"] == "explicit_tags"
    assert val_res["effective_identities"]["train"] == ["patient001_ED", "patient001_ES"]
    assert val_res["effective_identities"]["val"] == ["patient002_ED"]

    ds_train = ACDCDataset(tmp_path, split="train", depth_axis=0, image_size=8)
    ds_val = ACDCDataset(tmp_path, split="val", depth_axis=0, image_size=8)

    assert [r.case_id for r in ds_train.records] == val_res["effective_identities"]["train"]
    assert [r.case_id for r in ds_val.records] == val_res["effective_identities"]["val"]
    assert ds_train.split_signature == val_res["split_signature"]


def test_validator_and_loader_parity_untagged_deterministic_fallback(tmp_path: Path) -> None:
    for cid in ("patient001_ED", "patient002_ED", "patient003_ED"):
        _make_npy_pair(tmp_path, cid)

    config = {
        "dataset": "acdc",
        "data_root": str(tmp_path),
        "depth_axis": 0,
        "seed": 42,
    }

    val_res = validate_dataset_splits(config)
    assert val_res["strategy"] == "patient_level_fallback"
    assert val_res["test_available"] is True

    ds_train = ACDCDataset(tmp_path, split="train", depth_axis=0, image_size=8, seed=42)
    assert [r.case_id for r in ds_train.records] == val_res["effective_identities"]["train"]
    assert ds_train.split_signature == val_res["split_signature"]


# ==============================================================================
# 3. Missing/extra manifest entries, duplicate list entries and real duplicate files fail
# ==============================================================================


def test_manifest_missing_discovered_cases_rejected(tmp_path: Path) -> None:
    _make_npy_pair(tmp_path, "patient001_ED")
    _make_npy_pair(tmp_path, "patient002_ED")

    # Manifest lists patient999_ED which does not exist
    manifest = tmp_path / "missing_cases.json"
    manifest.write_text(
        json.dumps({
            "train_cases": ["patient001_ED", "patient999_ED"],
            "val_cases": ["patient002_ED"],
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match"):
        ACDCDataset(tmp_path, split="train", split_manifest=manifest, image_size=8)


def test_manifest_extra_discovered_cases_rejected(tmp_path: Path) -> None:
    _make_npy_pair(tmp_path, "patient001_ED")
    _make_npy_pair(tmp_path, "patient002_ED")
    _make_npy_pair(tmp_path, "patient003_ED")

    # Manifest only lists patient001 and patient002; patient003 is omitted
    manifest = tmp_path / "extra_cases.json"
    manifest.write_text(
        json.dumps({
            "train_cases": ["patient001_ED"],
            "val_cases": ["patient002_ED"],
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match"):
        ACDCDataset(tmp_path, split="train", split_manifest=manifest, image_size=8)


def test_manifest_duplicate_list_entries_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "dup_list.json"
    manifest.write_text(
        json.dumps({
            "train_cases": ["patient001_ED", "patient001_ED"],
            "val_cases": ["patient002_ED"],
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate case"):
        read_split_manifest(manifest)


def test_manifest_duplicate_across_splits_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "dup_cross.json"
    manifest.write_text(
        json.dumps({
            "train_cases": ["patient001_ED"],
            "val_cases": ["patient001_ED"],
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate case"):
        read_split_manifest(manifest)


def test_real_duplicate_files_collision_rejected(tmp_path: Path) -> None:
    # Place case1 in training/volumes/ and distinct file in validation/volumes/
    _make_npy_pair(tmp_path, "case1", split_dir="training")
    _make_npy_pair(tmp_path, "case1", split_dir="validation")

    with pytest.raises(ValueError, match="duplicate case"):
        discover_acdc_records(tmp_path)


def test_real_duplicate_nifti_files_collision_rejected(tmp_path: Path) -> None:
    # Distinct NIfTI files with identical case_id in two subdirectories
    dir_a = tmp_path / "folder_a"
    dir_b = tmp_path / "folder_b"
    dir_a.mkdir()
    dir_b.mkdir()
    (dir_a / "patient001_frame01.nii.gz").touch()
    (dir_a / "patient001_frame01_gt.nii.gz").touch()
    (dir_b / "patient001_frame01.nii.gz").touch()
    (dir_b / "patient001_frame01_gt.nii.gz").touch()

    with pytest.raises(ValueError, match="duplicate case"):
        discover_acdc_records(tmp_path)


# ==============================================================================
# 4. Manifest-vs-tag conflict and mixed tagged/untagged ambiguity fail clearly
# ==============================================================================


def test_manifest_vs_tag_conflict_fails_clearly(tmp_path: Path) -> None:
    _make_npy_pair(tmp_path, "patient001_ED", split_dir="training")
    _make_npy_pair(tmp_path, "patient002_ED", split_dir="training")

    # Manifest claims patient002_ED is val, conflicting with training directory tag
    manifest = tmp_path / "conflict.json"
    manifest.write_text(
        json.dumps({
            "train_cases": ["patient001_ED"],
            "val_cases": ["patient002_ED"],
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Manifest-vs-tag conflict"):
        ACDCDataset(tmp_path, split="train", split_manifest=manifest, image_size=8)


def test_mixed_tagged_and_untagged_ambiguity_fails(tmp_path: Path) -> None:
    # Tagged case in training/
    _make_npy_pair(tmp_path, "patient001_ED", split_dir="training")
    # Untagged case in root
    _make_npy_pair(tmp_path, "patient002_ED")

    with pytest.raises(ValueError, match="Mixed tagged and untagged"):
        ACDCDataset(tmp_path, split="train", image_size=8)


# ==============================================================================
# 5. Optional test absent accepted; configured train/val missing rejected
# ==============================================================================


def test_optional_test_absent_accepted(tmp_path: Path) -> None:
    _make_npy_pair(tmp_path, "patient001_ED")
    _make_npy_pair(tmp_path, "patient002_ED")

    manifest = tmp_path / "no_test.json"
    manifest.write_text(
        json.dumps({
            "train_cases": ["patient001_ED"],
            "val_cases": ["patient002_ED"],
        }),
        encoding="utf-8",
    )

    config = {
        "dataset": "acdc",
        "data_root": str(tmp_path),
        "split_manifest": str(manifest),
        "depth_axis": 0,
    }
    res = validate_dataset_splits(config)
    assert res["validated"] is True
    assert res["test_available"] is False
    assert "test" not in res["effective_identities"]


def test_missing_required_train_or_val_rejected(tmp_path: Path) -> None:
    _make_npy_pair(tmp_path, "patient001_ED")

    # Manifest has train but no val
    manifest = tmp_path / "no_val.json"
    manifest.write_text(
        json.dumps({
            "train_cases": ["patient001_ED"],
        }),
        encoding="utf-8",
    )

    config = {
        "dataset": "acdc",
        "data_root": str(tmp_path),
        "split_manifest": str(manifest),
        "depth_axis": 0,
    }
    with pytest.raises(ValueError, match="missing required 'val' split"):
        validate_dataset_splits(config)


def test_configured_aliases_respected(tmp_path: Path) -> None:
    _make_npy_pair(tmp_path, "patient001_ED")
    _make_npy_pair(tmp_path, "patient002_ED")

    manifest = tmp_path / "aliases.json"
    manifest.write_text(
        json.dumps({
            "training_cases": ["patient001_ED"],
            "validation_cases": ["patient002_ED"],
        }),
        encoding="utf-8",
    )

    config = {
        "dataset": "acdc",
        "data_root": str(tmp_path),
        "split_manifest": str(manifest),
        "train_split": "training",
        "val_split": "validation",
        "depth_axis": 0,
    }
    res = validate_dataset_splits(config)
    assert res["validated"] is True


def test_unsupported_alias_rejected(tmp_path: Path) -> None:
    _make_npy_pair(tmp_path, "patient001_ED")
    _make_npy_pair(tmp_path, "patient002_ED")

    manifest = tmp_path / "valid.json"
    manifest.write_text(
        json.dumps({
            "train_cases": ["patient001_ED"],
            "val_cases": ["patient002_ED"],
        }),
        encoding="utf-8",
    )

    config = {
        "dataset": "acdc",
        "data_root": str(tmp_path),
        "split_manifest": str(manifest),
        "train_split": "invalid_alias",
        "depth_axis": 0,
    }
    with pytest.raises(ValueError, match="Unsupported or ambiguous split alias"):
        validate_dataset_splits(config)


# ==============================================================================
# 6. Discovery order does not change record identities or signature
# ==============================================================================


def test_raw_nifti_recursive_traversal_deduplication(tmp_path: Path) -> None:
    """Raw NIfTI root.rglob plus split root discovers identical file twice."""
    train_patient_dir = tmp_path / "training" / "patient001"
    train_patient_dir.mkdir(parents=True)
    img_path = train_patient_dir / "patient001_frame01.nii.gz"
    mask_path = train_patient_dir / "patient001_frame01_gt.nii.gz"
    img_path.touch()
    mask_path.touch()

    records = discover_acdc_records(tmp_path)
    assert len(records) == 1
    rec = records[0]
    assert rec.case_id == "patient001_frame01"
    assert rec.split == "train"


def test_discovery_order_invariance_signature() -> None:
    rec1 = VolumeRecord("patient001_ED", "patient001", Path("/a/1.npy"), Path("/b/1.npy"), "train")
    rec2 = VolumeRecord("patient001_ES", "patient001", Path("/a/2.npy"), Path("/b/2.npy"), "train")
    rec3 = VolumeRecord("patient002_ED", "patient002", Path("/a/3.npy"), Path("/b/3.npy"), "val")

    splits_order1 = {"train": [rec1, rec2], "val": [rec3]}
    splits_order2 = {"val": [rec3], "train": [rec2, rec1]}

    sig1 = compute_split_signature(splits_order1)
    sig2 = compute_split_signature(splits_order2)
    assert sig1 == sig2


# ==============================================================================
# 7. Coordinator review points regressions
# ==============================================================================


def test_tracked_manifest_parse_regression() -> None:
    """Tracked manifest splits/acdc_patient_split_seed42.json must parse cleanly."""
    manifest_path = Path("splits/acdc_patient_split_seed42.json")
    assert manifest_path.is_file()
    parsed = read_split_manifest(manifest_path)
    assert len(parsed["train"]) == 160
    assert len(parsed["val"]) == 40


def test_manifest_redundant_identical_representations_accepted(tmp_path: Path) -> None:
    manifest = tmp_path / "redundant.json"
    manifest.write_text(
        json.dumps({
            "train_cases": ["patient001_ED"],
            "train_volumes": ["patient001_ED"],
            "val_cases": ["patient002_ED"],
        }),
        encoding="utf-8",
    )
    parsed = read_split_manifest(manifest)
    assert parsed["train"] == ["patient001_ED"]
    assert parsed["val"] == ["patient002_ED"]


def test_manifest_conflicting_representations_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "conflicting.json"
    manifest.write_text(
        json.dumps({
            "train_cases": ["patient001_ED"],
            "train_volumes": ["patient001_ES"],
            "val_cases": ["patient002_ED"],
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Conflicting memberships"):
        read_split_manifest(manifest)


def test_paired_files_stem_collision_rejected(tmp_path: Path) -> None:
    (tmp_path / "volumes").mkdir()
    (tmp_path / "masks").mkdir()
    (tmp_path / "volumes" / "case1.npy").touch()
    (tmp_path / "volumes" / "case1.npz").touch()
    (tmp_path / "masks" / "case1.npy").touch()

    with pytest.raises(ValueError, match="Collision"):
        discover_acdc_records(tmp_path)


def test_validator_result_is_strictly_json_serializable(tmp_path: Path) -> None:
    for cid in ("patient001_ED", "patient002_ED", "patient003_ED"):
        _make_npy_pair(tmp_path, cid)

    manifest = tmp_path / "split.json"
    manifest.write_text(
        json.dumps({
            "train_cases": ["patient001_ED"],
            "val_cases": ["patient002_ED"],
            "test_cases": ["patient003_ED"],
        }),
        encoding="utf-8",
    )

    config = {
        "dataset": "acdc",
        "data_root": str(tmp_path),
        "split_manifest": str(manifest),
        "depth_axis": 0,
    }
    val_res = validate_dataset_splits(config)
    serialized = json.dumps(val_res)
    assert isinstance(serialized, str)

    # Parity check on exact paths
    ds_train = ACDCDataset(tmp_path, split="train", split_manifest=manifest, depth_axis=0, image_size=8)
    descriptor = val_res["effective_records"]["train"][0]
    record = ds_train.records[0]
    assert descriptor["case_id"] == record.case_id
    assert descriptor["image_path"] == str(record.image_path)
    assert descriptor["mask_path"] == str(record.mask_path)


def test_manifest_unrecognized_split_keys_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "bad_keys.json"
    manifest.write_text(
        json.dumps({
            "train_cases": ["patient001_ED"],
            "val_cases": ["patient002_ED"],
            "eval_cases": ["patient003_ED"],
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unrecognized split key"):
        read_split_manifest(manifest)

    manifest_custom = tmp_path / "custom_key.json"
    manifest_custom.write_text(
        json.dumps({
            "train": ["patient001_ED"],
            "val": ["patient002_ED"],
            "holdout": ["patient003_ED"],
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unrecognized split key"):
        read_split_manifest(manifest_custom)


