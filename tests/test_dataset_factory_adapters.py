from __future__ import annotations

import json
import sys
from pathlib import Path

SRC_ROOT = str(Path(__file__).resolve().parents[1] / "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from self_audit.training._utils import build_patient_dataset

from test_cmrxmotion_adapter import _write_archive


def test_training_factory_builds_cmrxmotion_dataset_without_changing_model_flow(tmp_path: Path) -> None:
    _write_archive(tmp_path, with_label=True)
    manifest = tmp_path / "split.json"
    manifest.write_text(
        json.dumps({"train_cases": ["P001-1-ED"], "val_cases": ["P001-2-ED"]}),
        encoding="utf-8",
    )
    config = {
        "dataset": "cmr_motion",
        "data_root": str(tmp_path),
        "split_manifest": str(manifest),
        "image_size": 4,
        "preprocessing": {"clipping_min": 0.5, "clipping_max": 99.5},
    }
    dataset = build_patient_dataset(config, split="train", train=True)
    sample = dataset[0]
    assert tuple(sample["image"].shape) == (3, 4, 4)
    assert tuple(sample["mask"].shape) == (4, 4)
    assert sample["dataset"] == "cmr_motion"
    assert sample["subject_id"] == "P001"

def test_new_dataset_preflight_rejects_subject_overlap(tmp_path: Path) -> None:
    from self_audit.training._utils import validate_dataset_splits

    _write_archive(tmp_path, with_label=True)
    manifest = tmp_path / "leaky.json"
    manifest.write_text(
        json.dumps({"train_cases": ["P001-1-ED"], "val_cases": ["P001-2-ED"]}),
        encoding="utf-8",
    )
    config = {
        "dataset": "cmr_motion",
        "data_root": str(tmp_path),
        "split_manifest": str(manifest),
        "test_split": None,
    }
    import pytest
    with pytest.raises(ValueError, match="multiple splits"):
        validate_dataset_splits(config)
