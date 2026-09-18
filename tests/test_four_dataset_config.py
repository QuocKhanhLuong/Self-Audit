from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

SRC_ROOT = str(Path(__file__).resolve().parents[1] / "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from self_audit.training.unified_config import load_unified_config, parse_unified_config


def _raw_config() -> dict:
    return yaml.safe_load(Path("configs/self_audit_full.yaml").read_text(encoding="utf-8"))


def _mixed_dataset() -> dict:
    return {
        "name": "mixed",
        "data_root": ".",
        "split_manifest": None,
        "train_split": "train",
        "val_split": "val",
        "test_split": "test",
        "image_size": 64,
        "depth_axis": 2,
        "num_classes": 4,
        "class_mapping": {0: 0, 1: 1, 2: 2, 3: 3},
        "preprocessing": {
            "clipping_min": 0.5,
            "clipping_max": 99.5,
            "foreground_only": False,
        },
        "dataloader": {
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "prefetch_factor": 1,
        },
        "representation": "paired_3d",
        "sampling_strategy": "subject_dataset_balanced",
        "sources": {
            "acdc": {"data_root": "data/ACDC", "split_manifest": "splits/acdc.json"},
            "mnms": {"data_root": "data/M&Ms", "split_manifest": "splits/mnms.json"},
            "cmr_multi": {"data_root": "data/CMR-MULTI", "zt_report": "reports/cmr_multi_zt_inference.csv"},
            "cmr_motion": {"data_root": "data/CMRxMotion", "allow_affine_mismatch": False},
        },
    }


def test_mixed_four_dataset_config_is_strictly_parsed_and_forwarded() -> None:
    raw = _raw_config()
    raw["dataset"] = _mixed_dataset()
    config = parse_unified_config(raw)
    assert config.dataset.name == "mixed"
    assert config.dataset.sampling_strategy == "subject_dataset_balanced"
    assert set(config.dataset.sources) == {"acdc", "mnms", "cmr_multi", "cmr_motion"}
    legacy = config.to_legacy_dataset_config()
    assert legacy["sampling_strategy"] == "subject_dataset_balanced"
    assert legacy["sources"]["cmr_multi"]["zt_report"].endswith("cmr_multi_zt_inference.csv")


def test_existing_single_dataset_config_remains_compatible() -> None:
    config = load_unified_config("configs/self_audit_full.yaml")
    assert config.dataset.name == "acdc"
    assert config.dataset.sources == {}
    assert config.dataset.sampling_strategy == "none"


def test_unknown_dataset_name_is_rejected() -> None:
    raw = _raw_config()
    raw["dataset"]["name"] = "unknown_dataset"
    with pytest.raises(ValueError, match="Unsupported dataset"):
        parse_unified_config(raw)
