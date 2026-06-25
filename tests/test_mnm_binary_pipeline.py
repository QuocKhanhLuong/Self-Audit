from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.prepare_mnm_binary import prepare_mnm_binary_dataset
from src.training.train_s3r_acdc import apply_config_defaults, make_loaders


def _write_case(root: Path, split: str, case_id: str, fill: float = 0.5) -> None:
    volumes = root / split / "volumes"
    masks = root / split / "masks"
    volumes.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    volume = np.full((4, 4, 2), fill, dtype=np.float32)
    mask = np.zeros((4, 4, 2), dtype=np.uint8)
    mask[1:3, 1:3, 0] = 1
    mask[2:4, 2:4, 0] = 2
    mask[0:2, 2:4, 1] = 3
    np.save(volumes / f"{case_id}.npy", volume)
    np.save(masks / f"{case_id}.npy", mask)


def test_prepare_mnm_binary_binarizes_masks_and_preserves_splits(tmp_path: Path) -> None:
    src = tmp_path / "mnm"
    dst = tmp_path / "mnm_binary"
    _write_case(src, "training", "train_case", fill=0.25)
    _write_case(src, "validation", "val_case", fill=0.50)
    _write_case(src, "testing", "test_case", fill=0.75)

    summary = prepare_mnm_binary_dataset(src, dst)

    assert summary["splits"]["training"]["processed"] == 1
    assert summary["splits"]["validation"]["processed"] == 1
    assert summary["splits"]["testing"]["processed"] == 1
    for split, case_id in (("training", "train_case"), ("validation", "val_case"), ("testing", "test_case")):
        out_volume = np.load(dst / split / "volumes" / f"{case_id}.npy")
        out_mask = np.load(dst / split / "masks" / f"{case_id}.npy")
        in_volume = np.load(src / split / "volumes" / f"{case_id}.npy")
        assert np.array_equal(out_volume, in_volume)
        assert sorted(np.unique(out_mask).tolist()) == [0, 1]
        assert out_mask.dtype == np.uint8


def test_make_loaders_uses_explicit_mnm_training_and_validation_folders(tmp_path: Path) -> None:
    root = tmp_path / "mnm_binary"
    _write_case(root, "training", "train_a", fill=0.25)
    _write_case(root, "training", "train_b", fill=0.30)
    _write_case(root, "validation", "val_a", fill=0.50)

    cfg = apply_config_defaults(
        {
            "dataset": "mnm",
            "data_root": str(root),
            "split_mode": "folders",
            "input_mode": "2d",
            "image_size": 4,
            "foreground_only": False,
            "batch_size": 1,
            "num_workers": 0,
            "num_classes": 2,
            "class_names": ["BG", "FG"],
            "device": "cpu",
        }
    )

    train_loader, val_loader, split_info = make_loaders(cfg)

    assert split_info["train_cases"] == ["train_a", "train_b"]
    assert split_info["val_cases"] == ["val_a"]
    assert split_info["split_manifest"]["split_type"] == "folder"
    assert len(train_loader.dataset.case_ids) == 2
    assert len(val_loader.dataset.case_ids) == 1
