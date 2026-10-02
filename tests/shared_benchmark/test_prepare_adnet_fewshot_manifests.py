from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from prepare_adnet_fewshot_manifests import prepare_manifests
from shared_benchmark.adnet_fewshot import (
    ADNetContractError,
    load_gt_manifest,
    load_query_manifest,
    load_support_manifest,
)


def _write_npy(path: Path, array: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array, allow_pickle=False)
    return path


def _write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _asset_spec(tmp_path: Path) -> Path:
    assets = tmp_path / "assets"
    _write_npy(assets / "query" / "case001.npy", np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8))
    _write_npy(assets / "support" / "support001.npy", np.ones((3, 8, 8), dtype=np.float32))
    mask = np.zeros((3, 8, 8), dtype=np.uint8)
    mask[1, 2:6, 2:6] = 1
    _write_npy(assets / "support" / "support001_rv.npy", mask)
    gt = np.zeros((3, 8, 8), dtype=np.uint8)
    gt[1, 2:6, 2:6] = 1
    _write_npy(assets / "gt" / "case001.npy", gt)
    return _write_json(
        tmp_path / "adnet_assets.json",
        {
            "schema": "adnet.asset_spec.v1",
            "class_mapping": {"RV": 1},
            "queries": [
                {
                    "sample_id": "case001",
                    "image_path": "query/case001.npy",
                    "split": "dev",
                    "metadata": {"patient_id": "case001"},
                }
            ],
            "supports": [
                {
                    "support_id": "support_rv",
                    "class_name": "RV",
                    "canonical_class_id": 1,
                    "image_path": "support/support001.npy",
                    "mask_path": "support/support001_rv.npy",
                    "slice_indices": [1],
                }
            ],
            "gt": [{"sample_id": "case001", "gt_path": "gt/case001.npy"}],
        },
    )


def test_prepare_adnet_manifests_writes_valid_absolute_paths(tmp_path: Path):
    spec = _asset_spec(tmp_path)
    receipt = prepare_manifests(
        spec_path=spec,
        output_dir=tmp_path / "manifests",
        asset_root=tmp_path / "assets",
    )
    assert receipt["schema"] == "adnet.prepare_receipt.v1"
    assert receipt["query_count"] == 1
    assert receipt["support_count"] == 1
    assert receipt["gt_count"] == 1
    query = load_query_manifest(tmp_path / "manifests" / "query_manifest.json")
    support = load_support_manifest(tmp_path / "manifests" / "support_manifest.json")
    gt = load_gt_manifest(tmp_path / "manifests" / "gt_manifest.json")
    assert query.records[0].sample_id == "case001"
    assert support.supports[0].class_name == "RV"
    assert gt.records[0].sample_id == "case001"


def test_prepare_adnet_manifests_can_emit_relative_paths(tmp_path: Path):
    spec = _asset_spec(tmp_path)
    prepare_manifests(
        spec_path=spec,
        output_dir=tmp_path / "manifests",
        asset_root=tmp_path / "assets",
        path_style="relative-to-asset-root",
    )
    document = json.loads((tmp_path / "manifests" / "query_manifest.json").read_text(encoding="utf-8"))
    assert document["records"][0]["image_path"] == "query/case001.npy"
    load_query_manifest(tmp_path / "manifests" / "query_manifest.json", image_root=tmp_path / "assets")
    load_support_manifest(tmp_path / "manifests" / "support_manifest.json", asset_root=tmp_path / "assets")
    load_gt_manifest(tmp_path / "manifests" / "gt_manifest.json", gt_root=tmp_path / "assets")


def test_prepare_adnet_manifests_refuses_overwrite_without_force(tmp_path: Path):
    spec = _asset_spec(tmp_path)
    prepare_manifests(spec_path=spec, output_dir=tmp_path / "manifests", asset_root=tmp_path / "assets")
    with pytest.raises(ADNetContractError, match="refusing to overwrite"):
        prepare_manifests(spec_path=spec, output_dir=tmp_path / "manifests", asset_root=tmp_path / "assets")


def test_prepare_adnet_manifests_rejects_support_mapping_mismatch(tmp_path: Path):
    spec = _asset_spec(tmp_path)
    document = json.loads(spec.read_text(encoding="utf-8"))
    document["supports"][0]["canonical_class_id"] = 2
    broken = _write_json(tmp_path / "broken.json", document)
    with pytest.raises(ADNetContractError, match="disagrees with class_mapping"):
        prepare_manifests(spec_path=broken, output_dir=tmp_path / "manifests", asset_root=tmp_path / "assets")
