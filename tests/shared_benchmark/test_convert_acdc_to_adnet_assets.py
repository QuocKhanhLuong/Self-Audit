from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from convert_acdc_to_adnet_assets import convert_acdc_to_adnet_assets
from prepare_adnet_fewshot_manifests import prepare_manifests
from preflight_adnet_fewshot import preflight
from shared_benchmark.adnet_fewshot import load_gt_manifest, load_query_manifest, load_support_manifest


def _write_case(root: Path, case_id: str, labels: tuple[int, ...]) -> None:
    yy, xx, zz = np.meshgrid(
        np.arange(12, dtype=np.float32),
        np.arange(10, dtype=np.float32),
        np.arange(4, dtype=np.float32),
        indexing="ij",
    )
    image = yy + (2.0 * xx) + (3.0 * zz)
    mask = np.zeros((12, 10, 4), dtype=np.uint8)
    for offset, label in enumerate(labels):
        mask[2 + offset : 5 + offset, 2 + offset : 5 + offset, 1 + offset % 2] = label
    (root / "volumes").mkdir(parents=True, exist_ok=True)
    (root / "masks").mkdir(parents=True, exist_ok=True)
    np.save(root / "volumes" / f"{case_id}.npy", image, allow_pickle=False)
    np.save(root / "masks" / f"{case_id}.npy", mask, allow_pickle=False)


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    data = tmp_path / "acdc"
    _write_case(data, "patient001_ED", (1, 2, 3))
    _write_case(data, "patient002_ED", (1, 2, 3))
    _write_case(data, "patient003_ED", (1, 2, 3))
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps(
            {
                "train": ["patient001_ED"],
                "val": ["patient002_ED"],
                "test": ["patient003_ED"],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return data, split


def test_convert_acdc_to_adnet_assets_writes_valid_asset_spec(tmp_path: Path):
    data, split = _fixture(tmp_path)
    receipt = convert_acdc_to_adnet_assets(
        data_root=data,
        output_root=tmp_path / "adnet_assets",
        split_manifest=split,
        target_size=16,
    )
    assert receipt["schema"] == "adnet.acdc_conversion_receipt.v1"
    assert receipt["canonical_labels"] == {"0": "BG", "1": "RV", "2": "MYO", "3": "LV"}

    spec = json.loads((tmp_path / "adnet_assets" / "adnet_assets.json").read_text(encoding="utf-8"))
    assert spec["schema"] == "adnet.asset_spec.v1"
    assert spec["class_mapping"] == {"LV": 3, "MYO": 2, "RV": 1}
    assert len(spec["supports"]) == 3
    assert len(spec["queries"]) == 1
    assert len(spec["gt"]) == 1
    assert "source_image_sha256" in spec["queries"][0]["metadata"]
    assert all("mask" not in key.lower() for key in spec["queries"][0]["metadata"])

    image = np.load(tmp_path / "adnet_assets" / spec["queries"][0]["image_path"], allow_pickle=False)
    gt = np.load(tmp_path / "adnet_assets" / spec["gt"][0]["gt_path"], allow_pickle=False)
    assert image.shape == (4, 16, 16)
    assert gt.shape == (4, 16, 16)
    assert image.dtype == np.float32
    assert set(int(value) for value in np.unique(gt)) <= {0, 1, 2, 3}


def test_converted_acdc_assets_prepare_and_preflight(tmp_path: Path):
    data, split = _fixture(tmp_path)
    asset_root = tmp_path / "adnet_assets"
    convert_acdc_to_adnet_assets(
        data_root=data,
        output_root=asset_root,
        split_manifest=split,
        target_size=16,
    )
    manifest_dir = tmp_path / "manifests"
    prepare_manifests(
        spec_path=asset_root / "adnet_assets.json",
        output_dir=manifest_dir,
        asset_root=asset_root,
        path_style="relative-to-asset-root",
    )
    query = load_query_manifest(manifest_dir / "query_manifest.json", image_root=asset_root)
    support = load_support_manifest(manifest_dir / "support_manifest.json", asset_root=asset_root)
    gt = load_gt_manifest(manifest_dir / "gt_manifest.json", gt_root=asset_root)
    assert query.records[0].sample_id == gt.records[0].sample_id
    assert {item.canonical_class_id for item in support.supports} == {1, 2, 3}

    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"placeholder checkpoint for preflight")
    result = preflight(
        query_manifest_path=manifest_dir / "query_manifest.json",
        support_manifest_path=manifest_dir / "support_manifest.json",
        checkpoint=checkpoint,
        output_root=tmp_path / "run",
        query_image_root=asset_root,
        support_asset_root=asset_root,
        required_classes=(1, 2, 3),
    )
    assert result["status"] == "READY"
