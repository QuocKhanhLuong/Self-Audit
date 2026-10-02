from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import sys

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from preflight_adnet_fewshot import preflight
from shared_benchmark.adnet_fewshot import ADNetContractError, file_sha256


def _write_npy(path: Path, array: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array, allow_pickle=False)
    return path


def _write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _fixture(tmp_path: Path):
    query = _write_npy(tmp_path / "query.npy", np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8))
    support = _write_npy(tmp_path / "support.npy", (np.arange(3 * 8 * 8, dtype=np.float32) + 1).reshape(3, 8, 8))
    mask = np.zeros((3, 8, 8), dtype=np.uint8)
    mask[1, 2:6, 2:6] = 1
    support_mask = _write_npy(tmp_path / "support_mask.npy", mask)
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"fixture-checkpoint")
    query_manifest = _write_json(
        tmp_path / "query_manifest.json",
        {
            "schema": "adnet.query_manifest.v1",
            "records": [
                {
                    "sample_id": "case001",
                    "image_path": query.name,
                    "image_sha256": file_sha256(query),
                    "split": "dev",
                    "metadata": {},
                }
            ],
        },
    )
    support_manifest = _write_json(
        tmp_path / "support_manifest.json",
        {
            "schema": "adnet.support_manifest.v1",
            "class_mapping": {"RV": 1},
            "supports": [
                {
                    "support_id": "support_rv",
                    "class_name": "RV",
                    "canonical_class_id": 1,
                    "image_path": support.name,
                    "image_sha256": file_sha256(support),
                    "mask_path": support_mask.name,
                    "mask_sha256": file_sha256(support_mask),
                    "slice_indices": [1],
                }
            ],
        },
    )
    return query_manifest, support_manifest, checkpoint


def test_adnet_preflight_ready_for_valid_fixture(tmp_path: Path):
    query_manifest, support_manifest, checkpoint = _fixture(tmp_path)
    result = preflight(
        query_manifest_path=query_manifest,
        support_manifest_path=support_manifest,
        checkpoint=checkpoint,
        output_root=tmp_path / "out",
        required_classes=(1,),
    )
    assert result["status"] == "READY"
    assert result["query_count"] == 1
    assert result["support_count"] == 1


def test_adnet_preflight_requires_declared_classes(tmp_path: Path):
    query_manifest, support_manifest, checkpoint = _fixture(tmp_path)
    with pytest.raises(ADNetContractError, match="missing required classes"):
        preflight(
            query_manifest_path=query_manifest,
            support_manifest_path=support_manifest,
            checkpoint=checkpoint,
            output_root=tmp_path / "out",
            required_classes=(1, 2, 3),
        )


def test_adnet_preflight_rejects_non_empty_output_root(tmp_path: Path):
    query_manifest, support_manifest, checkpoint = _fixture(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    (output / "existing.txt").write_text("old")
    with pytest.raises(ADNetContractError, match="output root is not empty"):
        preflight(
            query_manifest_path=query_manifest,
            support_manifest_path=support_manifest,
            checkpoint=checkpoint,
            output_root=output,
            required_classes=(1,),
        )
