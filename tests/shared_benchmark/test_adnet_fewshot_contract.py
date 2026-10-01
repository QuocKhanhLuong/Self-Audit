from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from shared_benchmark.adnet_fewshot import (
    ADNetContractError,
    evaluate_adnet_outputs,
    file_sha256,
    load_gt_manifest,
    load_query_manifest,
    load_support_manifest,
    merge_class_binary_predictions,
    seal_adnet_outputs,
)


def _write_npy(path: Path, array: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array, allow_pickle=False)
    return path


def _write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _fixtures(tmp_path: Path):
    image = _write_npy(tmp_path / "query.npy", np.arange(48, dtype=np.float32).reshape(3, 4, 4))
    support_image = _write_npy(tmp_path / "support.npy", np.ones((3, 4, 4), dtype=np.float32))
    support_mask = _write_npy(tmp_path / "support_mask.npy", np.pad(np.ones((1, 2, 2), dtype=np.uint8), ((1, 1), (1, 1), (1, 1))))
    query_manifest = _write_json(
        tmp_path / "query_manifest.json",
        {
            "schema": "adnet.query_manifest.v1",
            "records": [
                {
                    "sample_id": "patient001_frame0001",
                    "image_path": image.name,
                    "image_sha256": file_sha256(image),
                    "split": "dev",
                    "metadata": {"patient_id": "patient001"},
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
                    "support_id": "support_patient001_rv",
                    "class_name": "RV",
                    "canonical_class_id": 1,
                    "image_path": support_image.name,
                    "image_sha256": file_sha256(support_image),
                    "mask_path": support_mask.name,
                    "mask_sha256": file_sha256(support_mask),
                    "slice_indices": [1],
                }
            ],
        },
    )
    return query_manifest, support_manifest


def test_query_manifest_rejects_query_gt_or_mask_fields(tmp_path: Path):
    image = _write_npy(tmp_path / "query.npy", np.zeros((1, 4, 4), dtype=np.float32))
    manifest = _write_json(
        tmp_path / "query_manifest.json",
        {
            "schema": "adnet.query_manifest.v1",
            "records": [
                {
                    "sample_id": "case001",
                    "image_path": image.name,
                    "image_sha256": file_sha256(image),
                    "mask_path": "query_mask.npy",
                }
            ],
        },
    )
    with pytest.raises(ADNetContractError, match="forbidden query evidence"):
        load_query_manifest(manifest)


def test_support_manifest_binds_labeled_support_assets(tmp_path: Path):
    _query_manifest, support_manifest = _fixtures(tmp_path)
    loaded = load_support_manifest(support_manifest)
    assert loaded.class_mapping == {"RV": 1}
    assert loaded.supports[0].slice_indices == (1,)
    document = json.loads(support_manifest.read_text(encoding="utf-8"))
    document["supports"][0]["mask_sha256"] = "0" * 64
    broken = _write_json(tmp_path / "broken_support.json", document)
    with pytest.raises(ADNetContractError, match="support mask hash mismatch"):
        load_support_manifest(broken)


def test_class_merge_fails_closed_on_overlap():
    first = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    second = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    with pytest.raises(ADNetContractError, match="overlapping"):
        merge_class_binary_predictions({1: first, 3: second})


def test_seal_adnet_outputs_binds_support_query_and_class_predictions(tmp_path: Path):
    query_manifest_path, support_manifest_path = _fixtures(tmp_path)
    support = load_support_manifest(support_manifest_path)
    query = load_query_manifest(query_manifest_path)
    record = query.records[0]
    metadata = seal_adnet_outputs(
        tmp_path / "out",
        sample_id=record.sample_id,
        class_predictions={1: np.array([[0, 1], [0, 0]], dtype=np.uint8)},
        support_manifest=support,
        query_manifest=query,
        query_record=record,
        checkpoint_sha256="a" * 64,
        code_identity={"sha256": "code"},
        run_config={"seed": 2021},
    )
    assert metadata["completion_status"] == "SEMANTIC_COMPLETE"
    assert metadata["support_manifest_sha256"] == support.sha256
    assert metadata["query_manifest_sha256"] == query.sha256
    assert (tmp_path / "out" / record.sample_id / "semantic_map.npy").is_file()
    assert (tmp_path / "out" / record.sample_id / "metadata.json").is_file()


def test_gt_only_evaluator_reads_gt_after_sealed_output(tmp_path: Path):
    query_manifest_path, support_manifest_path = _fixtures(tmp_path)
    support = load_support_manifest(support_manifest_path)
    query = load_query_manifest(query_manifest_path)
    record = query.records[0]
    seal_adnet_outputs(
        tmp_path / "out",
        sample_id=record.sample_id,
        class_predictions={1: np.array([[0, 1], [0, 0]], dtype=np.uint8)},
        support_manifest=support,
        query_manifest=query,
        query_record=record,
        checkpoint_sha256="a" * 64,
        code_identity={"sha256": "code"},
        run_config={"seed": 2021},
    )
    gt = _write_npy(tmp_path / "gt.npy", np.array([[0, 1], [0, 0]], dtype=np.uint8))
    gt_manifest_path = _write_json(
        tmp_path / "gt_manifest.json",
        {
            "schema": "adnet.gt_manifest.v1",
            "records": [
                {
                    "sample_id": record.sample_id,
                    "gt_path": gt.name,
                    "gt_sha256": file_sha256(gt),
                }
            ],
        },
    )
    gt_manifest = load_gt_manifest(gt_manifest_path)
    result = evaluate_adnet_outputs(tmp_path / "out", gt_manifest, class_ids=(1,))
    assert result["schema"] == "adnet.fewshot_evaluation.v1"
    assert result["macro_dice"] == 1.0
    assert result["macro_iou"] == 1.0
    assert result["samples"][0]["scores"]["1"]["tp"] == 1
