"""Focused contracts for metadata-only nested few-shot manifests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from self_audit.candidate_a3d.data import synthetic_fixture
from self_audit.data import TrainingData, digest
from self_audit.fewshot import build_subset_manifest


def _source(tmp_path: Path) -> tuple[Path, dict]:
    fixture = synthetic_fixture(
        tmp_path / "cache", n_labeled=10, n_unlabeled=2, n_evaluation=1,
        shape_tzyx=(1, 2, 4, 5),
    )
    source = Path(fixture["train"])
    return source, json.loads(source.read_text())


def test_nested_k_selection_and_payload_contract(tmp_path):
    source, original = _source(tmp_path)
    source_bytes = source.read_bytes()
    manifests = {
        k: build_subset_manifest(source, digest(source), k=k, seed=42,
                                 dataset="synthetic", synthetic=True)
        for k in (2, 5, 8, 10)
    }
    selected = {k: set(value["fewshot"]["selected_ids"]) for k, value in manifests.items()}
    assert selected[2] < selected[5] < selected[8] < selected[10]
    assert manifests[10]["fewshot"]["eligible_labeled_ids"] == sorted(
        record["id"] for record in original["labeled"]
    )
    for k, manifest in manifests.items():
        assert manifest["label_budget"] == k
        assert manifest["label_seed"] == 42
        assert manifest["evaluation_ids"] == original["evaluation_ids"]
        assert manifest["geometry"] == original["geometry"]
        assert len(manifest["labeled"]) == k
        assert all(set(record) == {"id", "image", "image_sha256", "masks"}
                   for record in manifest["labeled"])
        assert all(set(record) == {"id", "image", "image_sha256"}
                   for record in manifest["unlabeled"])
        assert all(Path(record["image"]).is_absolute() for record in manifest["labeled"] + manifest["unlabeled"])
        assert all(Path(mask["path"]).is_absolute()
                   for record in manifest["labeled"] for mask in record["masks"].values())
    assert source.read_bytes() == source_bytes


def test_helper_does_not_open_excluded_masks_and_loader_reads_only_selected(tmp_path):
    source, _ = _source(tmp_path)
    derived = build_subset_manifest(source, digest(source), k=2, seed=42,
                                    dataset="synthetic", synthetic=True)
    selected = set(derived["fewshot"]["selected_ids"])
    source_manifest = json.loads(source.read_text())
    excluded = [record for record in source_manifest["labeled"] if record["id"] not in selected]
    # Remove every excluded mask after the metadata-only transformation.  A
    # normal loader would reject the full source, but the derived budget must
    # remain loadable because those masks are no longer referenced.
    for record in excluded:
        for mask in record["masks"].values():
            (source.parent / mask["path"]).resolve().unlink()
    derived_path = tmp_path / "derived" / "train.json"
    derived_path.parent.mkdir()
    derived_path.write_text(json.dumps(derived, indent=2) + "\n")
    data = TrainingData(derived_path, digest(derived_path), synthetic=True)
    assert set(data.lids) == selected
    assert data.label_reads == set()
    keys = data.labeled_slice_keys() if hasattr(data, "labeled_slice_keys") else data.labeled_frame_keys()
    assert keys


def test_selected_mask_corruption_is_rejected_by_loader_not_helper(tmp_path):
    source, _ = _source(tmp_path)
    derived = build_subset_manifest(source, digest(source), k=2, seed=42,
                                    dataset="synthetic", synthetic=True)
    selected = derived["labeled"][0]
    mask_path = Path(next(iter(selected["masks"].values()))["path"])
    mask_path.write_bytes(b"corrupt")
    derived_path = tmp_path / "derived" / "train.json"
    derived_path.parent.mkdir()
    derived_path.write_text(json.dumps(derived, indent=2) + "\n")
    with pytest.raises(ValueError, match="labeled mask hash mismatch|data payload hash mismatch"):
        TrainingData(derived_path, digest(derived_path), synthetic=True)


@pytest.mark.parametrize("kwargs,pattern", [
    ({"k": 1}, "2 <= k"),
    ({"k": 11}, "2 <= k"),
    ({"k": 2, "dataset": "acdc"}, "dataset identity"),
    ({"k": 2, "expected_sha256": "0" * 64}, "source manifest hash"),
])
def test_source_and_budget_validation(tmp_path, kwargs, pattern):
    source, _ = _source(tmp_path)
    args = {"source_manifest": source, "expected_sha256": digest(source), "k": 2,
            "seed": 42, "dataset": "synthetic", "synthetic": True}
    args.update(kwargs)
    with pytest.raises(ValueError, match=pattern):
        build_subset_manifest(**args)


def test_source_duplicate_and_invalid_u_records_are_rejected(tmp_path):
    source, manifest = _source(tmp_path)
    manifest["unlabeled"][0]["id"] = manifest["labeled"][0]["id"]
    source.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unique"):
        build_subset_manifest(source, digest(source), k=2, dataset="synthetic", synthetic=True)

    source, manifest = _source(tmp_path / "second")
    manifest["unlabeled"][0]["masks"] = {}
    source.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unlabeled"):
        build_subset_manifest(source, digest(source), k=2, dataset="synthetic", synthetic=True)
