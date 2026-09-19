
from __future__ import annotations

import copy
from pathlib import Path

import sys

BASELINE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BASELINE_ROOT.parents[1]
for name in list(sys.modules):
    if name == "cardiac_benchmark" or name.startswith("cardiac_benchmark."):
        sys.modules.pop(name)
for import_path in (BASELINE_ROOT / "src", BASELINE_ROOT, REPO_ROOT / "src"):
    if str(import_path) in sys.path:
        sys.path.remove(str(import_path))
for import_path in (BASELINE_ROOT / "src", BASELINE_ROOT, REPO_ROOT / "src"):
    sys.path.insert(0, str(import_path))

import numpy as np
import pytest

import sys

BASELINE_SRC = Path(__file__).resolve().parents[2] / "src"
for name in list(sys.modules):
    if name == "cardiac_benchmark" or name.startswith("cardiac_benchmark."):
        sys.modules.pop(name)
if str(BASELINE_SRC) in sys.path:
    sys.path.remove(str(BASELINE_SRC))
sys.path.insert(0, str(BASELINE_SRC))

from shared_benchmark.manifest import SharedManifestError, build_shared_manifest
from shared_benchmark.spatial import build_grid_spec
from self_audit_maskfree.data.discovery import discover_dataset


def write_image(root: Path, relative: str, *, shape: tuple[int, ...] = (7, 11, 3), seed: int = 0) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    np.save(path, rng.normal(size=shape).astype(np.float32), allow_pickle=False)
    return path


def shared_manifest(root: Path, *, target_hw: tuple[int, int] = (8, 8)) -> dict:
    for index in range(4):
        write_image(root, f"patient{index:03d}.npy", seed=index)
    upstream = discover_dataset(root, "acdc", seed=42, protocol="auto", depth_axis=2)
    return build_shared_manifest(
        upstream,
        build_grid_spec(target_hw, config_provenance={"source": "stego-picie-test"}),
        fixture=True,
        scientific=False,
        local_source_root=root,
    )

from cardiac_benchmark.dataset import STEGOCardiacDataset, NORMALIZATION_VERSION
from cardiac_benchmark.manifest import counts_by_split, validate_manifest


def test_stego_shared_manifest_view_preserves_identity_and_grid(tmp_path):
    manifest = shared_manifest(tmp_path, target_hw=(8, 8))
    record = next(row for row in manifest["records"] if row["slice_index"] == 0)
    ds = STEGOCardiacDataset(manifest, split=record["split"], source_root=tmp_path)
    sample = next(ds[index] for index, row in enumerate(ds.records) if row["sample_id"] == record["sample_id"])
    assert tuple(sample.image.shape) == (3, 8, 8)
    assert sample.provenance["sample_id"] == record["sample_id"]
    assert sample.provenance["patient_id"] == record["patient_id"]
    assert sample.provenance["split"] == record["split"]
    assert sample.provenance["frame_index"] == record["frame_index"]
    assert sample.provenance["slice_index"] == 0
    assert sample.provenance["context_indices"] == [0, 0, 1]
    assert sample.provenance["target_shape"] == manifest["shared_grid"]["target_hw"]
    assert sample.provenance["normalization"] == NORMALIZATION_VERSION


def test_stego_rejects_grid_override(tmp_path):
    manifest = shared_manifest(tmp_path, target_hw=(8, 8))
    record = manifest["records"][0]
    with pytest.raises(SharedManifestError):
        STEGOCardiacDataset(manifest, split=record["split"], source_root=tmp_path, target_hw=(16, 16))


def test_stego_manifest_rejects_missing_source_hash(tmp_path):
    manifest = shared_manifest(tmp_path, target_hw=(8, 8))
    bad = copy.deepcopy(manifest)
    bad["records"][0]["source"]["sha256"] = ""
    with pytest.raises(SharedManifestError):
        validate_manifest(bad)


def test_stego_counts_by_split_uses_shared_records(tmp_path):
    manifest = shared_manifest(tmp_path, target_hw=(8, 8))
    counts = counts_by_split(manifest)
    assert sum(row["samples"] for row in counts.values()) == len(manifest["records"])
