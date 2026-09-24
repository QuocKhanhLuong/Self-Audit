"""Contract artifact tests for STEGO cardiac benchmark.

Verifies:
- Dataset adapter provenance fields
- Fixed Affine normalization contract
- KNN isolation (train-only)
- GT firewall
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

BASELINE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(BASELINE_ROOT / "src"))


def _purge_cardiac_benchmark_modules() -> None:
    for name in list(sys.modules):
        if name == "cardiac_benchmark" or name.startswith("cardiac_benchmark."):
            sys.modules.pop(name, None)


_purge_cardiac_benchmark_modules()

from self_audit_maskfree.data.discovery import discover_dataset
from shared_benchmark.manifest import build_shared_manifest
from shared_benchmark.spatial import build_grid_spec

from cardiac_benchmark.config import STEGOConfig
from cardiac_benchmark.dataset import (
    STEGOCardiacDataset,
    fixed_affine_normalize,
    NORMALIZATION_VERSION,
)
from cardiac_benchmark.manifest import ManifestError, load_manifest, write_manifest


def make_mock_manifest(root: Path, *, dataset: str = "acdc", target_hw: tuple[int, int] = (16, 16)) -> Path:
    for index in range(3):
        source = root / f"patient{index:03d}.npy"
        array = np.full((16, 16, 1), float(index), dtype=np.float32)
        np.save(source, array, allow_pickle=False)
    upstream = discover_dataset(root, dataset, seed=42, protocol="auto", depth_axis=2)
    payload = build_shared_manifest(
        upstream,
        build_grid_spec(target_hw, config_provenance={"source": "stego-test"}),
        fixture=True,
        scientific=False,
        local_source_root=root,
    )
    path = root / "mock_manifest.json"
    write_manifest(payload, path)
    return path


class ContractArtifactTests(unittest.TestCase):

    def test_provenance_fields_present(self):
        """Dataset adapter returns all required provenance fields."""
        with tempfile.TemporaryDirectory() as tmp:
            path = make_mock_manifest(Path(tmp))
            manifest = load_manifest(path, check_paths=True)
            ds = STEGOCardiacDataset(manifest, split="train", profile="STEGO-2D", resolution=16)
            prov = ds[0].provenance
            required = {
                "sample_id", "patient_id", "study_id", "volume_id", "split", "dataset",
                "manifest_hash", "profile", "normalization", "input_channels",
                "slice_index", "source_image_hash", "source_shape", "target_shape",
                "shared_grid_version", "shared_grid_hash", "spatial_transform",
            }
            missing = required - set(prov)
            self.assertFalse(missing, f"missing provenance keys: {missing}")
            self.assertEqual(prov["normalization"], NORMALIZATION_VERSION)
            self.assertEqual(prov["target_shape"], [16, 16])

    def test_grid_hash_consistent_across_samples(self):
        """manifest_hash is same for all samples from same manifest."""
        with tempfile.TemporaryDirectory() as tmp:
            path = make_mock_manifest(Path(tmp))
            manifest = load_manifest(path, check_paths=True)
            ds_train = STEGOCardiacDataset(manifest, split="train", profile="STEGO-2D", resolution=16)
            ds_dev = STEGOCardiacDataset(manifest, split="dev", profile="STEGO-2D", resolution=16)
            self.assertEqual(
                ds_train[0].provenance["manifest_hash"],
                ds_dev[0].provenance["manifest_hash"],
            )

    def test_no_gt_keys_in_dataset(self):
        """Provenance must not contain GT keys."""
        with tempfile.TemporaryDirectory() as tmp:
            path = make_mock_manifest(Path(tmp))
            manifest = load_manifest(path, check_paths=True)
            ds = STEGOCardiacDataset(manifest, split="train", profile="STEGO-2D", resolution=16)
            prov = ds[0].provenance
            for key in prov:
                self.assertFalse(
                    any(token in key.lower() for token in ("mask", "label", "annotation")),
                    f"GT key found: {key}",
                )

    def test_fixed_affine_uint8_output(self):
        """Fixed Affine produces uint8 [0, 255] output."""
        for val in [-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0]:
            plane = np.full((4, 4), val, dtype=np.float32)
            result = fixed_affine_normalize(plane)
            self.assertEqual(result.dtype, np.uint8)
            self.assertTrue(0 <= result.min() <= result.max() <= 255)

    def test_fixed_affine_clip_boundary(self):
        """Values outside [-3,3] are clipped, not scaled per-image."""
        lo = np.full((2, 2), -10.0, dtype=np.float32)
        hi = np.full((2, 2), 10.0, dtype=np.float32)
        self.assertTrue(np.all(fixed_affine_normalize(lo) == 0))
        self.assertTrue(np.all(fixed_affine_normalize(hi) == 255))

    def test_fixed_affine_deterministic(self):
        """Same input always gives same output (no random component)."""
        plane = np.random.RandomState(42).randn(32, 32).astype(np.float32)
        r1 = fixed_affine_normalize(plane.copy())
        r2 = fixed_affine_normalize(plane.copy())
        self.assertTrue(np.array_equal(r1, r2))

    def test_three_channel_image(self):
        """Dataset produces 3-channel images from grayscale input."""
        with tempfile.TemporaryDirectory() as tmp:
            path = make_mock_manifest(Path(tmp))
            manifest = load_manifest(path, check_paths=True)
            ds = STEGOCardiacDataset(manifest, split="train", profile="STEGO-2D", resolution=16)
            img = ds[0].image
            self.assertEqual(img.shape[0], 3)
            self.assertTrue(torch.equal(img[0], img[1]))
            self.assertTrue(torch.equal(img[1], img[2]))

    def test_manifest_firewall_rejects_gt_key(self):
        """Manifest with annotation keys is rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            path = make_mock_manifest(Path(tmp))
            original = load_manifest(path)
            bad = copy.deepcopy(original)
            bad["records"][0]["annotation_path"] = "forbidden"
            bad_path = Path(tmp) / "bad.json"
            bad_path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(ManifestError):
                load_manifest(bad_path)

    def test_knn_split_constraint(self):
        """Config rejects non-train KNN split."""
        with self.assertRaises(ValueError):
            STEGOConfig(knn_split="test").validate()
        with self.assertRaises(ValueError):
            STEGOConfig(knn_split="dev").validate()
        STEGOConfig(knn_split="train").validate()

    def test_config_seed_constraint(self):
        """Config rejects non-42 benchmark seed."""
        with self.assertRaises(ValueError):
            STEGOConfig(benchmark_seed=0).validate()


if __name__ == "__main__":
    unittest.main()
