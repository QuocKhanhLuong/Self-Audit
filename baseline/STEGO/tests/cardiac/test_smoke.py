"""Smoke tests: STEGO runner produces valid partitions on synthetic input."""

from __future__ import annotations

import copy
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
from cardiac_benchmark.manifest import ManifestError, write_manifest, load_manifest, counts_by_split
from cardiac_benchmark.stego_runner import run_inference


def make_mock_manifest(root: Path, *, dataset: str = "acdc", target_hw: tuple[int, int] = (16, 16)) -> Path:
    """Create a valid shared manifest with one sample per split."""
    for index in range(3):
        source = root / f"patient{index:03d}.npy"
        array = np.full((16, 16, 1), float(index), dtype=np.float32)
        np.save(source, array, allow_pickle=False)
    upstream = discover_dataset(root, dataset, seed=42, protocol="auto", depth_axis=2)
    payload = build_shared_manifest(
        upstream,
        build_grid_spec(target_hw, config_provenance={"source": "stego-smoke-test"}),
        fixture=True,
        scientific=False,
        local_source_root=root,
    )
    path = root / "mock_manifest.json"
    write_manifest(payload, path)
    return path


class SmokeTests(unittest.TestCase):

    def test_manifest_load_and_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_mock_manifest(Path(tmp))
            manifest = load_manifest(path, check_paths=True)
            counts = counts_by_split(manifest)
            self.assertEqual(counts["train"], {"patients": 1, "samples": 1})
            self.assertEqual(counts["dev"], {"patients": 1, "samples": 1})
            self.assertEqual(counts["test"], {"patients": 1, "samples": 1})

    def test_dataset_returns_valid_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_mock_manifest(Path(tmp))
            manifest = load_manifest(path, check_paths=True)
            dataset = STEGOCardiacDataset(manifest, split="train", profile="STEGO-2D", resolution=16)
            self.assertEqual(len(dataset), 1)
            sample = dataset[0]
            self.assertEqual(tuple(sample.image.shape), (3, 16, 16))
            self.assertTrue(sample.image.dtype == torch.float32)
            self.assertIn("sample_id", sample.provenance)
            self.assertIn("normalization", sample.provenance)
            self.assertEqual(sample.provenance["normalization"], NORMALIZATION_VERSION)
            self.assertEqual(sample.provenance["input_channels"], 3)
            self.assertEqual(sample.provenance["target_shape"], [16, 16])

    def test_fixed_affine_output_range(self):
        plane = np.array([[-5.0, -3.0, 0.0], [1.5, 3.0, 5.0]], dtype=np.float32)
        result = fixed_affine_normalize(plane)
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.min(), 0)
        self.assertEqual(result.max(), 255)
        self.assertEqual(int(result[0, 1]), 0)
        self.assertEqual(int(result[1, 1]), 255)
        center_val = int(result[0, 2])
        self.assertTrue(125 <= center_val <= 130)

    def test_three_channel_replication(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_mock_manifest(Path(tmp))
            manifest = load_manifest(path, check_paths=True)
            dataset = STEGOCardiacDataset(manifest, split="train", profile="STEGO-2D", resolution=16)
            sample = dataset[0]
            self.assertTrue(torch.equal(sample.image[0], sample.image[1]))
            self.assertTrue(torch.equal(sample.image[1], sample.image[2]))

    def test_no_gt_keys_in_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_mock_manifest(Path(tmp))
            manifest = load_manifest(path, check_paths=True)
            dataset = STEGOCardiacDataset(manifest, split="train", profile="STEGO-2D", resolution=16)
            sample = dataset[0]
            forbidden = {"mask", "label", "annotation", "foreground_hint", "reference_path"}
            for key in sample.provenance:
                self.assertFalse(
                    any(token in key.lower() for token in forbidden),
                    f"provenance contains forbidden key: {key}",
                )

    def test_synthetic_inference_shape(self):
        """Verify run_inference produces correct shape on synthetic input.

        Uses a minimal mock model to avoid loading the real DINO checkpoint.
        """
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.patch_size = 8
                self.cfg = type("C", (), {"dropout": False})()

            def forward(self, img, n=1, return_class_feat=False):
                b, c, h, w = img.shape
                fh, fw = h // self.patch_size, w // self.patch_size
                feat = torch.randn(b, 768, fh, fw)
                code = torch.randn(b, 70, fh, fw)
                return feat, code

        model = MockModel()
        image = torch.rand(3, 224, 224) * 255.0
        partition = run_inference(model, image, resolution=224, device="cpu")
        self.assertEqual(partition.shape, (224, 224))
        self.assertEqual(partition.dtype, np.int32)
        self.assertFalse(np.any(np.isnan(partition.astype(float))))

    def test_profile_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_mock_manifest(Path(tmp))
            manifest = load_manifest(path, check_paths=True)
            with self.assertRaises(ValueError):
                STEGOCardiacDataset(manifest, split="train", profile="INVALID")

    def test_empty_split_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_mock_manifest(Path(tmp))
            manifest = copy.deepcopy(load_manifest(path, check_paths=True))
            for record in manifest["records"]:
                record["split"] = "train"
            with self.assertRaises(ManifestError):
                STEGOCardiacDataset(manifest, split="test", profile="STEGO-2D", resolution=16)


if __name__ == "__main__":
    unittest.main()
