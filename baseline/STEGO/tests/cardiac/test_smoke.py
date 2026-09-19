"""Smoke tests: STEGO runner produces valid partitions on synthetic input."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cardiac_benchmark.config import STEGOConfig
from cardiac_benchmark.dataset import (
    STEGOCardiacDataset,
    fixed_affine_normalize,
    NORMALIZATION_VERSION,
    PROFILES,
)
from cardiac_benchmark.manifest import ManifestError, write_manifest, load_manifest, counts_by_split
from cardiac_benchmark.stego_runner import run_inference


def make_mock_manifest(root: Path, *, dataset: str = "acdc") -> Path:
    """Create a 3-sample mock manifest with synthetic 3×16×16 volumes."""
    source = root / "image.npy"
    array = np.stack(
        [np.full((16, 16), value, dtype=np.float32) for value in range(3)],
        axis=0,
    )
    np.save(source, array, allow_pickle=False)
    records = []
    for patient, split, z in (("p_train", "train", 0), ("p_dev", "dev", 1), ("p_test", "test", 2)):
        records.append({
            "dataset": dataset,
            "patient_id": patient,
            "split": split,
            "sample_id": f"{patient}:z{z:04d}",
            "acquisition_id": "acq",
            "source_path": str(source),
            "image_checksum": "fixture-image-sha",
            "slice_index": z,
            "context_indices": [max(0, z - 1), z, min(2, z + 1)],
            "frame_index": None,
            "frame_axis": None,
            "depth_axis": 0,
            "native_shape": [3, 16, 16],
            "spacing": [1.0, 1.0, 1.0],
            "affine": None,
            "orientation": None,
        })
    payload = {
        "manifest_kind": "mock",
        "dataset": dataset,
        "freemask_source_sha": "96c32b10fc7b8e09b48822e10ae9eb6cc149e253",
        "freemask_discovery_contract": {"fixture_only": True},
        "image_roots": [str(root)],
        "records": records,
    }
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
            source = Path(tmp) / "image.npy"
            np.save(source, np.zeros((3, 16, 16), dtype=np.float32))
            payload = {
                "manifest_kind": "mock",
                "dataset": "acdc",
                "freemask_source_sha": "96c32b10fc7b8e09b48822e10ae9eb6cc149e253",
                "freemask_discovery_contract": {"fixture_only": True},
                "image_roots": [tmp],
                "records": [{
                    "dataset": "acdc", "patient_id": "p1", "split": "train",
                    "sample_id": "p1:z0000", "acquisition_id": "acq",
                    "source_path": str(source), "image_checksum": "sha",
                    "slice_index": 0, "context_indices": [0, 0, 1],
                    "frame_index": None, "frame_axis": None, "depth_axis": 0,
                    "native_shape": [3, 16, 16], "spacing": [1.0, 1.0, 1.0],
                }],
            }
            path = Path(tmp) / "m.json"
            write_manifest(payload, path)
            manifest = load_manifest(path)
            with self.assertRaises(ManifestError):
                STEGOCardiacDataset(manifest, split="test")


if __name__ == "__main__":
    unittest.main()
