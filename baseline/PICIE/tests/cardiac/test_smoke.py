"""Smoke tests: PICIE runner produces valid partitions on synthetic input."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cardiac_benchmark.config import PICIEConfig
from cardiac_benchmark.dataset import (
    PICIECardiacDataset,
    NORMALIZATION_VERSION,
    PROFILES,
)
from cardiac_benchmark.manifest import ManifestError, write_manifest, load_manifest, counts_by_split
from cardiac_benchmark.picie_runner import run_inference


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
        })
    payload = {
        "manifest_kind": "mock",
        "dataset": dataset,
        "freemask_source_sha": "fixture",
        "freemask_discovery_contract": {"fixture": True},
        "image_roots": [str(root)],
        "records": records,
    }
    path = root / "manifest.json"
    write_manifest(payload, path)
    return path


class RunnerSmokeTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)
        self.manifest_path = make_mock_manifest(self.tmp)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_picie_config_validation(self):
        """PICIEConfig enforces K_train/K_test=4 and pretrain=False."""
        PICIEConfig().validate()
        with self.assertRaises(ValueError):
            PICIEConfig(K_train=5).validate()
        with self.assertRaises(ValueError):
            PICIEConfig(pretrain=True).validate()
        with self.assertRaises(ValueError):
            PICIEConfig(arch="dino").validate()

    def test_picie_dataset_yields_correct_tensors(self):
        """PICIECardiacDataset yields 3-channel RGB image."""
        manifest = load_manifest(self.manifest_path)
        ds = PICIECardiacDataset(manifest, split="train", target_hw=(224, 224))
        self.assertEqual(len(ds), 1)
        sample = ds[0]
        self.assertEqual(sample.image.shape, (3, 224, 224))
        self.assertEqual(sample.provenance["input_channels"], 3)
        self.assertEqual(sample.provenance["augmentation_policy"], "geometric_only")

if __name__ == "__main__":
    unittest.main()
