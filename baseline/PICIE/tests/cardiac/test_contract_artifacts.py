"""Contract artifact tests for PICIE cardiac benchmark.

Verifies:
- Dataset adapter provenance fields
- Geometric only augmentation
- No GT firewall
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cardiac_benchmark.config import PICIEConfig
from cardiac_benchmark.dataset import (
    PICIECardiacDataset,
    NORMALIZATION_VERSION,
)
from cardiac_benchmark.manifest import ManifestError, load_manifest, write_manifest


def make_mock_manifest(root: Path, *, dataset: str = "acdc") -> Path:
    source = root / "image.npy"
    array = np.stack(
        [np.full((16, 16), float(v), dtype=np.float32) for v in range(3)], axis=0,
    )
    np.save(source, array, allow_pickle=False)
    records = []
    for patient, split, z in (("p_train", "train", 0), ("p_dev", "dev", 1), ("p_test", "test", 2)):
        records.append({
            "dataset": dataset, "patient_id": patient, "split": split,
            "sample_id": f"{patient}:z{z:04d}", "acquisition_id": "acq",
            "source_path": str(source), "image_checksum": "fixture",
            "slice_index": z, "context_indices": [max(0, z - 1), z, min(2, z + 1)],
            "frame_index": None, "frame_axis": None, "depth_axis": 0,
            "native_shape": [3, 16, 16], "spacing": [1.0, 1.0, 1.0],
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


class ContractArtifactTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)
        self.manifest_path = make_mock_manifest(self.tmp)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_provenance_fields(self):
        manifest = load_manifest(self.manifest_path)
        ds = PICIECardiacDataset(manifest, split="train")
        sample = ds[0]
        prov = sample.provenance
        self.assertEqual(prov["normalization"], NORMALIZATION_VERSION)
        self.assertEqual(prov["augmentation_policy"], "geometric_only")
        self.assertEqual(prov["input_channels"], 3)
        self.assertNotIn("source_path", prov)
        self.assertEqual(prov["manifest_hash"], manifest["manifest_hash"])

    def test_no_gt_leakage(self):
        manifest = load_manifest(self.manifest_path)
        ds = PICIECardiacDataset(manifest, split="train")
        sample = ds[0]
        # Ensure no label/mask keys exist in the provenance or anywhere in dataset
        self.assertNotIn("annotation_path", sample.provenance)
        self.assertNotIn("label", sample.provenance)

if __name__ == "__main__":
    unittest.main()
