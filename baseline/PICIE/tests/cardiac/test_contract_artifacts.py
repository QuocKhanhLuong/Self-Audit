"""Contract artifact tests for PICIE cardiac benchmark.

Verifies:
- Dataset adapter provenance fields
- Geometric only augmentation
- No GT firewall
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

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

from cardiac_benchmark.config import PICIEConfig
from cardiac_benchmark.dataset import (
    PICIECardiacDataset,
    NORMALIZATION_VERSION,
)
from cardiac_benchmark.manifest import load_manifest, write_manifest


def make_mock_manifest(root: Path, *, dataset: str = "acdc", target_hw: tuple[int, int] = (224, 224)) -> Path:
    for index in range(3):
        source = root / f"patient{index:03d}.npy"
        array = np.full((16, 16, 1), float(index), dtype=np.float32)
        np.save(source, array, allow_pickle=False)
    upstream = discover_dataset(root, dataset, seed=42, protocol="auto", depth_axis=2)
    payload = build_shared_manifest(
        upstream,
        build_grid_spec(target_hw, config_provenance={"source": "picie-test"}),
        fixture=True,
        scientific=False,
        local_source_root=root,
    )
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
        manifest = load_manifest(self.manifest_path, check_paths=True)
        ds = PICIECardiacDataset(manifest, split="train")
        sample = ds[0]
        prov = sample.provenance
        self.assertEqual(prov["normalization"], NORMALIZATION_VERSION)
        self.assertEqual(prov["augmentation_policy"], "none_scientific")
        self.assertEqual(prov["input_channels"], 3)
        self.assertNotIn("source_path", prov)
        self.assertEqual(prov["manifest_hash"], manifest["manifest_hash"])
        self.assertEqual(prov["target_shape"], [224, 224])
        self.assertIn("source_image_hash", prov)

    def test_no_gt_leakage(self):
        manifest = load_manifest(self.manifest_path, check_paths=True)
        ds = PICIECardiacDataset(manifest, split="train")
        sample = ds[0]
        self.assertNotIn("annotation_path", sample.provenance)
        self.assertNotIn("label", sample.provenance)


if __name__ == "__main__":
    unittest.main()
