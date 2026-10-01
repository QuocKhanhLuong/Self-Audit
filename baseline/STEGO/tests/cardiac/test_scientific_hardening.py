"""Scientific hardening tests for STEGO cardiac benchmark.

Verifies:
- Freeze validation: config hash stability
- Deterministic inference: same input → same partition hash
- Checkpoint provenance: SHA-256 binding
- Bundle seal/verify integrity
"""

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
from cardiac_benchmark.freeze import (
    FreezeError,
    create_freeze,
    evaluator_skeleton,
    seal_raw_bundle,
    validate_freeze,
    verify_raw_bundle,
)
from cardiac_benchmark.provenance import sha256_array, sha256_file, sha256_json
from cardiac_benchmark.stego_runner import run_inference


class FreezeTests(unittest.TestCase):

    def test_config_hash_stable(self):
        """Same config produces same hash."""
        config = STEGOConfig()
        h1 = config.config_hash()
        h2 = config.config_hash()
        self.assertEqual(h1, h2)

    def test_config_hash_changes_on_param_change(self):
        """Different config produces different hash."""
        c1 = STEGOConfig(dim=70)
        c2 = STEGOConfig(dim=100)
        self.assertNotEqual(c1.config_hash(), c2.config_hash())

    def test_create_and_validate_freeze(self):
        """Freeze creation and validation round-trips."""
        config_dict = {
            "model_type": "vit_base",
            "checkpoint_sha256": "abc123",
            "n_classes": 4,
            "resolution": 224,
            "dim": 70,
            "normalization": "stego.cardiac.fixed_affine_v1",
            "projection_type": "nonlinear",
            "benchmark_seed": 42,
        }
        freeze = create_freeze(config_dict)
        self.assertIn("config_hash", freeze)
        validate_freeze(freeze)

    def test_freeze_detects_tampering(self):
        """Modified freeze is detected."""
        config_dict = {
            "model_type": "vit_base",
            "checkpoint_sha256": "abc123",
            "n_classes": 4,
            "resolution": 224,
            "dim": 70,
            "normalization": "stego.cardiac.fixed_affine_v1",
            "projection_type": "nonlinear",
            "benchmark_seed": 42,
        }
        freeze = create_freeze(config_dict)
        freeze["resolution"] = 128
        with self.assertRaises(FreezeError):
            validate_freeze(freeze)

    def test_freeze_expected_hash_mismatch(self):
        """validate_freeze rejects wrong expected hash."""
        config_dict = {
            "model_type": "vit_base",
            "checkpoint_sha256": "abc123",
            "n_classes": 4,
            "resolution": 224,
            "dim": 70,
            "normalization": "stego.cardiac.fixed_affine_v1",
            "projection_type": "nonlinear",
            "benchmark_seed": 42,
        }
        freeze = create_freeze(config_dict)
        with self.assertRaises(FreezeError):
            validate_freeze(freeze, expected_hash="wrong")


class DeterminismTests(unittest.TestCase):

    def test_same_input_same_partition(self):
        """Same input + same model → same partition hash."""
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.patch_size = 8
                self.cfg = type("C", (), {"dropout": False})()
                self.linear = torch.nn.Conv2d(768, 70, (1, 1))
                torch.manual_seed(42)
                torch.nn.init.constant_(self.linear.weight, 0.01)
                torch.nn.init.constant_(self.linear.bias, 0.0)

            def forward(self, img, n=1, return_class_feat=False):
                b, c, h, w = img.shape
                fh, fw = h // self.patch_size, w // self.patch_size
                feat = torch.ones(b, 768, fh, fw)
                code = self.linear(feat)
                return feat, code

        model = MockModel()
        model.eval()
        image = torch.rand(3, 224, 224) * 255.0

        p1 = run_inference(model, image.clone(), resolution=224, device="cpu")
        p2 = run_inference(model, image.clone(), resolution=224, device="cpu")
        self.assertEqual(sha256_array(p1), sha256_array(p2))


class CheckpointProvenanceTests(unittest.TestCase):

    def test_sha256_binding(self):
        """Checkpoint file hash is deterministic."""
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "model.ckpt"
            torch.save({"state_dict": {"w": torch.ones(3)}}, ckpt)
            h1 = sha256_file(ckpt)
            h2 = sha256_file(ckpt)
            self.assertEqual(h1, h2)
            self.assertEqual(len(h1), 64)


class BundleSealTests(unittest.TestCase):

    def test_seal_and_verify(self):
        """Seal → verify round-trip."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            npy = root / "p.npy"
            np.save(npy, np.zeros((8, 8), dtype=np.int32), allow_pickle=False)
            entry = {"sample_id": "p", "status": "success", "raw_partition_path": str(npy)}
            seal_raw_bundle([entry], root, required_sample_ids={"p"})
            bundle_path = root / "raw_bundle.json"
            verified = verify_raw_bundle(bundle_path, expected_sample_ids={"p"})
            self.assertIn("bundle_hash", verified)

    def test_mutation_detected(self):
        """Modifying raw partition file triggers FreezeError."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            npy = root / "p.npy"
            np.save(npy, np.zeros((8, 8), dtype=np.int32), allow_pickle=False)
            entry = {"sample_id": "p", "status": "success", "raw_partition_path": str(npy)}
            seal_raw_bundle([entry], root, required_sample_ids={"p"})
            npy.write_bytes(npy.read_bytes() + b"x")
            with self.assertRaises(FreezeError):
                verify_raw_bundle(root / "raw_bundle.json", expected_sample_ids={"p"})

    def test_missing_sample_detected(self):
        """Bundle with wrong sample set is rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            npy = root / "p.npy"
            np.save(npy, np.zeros((8, 8), dtype=np.int32), allow_pickle=False)
            entry = {"sample_id": "p", "status": "success", "raw_partition_path": str(npy)}
            with self.assertRaises(FreezeError):
                seal_raw_bundle([entry], root, required_sample_ids={"p", "q"})

    def test_evaluator_skeleton(self):
        """Evaluator skeleton verifies bundle before GT mount."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            npy = root / "p.npy"
            np.save(npy, np.zeros((8, 8), dtype=np.int32), allow_pickle=False)
            entry = {"sample_id": "p", "status": "success", "raw_partition_path": str(npy)}
            seal_raw_bundle([entry], root, required_sample_ids={"p"})
            receipt = evaluator_skeleton(root / "raw_bundle.json", root / "eval", expected_sample_ids={"p"})
            self.assertFalse(receipt["gt_opened"])
            self.assertFalse(receipt["metrics_written"])
            self.assertEqual(receipt["verified_samples"], 1)


if __name__ == "__main__":
    unittest.main()
