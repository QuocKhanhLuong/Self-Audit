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

from cardiac_benchmark.cluster_kmeans import cluster_latent
from cardiac_benchmark.dataset import ImageOnlyCardiacDataset
from cardiac_benchmark.freeze import FreezeError, evaluator_skeleton, seal_raw_bundle
from cardiac_benchmark.manifest import ManifestError, counts_by_split, load_manifest, require_scientific_manifest, write_manifest
from cardiac_benchmark.train_stage1 import (Stage1Config, build_loaders, build_model, build_optimization,
                                            load_checkpoint_for_export, seed_primary, validate_epoch)
from data_utils.patch_sampler import PatchSampler
from model import CUTSEncoder
from shared_benchmark.manifest import build_shared_manifest
from shared_benchmark.provenance import sha256_file
from shared_benchmark.spatial import build_grid_spec


def make_mock_manifest(root: Path, *, dataset: str = "acdc") -> Path:
    source = root / "image.npy"
    array = np.stack([np.full((16, 16), value, dtype=np.float32) for value in range(3)], axis=0)
    np.save(source, array, allow_pickle=False)
    records = []
    for patient, split, z in (("p_train", "train", 0), ("p_dev", "dev", 1), ("p_test", "test", 2)):
        records.append({"dataset": dataset, "patient_id": patient, "study_id": f"{dataset}:{patient}",
                        "volume_id": f"{dataset}:vol-{patient}:t0000", "unit_id": f"{patient}:z{z:04d}",
                        "split": split, "path": str(source), "source_path": str(source), "relative_path": source.name,
                        "source_format": "npy", "shape": [3,16,16], "native_shape": [3,16,16], "dtype": "float32",
                        "native_hw": [16,16], "depth": 3, "num_slices": 3, "depth_axis": 0, "frame_axis": None,
                        "slice_index": z, "frame_index": 0, "frame_selection_rule": "single_acquired_frame_index_0_image_only",
                        "native_geometry": "unavailable", "native_affine": None, "orientation": None, "spacing_mm": None,
                        "spacing_valid": False, "native_grid_export": False, "export_grid": "stored", "spatial_unit": "unknown",
                        "source_hash": sha256_file(source), "source_fingerprint": sha256_file(source),
                        "frame_fingerprint": f"fixture-{patient}", "study_grid_compatibility": None})
    upstream = {"schema_version": "maskfree150.data.v2", "dataset": dataset, "seed": 42,
                "manifest_id": "cuts-fixture-upstream-v1", "records": records,
                "discovery_contract": {"version": "fixture"},
                "split_provenance": {"rule": "fixture", "seed": 42, "ratios": {},
                                     "patient_level_disjoint": True, "split_identity": "patient_id",
                                     "selection_inputs": ["fixture"], "content_fingerprint_used": False}}
    payload = build_shared_manifest(upstream, build_grid_spec((16,16), config_provenance={"source":"fixture"}),
                                    fixture=True, scientific=False, local_source_root=root)
    path = root / "mock_manifest.json"
    write_manifest(payload, path)
    return path


class P0Tests(unittest.TestCase):
    def test_manifest_loader_profiles_and_scientific_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = make_mock_manifest(Path(temporary))
            manifest = load_manifest(path, check_paths=True)
            self.assertEqual(counts_by_split(manifest)["train"], {"patients": 1, "samples": 1})
            with self.assertRaises(ManifestError):
                require_scientific_manifest(path)
            with self.assertRaises(ManifestError):
                build_loaders(Stage1Config(profile="CUTS-2D", dataset="acdc", manifest_path=str(path),
                                            max_epochs=200, scientific_run=True))
            image_2d = ImageOnlyCardiacDataset(manifest, split="train", profile="CUTS-2D")[0].image
            image_25d = ImageOnlyCardiacDataset(manifest, split="train", profile="CUTS-2.5D")[0].image
            self.assertEqual(tuple(image_2d.shape), (1, 16, 16))
            self.assertEqual(tuple(image_25d.shape), (3, 16, 16))
            self.assertTrue(torch.equal(image_25d[0], image_25d[1]))

    def test_sampler_and_one_step_algebra_parity(self):
        config = Stage1Config(profile="CUTS-2D", dataset="acdc", manifest_path="unused", max_epochs=1,
                              sampled_patches_per_image=2, patch_size=5, batch_size=1)
        image = torch.linspace(0, 1, 16 * 16, dtype=torch.float32).reshape(1, 1, 16, 16)
        seed_primary(42)
        wrapper = build_model(config)
        seed_primary(42)
        official = CUTSEncoder(in_channels=1, num_kernels=16, random_seed=42, sampled_patches_per_image=2, patch_size=5)
        for left, right in zip(wrapper.parameters(), official.parameters()):
            self.assertTrue(torch.equal(left, right))
        self.assertTrue(np.array_equal(PatchSampler(42, 5, 2).sample(image)[0], PatchSampler(42, 5, 2).sample(image)[0]))
        optim_a, sched_a, mse_a, ntx_a = build_optimization(wrapper, config)
        optim_b, sched_b, mse_b, ntx_b = build_optimization(official, config)
        def step(model, optimizer, mse, ntx):
            _, real, recon, anchors, positives = model(image)
            loss = .001 * ntx(anchors, positives) + .999 * mse(real, recon)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            return loss.detach()
        loss_a, loss_b = step(wrapper, optim_a, mse_a, ntx_a), step(official, optim_b, mse_b, ntx_b)
        self.assertTrue(torch.allclose(loss_a, loss_b, atol=0, rtol=0))
        sched_a.step(); sched_b.step()
        self.assertEqual(sched_a.state_dict(), sched_b.state_dict())
        for left, right in zip(wrapper.parameters(), official.parameters()):
            self.assertTrue(torch.allclose(left, right, atol=0, rtol=0))

    def test_train_dev_isolation_and_architecture(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = make_mock_manifest(Path(temporary))
            config = Stage1Config(profile="CUTS-2D", dataset="acdc", manifest_path=str(path), max_epochs=1,
                                  batch_size=1, sampled_patches_per_image=2)
            train_loader, dev_loader, manifest = build_loaders(config)
            train_patients = {item.provenance["patient_id"] for item in train_loader.dataset.dataset}
            dev_patients = {item.provenance["patient_id"] for item in dev_loader.dataset}
            self.assertEqual(train_patients, {"p_train"}); self.assertEqual(dev_patients, {"p_dev"})
            model = build_model(config); _, _, mse, ntx = build_optimization(model, config)
            before = [module.running_mean.clone() for module in model.modules() if isinstance(module, torch.nn.BatchNorm2d)]
            validate_epoch(model, dev_loader, mse, ntx, config, torch.device("cpu"))
            after = [module.running_mean for module in model.modules() if isinstance(module, torch.nn.BatchNorm2d)]
            self.assertFalse(model.training)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, after)))
            convolutions = [module for module in model.modules() if isinstance(module, torch.nn.Conv2d)]
            self.assertEqual(len(convolutions), 4)
            self.assertTrue(all(module.kernel_size == (5, 5) and module.stride == (1, 1) for module in convolutions))
            self.assertFalse(any(isinstance(module, (torch.nn.MaxPool2d, torch.nn.AvgPool2d)) for module in model.modules()))

    def test_manifest_firewall_and_cohort_hash_immutability(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = make_mock_manifest(Path(temporary))
            original = load_manifest(path)
            # Model/clustering configuration is deliberately external to the manifest.
            self.assertEqual(original["manifest_hash"], load_manifest(path)["manifest_hash"])
            bad = copy.deepcopy(original)
            bad["records"][0]["annotation_path"] = "forbidden"
            # Rewriting its hash does not make an annotation-bearing schema valid.
            bad_path = Path(temporary) / "bad.json"
            bad_path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(ManifestError):
                load_manifest(bad_path)

    def test_clustering_retry_policy(self):
        latent = np.ones((128, 16, 16), dtype=np.float32)
        calls: list[int] = []
        def first_fails(flat, *, random_seed, num_workers):
            calls.append(random_seed)
            if random_seed == 1: raise RuntimeError("seed one")
            return np.zeros(flat.shape[0], dtype=np.int64)
        success = cluster_latent(latent, clustering_fn=first_fails)
        self.assertEqual(calls, [1, 2]); self.assertTrue(success["retry_occurred"]); self.assertEqual(success["actual_clustering_seed_used"], 2)
        calls.clear()
        def always_fails(flat, *, random_seed, num_workers):
            calls.append(random_seed); raise RuntimeError("fail")
        failure = cluster_latent(latent, clustering_fn=always_fails)
        self.assertEqual(calls, [1, 2]); self.assertEqual(failure["status"], "failure")

    def test_inference_latent_parity(self):
        config = Stage1Config(profile="CUTS-2.5D", dataset="acdc", manifest_path="unused", max_epochs=1,
                              sampled_patches_per_image=2)
        seed_primary(42); training_model = build_model(config)
        seed_primary(42); inference_model = build_model(config, inference=True)
        inference_model.load_state_dict(training_model.state_dict()); training_model.eval(); inference_model.eval()
        image = torch.rand(1, 3, 16, 16)
        with torch.no_grad():
            latent_from_training = training_model(image)[0]
            latent_from_inference = inference_model(image)
        self.assertTrue(torch.equal(latent_from_training, latent_from_inference))

    def test_freeze_integrity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "p.npy"; np.save(raw, np.zeros((8, 8), dtype=np.int64), allow_pickle=False)
            entry = {"sample_id": "p", "status": "success", "raw_partition_path": str(raw)}
            seal_raw_bundle([entry], root, required_sample_ids={"p"})
            bundle = root / "raw_bundle.json"
            evaluator_skeleton(bundle, root / "evaluation", expected_sample_ids={"p"})
            raw.write_bytes(raw.read_bytes() + b"x")
            with self.assertRaises(FreezeError):
                evaluator_skeleton(bundle, root / "evaluation2", expected_sample_ids={"p"})

    def test_checkpoint_profile_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); path = make_mock_manifest(root)
            config_2d = Stage1Config(profile="CUTS-2D", dataset="acdc", manifest_path=str(path), max_epochs=1,
                                     sampled_patches_per_image=2)
            seed_primary(42); model = build_model(config_2d)
            checkpoint = root / "checkpoint.pt"
            torch.save({"state_dict": model.state_dict(), "dataset": "acdc", "profile": "CUTS-2D",
                        "manifest_hash": load_manifest(path)["manifest_hash"]}, checkpoint)
            config_25d = Stage1Config(profile="CUTS-2.5D", dataset="acdc", manifest_path=str(path), max_epochs=1,
                                      sampled_patches_per_image=2)
            with self.assertRaises(ValueError):
                load_checkpoint_for_export(config_25d, checkpoint)


if __name__ == "__main__":
    unittest.main()
