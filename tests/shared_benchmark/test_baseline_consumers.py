from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# from main_tbc
from shared_benchmark.manifest import write_shared_manifest
from helpers import discovered_projection, write_image

# from backup_tbc
REPO_ROOT = Path(__file__).resolve().parents[2]
CUTS_SRC = REPO_ROOT / "baseline" / "CUTS" / "src"
STEGO_SRC = REPO_ROOT / "baseline" / "STEGO" / "src"
PICIE_SRC = REPO_ROOT / "baseline" / "PICIE" / "src"
def _consumer_row(repo_root, baseline, manifest_path):
    code = """
import json, sys
from pathlib import Path
repo = Path(sys.argv[1]); manifest_path = Path(sys.argv[2])
sys.path.insert(0, str(repo / 'src')); sys.path.insert(0, str(repo / 'baseline' / sys.argv[3] / 'src'))
from shared_benchmark.manifest import load_shared_manifest
manifest = load_shared_manifest(manifest_path)
baseline = sys.argv[3]
if baseline == 'CUTS':
    from cardiac_benchmark.dataset import ImageOnlyCardiacDataset
    sample = ImageOnlyCardiacDataset(manifest, split='test', profile='CUTS-2D')[0]
    row = sample.provenance
elif baseline == 'DFC':
    from cardiac_benchmark.dataset import load_primary_2d
    record = next(row for row in manifest['records'] if row['split'] == 'test')
    _, row = load_primary_2d(record, manifest['local_receipt']['local_source_root'])
elif baseline == 'STEGO':
    from cardiac_benchmark.dataset import STEGOCardiacDataset
    sample = STEGOCardiacDataset(manifest, split='test', profile='STEGO-2D')[0]
    row = sample.provenance
elif baseline == 'PICIE':
    from cardiac_benchmark.dataset import PICIECardiacDataset
    sample = PICIECardiacDataset(manifest, split='test', profile='PICIE-2D')[0]
    row = sample.provenance
else:
    raise AssertionError(f'unknown baseline: {baseline}')
print(json.dumps(row, sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(repo_root), str(manifest_path), baseline],
        check=True, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(repo_root / "src")},
    )
    return json.loads(result.stdout)


def test_all_baselines_consume_identical_shared_sample_identity_and_grid(tmp_path):
    for index in range(7):
        write_image(tmp_path, f"patient{index:03d}.npy", seed=index)
    _, manifest = discovered_projection(tmp_path, target_hw=(8, 8))
    path = tmp_path / "shared.json"
    write_shared_manifest(manifest, path)
    repo_root = __import__("pathlib").Path(__file__).resolve().parents[2]
    expected = next(row for row in manifest["records"] if row["split"] == "test")
    rows = [_consumer_row(repo_root, baseline, path) for baseline in ("CUTS", "DFC", "STEGO", "PICIE")]
    for row in rows:
        assert row["sample_id"] == expected["sample_id"]
        assert row["patient_id"] == expected["patient_id"]
        assert row["split"] == expected["split"]
        assert row["frame_index"] == expected["frame_index"]
        assert row["slice_index"] == expected["slice_index"]
        assert row["context_indices"] == expected["context_indices"]
        assert row["target_shape"] == expected["shared_grid"]["target_hw"]
def _import_cuts():
    import sys
    sys.modules.pop("cardiac_benchmark", None)
    sys.modules.pop("cardiac_benchmark.manifest", None)
    sys.modules.pop("cardiac_benchmark.dataset", None)
    sys.modules.pop("cardiac_benchmark.config", None)
    if str(CUTS_SRC) not in sys.path:
        sys.path.insert(0, str(CUTS_SRC))
    from cardiac_benchmark import manifest as cuts_manifest
    from cardiac_benchmark import dataset as cuts_dataset
    return cuts_manifest, cuts_dataset


def _import_stego():
    import sys
    sys.modules.pop("cardiac_benchmark", None)
    sys.modules.pop("cardiac_benchmark.manifest", None)
    sys.modules.pop("cardiac_benchmark.dataset", None)
    sys.modules.pop("cardiac_benchmark.config", None)
    if str(STEGO_SRC) not in sys.path:
        sys.path.insert(0, str(STEGO_SRC))
    from cardiac_benchmark import manifest as stego_manifest
    from cardiac_benchmark import dataset as stego_dataset
    return stego_manifest, stego_dataset

def _import_picie():
    import sys
    sys.modules.pop("cardiac_benchmark", None)
    sys.modules.pop("cardiac_benchmark.manifest", None)
    sys.modules.pop("cardiac_benchmark.dataset", None)
    sys.modules.pop("cardiac_benchmark.config", None)
    if str(PICIE_SRC) not in sys.path:
        sys.path.insert(0, str(PICIE_SRC))
    from cardiac_benchmark import manifest as picie_manifest
    from cardiac_benchmark import dataset as picie_dataset
    return picie_manifest, picie_dataset


def _make_shared_manifest(root: Path) -> Path:
    """Create a minimal manifest that baselines can consume."""
    source = root / "volume.npy"
    array = np.stack(
        [np.full((16, 16), float(v), dtype=np.float32) for v in range(5)],
        axis=0,
    )
    np.save(source, array, allow_pickle=False)
    records = []
    for patient, split, z in [
        ("patient001", "train", 0),
        ("patient001", "train", 1),
        ("patient002", "dev", 2),
        ("patient003", "test", 3),
        ("patient003", "test", 4),
    ]:
        records.append({
            "dataset": "acdc",
            "patient_id": patient,
            "split": split,
            "sample_id": f"{patient}:z{z:04d}",
            "acquisition_id": "acq_shared",
            "source_path": str(source),
            "image_checksum": "shared-fixture-hash",
            "slice_index": z,
            "context_indices": [max(0, z - 1), z, min(4, z + 1)],
            "frame_index": None,
            "frame_axis": None,
            "depth_axis": 0,
            "native_shape": [5, 16, 16],
            "spacing": [1.5, 1.25, 1.25],
        })
    # Use STEGO's writer (all should produce identical manifests)
    stego_manifest, _ = _import_stego()
    payload = {
        "manifest_kind": "mock",
        "dataset": "acdc",
        "freemask_source_sha": "96c32b10fc7b8e09b48822e10ae9eb6cc149e253",
        "freemask_discovery_contract": {"fixture_only": True},
        "image_roots": [str(root)],
        "records": records,
    }
    path = root / "shared_manifest.json"
    stego_manifest.write_manifest(payload, path)
    return path


class CrossBaselineManifestTests(unittest.TestCase):
    """Baselines must read the same manifest identically."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)
        self.manifest_path = _make_shared_manifest(self.tmp)

    def tearDown(self):
        self._tmpdir.cleanup()

    def _load_with_cuts(self):
        cuts_manifest, _ = _import_cuts()
        return cuts_manifest.load_manifest(self.manifest_path, check_paths=True)

    def _load_with_stego(self):
        stego_manifest, _ = _import_stego()
        return stego_manifest.load_manifest(self.manifest_path, check_paths=True)

    def _load_with_picie(self):
        picie_manifest, _ = _import_picie()
        return picie_manifest.load_manifest(self.manifest_path, check_paths=True)

    def test_same_schema_version(self):
        cuts_manifest, _ = _import_cuts()
        stego_manifest, _ = _import_stego()
        picie_manifest, _ = _import_picie()
        self.assertEqual(
            cuts_manifest.MANIFEST_SCHEMA_VERSION,
            stego_manifest.MANIFEST_SCHEMA_VERSION,
        )
        self.assertEqual(
            cuts_manifest.MANIFEST_SCHEMA_VERSION,
            picie_manifest.MANIFEST_SCHEMA_VERSION,
        )

    def test_both_load_same_manifest(self):
        """Baselines load the manifest without error."""
        cuts_m = self._load_with_cuts()
        stego_m = self._load_with_stego()
        picie_m = self._load_with_picie()
        self.assertIsNotNone(cuts_m)
        self.assertIsNotNone(stego_m)
        self.assertIsNotNone(picie_m)

    def test_same_manifest_hash(self):
        """Baselines compute the same manifest_hash (shared_grid_hash)."""
        cuts_m = self._load_with_cuts()
        stego_m = self._load_with_stego()
        picie_m = self._load_with_picie()
        self.assertEqual(cuts_m["manifest_hash"], stego_m["manifest_hash"])
        self.assertEqual(cuts_m["manifest_hash"], picie_m["manifest_hash"])

    def test_same_sample_ids(self):
        """Baselines see identical sample_id sets."""
        cuts_m = self._load_with_cuts()
        stego_m = self._load_with_stego()
        picie_m = self._load_with_picie()
        cuts_ids = {r["sample_id"] for r in cuts_m["records"]}
        stego_ids = {r["sample_id"] for r in stego_m["records"]}
        picie_ids = {r["sample_id"] for r in picie_m["records"]}
        self.assertEqual(cuts_ids, stego_ids)
        self.assertEqual(cuts_ids, picie_ids)

    def test_same_patient_ids(self):
        """Baselines see identical patient_id sets."""
        cuts_m = self._load_with_cuts()
        stego_m = self._load_with_stego()
        picie_m = self._load_with_picie()
        cuts_pids = {r["patient_id"] for r in cuts_m["records"]}
        stego_pids = {r["patient_id"] for r in stego_m["records"]}
        picie_pids = {r["patient_id"] for r in picie_m["records"]}
        self.assertEqual(cuts_pids, stego_pids)
        self.assertEqual(cuts_pids, picie_pids)

    def test_same_split_counts(self):
        """Baselines report identical split counts."""
        cuts_manifest, _ = _import_cuts()
        stego_manifest, _ = _import_stego()
        picie_manifest, _ = _import_picie()
        cuts_m = self._load_with_cuts()
        stego_m = self._load_with_stego()
        picie_m = self._load_with_picie()
        cuts_counts = cuts_manifest.counts_by_split(cuts_m)
        stego_counts = stego_manifest.counts_by_split(stego_m)
        picie_counts = picie_manifest.counts_by_split(picie_m)
        self.assertEqual(cuts_counts, stego_counts)
        self.assertEqual(cuts_counts, picie_counts)

    def test_record_manifest_hash_matches_top_level(self):
        """Each record's manifest_hash equals the top-level hash in all baselines."""
        for label, manifest in [
            ("cuts", self._load_with_cuts()),
            ("stego", self._load_with_stego()),
            ("picie", self._load_with_picie())
        ]:
            for record in manifest["records"]:
                self.assertEqual(
                    record["manifest_hash"],
                    manifest["manifest_hash"],
                    f"{label} record {record['sample_id']} has wrong manifest_hash",
                )

    def test_both_reject_gt_contaminated_manifest(self):
        """All baselines reject manifests with GT keys."""
        import copy
        import json

        stego_m = self._load_with_stego()
        bad = copy.deepcopy(stego_m)
        bad["records"][0]["annotation_path"] = "/forbidden/mask.npy"
        bad_path = self.tmp / "bad_manifest.json"
        bad_path.write_text(json.dumps(bad), encoding="utf-8")

        cuts_manifest, _ = _import_cuts()
        stego_manifest, _ = _import_stego()
        picie_manifest, _ = _import_picie()

        with self.assertRaises(Exception):
            cuts_manifest.load_manifest(bad_path)
        with self.assertRaises(Exception):
            stego_manifest.load_manifest(bad_path)
        with self.assertRaises(Exception):
            picie_manifest.load_manifest(bad_path)


class CrossBaselineDatasetTests(unittest.TestCase):
    """Dataset adapters produce compatible provenance fields."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)
        self.manifest_path = _make_shared_manifest(self.tmp)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_stego_dataset_provenance_has_required_fields(self):
        """STEGO dataset provenance has the same core keys as CUTS."""
        _, stego_dataset = _import_stego()
        stego_manifest, _ = _import_stego()
        manifest = stego_manifest.load_manifest(self.manifest_path, check_paths=True)
        ds = stego_dataset.STEGOCardiacDataset(manifest, split="train", resolution=16)
        sample = ds[0]
        shared_keys = {
            "sample_id", "patient_id", "split", "dataset",
            "manifest_hash", "normalization", "input_channels",
        }
        missing = shared_keys - set(sample.provenance)
        self.assertFalse(missing, f"STEGO missing shared provenance keys: {missing}")

    def test_stego_manifest_hash_matches_manifest(self):
        """STEGO dataset reports the same manifest_hash as the loaded manifest."""
        _, stego_dataset = _import_stego()
        stego_manifest, _ = _import_stego()
        manifest = stego_manifest.load_manifest(self.manifest_path, check_paths=True)
        ds = stego_dataset.STEGOCardiacDataset(manifest, split="train", resolution=16)
        sample = ds[0]
        self.assertEqual(sample.provenance["manifest_hash"], manifest["manifest_hash"])

    def test_picie_dataset_provenance_has_required_fields(self):
        """PICIE dataset provenance has the same core keys as CUTS."""
        _, picie_dataset = _import_picie()
        picie_manifest, _ = _import_picie()
        manifest = picie_manifest.load_manifest(self.manifest_path, check_paths=True)
        ds = picie_dataset.PICIECardiacDataset(manifest, split="train", target_hw=(16, 16))
        sample = ds[0]
        shared_keys = {
            "sample_id", "patient_id", "split", "dataset",
            "manifest_hash", "normalization", "input_channels", "augmentation_policy"
        }
        missing = shared_keys - set(sample.provenance)
        self.assertFalse(missing, f"PICIE missing shared provenance keys: {missing}")

    def test_picie_manifest_hash_matches_manifest(self):
        """PICIE dataset reports the same manifest_hash as the loaded manifest."""
        _, picie_dataset = _import_picie()
        picie_manifest, _ = _import_picie()
        manifest = picie_manifest.load_manifest(self.manifest_path, check_paths=True)
        ds = picie_dataset.PICIECardiacDataset(manifest, split="train", target_hw=(16, 16))
        sample = ds[0]
        self.assertEqual(sample.provenance["manifest_hash"], manifest["manifest_hash"])


if __name__ == "__main__":
    unittest.main()
