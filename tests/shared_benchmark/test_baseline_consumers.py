from __future__ import annotations

import json
import os
import subprocess
import sys

from shared_benchmark.manifest import write_shared_manifest

from helpers import discovered_projection, write_image


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
