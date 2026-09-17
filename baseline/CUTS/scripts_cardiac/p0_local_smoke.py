"""Fixture-only end-to-end P0 smoke: no clinical data and no GT."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cardiac_benchmark.cluster_kmeans import export_raw_partition
from cardiac_benchmark.export_latents import export_latents
from cardiac_benchmark.freeze import evaluator_skeleton, seal_raw_bundle
from cardiac_benchmark.manifest import counts_by_split, load_manifest, write_manifest
from cardiac_benchmark.provenance import write_json
from cardiac_benchmark.train_stage1 import Stage1Config, train_stage1


def make_fixture_manifest(root: Path) -> Path:
    source = root / "fixture_image.npy"
    # Three image-only Z slices. This is explicitly a non-scientific fixture.
    np.save(source, np.stack([np.full((16, 16), z, dtype=np.float32) for z in range(3)]), allow_pickle=False)
    records = []
    for patient, split, z in (("fixture_train", "train", 0), ("fixture_dev", "dev", 1), ("fixture_test", "test", 2)):
        records.append({"dataset": "acdc", "patient_id": patient, "split": split, "sample_id": f"{patient}:z{z:04d}",
                        "acquisition_id": "fixture_acquisition", "source_path": str(source), "image_checksum": "fixture-only",
                        "slice_index": z, "context_indices": [max(0, z - 1), z, min(2, z + 1)], "frame_index": None,
                        "frame_axis": None, "depth_axis": 0, "native_shape": [3, 16, 16], "spacing": [1.0, 1.0, 1.0],
                        "affine": None, "orientation": None})
    path = root / "fixture_manifest.json"
    write_manifest({"manifest_kind": "mock", "dataset": "acdc", "freemask_source_sha": "96c32b10fc7b8e09b48822e10ae9eb6cc149e253",
                    "freemask_discovery_contract": {"fixture_only": True}, "image_roots": [str(root)], "records": records}, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output).resolve(); root.mkdir(parents=True, exist_ok=True)
    manifest_path = make_fixture_manifest(root)
    config = Stage1Config(profile="CUTS-2D", dataset="acdc", manifest_path=str(manifest_path), batch_size=1,
                          num_workers=0, max_epochs=1, sampled_patches_per_image=2, scientific_run=False)
    checkpoint = root / "checkpoint.pt"
    train_result = train_stage1(config, checkpoint, device="cpu")
    latents = export_latents(config, checkpoint, root / "latents", split="train", device="cpu")
    partitions = [export_raw_partition(item, root / "partitions", num_workers=1) for item in latents]
    expected = {item["sample_id"] for item in latents}
    bundle = seal_raw_bundle(partitions, root / "freeze", required_sample_ids=expected)
    receipt = evaluator_skeleton(root / "freeze" / "raw_bundle.json", root / "evaluation", expected_sample_ids=expected)
    manifest = load_manifest(manifest_path, check_paths=True)
    result = {"fixture_only": True, "manifest_hash": manifest["manifest_hash"], "counts": counts_by_split(manifest),
              "train": train_result, "latent_count": len(latents), "partition_statuses": [p["status"] for p in partitions],
              "bundle_hash": bundle["bundle_hash"], "evaluator": receipt}
    write_json(root / "smoke_receipt.json", result)
    print(root / "smoke_receipt.json")


if __name__ == "__main__":
    main()
