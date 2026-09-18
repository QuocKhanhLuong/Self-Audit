"""DFC accessors for the project-level shared benchmark manifest."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from shared_benchmark.manifest import (
    MANIFEST_SCHEMA_VERSION as SCHEMA_VERSION,
    SharedManifestError as ManifestError,
    build_shared_manifest,
    manifest_hash,
    validate_manifest,
    validate_scientific_manifest,
)
from shared_benchmark.provenance import sha256_file
from shared_benchmark.spatial import build_grid_spec


def validate_scientific_run(manifest: Mapping[str, Any], root: str | Path) -> None:
    """DFC's scientific gate is the common gate, not a parallel schema."""
    validate_scientific_manifest(manifest, image_root=root)


def make_fixture_manifest(root: str | Path, sample_ids: list[str] | None = None) -> dict[str, Any]:
    """Build a marked fixture projection from a FreeMask-shaped image-only source.

    This helper exists exclusively for bounded local tests.  It cannot create a
    scientific manifest because the shared projector requires exactly one status.
    """
    root = Path(root)
    sample_ids = sample_ids or ["fixture:patient-001:frame-00:slice-01"]
    records = []
    for index, _ in enumerate(sample_ids):
        path = root / f"image_{index}.npy"
        if not path.is_file():
            raise ManifestError(f"fixture image missing: {path}")
        import numpy as np

        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.ndim != 3:
            raise ManifestError("fixture image must be rank-3 [Z,H,W]")
        patient = f"fixture-p{index:03d}"
        records.extend(
            {
                "volume_id": f"fixture:vol-{index:04d}:t0000",
                "unit_id": f"fixture:vol-{index:04d}:t0000:z{z:04d}",
                "study_id": f"fixture:{patient}", "patient_id": patient,
                "dataset": "fixture", "split": "test", "path": str(path),
                "source_path": str(path), "relative_path": path.name,
                "source_format": "npy", "shape": list(array.shape),
                "native_shape": list(array.shape), "dtype": str(array.dtype),
                "native_hw": [int(array.shape[1]), int(array.shape[2])],
                "depth": int(array.shape[0]), "num_slices": int(array.shape[0]),
                "depth_axis": 0, "frame_axis": None, "slice_index": z,
                "frame_index": 0, "frame_selection_rule": "single_acquired_frame_index_0_image_only",
                "native_geometry": "unavailable", "native_affine": None,
                "orientation": None, "spacing_mm": None, "spacing_valid": False,
                "native_grid_export": False, "export_grid": "stored", "spatial_unit": "unknown",
                "source_hash": sha256_file(path), "source_fingerprint": sha256_file(path),
                "frame_fingerprint": f"fixture-frame-{index:04d}",
                "study_grid_compatibility": None,
            }
            for z in range(array.shape[0])
        )
    upstream = {
        "schema_version": "maskfree150.data.v2", "dataset": "fixture", "seed": 42,
        "manifest_id": "fixture-freemask-manifest-v1", "records": records,
        "discovery_contract": {"version": "fixture-image-only-v1"},
        "split_provenance": {
            "rule": "fixture", "seed": 42, "ratios": {}, "patient_level_disjoint": True,
            "split_identity": "patient_id", "selection_inputs": ["fixture"],
            "content_fingerprint_used": False,
        },
    }
    grid = build_grid_spec((int(array.shape[1]), int(array.shape[2])), config_provenance={"source": "fixture"})
    return build_shared_manifest(upstream, grid, fixture=True, scientific=False, local_source_root=root)
