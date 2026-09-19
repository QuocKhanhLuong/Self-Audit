"""PiCIE accessors for the project-level shared benchmark manifest."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from shared_benchmark.manifest import (
    MANIFEST_SCHEMA_VERSION,
    SharedManifestError as ManifestError,
    load_shared_manifest,
    validate_manifest as _validate_manifest,
    validate_scientific_manifest,
    write_shared_manifest,
)


SPLITS = ("train", "dev", "test")


def validate_manifest(payload: Mapping[str, Any], *, check_paths: bool = False) -> None:
    _validate_manifest(payload)
    if check_paths:
        root = payload.get("local_receipt", {}).get("local_source_root")
        if root is None:
            raise ManifestError("check_paths requires a declared local image-only root")
        for record in payload["records"]:
            if not (Path(root) / record["source"]["locator"]).is_file():
                raise ManifestError(f"image source does not exist: {record['sample_id']}")


def write_manifest(payload: Mapping[str, Any], path: str | Path) -> str:
    return write_shared_manifest(payload, path)


def load_manifest(path: str | Path, *, check_paths: bool = False) -> dict[str, Any]:
    payload = load_shared_manifest(path)
    validate_manifest(payload, check_paths=check_paths)
    return payload


def require_scientific_manifest(path: str | Path, *, image_root: str | Path | None = None) -> dict[str, Any]:
    payload = load_shared_manifest(path)
    root = image_root or payload.get("local_receipt", {}).get("local_source_root")
    if root is None:
        raise ManifestError("scientific PiCIE execution requires an explicit image-only root")
    validate_scientific_manifest(payload, image_root=root)
    return payload


def counts_by_split(manifest: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    return {
        split: {
            "patients": len({record["patient_id"] for record in manifest["records"] if record["split"] == split}),
            "samples": sum(record["split"] == split for record in manifest["records"]),
        }
        for split in SPLITS
    }
