"""Exact image-index correspondence for isolated native reference evaluation.

Private to the post-freeze reference process. A mask on an ACDC 3-D image
re-export can be transferred without interpolation only when that image is
exactly the frozen canonical cine frame and its mask shares the re-export grid.
The changed header is explicit provenance; original files are never modified.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def transfer_reference_indices(mask_path: str | Path, original_image_path: str | Path,
                               canonical_record: Mapping[str, Any],
                               output_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    """Caller must validate the complete epoch freeze before invoking this.

    Re-hash the canonical image against the frozen image manifest, compare the
    exact frame arrays, require the original mask/image grids to agree, and
    attach the canonical native affine to identical label indices. No spatial
    interpolation, registration or mask-dependent alignment is performed.
    """
    import nibabel as nib

    mask_path, original_image_path = Path(mask_path), Path(original_image_path)
    canonical_path = Path(canonical_record.get("source_path") or canonical_record["path"])
    source_hash = _sha256(canonical_path)
    expected_hash = canonical_record.get("source_hash") or canonical_record.get("source_fingerprint")
    if source_hash != expected_hash:
        raise ValueError("canonical source image changed since the frozen image manifest")
    canonical_image = nib.load(str(canonical_path))
    original_image = nib.load(str(original_image_path))
    if len(original_image.shape) != 3 or len(canonical_image.shape) != 4:
        raise ValueError("index transfer requires a native 3-D image re-export and 4-D canonical cine")
    frame = int(canonical_record["frame_index"])
    if frame < 0 or frame >= canonical_image.shape[3]:
        raise ValueError("canonical frame index is out of bounds")
    expected_affine = np.asarray(canonical_record["native_affine"], dtype=np.float64)
    if not np.allclose(canonical_image.affine, expected_affine, rtol=1e-5, atol=1e-5):
        raise ValueError("canonical image affine differs from its frozen image manifest")
    original_values = np.asanyarray(original_image.dataobj)
    canonical_values = np.asanyarray(canonical_image.dataobj[..., frame])
    if original_values.shape != canonical_values.shape or not np.array_equal(original_values, canonical_values):
        raise ValueError("3-D image does not exactly equal the canonical cine frame; no index transfer")

    mask_image = nib.load(str(mask_path))
    if mask_image.shape != original_image.shape or not np.allclose(
            mask_image.affine, original_image.affine, rtol=1e-5, atol=1e-5):
        raise ValueError("reference mask grid differs from its original 3-D image")
    if mask_image.header.get_xyzt_units()[0] != original_image.header.get_xyzt_units()[0]:
        raise ValueError("reference mask spatial units differ from its original 3-D image")
    original_hash, mask_hash = _sha256(original_image_path), _sha256(mask_path)
    proof = {
        "operation": "exact_image_index_transfer", "no_resampling": True,
        "canonical_source_path": str(canonical_path), "canonical_source_sha256": source_hash,
        "canonical_volume_id": canonical_record["volume_id"], "canonical_frame_index": frame,
        "original_image_path": str(original_image_path), "original_image_sha256": original_hash,
        "original_mask_path": str(mask_path), "original_mask_sha256": mask_hash,
        "original_affine": original_image.affine.tolist(), "canonical_affine": canonical_image.affine.tolist(),
        "proof": "exact image array equality and matching original mask/image native grid",
    }
    token = hashlib.sha256(json.dumps(proof, sort_keys=True).encode()).hexdigest()[:24]
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    output_path = root / f"index_transfer_{token}.nii.gz"
    sidecar_path = root / f"index_transfer_{token}.json"
    if not output_path.exists():
        image = nib.Nifti1Image(np.asanyarray(mask_image.dataobj), canonical_image.affine)
        image.header.set_xyzt_units(canonical_image.header.get_xyzt_units()[0])
        staging = root / f"index_transfer_{token}.tmp-{os.getpid()}.nii.gz"
        nib.save(image, staging)
        os.replace(staging, output_path)
    else:
        existing = nib.load(str(output_path))
        if ((sidecar_path.is_file() and json.loads(sidecar_path.read_text()) != proof) or
                not np.array_equal(np.asanyarray(existing.dataobj), np.asanyarray(mask_image.dataobj)) or
                not np.array_equal(existing.affine, canonical_image.affine)):
            raise ValueError("existing aligned reference differs from its exact index proof")
    if not sidecar_path.exists():
        staging = sidecar_path.with_suffix(f".tmp-{os.getpid()}")
        staging.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
        os.replace(staging, sidecar_path)
    return output_path, proof
