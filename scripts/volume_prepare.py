#!/usr/bin/env python3
"""Prepare a hash-pinned native geometry sidecar from a TRAIN manifest.

The default path inspects image arrays only and never opens U/VAL/TEST masks.
For real data, ``--physical-metadata`` supplies explicit hash-pinned NIfTI
image/mask/frame sources; those are validated for units, orientation, affine,
frame mapping and exact frame/cine voxel correspondence.  Missing physical
metadata fails closed.  ``--synthetic`` is accepted only for a manifest marked
``synthetic: true``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from self_audit.model import digest
from self_audit.volume.geometry import (
    AXIS_ORDER,
    SCHEMA,
    forbidden_mask_reference,
    sha256_file,
    validate_nifti_correspondence,
    validate_sidecar,
)


def build_sidecar(manifest_path: str | Path, manifest_sha256: str, *, synthetic: bool = False,
                  affine_overrides: dict[str, object] | None = None,
                  physical_metadata: dict[str, object] | None = None) -> dict:
    manifest_path = Path(manifest_path).resolve()
    if digest(manifest_path) != manifest_sha256:
        raise ValueError("training manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "self_audit.data.v2" or manifest.get("role") != "training":
        raise ValueError("expected a v2 training manifest")
    if type(manifest.get("synthetic")) is not bool:
        raise ValueError("training manifest requires an explicit synthetic identity")
    if bool(manifest.get("synthetic")) != bool(synthetic):
        raise ValueError("synthetic caller/manifest identity mismatch")
    records = list(manifest.get("labeled", [])) + list(manifest.get("unlabeled", []))
    record_ids = [str(rec.get("id")) for rec in records]
    if any(not pid or pid == "None" for pid in record_ids) or len(set(record_ids)) != len(record_ids):
        raise ValueError("TRAIN patient IDs must be unique across labeled and unlabeled roles")
    unlabeled_ids = {str(r.get("id")) for r in manifest.get("unlabeled", [])}
    patients = {}
    for rec in records:
        # U records are intentionally checked before any other operation: they
        # may carry image metadata only, never mask references.
        is_unlabeled = str(rec.get("id")) in unlabeled_ids
        if is_unlabeled and set(rec) != {"id", "image", "image_sha256"}:
            raise ValueError("unlabeled record contains forbidden non-image fields")
        supplied = rec.get("geometry")
        supplied = supplied if isinstance(supplied, dict) else {}
        supplied_extra = (physical_metadata or {}).get(rec["id"], {}) if isinstance(physical_metadata, dict) else {}
        if isinstance(supplied_extra, dict):
            supplied = {**supplied, **supplied_extra}
        if is_unlabeled:
            forbidden = forbidden_mask_reference(supplied)
            if forbidden is not None:
                raise ValueError(f"unlabeled physical metadata contains forbidden mask reference: {forbidden}")
        image_path = Path(rec["image"])
        if not image_path.is_absolute():
            image_path = manifest_path.parent / image_path
        if digest(image_path) != rec["image_sha256"]:
            raise ValueError(f"image hash mismatch for {rec['id']}")
        image = np.load(image_path, mmap_mode="r", allow_pickle=False)
        if image.ndim != 4 or any(x < 1 for x in image.shape) or not np.isfinite(image).all():
            raise ValueError(f"invalid normalized TZYX image for {rec['id']}")
        override = (affine_overrides or {}).get(rec["id"])
        affine = override if override is not None else supplied.get("affine_xyz_to_world_mm") or rec.get("affine_xyz_to_world_mm")
        if affine is None:
            if not synthetic:
                raise ValueError(f"missing physical affine for {rec['id']}; refusing to guess spacing")
            affine = np.eye(4).tolist()
        if not synthetic and "orientation" not in supplied:
            raise ValueError(f"missing explicit orientation metadata for {rec['id']}")
        if not synthetic and "slice_order" not in supplied:
            raise ValueError(f"missing explicit slice_order metadata for {rec['id']}")
        orientation = supplied.get("orientation", "affine_direction_cosines")
        slice_order = supplied.get("slice_order", "increasing_native_z_index")
        frame_mapping = supplied.get("temporal_frame_mapping")
        if frame_mapping is None:
            if not synthetic:
                raise ValueError(f"missing explicit temporal_frame_mapping for {rec['id']}")
            frame_mapping = list(range(int(image.shape[0])))
        source_image = rec.get("source_image")
        if source_image is None:
            source_image = supplied.get("source_image")
        if not synthetic and source_image is None:
            raise ValueError(f"missing hash-pinned source_image metadata for {rec['id']}")
        mask_sources = supplied.get("mask_sources")
        if mask_sources is not None and not isinstance(mask_sources, dict):
            raise ValueError(f"mask_sources for {rec['id']} must be an object")
        if is_unlabeled and mask_sources is not None:
            raise ValueError(f"unlabeled patient {rec['id']} cannot carry mask physical metadata")
        if not synthetic and not is_unlabeled:
            manifest_frames = {str(k) for k in rec.get("masks", {})}
            if not isinstance(mask_sources, dict) or set(str(k) for k in mask_sources) != manifest_frames:
                raise ValueError(f"missing or incomplete hash-pinned mask_sources for labeled patient {rec['id']}")
        if source_image is not None:
            if not isinstance(source_image, dict):
                raise ValueError(f"source_image physical metadata for {rec['id']} must be an object")
            source_path = source_image.get("path", source_image.get("source_path"))
            source_sha = source_image.get("sha256", source_image.get("source_sha256"))
            if not isinstance(source_path, str) or not isinstance(source_sha, str):
                raise ValueError(f"source_image for {rec['id']} requires path and sha256")
            validate_nifti_correspondence(
                source_path, image_sha256=source_sha,
                expected_shape_tzyx=tuple(int(x) for x in image.shape),
                expected_affine_xyz_to_world_mm=affine,
                expected_orientation=str(orientation),
                expected_frame_mapping=[int(x) for x in frame_mapping],
                masks=mask_sources if mask_sources else None,
                base_dir=manifest_path.parent,
                prepared_image_tzyx=image,
                prepared_masks_tzyx=(
                    {str(frame): np.asarray(
                        np.load((manifest_path.parent / item["path"]) if not Path(item["path"]).is_absolute() else item["path"],
                                mmap_mode="r", allow_pickle=False)
                    ) for frame, item in rec.get("masks", {}).items()}
                    if mask_sources else None
                ),
            )
        patients[rec["id"]] = {
            "native_shape_tzyx": list(image.shape), "affine_xyz_to_world_mm": affine,
            "axis_order": AXIS_ORDER, "units": "mm",
            "orientation": orientation,
            "slice_order": slice_order,
            "temporal_frame_mapping": list(frame_mapping),
            "image_sha256": rec["image_sha256"],
        }
        if source_image is not None:
            patients[rec["id"]]["source_image"] = dict(source_image)
        if mask_sources is not None:
            patients[rec["id"]]["mask_sources"] = dict(mask_sources)
    return validate_sidecar({
        "schema": SCHEMA, "axis_order": AXIS_ORDER, "units": "mm", "synthetic": bool(synthetic),
        "source_manifest": manifest_path.name, "source_manifest_sha256": manifest_sha256,
        "patients": patients,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--synthetic", action="store_true", help="allow identity affine for manufactured fixtures only")
    parser.add_argument("--physical-metadata", type=Path,
                        help="JSON mapping patient IDs to hash-pinned source_image/mask_sources metadata")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists; choose a new sidecar path")
    physical = None
    if args.physical_metadata is not None:
        try:
            physical = json.loads(args.physical_metadata.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"cannot read --physical-metadata: {exc}")
        if not isinstance(physical, dict):
            parser.error("--physical-metadata must contain a patient-ID object")
    sidecar = build_sidecar(args.manifest, args.manifest_sha256, synthetic=args.synthetic,
                            physical_metadata=physical)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(sidecar, indent=2, sort_keys=True, allow_nan=False) + "\n")
    mask_paths_read = sum(
        len(item.get("mask_sources", {}))
        for item in (physical or {}).values()
        if isinstance(item, dict) and isinstance(item.get("mask_sources", {}), dict)
    )
    print(json.dumps({"geometry": str(args.out.resolve()), "geometry_sha256": sha256_file(args.out),
                      "mask_paths_read": mask_paths_read, "synthetic": bool(args.synthetic)}, indent=2))


if __name__ == "__main__":
    main()
