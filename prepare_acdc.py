#!/usr/bin/env python3
"""Create annotation-budgeted caches. Only selected TRAIN masks are opened.

Unlabeled images are complete cines, with no ED/ES lookup or U mask discovery.
Evaluation caches require a separate explicit --evaluation authorization.
"""
import argparse
import hashlib
import re
from pathlib import Path

import numpy as np

from self_audit.data import (
    CLASS_MAPPING,
    GEOMETRY_MAP_SCHEMA,
    GEOMETRY_SCHEMA,
    label_subset,
    write_json,
)
from self_audit.model import digest
from self_audit.preparation import SPATIAL_UNIT_SCALE_MM, affine_in_mm, spatial_unit
from self_audit.preparation import native_geometry_zyx as _native_geometry_zyx
from self_audit.preparation import read_patient_split as _read_patient_split


def one(paths):
    paths = list(paths)
    if len(paths) != 1: raise ValueError(f"expected exactly one input, found {len(paths)}")
    return paths[0]


def native_geometry_zyx(affine, shape_zyx):
    """Compatibility wrapper for the shared native geometry validator."""
    return _native_geometry_zyx(affine, shape_zyx)



def read_patient_split(path):
    """Compatibility wrapper for the shared patient split reader."""
    return _read_patient_split(path)


def _header_sha256(image):
    """Hash the source NIfTI header without rewriting or normalizing it."""
    header = getattr(image, "header", None)
    block = getattr(header, "binaryblock", None)
    if block is None:
        block = repr(header).encode("utf-8")
    return hashlib.sha256(bytes(block)).hexdigest()


def _zooms_mm(image, unit):
    """Read declared voxel spacing, falling back only for tiny test doubles."""
    header = getattr(image, "header", None)
    getter = getattr(header, "get_zooms", None)
    if callable(getter):
        values = np.asarray(getter()[:3], dtype=np.float64)
    else:  # pragma: no cover - only minimal test doubles omit get_zooms
        values = np.linalg.norm(np.asarray(image.affine, dtype=np.float64)[:3, :3], axis=0)
    if values.shape != (3,) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("source image has invalid voxel spacing")
    values *= SPATIAL_UNIT_SCALE_MM[unit]
    return values


def _same_spacing(left, right):
    return bool(np.allclose(left, right, rtol=1e-5, atol=1e-5))


def _resolve_aux_unit(image, cine_unit, *, override, bridge_allowed, name):
    """Resolve a frame/mask unit, allowing inheritance only after a bridge."""
    try:
        return spatial_unit(image, override), "declared_or_override"
    except ValueError:
        if override is None and bridge_allowed:
            return cine_unit, "inherited_from_cine_after_exact_bridge"
        raise ValueError(
            f"{name} spatial units are unknown; verified-cine-bridge requires "
            "a known cine unit or an explicit --spatial-unit override"
        )


def _geometry_audit(*, pid, frame_index, cine_path, frame_path, mask_path,
                    cine, frame, mask, cine_unit, frame_unit, mask_unit,
                    frame_unit_source, mask_unit_source, exact_bridge,
                    frame_mask_affine_equal, frame_cine_affine_equal,
                    cine_spacing, frame_spacing, mask_spacing, policy):
    def _affine_hash(value):
        return hashlib.sha256(np.asarray(value, dtype=np.float64).tobytes()).hexdigest()

    return {
        "patient_id": pid,
        "frame": int(frame_index),
        "policy": policy,
        "geometry_source": "cine_affine",
        "cine_file": str(cine_path),
        "frame_file": str(frame_path),
        "mask_file": str(mask_path),
        "cine_sha256": digest(cine_path),
        "frame_sha256": digest(frame_path),
        "mask_sha256": digest(mask_path),
        "cine_header_sha256": _header_sha256(cine),
        "frame_header_sha256": _header_sha256(frame),
        "mask_header_sha256": _header_sha256(mask),
        "cine_affine_sha256": _affine_hash(cine.affine),
        "frame_affine_sha256": _affine_hash(frame.affine),
        "mask_affine_sha256": _affine_hash(mask.affine),
        "cine_affine": np.asarray(cine.affine, dtype=np.float64).tolist(),
        "frame_affine": np.asarray(frame.affine, dtype=np.float64).tolist(),
        "mask_affine": np.asarray(mask.affine, dtype=np.float64).tolist(),
        "cine_header_units": getattr(cine.header, "get_xyzt_units", lambda: (None, None))()[0],
        "frame_header_units": getattr(frame.header, "get_xyzt_units", lambda: (None, None))()[0],
        "mask_header_units": getattr(mask.header, "get_xyzt_units", lambda: (None, None))()[0],
        "resolved_units": {"cine": cine_unit, "frame": frame_unit, "mask": mask_unit},
        "unit_resolution": {"frame": frame_unit_source, "mask": mask_unit_source},
        "spacing_mm": {
            "cine_xyz": np.asarray(cine_spacing, dtype=float).tolist(),
            "frame_xyz": np.asarray(frame_spacing, dtype=float).tolist(),
            "mask_xyz": np.asarray(mask_spacing, dtype=float).tolist(),
        },
        "exact_frame_cine_voxel_bridge": bool(exact_bridge),
        "frame_mask_affine_equal": bool(frame_mask_affine_equal),
        "frame_cine_affine_equal": bool(frame_cine_affine_equal),
        "accepted_header_affine_mismatch": bool(not frame_cine_affine_equal),
    }


def prepare(root, split_path, out, budget, seed, evaluation=False,
            spatial_unit_override=None, frame_geometry_policy="strict"):
    import nibabel as nib  # Optional preparation dependency; never silently installed.
    root, out = Path(root).resolve(), Path(out).resolve()
    split = read_patient_split(split_path)
    all_ids = [patient_id for role in ("train", "val", "test") for patient_id in split[role]]
    if len(set(all_ids)) != len(all_ids) or any(not re.fullmatch(r"patient\d{3}", x) for x in all_ids):
        raise ValueError("unsafe or overlapping patient split")
    if frame_geometry_policy not in {"strict", "verified-cine-bridge"}:
        raise ValueError("frame_geometry_policy must be strict or verified-cine-bridge")
    labeled_ids = label_subset(split["train"], budget, seed)
    if out.exists() or out == root or root.is_relative_to(out):
        raise ValueError("new separate output directory required")
    out.mkdir(parents=True)
    groups = {"labeled": [], "unlabeled": [], "evaluation": []}; source_images = {}; geometry_by_patient = {}; spatial_units = {}
    geometry_audits = []
    ids = split["train"] + (split["val"] if evaluation else [])
    for pid in ids:
        folder = root / pid
        cine_path = one(folder.glob(f"{pid}_4d.nii*")); cine_hash = digest(cine_path)
        ci = nib.load(cine_path); raw = np.asarray(ci.dataobj)
        if raw.ndim != 4 or not np.isfinite(raw).all(): raise ValueError("invalid raw cine")
        lo, hi = np.percentile(raw, [.5, 99.5])
        arr = np.clip((raw-lo)/max(float(hi-lo), 1e-6), 0, 1).transpose(3, 2, 1, 0).astype(np.float32)
        ip = out / f"{pid}.npy"; np.save(ip, arr)
        r = {"id": pid, "image": ip.name, "image_sha256": digest(ip)}
        unit = spatial_unit(ci, spatial_unit_override)
        spatial_units[pid] = unit
        native_geometry = native_geometry_zyx(affine_in_mm(ci.affine, unit), arr.shape[1:])
        geometry_by_patient[pid] = native_geometry
        role = "evaluation" if pid in split["val"] else ("labeled" if pid in labeled_ids else "unlabeled")
        if role != "unlabeled":
            r["masks"] = {}
            for mp in sorted(folder.glob(f"{pid}_frame*_gt.nii*")):
                t = int(re.search(r"_frame(\d+)_gt", mp.name)[1])-1
                mask = nib.load(mp)
                frame = nib.load(Path(str(mp).replace("_gt.nii", ".nii")))
                frame_data = np.asarray(frame.dataobj)
                exact_bridge = (
                    frame.shape == raw.shape[:3]
                    and 0 <= t < raw.shape[3]
                    and np.array_equal(frame_data, raw[..., t])
                )
                frame_mask_affine_equal = np.allclose(
                    mask.affine, frame.affine, rtol=0., atol=1e-6,
                )
                if mask.shape != frame.shape or not frame_mask_affine_equal:
                    raise ValueError("reference/frame geometry mismatch")
                frame_unit, frame_unit_source = _resolve_aux_unit(
                    frame, unit, override=spatial_unit_override,
                    bridge_allowed=(frame_geometry_policy == "verified-cine-bridge" and exact_bridge),
                    name="frame",
                )
                mask_unit, mask_unit_source = _resolve_aux_unit(
                    mask, unit, override=spatial_unit_override,
                    bridge_allowed=(frame_geometry_policy == "verified-cine-bridge" and exact_bridge),
                    name="mask",
                )
                if frame_unit != unit or mask_unit != unit:
                    raise ValueError("frame/cine spatial units differ; normalize source headers explicitly")
                cine_spacing = _zooms_mm(ci, unit)
                frame_spacing = _zooms_mm(frame, frame_unit)
                mask_spacing = _zooms_mm(mask, mask_unit)
                if not (_same_spacing(cine_spacing, frame_spacing)
                        and _same_spacing(frame_spacing, mask_spacing)):
                    raise ValueError("frame/mask/cine voxel spacing mismatch")
                frame_cine_affine_equal = np.allclose(
                    affine_in_mm(frame.affine, frame_unit),
                    affine_in_mm(ci.affine, unit), rtol=1e-5, atol=1e-5,
                )
                if frame_geometry_policy == "strict" and not frame_cine_affine_equal:
                    raise ValueError("frame/cine geometry mismatch")
                # The voxel bridge is a source-integrity invariant for both
                # policies.  The opt-in policy only relaxes the header-affine
                # equality check; it never permits a frame whose payload is
                # different from the cine time point.
                if not exact_bridge:
                    raise ValueError("frame/cine exact voxel bridge failed")
                geometry_audits.append(_geometry_audit(
                    pid=pid, frame_index=t, cine_path=cine_path,
                    frame_path=Path(str(mp).replace("_gt.nii", ".nii")), mask_path=mp,
                    cine=ci, frame=frame, mask=mask, cine_unit=unit,
                    frame_unit=frame_unit, mask_unit=mask_unit,
                    frame_unit_source=frame_unit_source, mask_unit_source=mask_unit_source,
                    exact_bridge=exact_bridge,
                    frame_mask_affine_equal=frame_mask_affine_equal,
                    frame_cine_affine_equal=frame_cine_affine_equal,
                    cine_spacing=cine_spacing, frame_spacing=frame_spacing,
                    mask_spacing=mask_spacing, policy=frame_geometry_policy,
                ))
                y = np.asarray(mask.dataobj)
                if not np.isin(y, [int(x) for x in CLASS_MAPPING]).all(): raise ValueError("unknown annotation class")
                op = out / f"{pid}_t{t:03d}_mask.npy"; np.save(op, y.transpose(2, 1, 0).astype(np.uint8))
                r["masks"][str(t)] = {"path": op.name, "sha256": digest(op)}
            if not r["masks"]: raise ValueError("budgeted/evaluation patient has no reference frames")
        groups[role].append(r); source_images[pid] = cine_hash
    if len(set(source_images.values())) != len(source_images):
        raise ValueError("duplicate raw patient content across roles; resolve split explicitly, never silently drop")
    train_manifest = {"schema": "self_audit.data.v2", "role": "training", "synthetic": False,
                      "dataset": "acdc",
                      "label_budget": budget, "label_seed": seed, "labeled": groups["labeled"],
                      "unlabeled": groups["unlabeled"], "evaluation_ids": split["val"]+split["test"],
                      "class_mapping": dict(CLASS_MAPPING),
                      "geometry": {"schema": GEOMETRY_MAP_SCHEMA,
                                   "by_patient": {r["id"]: geometry_by_patient[r["id"]]
                                                   for group in (groups["labeled"], groups["unlabeled"])
                                                   for r in group}}}
    write_json(out / "train.json", train_manifest)
    if evaluation:
        write_json(out / "development.json", {"schema": "self_audit.data.v2", "role": "development",
                     "synthetic": False, "dataset": "acdc", "patients": groups["evaluation"], "scope": "exposed_development",
                     "class_mapping": dict(CLASS_MAPPING),
                     "geometry": {"schema": GEOMETRY_MAP_SCHEMA,
                                  "by_patient": {r["id"]: geometry_by_patient[r["id"]] for r in groups["evaluation"]}}})
    write_json(out / "PREPARATION.json", {"schema": "self_audit.preparation.acdc.v1", "dataset": "acdc",
               "source_split_sha256": digest(split_path), "raw_images": source_images,
               "read_mask_patient_ids": labeled_ids + (split["val"] if evaluation else []),
               "unlabeled_mask_reads": 0, "normalization": "per_cine_image_only_percentiles_0.5_99.5",
               "train_manifest_sha256": digest(out/"train.json"), "evaluation_authorized": evaluation,
               "geometry_schema": GEOMETRY_SCHEMA, "geometry_axis_order": ["T", "Z", "Y", "X"],
               "spatial_unit_override": spatial_unit_override, "spatial_units": spatial_units,
               "frame_geometry_policy": frame_geometry_policy,
               "geometry_audits": geometry_audits,
               "geometry_policy_summary": {
                   "accepted_header_affine_mismatches": sum(
                       int(item["accepted_header_affine_mismatch"])
                       for item in geometry_audits
                   ),
                   "geometry_source": "cine_affine",
                   "raw_headers_unchanged": True,
               },
               "class_mapping": dict(CLASS_MAPPING)})
    return digest(out / "train.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True); p.add_argument("--split", required=True)
    p.add_argument("--out", required=True); p.add_argument("--labels", type=int, default=8)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--evaluation", action="store_true")
    p.add_argument("--spatial-unit", choices=("mm", "m", "micron"))
    p.add_argument("--frame-geometry-policy", choices=("strict", "verified-cine-bridge"), default="strict")
    a = p.parse_args(); print(prepare(a.root, a.split, a.out, a.labels, a.seed, a.evaluation,
                                      a.spatial_unit, a.frame_geometry_policy))
