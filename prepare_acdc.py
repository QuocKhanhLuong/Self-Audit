#!/usr/bin/env python3
"""Create annotation-budgeted caches. Only selected TRAIN masks are opened.

Unlabeled images are complete cines, with no ED/ES lookup or U mask discovery.
Evaluation caches require a separate explicit --evaluation authorization.
"""
from pathlib import Path
import argparse
import re
import numpy as np
from self_audit.data import (
    CLASS_MAPPING,
    GEOMETRY_MAP_SCHEMA,
    GEOMETRY_SCHEMA,
    label_subset,
    write_json,
)
from self_audit.model import digest
from self_audit.preparation import native_geometry_zyx as _native_geometry_zyx
from self_audit.preparation import read_patient_split as _read_patient_split
from self_audit.preparation import affine_in_mm, spatial_unit


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


def prepare(root, split_path, out, budget, seed, evaluation=False, spatial_unit_override=None):
    import nibabel as nib  # Optional preparation dependency; never silently installed.
    root, out = Path(root).resolve(), Path(out).resolve()
    split = read_patient_split(split_path)
    all_ids = sum([split[k] for k in ("train", "val", "test")], [])
    if len(set(all_ids)) != len(all_ids) or any(not re.fullmatch(r"patient\d{3}", x) for x in all_ids):
        raise ValueError("unsafe or overlapping patient split")
    labeled_ids = label_subset(split["train"], budget, seed)
    if out.exists() or out == root or root.is_relative_to(out):
        raise ValueError("new separate output directory required")
    out.mkdir(parents=True)
    groups = {"labeled": [], "unlabeled": [], "evaluation": []}; source_images = {}; geometry_by_patient = {}; spatial_units = {}
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
                if mask.shape != frame.shape or not np.allclose(mask.affine, frame.affine):
                    raise ValueError("reference/frame geometry mismatch")
                if not np.allclose(frame.affine, ci.affine):
                    raise ValueError("frame/cine geometry mismatch")
                frame_unit = spatial_unit(frame, spatial_unit_override)
                if frame_unit != unit:
                    raise ValueError("frame/cine spatial units differ; normalize source headers explicitly")
                native_geometry_zyx(affine_in_mm(frame.affine, frame_unit), arr.shape[1:])
                if not 0 <= t < raw.shape[3] or not np.array_equal(np.asarray(frame.dataobj), raw[..., t]):
                    raise ValueError("frame/cine exact voxel bridge failed")
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
               "class_mapping": dict(CLASS_MAPPING)})
    return digest(out / "train.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True); p.add_argument("--split", required=True)
    p.add_argument("--out", required=True); p.add_argument("--labels", type=int, default=8)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--evaluation", action="store_true")
    p.add_argument("--spatial-unit", choices=("mm", "m", "micron"))
    a = p.parse_args(); print(prepare(a.root, a.split, a.out, a.labels, a.seed, a.evaluation, a.spatial_unit))
