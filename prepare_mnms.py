#!/usr/bin/env python3
"""Prepare the original M&Ms (2020) challenge from an explicit index CSV.

The index is intentionally required because public M&Ms mirrors use different
filenames and may omit restricted cases.  A row is one patient and must contain
``patient_id,split,image,ed_frame,es_frame`` plus ``mask_ed,mask_es`` for an
annotated patient.  ``axis_order`` must be ``XYZT`` and
``mask_axis_order`` must be ``XYZ`` or ``XYZT``; arbitrary permutations are
rejected rather than inferred.  No axis, ED/ES, or mask discovery is inferred
from filenames.

The original challenge annotates LV, MYO and RV at ED/ES.  Pass its source map
explicitly, for example ``'{"0":"background","1":"LV","2":"MYO","3":"RV"}'``.
The output remaps those classes to the repository ontology
``0=background,1=RV,2=MYO,3=LV``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from self_audit.data import CLASS_MAPPING, GEOMETRY_MAP_SCHEMA, GEOMETRY_SCHEMA, label_subset, write_json
from self_audit.model import digest
from self_audit.preparation import (
    native_geometry_zyx,
    normalize_cine,
    parse_class_map,
    parse_frame,
    read_index_csv,
    remap_mask,
    resolve_path,
    affine_in_mm,
    spatial_unit,
    spatial_to_zyx,
)


def _import_nibabel():
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError("M&Ms preparation requires nibabel; install requirements.txt") from exc
    return nib


def _has_annotations(row: dict[str, str]) -> bool:
    return bool((row.get("mask_ed") or row.get("mask") or "").strip() and
                (row.get("mask_es") or row.get("mask") or "").strip())


def _axis(row: dict[str, str], key: str, explicit: str | None) -> str:
    value = (row.get(key) or explicit or "").strip().upper()
    if not value:
        raise ValueError(f"index row {row['patient_id']} needs explicit {key}")
    allowed = {"axis_order": {"XYZT"}, "mask_axis_order": {"XYZ", "XYZT"}}
    if key in allowed and value not in allowed[key]:
        raise ValueError(f"{key}={value!r} is unsupported; use {sorted(allowed[key])}")
    return value


def _source_spatial_shape(raw: np.ndarray, axis_order: str) -> tuple[int, int, int]:
    axes = tuple(axis_order.upper())
    return tuple(raw.shape[axes.index(axis)] for axis in "XYZ")


def _load_image(nib, path: Path, axis_order: str):
    obj = nib.load(path)
    raw = np.asarray(obj.dataobj)
    if raw.ndim != 4:
        raise ValueError(f"M&Ms cine must be 4-D: {path}")
    return obj, raw, normalize_cine(raw, axis_order), _source_spatial_shape(raw, axis_order)


def _load_mask_frame(nib, path: Path, *, target_frame: int, image_obj: Any,
                     image_spatial_shape: tuple[int, int, int], mask_axis_order: str,
                     source_map: dict[int, str], image_unit: str,
                     unit_override: str | None, name: str) -> np.ndarray:
    obj = nib.load(path)
    raw = np.asarray(obj.dataobj)
    mask_unit = spatial_unit(obj, unit_override)
    if not np.allclose(affine_in_mm(obj.affine, mask_unit), affine_in_mm(image_obj.affine, image_unit),
                       rtol=1e-5, atol=1e-5):
        raise ValueError(f"{name} affine does not match image affine")
    axes = mask_axis_order.upper()
    if raw.ndim == 4:
        if len(axes) != 4 or set(axes) != set("XYZT"):
            raise ValueError(f"{name} 4-D mask axis_order must be the explicit native XYZT order")
        ordered = np.transpose(raw, tuple(axes.index(axis) for axis in "XYZT"))
        if not 0 <= target_frame < ordered.shape[-1]:
            raise ValueError(f"{name} frame is outside mask time dimension")
        spatial = ordered[..., target_frame]
    elif raw.ndim == 3:
        if len(axes) != 3 or set(axes) != set("XYZ"):
            raise ValueError(f"{name} 3-D mask axis_order must be the explicit native XYZ order")
        # A separate 3-D ED/ES mask already represents the selected frame.
        spatial = np.transpose(raw, tuple(axes.index(axis) for axis in "XYZ"))
    else:
        raise ValueError(f"{name} mask must be 3-D or 4-D")
    if spatial.shape != image_spatial_shape:
        raise ValueError(f"{name} spatial shape {spatial.shape} != image {image_spatial_shape}")
    return spatial_to_zyx(remap_mask(spatial, source_map), "XYZ")


def _frame_mask(nib, row: dict[str, str], phase: str, frame: int, *, root: Path,
                image_obj: Any, image_spatial_shape: tuple[int, int, int],
                source_map: dict[int, str], default_mask_axis: str | None,
                image_unit: str, unit_override: str | None) -> np.ndarray:
    path_value = (row.get(f"mask_{phase}") or row.get("mask") or "").strip()
    if not path_value:
        raise ValueError(f"annotated patient {row['patient_id']} lacks mask_{phase}")
    path = resolve_path(root, path_value)
    if not path.is_file():
        raise FileNotFoundError(path)
    mask_axis = _axis(row, "mask_axis_order", default_mask_axis)
    suffix = "_single" if row.get(f"mask_{phase}") else ""
    return _load_mask_frame(nib, path, target_frame=frame, image_obj=image_obj,
                            image_spatial_shape=image_spatial_shape,
                            mask_axis_order=mask_axis, source_map=source_map,
                            image_unit=image_unit, unit_override=unit_override,
                            name=f"{row['patient_id']}_{phase}{suffix}")


def _select_annotated(rows: list[dict[str, str]], budget: int, seed: int) -> set[str]:
    candidates = [r["patient_id"] for r in rows if _has_annotations(r)]
    if len(set(candidates)) != len(candidates):
        raise ValueError("duplicate annotated M&Ms patient IDs")
    if type(budget) is not int or budget < 2 or budget > len(candidates):
        raise ValueError("labels must be at least two and no more than annotated TRAIN patients")
    return set(label_subset(candidates, budget, seed))


def prepare(index: str | Path, root: str | Path, out: str | Path, labels: int, seed: int,
            source_class_map: str | Path | dict, *, frame_base: int | None = None,
            axis_order: str | None = None, mask_axis_order: str | None = None,
            evaluation: bool = False, spatial_unit_override: str | None = None,
            dataset_release: str = "original_2020") -> str:
    """Build hash-pinned train/development manifests from an M&Ms index."""
    if frame_base is None:
        raise ValueError("frame_base must be explicit (M&Ms index conventions differ across mirrors)")
    if dataset_release != "original_2020":
        raise ValueError("this preparer requires an explicit original_2020 M&Ms release declaration")
    nib = _import_nibabel()
    rows = read_index_csv(index)
    source_map = parse_class_map(source_class_map)
    root, out = Path(root).resolve(), Path(out).resolve()
    if out.exists() or out == root or root.is_relative_to(out):
        raise ValueError("new output directory outside source data is required")
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    test_rows = [r for r in rows if r["split"] == "test"]
    if len(train_rows) < 2:
        raise ValueError("M&Ms index needs at least two TRAIN patients")
    labeled_ids = _select_annotated(train_rows, labels, seed)
    out.mkdir(parents=True)
    geometry_by_patient: dict[str, dict] = {}
    source_images: dict[str, dict] = {}
    groups = {"labeled": [], "unlabeled": [], "evaluation": []}

    def process(row: dict[str, str], role: str) -> None:
        pid = row["patient_id"]
        image_path = resolve_path(root, row["image"])
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        image_axis = _axis(row, "axis_order", axis_order)
        image_obj, raw, cine, spatial_shape = _load_image(nib, image_path, image_axis)
        image_unit = spatial_unit(image_obj, spatial_unit_override)
        geometry = native_geometry_zyx(affine_in_mm(image_obj.affine, image_unit), cine.shape[1:])
        geometry_by_patient[pid] = geometry
        source_images[pid] = {"path": str(image_path), "sha256": digest(image_path),
                              "shape_source": list(raw.shape), "axis_order": image_axis,
                              "spatial_unit": image_unit}
        image_out = out / f"{pid}.npy"
        np.save(image_out, cine)
        record = {"id": pid, "image": image_out.name, "image_sha256": digest(image_out)}
        if role != "unlabeled":
            ed = parse_frame(row["ed_frame"], frame_base=frame_base, name=f"{pid}.ed_frame")
            es = parse_frame(row["es_frame"], frame_base=frame_base, name=f"{pid}.es_frame")
            if ed == es or ed >= cine.shape[0] or es >= cine.shape[0]:
                raise ValueError(f"invalid ED/ES frames for {pid}: {ed}, {es}")
            record["masks"] = {}
            for phase, frame in (("ed", ed), ("es", es)):
                mask = _frame_mask(nib, row, phase, frame, root=root, image_obj=image_obj,
                                   image_spatial_shape=spatial_shape, source_map=source_map,
                                   default_mask_axis=mask_axis_order, image_unit=image_unit,
                                   unit_override=spatial_unit_override)
                mask_path = out / f"{pid}_t{frame:03d}_mask.npy"
                np.save(mask_path, mask.astype(np.uint8))
                record["masks"][str(frame)] = {"path": mask_path.name, "sha256": digest(mask_path)}
        groups[role].append(record)

    for row in train_rows:
        process(row, "labeled" if row["patient_id"] in labeled_ids else "unlabeled")
    if evaluation:
        for row in val_rows:
            if not _has_annotations(row):
                raise ValueError(f"development patient {row['patient_id']} needs explicit ED/ES masks")
            process(row, "evaluation")

    train_manifest = {
        "schema": "self_audit.data.v2", "role": "training", "synthetic": False,
        "dataset": "mnms", "dataset_release": dataset_release, "label_budget": labels, "label_seed": seed,
        "labeled": groups["labeled"], "unlabeled": groups["unlabeled"],
        "evaluation_ids": [r["patient_id"] for r in val_rows + test_rows],
        "class_mapping": dict(CLASS_MAPPING),
        "geometry": {"schema": GEOMETRY_MAP_SCHEMA,
                     "by_patient": {r["id"]: geometry_by_patient[r["id"]]
                                     for g in (groups["labeled"], groups["unlabeled"]) for r in g}},
    }
    write_json(out / "train.json", train_manifest)
    if evaluation:
        write_json(out / "development.json", {
            "schema": "self_audit.data.v2", "role": "development", "synthetic": False,
            "dataset": "mnms", "dataset_release": dataset_release, "scope": "exposed_development",
            "patients": groups["evaluation"], "class_mapping": dict(CLASS_MAPPING),
            "geometry": {"schema": GEOMETRY_MAP_SCHEMA,
                         "by_patient": {r["id"]: geometry_by_patient[r["id"]] for r in groups["evaluation"]}},
        })
    write_json(out / "PREPARATION.json", {
        "schema": "self_audit.preparation.mnms.v1", "dataset": "mnms", "dataset_release": dataset_release,
        "source_index": str(Path(index).resolve()), "source_index_sha256": digest(index),
        "source_class_map": {str(k): v for k, v in source_map.items()},
        "frame_base": frame_base, "axis_order_default": axis_order,
        "spatial_unit_override": spatial_unit_override,
        "mask_axis_order_default": mask_axis_order, "raw_images": source_images,
        "read_mask_patient_ids": sorted(labeled_ids) + ([r["patient_id"] for r in val_rows] if evaluation else []),
        "unlabeled_mask_reads": 0,
        "normalization": "per_cine_image_only_percentiles_0.5_99.5",
        "geometry_schema": GEOMETRY_SCHEMA, "geometry_axis_order": ["T", "Z", "Y", "X"],
        "class_mapping": dict(CLASS_MAPPING), "train_manifest_sha256": digest(out / "train.json"),
        "evaluation_authorized": evaluation,
    })
    return digest(out / "train.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--labels", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-class-map", required=True)
    parser.add_argument("--frame-base", type=int, required=True, choices=(0, 1))
    parser.add_argument("--axis-order")
    parser.add_argument("--mask-axis-order")
    parser.add_argument("--spatial-unit", choices=("mm", "m", "micron"))
    parser.add_argument("--dataset-release", required=True, choices=("original_2020",))
    parser.add_argument("--evaluation", action="store_true")
    args = parser.parse_args()
    print(prepare(args.index, args.root, args.out, args.labels, args.seed,
                   args.source_class_map, frame_base=args.frame_base,
                   axis_order=args.axis_order, mask_axis_order=args.mask_axis_order,
                   evaluation=args.evaluation, spatial_unit_override=args.spatial_unit,
                   dataset_release=args.dataset_release))
