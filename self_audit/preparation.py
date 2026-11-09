"""Strict NIfTI-to-cache preparation primitives shared by ACDC and M&Ms.

Every source ontology and source axis order is explicit.  The cache protocol is
``[T, Z, Y, X]`` with native orthogonal geometry and canonical labels
``0=background, 1=RV, 2=MYO, 3=LV, 255=ignore``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .data import CLASS_MAPPING, GEOMETRY_SCHEMA, validate_geometry

CANONICAL_NAME_TO_ID = {
    "background": 0, "bg": 0,
    "rv": 1, "right_ventricle": 1, "right ventricle": 1,
    "myo": 2, "myocardium": 2,
    "lv": 3, "left_ventricle": 3, "left ventricle": 3,
    "ignore": 255,
}
AXES_TZYX = ("T", "Z", "Y", "X")
SPATIAL_UNIT_SCALE_MM = {
    "mm": 1.0,
    "m": 1000.0,
    "meter": 1000.0,
    "meters": 1000.0,
    "micron": 0.001,
    "um": 0.001,
}


def spatial_unit(image: Any, override: str | None = None) -> str:
    """Read NIfTI spatial units and normalize aliases; unknown units fail closed."""
    value = override
    if value is None:
        header = getattr(image, "header", None)
        getter = getattr(header, "get_xyzt_units", None)
        if getter is None:
            raise ValueError("NIfTI header units are unavailable; pass --spatial-unit explicitly")
        value = getter()[0]
    normalized = str(value or "").strip().lower()
    if normalized not in SPATIAL_UNIT_SCALE_MM:
        raise ValueError(f"unsupported/unknown spatial unit {value!r}; use mm, m or micron")
    if normalized in {"m", "meter", "meters"}:
        return "m"
    return "micron" if normalized in {"um", "micron"} else normalized


def affine_in_mm(affine: np.ndarray, unit: str) -> np.ndarray:
    """Convert an affine's world coordinates from its declared unit to mm."""
    normalized = spatial_unit(None, unit)
    converted = np.asarray(affine, dtype=np.float64).copy()
    converted[:3, :] *= SPATIAL_UNIT_SCALE_MM[normalized]
    return converted


def _safe_patient_id(value: Any) -> str:
    patient_id = str(value).strip()
    if not patient_id or patient_id in {".", ".."} or "/" in patient_id or "\\" in patient_id:
        raise ValueError("patient_id must be a non-empty path-safe identifier")
    return patient_id


def parse_class_map(value: str | Path | Mapping[Any, Any]) -> dict[int, str]:
    """Parse source label -> semantic name and require all four classes."""
    if isinstance(value, Mapping):
        raw = dict(value)
    else:
        text = str(value)
        path = Path(text[1:]) if text.startswith("@") else Path(text)
        raw = json.loads(path.read_text()) if path.is_file() else json.loads(text)
    if not isinstance(raw, Mapping):
        raise ValueError("source class map must be a JSON object label -> semantic name")
    parsed: dict[int, str] = {}
    used: set[int] = set()
    for key, name in raw.items():
        try:
            label = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"source class label is not an integer: {key!r}") from exc
        semantic_key = str(name).strip().lower().replace("-", "_")
        if label in parsed or semantic_key not in CANONICAL_NAME_TO_ID:
            raise ValueError(f"invalid or duplicate source class mapping: {key!r} -> {name!r}")
        target = CANONICAL_NAME_TO_ID[semantic_key]
        if target in used:
            raise ValueError("source class map contains duplicate semantic classes")
        semantic = {0: "background", 1: "rv", 2: "myo", 3: "lv", 255: "ignore"}[target]
        parsed[label] = semantic
        used.add(target)
    required = {"background", "rv", "myo", "lv"}
    if not required.issubset(parsed.values()):
        raise ValueError("source class map must define background, RV, MYO and LV")
    return parsed


def remap_mask(mask: np.ndarray, source_class_map: Mapping[Any, Any]) -> np.ndarray:
    """Remap a source mask to the canonical ontology, rejecting unknown labels."""
    mapping = parse_class_map(source_class_map)
    source = np.asarray(mask)
    if source.ndim not in (3, 4) or not np.isfinite(source).all():
        raise ValueError("mask must be finite 3-D or 4-D data")
    if not np.equal(source, np.floor(source)).all():
        raise ValueError("segmentation labels must be integer-valued")
    output = np.full(source.shape, 255, dtype=np.uint8)
    for label, semantic in mapping.items():
        output[source == label] = CANONICAL_NAME_TO_ID[semantic]
    unknown = ~np.isin(source, list(mapping))
    if unknown.any():
        raise ValueError(f"unknown source annotation labels: {np.unique(source[unknown]).tolist()}")
    return output


def normalize_cine(raw: np.ndarray, axis_order: str = "XYZT") -> np.ndarray:
    """Normalize finite cine data in the native NIfTI ``XYZT`` order."""
    source = np.asarray(raw)
    axes = tuple(str(axis_order).upper())
    if source.ndim != 4 or axes != tuple("XYZT"):
        raise ValueError("cine axis_order must be the explicit native XYZT order")
    if not np.isfinite(source).all() or source.size == 0:
        raise ValueError("cine must be finite and non-empty")
    ordered = np.transpose(source, tuple(axes.index(axis) for axis in AXES_TZYX))
    lo, hi = np.percentile(ordered, [0.5, 99.5])
    return np.clip((ordered.astype(np.float32) - float(lo)) / max(float(hi - lo), 1e-6), 0., 1.).astype(np.float32)


def spatial_to_zyx(mask: np.ndarray, axis_order: str = "XYZ") -> np.ndarray:
    source = np.asarray(mask)
    axes = tuple(str(axis_order).upper())
    if source.ndim != 3 or axes != tuple("XYZ"):
        raise ValueError("spatial axis_order must be the explicit native XYZ order")
    return np.transpose(source, tuple(axes.index(axis) for axis in ("Z", "Y", "X")))


def native_geometry_zyx(affine: np.ndarray, shape_zyx: Iterable[int],
                        source_axis_order: str = "XYZ") -> dict:
    """Validate an orthogonal NIfTI affine and express it in cached ZYX order."""
    source = np.asarray(affine, dtype=np.float64)
    shape = tuple(int(x) for x in shape_zyx)
    if source.shape != (4, 4) or not np.isfinite(source).all():
        raise ValueError("native affine must be finite 4x4")
    if not np.allclose(source[3], [0., 0., 0., 1.], rtol=0., atol=1e-6):
        raise ValueError("native affine must be homogeneous")
    source_axes = tuple(str(source_axis_order).upper())
    if source_axes != tuple("XYZ"):
        raise ValueError("source_axis_order must be the explicit native XYZ order")
    linear = source[:3, :3][:, [source_axes.index(axis) for axis in "XYZ"]]
    spacing_xyz = np.linalg.norm(linear, axis=0)
    if (spacing_xyz <= 0).any() or not np.isfinite(spacing_xyz).all():
        raise ValueError("native affine has invalid voxel spacing")
    directions = linear / spacing_xyz
    if not np.allclose(directions.T @ directions, np.eye(3), rtol=1e-5, atol=1e-5):
        raise ValueError("sheared native affine is unsupported")
    permutation = np.eye(4, dtype=np.float64)
    permutation[:3, :3] = np.asarray([[0., 0., 1.], [0., 1., 0.], [1., 0., 0.]])
    spacing_zyx = spacing_xyz[[2, 1, 0]]
    canonical_affine = source.copy()
    canonical_affine[:3, :3] = linear
    return validate_geometry({
        "schema": GEOMETRY_SCHEMA, "array_axes": list(AXES_TZYX),
        "spatial_axes": ["Z", "Y", "X"], "shape_zyx": list(shape),
        "spacing_zyx_mm": spacing_zyx.tolist(), "spacing_yx_mm": spacing_zyx[1:].tolist(),
        "affine_zyx": (canonical_affine @ permutation).tolist(), "class_mapping": dict(CLASS_MAPPING),
    }, shape_zyx=shape)


def read_patient_split(path: str | Path, *, require_test: bool = False) -> dict[str, list[str]]:
    obj = json.loads(Path(path).read_text())
    result: dict[str, list[str]] = {}
    for role in ("train", "val", "test"):
        variants = []
        for key in (role, role + "_patients"):
            if isinstance(obj.get(key), list):
                variants.append([_safe_patient_id(x) for x in obj[key]])
        for key in ("splits", "locked_splits"):
            group = obj.get(key, {}).get(role) if isinstance(obj.get(key), dict) else None
            if isinstance(group, dict):
                group = group.get("patients")
            if isinstance(group, list):
                variants.append([_safe_patient_id(x) for x in group])
        if not variants:
            if role == "test" and not require_test:
                result[role] = []
                continue
            raise ValueError(f"missing patient-level {role} split")
        if any(sorted(v) != sorted(variants[0]) for v in variants):
            raise ValueError(f"conflicting {role} split representations")
        result[role] = variants[0]
    all_ids = sum(result.values(), [])
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("patient-level splits overlap")
    return result


def read_index_csv(path: str | Path) -> list[dict[str, str]]:
    """Read a strict one-row-per-patient M&Ms index."""
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("index CSV is empty")
    required = {"patient_id", "split", "image", "ed_frame", "es_frame"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"index CSV missing required columns: {sorted(missing)}")
    seen = set()
    for row in rows:
        pid = _safe_patient_id(row.get("patient_id", ""))
        if pid in seen:
            raise ValueError("index CSV must contain one row per patient")
        seen.add(pid)
        row["patient_id"] = pid
        row["split"] = row.get("split", "").strip().lower()
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError(f"invalid split for {pid}: {row['split']!r}")
        if not row.get("image", "").strip():
            raise ValueError(f"missing image path for {pid}")
    return rows


def resolve_path(root: str | Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else Path(root) / path).resolve()


def parse_frame(value: str, *, frame_base: int, name: str) -> int:
    try:
        frame = int(value) - int(frame_base)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer frame index") from exc
    if frame < 0:
        raise ValueError(f"{name} is below the declared frame base")
    return frame
