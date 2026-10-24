"""Offline, image-only construction of a fixed teacher-memory bank.

This module never follows or opens a mask path.  A v2 TRAIN manifest is used
only for patient IDs and image references; native image geometry comes from
the corresponding 4-D image NIfTI under an explicitly supplied root.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import math
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

CLASSES = ("BG", "RV", "MYO", "LV")
PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
FEATURE_DIM = 32
MAX_PAIRS_PER_GROUP = 8
MIN_CONFIDENCE = 0.8
MAX_DISTANCE_MM = 3.0
_SPATIAL_TO_MM = {"mm": 1.0, "meter": 1000.0, "micron": 0.001}
_PATIENT_ID = re.compile(r"patient\d{3}\Z")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _canonical_json_sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _train_records(manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    if manifest.get("schema") != "self_audit.data.v2" or manifest.get("role") != "training" or manifest.get("synthetic") is not False:
        raise ValueError("an image-only patient self_audit.data.v2 TRAIN manifest is required")
    labeled, unlabeled = manifest.get("labeled"), manifest.get("unlabeled")
    if not isinstance(labeled, list) or not isinstance(unlabeled, list):
        raise ValueError("TRAIN manifest must contain labeled and unlabeled record lists")
    records = []
    for group, is_labeled in ((labeled, True), (unlabeled, False)):
        for record in group:
            if not isinstance(record, dict):
                raise ValueError("invalid TRAIN record")
            required = {"id", "image", "image_sha256"} | ({"masks"} if is_labeled else set())
            if set(record) != required:
                raise ValueError("TRAIN records may contain only patient/image fields and labeled mask references")
            pid, image, image_hash = record["id"], record["image"], record["image_sha256"]
            if not isinstance(pid, str) or not _PATIENT_ID.fullmatch(pid):
                raise ValueError("patient IDs must use patientNNN form")
            if not isinstance(image, str) or not image or not re.fullmatch(r"[0-9a-f]{64}", str(image_hash)):
                raise ValueError("invalid TRAIN image reference or SHA256")
            # Deliberately do not inspect, resolve, hash, or open record['masks'].
            records.append({"id": pid, "image": image, "image_sha256": image_hash,
                            "training_role": "labeled" if is_labeled else "unlabeled"})
    ids = [r["id"] for r in records]
    evaluation_ids = manifest.get("evaluation_ids")
    if not isinstance(evaluation_ids, list) or any(not isinstance(x, str) for x in evaluation_ids):
        raise ValueError("explicit evaluation_ids are required")
    if len(ids) != len(set(ids)) or len(evaluation_ids) != len(set(evaluation_ids)) or set(ids) & set(evaluation_ids):
        raise ValueError("duplicate patient IDs or TRAIN/evaluation overlap")
    if not records:
        raise ValueError("empty TRAIN manifest")
    return sorted(records, key=lambda r: r["id"]), sorted(evaluation_ids)


def _resolve_train_image(manifest_path: Path, ref: str) -> Path:
    path = Path(ref)
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def create_native_image_manifest(
    train_manifest_path: str | Path,
    train_manifest_sha256: str,
    native_root: str | Path,
    output_path: str | Path,
    *,
    max_patients: int | None = None,
) -> dict[str, Any]:
    """Create a hash-bound TRAIN-only manifest using exact patient cine paths.

    Hashes for every prepared image and native NIfTI are checked/computed before
    any image arrays or NIfTI headers are opened.  The only NIfTI path considered
    is ``<root>/<patient>/<patient>_4d.nii[.gz]``.
    """
    import nibabel as nib  # Optional dependency, imported only by this offline step.

    train_path = Path(train_manifest_path).resolve()
    if sha256_file(train_path) != train_manifest_sha256:
        raise ValueError("TRAIN manifest hash mismatch")
    source = json.loads(train_path.read_text())
    records, evaluation_ids = _train_records(source)
    if max_patients is not None and (type(max_patients) is not int or max_patients < 1):
        raise ValueError("max_patients must be a positive integer")
    selected = records if max_patients is None else records[:max_patients]
    root = Path(native_root).resolve()

    # Verify all prepared image references and establish exact native image paths first.
    paths = []
    prepared_hashes: dict[str, str] = {}
    native_hashes: dict[str, str] = {}
    for record in selected:
        prepared_path = _resolve_train_image(train_path, record["image"])
        if sha256_file(prepared_path) != record["image_sha256"]:
            raise ValueError(f"prepared TRAIN image hash mismatch for {record['id']}")
        prior = prepared_hashes.setdefault(record["image_sha256"], record["id"])
        if prior != record["id"]:
            raise ValueError(f"duplicate prepared TRAIN image content: {prior}, {record['id']}")
        folder = root / record["id"]
        candidates = [p for p in (folder / f"{record['id']}_4d.nii.gz", folder / f"{record['id']}_4d.nii") if p.is_file()]
        if len(candidates) != 1:
            raise ValueError(f"expected one exact native 4-D image for {record['id']}")
        native_path = candidates[0]
        native_hash = sha256_file(native_path)
        prior = native_hashes.setdefault(native_hash, record["id"])
        if prior != record["id"]:
            raise ValueError(f"duplicate native TRAIN image content: {prior}, {record['id']}")
        paths.append((record, prepared_path, native_path, native_hash))

    native_records = []
    for record, prepared_path, native_path, native_hash in paths:
        image = nib.load(str(native_path), mmap=True)
        if len(image.shape) != 4 or min(image.shape) < 1:
            raise ValueError(f"native TRAIN image must be XYZT for {record['id']}")
        affine = np.asarray(image.affine, dtype=np.float64)
        if affine.shape != (4, 4) or not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-12:
            raise ValueError(f"invalid native affine for {record['id']}")
        spatial_units = image.header.get_xyzt_units()[0]
        if spatial_units not in _SPATIAL_TO_MM:
            raise ValueError(f"unsupported/unknown NIfTI spatial units for {record['id']}: {spatial_units!r}")
        unit_to_mm = _SPATIAL_TO_MM[spatial_units]
        affine_mm = affine.copy()
        affine_mm[:3, :] *= unit_to_mm
        native_records.append({
            "id": record["id"], "training_role": record["training_role"],
            "prepared_image_path": str(prepared_path), "image_sha256": record["image_sha256"],
            "native_image_path": str(native_path), "native_image_sha256": native_hash,
            "shape_xyzt": list(image.shape), "affine_xyz_to_world_mm": affine_mm.tolist(),
            "native_affine_xyz_to_world_header_units": affine.tolist(),
            "spatial_units": spatial_units, "unit_to_mm_scale": unit_to_mm,
            "timepoints": list(range(int(image.shape[3]))),
            "spacing_xyz_mm": [float(x) * unit_to_mm for x in image.header.get_zooms()[:3]],
        })
        del image
    out = {
        "schema": "self_audit.native_teacher_images.v1", "role": "training", "synthetic": False,
        "train_manifest_sha256": train_manifest_sha256, "train_manifest_path": str(train_path),
        "native_root": str(root), "evaluation_ids": evaluation_ids, "patients": native_records,
        "coverage": {"train_patients_total": len(records), "train_patients_selected": len(selected),
                     "complete": len(selected) == len(records), "bounded": max_patients is not None},
        "image_reads": {"prepared_image_arrays": 0, "native_headers": len(native_records), "mask_paths_read": 0},
        "geometry_source": "native_NIfTI_affine_and_header; no geometry inferred from TZYX arrays",
    }
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    return out


def load_native_manifest(path: str | Path, expected_sha256: str) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    if sha256_file(manifest_path) != expected_sha256:
        raise ValueError("native image manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "self_audit.native_teacher_images.v1" or manifest.get("role") != "training" or manifest.get("synthetic") is not False:
        raise ValueError("not a patient TRAIN native-image manifest")
    records = manifest.get("patients")
    evaluation_ids = manifest.get("evaluation_ids")
    if not isinstance(records, list) or not records or not isinstance(evaluation_ids, list) or any(not isinstance(x, str) for x in evaluation_ids):
        raise ValueError("invalid native-image manifest records")
    ids = [r.get("id") for r in records if isinstance(r, dict)]
    if (len(ids) != len(records) or len(ids) != len(set(ids)) or len(evaluation_ids) != len(set(evaluation_ids))
            or set(ids) & set(evaluation_ids)):
        raise ValueError("duplicate patient IDs or TRAIN/evaluation overlap")
    if any(not _PATIENT_ID.fullmatch(str(pid)) for pid in ids):
        raise ValueError("invalid TRAIN patient ID")
    source_manifest_path = Path(manifest.get("train_manifest_path", "")).resolve()
    source_sha256 = manifest.get("train_manifest_sha256")
    if (not source_manifest_path.is_file() or not isinstance(source_sha256, str)
            or sha256_file(source_manifest_path) != source_sha256):
        raise ValueError("source TRAIN manifest hash mismatch")
    source_records, source_evaluation_ids = _train_records(json.loads(source_manifest_path.read_text()))
    source_by_id = {record["id"]: record for record in source_records}
    if set(source_evaluation_ids) != set(evaluation_ids):
        raise ValueError("native manifest evaluation IDs differ from source TRAIN manifest")
    coverage = manifest.get("coverage")
    expected_complete = set(ids) == set(source_by_id) and len(ids) == len(source_records)
    if (not isinstance(coverage, dict)
            or type(coverage.get("train_patients_total")) is not int
            or type(coverage.get("train_patients_selected")) is not int
            or type(coverage.get("complete")) is not bool
            or type(coverage.get("bounded")) is not bool
            or coverage["train_patients_total"] != len(source_records)
            or coverage["train_patients_selected"] != len(records)
            or coverage["complete"] != expected_complete):
        raise ValueError("native manifest coverage is inconsistent with source TRAIN patient membership")
    prepared_hashes: dict[str, str] = {}
    native_hashes: dict[str, str] = {}
    # Validate every data-file hash before any image array/header is opened.
    for record in records:
        required = {"id", "training_role", "prepared_image_path", "image_sha256", "native_image_path",
                    "native_image_sha256", "shape_xyzt", "affine_xyz_to_world_mm",
                    "native_affine_xyz_to_world_header_units", "spatial_units", "unit_to_mm_scale",
                    "timepoints", "spacing_xyz_mm"}
        if set(record) != required:
            raise ValueError("native patient entries may contain image fields only")
        source_record = source_by_id.get(record["id"])
        if (source_record is None or source_record["image_sha256"] != record.get("image_sha256")
                or source_record["training_role"] != record.get("training_role")):
            raise ValueError("native patient membership/image hash differs from source TRAIN manifest")
        shape = record.get("shape_xyzt")
        if (not isinstance(shape, list) or len(shape) != 4 or any(type(v) is not int or v < 1 for v in shape)
                or record.get("timepoints") != list(range(shape[3]))):
            raise ValueError("native XYZT shape and explicit timepoints are inconsistent")
        scale = _SPATIAL_TO_MM.get(record.get("spatial_units"))
        native_affine = np.asarray(record.get("native_affine_xyz_to_world_header_units"), dtype=np.float64)
        affine_mm = np.asarray(record.get("affine_xyz_to_world_mm"), dtype=np.float64)
        if (scale is None or record.get("unit_to_mm_scale") != scale or native_affine.shape != (4, 4)
                or affine_mm.shape != (4, 4) or not np.isfinite(native_affine).all()
                or not np.isfinite(affine_mm).all()):
            raise ValueError("native spatial units/affines are invalid")
        expected_affine_mm = native_affine.copy()
        expected_affine_mm[:3, :] *= scale
        if not np.allclose(affine_mm, expected_affine_mm, atol=1e-8, rtol=0):
            raise ValueError("native affine unit conversion to mm is inconsistent")
        for key in ("prepared_image_path", "native_image_path"):
            if key not in record:
                raise ValueError(f"missing {key}")
        if sha256_file(record["prepared_image_path"]) != record.get("image_sha256"):
            raise ValueError(f"prepared image hash mismatch for {record['id']}")
        if sha256_file(record["native_image_path"]) != record.get("native_image_sha256"):
            raise ValueError(f"native image hash mismatch for {record['id']}")
        for digest_value, seen_hashes, label in (
            (record["image_sha256"], prepared_hashes, "prepared"),
            (record["native_image_sha256"], native_hashes, "native"),
        ):
            prior = seen_hashes.setdefault(digest_value, record["id"])
            if prior != record["id"]:
                raise ValueError(f"duplicate {label} TRAIN image content: {prior}, {record['id']}")
        expected_native_dir = Path(manifest["native_root"]).resolve() / record["id"]
        native_path = Path(record["native_image_path"]).resolve()
        if native_path.parent != expected_native_dir or native_path.name not in {f"{record['id']}_4d.nii", f"{record['id']}_4d.nii.gz"}:
            raise ValueError("native path must be the exact patient 4-D image under native_root")
    return manifest


def _validate_prediction(
    probabilities: Any, features: Any, shape_xyz: tuple[int, int, int],
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(probabilities, dtype=np.float32)
    feats = np.asarray(features, dtype=np.float32)
    if probs.shape != (4, *shape_xyz) or feats.shape != (FEATURE_DIM, *shape_xyz):
        raise ValueError("prediction must be probabilities [4,X,Y,Z] and features [32,X,Y,Z]")
    if valid_mask is None:
        support = np.ones(shape_xyz, dtype=bool)
    else:
        support = np.asarray(valid_mask)
        if support.dtype != np.bool_ or support.shape != shape_xyz:
            raise ValueError("valid_mask must match native XYZ prediction geometry")
    if (not np.isfinite(probs).all() or np.any(probs < 0)
            or (support.any() and not np.allclose(probs.sum(0)[support], 1.0, atol=2e-3))):
        raise ValueError("invalid class probabilities")
    if np.any(probs[:, ~support] != 0.0):
        raise ValueError("unsupported teacher voxels must have zero probabilities")
    if not np.isfinite(feats).all():
        raise ValueError("nonfinite feature map")
    return probs, feats


def _world_xyz(coords_xyz: np.ndarray, affine: np.ndarray) -> np.ndarray:
    hom = np.concatenate((coords_xyz.astype(np.float64), np.ones((len(coords_xyz), 1))), axis=1)
    return (hom @ affine.T)[:, :3]


def _rank_candidate(candidate: dict[str, Any]) -> str:
    stable = (f"{candidate['patient']}|{candidate['timepoint']}|{candidate['class_pair']}|"
              f"{candidate['coord_c_xyz']}|{candidate['coord_d_xyz']}|{candidate['image_sha256']}")
    return hashlib.sha256(stable.encode()).hexdigest()


def _same_pair_segment(
    coord_c: np.ndarray, coord_d: np.ndarray, c: int, d: int,
    labels: np.ndarray, support: np.ndarray,
) -> bool:
    """Reject endpoint pairs whose native voxel segment crosses a third class."""
    delta = coord_d.astype(np.float64) - coord_c.astype(np.float64)
    steps = max(1, int(math.ceil(float(np.max(np.abs(delta))) * 2.0)))
    sequence: list[int] = []
    for alpha in np.linspace(0.0, 1.0, steps + 1):
        point = np.rint(coord_c + alpha * delta).astype(np.int64)
        index = tuple(int(v) for v in point)
        label = int(labels[index])
        if not support[index] or label not in (c, d):
            return False
        if not sequence or sequence[-1] != label:
            sequence.append(label)
    # A farther c/d pair cannot bridge two or more distinct interfaces, even
    # when the intervening labels happen to alternate only between c and d.
    return sequence == [c, d]


def boundary_pair_candidates(
    probabilities: np.ndarray,
    features: np.ndarray,
    affine_xyz_to_world_mm: np.ndarray,
    *, patient: str, timepoint: int, image_sha256: str,
    source_sha256: str, source_kind: str,
    valid_mask: np.ndarray | None = None,
    max_distance_mm: float = MAX_DISTANCE_MM,
    min_confidence: float = MIN_CONFIDENCE,
    per_patient_cap: int = MAX_PAIRS_PER_GROUP,
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    """Collect confident, physically close endpoints from predicted class boundaries."""
    from scipy.ndimage import binary_erosion
    from scipy.spatial import cKDTree

    if max_distance_mm <= 0 or not 0 <= min_confidence <= 1 or per_patient_cap < 1:
        raise ValueError("invalid boundary-pair extraction limits")
    probs, feats = _validate_prediction(probabilities, features, tuple(probabilities.shape[1:]), valid_mask)
    if valid_mask is None:
        support = np.ones(probs.shape[1:], dtype=bool)
    else:
        support = np.asarray(valid_mask, dtype=bool)
        if support.shape != probs.shape[1:]:
            raise ValueError("valid_mask must match native XYZ prediction geometry")
    affine = np.asarray(affine_xyz_to_world_mm, dtype=np.float64)
    if affine.shape != (4, 4) or not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-12:
        raise ValueError("valid native affine required for physical coordinates")
    labels = probs.argmax(0)
    confidence = probs.max(0)
    # Convert channels-last voxel features to normalized endpoint vectors.
    feature_vectors = np.moveaxis(feats, 0, -1)
    norms = np.linalg.norm(feature_vectors, axis=-1, keepdims=True)
    feature_vectors = feature_vectors / np.maximum(norms, 1e-12)
    result: dict[tuple[int, int], list[dict[str, Any]]] = {pair: [] for pair in PAIRS}
    structure = np.ones((3, 3, 3), dtype=bool)
    # Apply the adapter's explicit inverse-resampling coverage only to endpoint
    # voxels; do not manufacture an extra erosion margin at valid native edges.
    support_interior = support

    for c, d in PAIRS:
        c_mask, d_mask = labels == c, labels == d
        # Only predicted interfaces are eligible; endpoints remain native voxels.
        c_boundary = c_mask & ~binary_erosion(c_mask, structure=structure, border_value=0) & support_interior
        d_boundary = d_mask & ~binary_erosion(d_mask, structure=structure, border_value=0) & support_interior
        c_xyz = np.argwhere(c_boundary & (confidence >= min_confidence))
        d_xyz = np.argwhere(d_boundary & (confidence >= min_confidence))
        if not len(c_xyz) or not len(d_xyz):
            continue
        c_world, d_world = _world_xyz(c_xyz, affine), _world_xyz(d_xyz, affine)
        tree = cKDTree(d_world)
        candidate_heap = []
        sequence_id = 0
        for ci, c_point in enumerate(c_world):
            neighbors = tree.query_ball_point(c_point, r=max_distance_mm)
            if not neighbors:
                continue
            # Search in deterministic physical-distance order and keep only a
            # direct c/d interface path; a third class cannot be jumped over.
            ordered = sorted(neighbors, key=lambda j: (
                float(np.linalg.norm(d_world[j] - c_world[ci])), tuple(int(v) for v in d_xyz[j])))
            chosen = None
            for candidate_j in ordered:
                candidate_distance = float(np.linalg.norm(d_world[candidate_j] - c_world[ci]))
                if _same_pair_segment(c_xyz[ci], d_xyz[candidate_j], c, d, labels, support):
                    chosen = (candidate_j, candidate_distance)
                    break
            if chosen is None:
                continue
            dj, best_distance = chosen
            coord_c, coord_d = c_xyz[ci], d_xyz[dj]
            conf_c = float(confidence[tuple(coord_c)])
            conf_d = float(confidence[tuple(coord_d)])
            vec_c = feature_vectors[tuple(coord_c)].astype(np.float32, copy=True)
            vec_d = feature_vectors[tuple(coord_d)].astype(np.float32, copy=True)
            if np.linalg.norm(vec_c) < 1e-10 or np.linalg.norm(vec_d) < 1e-10:
                continue
            item = {
                "endpoints": [vec_c, vec_d], "confidence": [conf_c, conf_d],
                "patient": patient, "timepoint": int(timepoint), "class_pair": [c, d],
                "coord_c_xyz": [int(v) for v in coord_c], "coord_d_xyz": [int(v) for v in coord_d],
                "world_c_mm": _world_xyz(coord_c[None], affine)[0].tolist(),
                "world_d_mm": _world_xyz(coord_d[None], affine)[0].tolist(),
                "distance_mm": best_distance, "image_sha256": image_sha256,
                "source_sha256": source_sha256, "feature_source": source_kind,
            }
            item["_rank"] = _rank_candidate(item)
            rank_value = int(item["_rank"], 16)
            heap_item = (-rank_value, sequence_id, item)
            sequence_id += 1
            # Keep only the smallest stable hashes; do not retain dense boundary
            # feature lists from every voxel in a full cine.
            if len(candidate_heap) < per_patient_cap:
                heapq.heappush(candidate_heap, heap_item)
            elif rank_value < -candidate_heap[0][0]:
                heapq.heapreplace(candidate_heap, heap_item)
        result[(c, d)] = sorted((entry[2] for entry in candidate_heap), key=lambda item: item["_rank"])
    return result


def patient_balanced_select(
    candidates: dict[str, dict[tuple[int, int], list[dict[str, Any]]]],
    *, max_per_group: int = MAX_PAIRS_PER_GROUP,
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    """Select by sorted-patient round robin, then deterministic candidate rank."""
    selected = {pair: [] for pair in PAIRS}
    for pair in PAIRS:
        patient_pools = {}
        for patient in sorted(candidates):
            pool = list(candidates[patient].get(pair, ()))
            pool.sort(key=lambda item: item.get("_rank", _rank_candidate(item)))
            if pool:
                patient_pools[patient] = pool
        # Round-robin gives each patient one opportunity before any patient gets a second.
        while len(selected[pair]) < max_per_group and patient_pools:
            for patient in list(patient_pools):
                selected[pair].append(patient_pools[patient].pop(0))
                if not patient_pools[patient]:
                    del patient_pools[patient]
                if len(selected[pair]) == max_per_group:
                    break
    return selected


def pack_bank(selected: dict[tuple[int, int], list[dict[str, Any]]], provenance: dict[str, Any]) -> dict[str, Any]:
    """Return fields accepted by ``MemoryBank(endpoints, confidences, valid, metadata)``."""
    endpoints = torch.zeros((len(PAIRS), MAX_PAIRS_PER_GROUP, 2, FEATURE_DIM), dtype=torch.float32)
    confidences = torch.zeros((len(PAIRS), MAX_PAIRS_PER_GROUP, 2), dtype=torch.float32)
    valid = torch.zeros((len(PAIRS), MAX_PAIRS_PER_GROUP), dtype=torch.bool)
    entry_metadata: list[list[list[dict[str, Any]]]] = []
    for pi, pair in enumerate(PAIRS):
        group_meta = []
        entries = selected.get(pair, [])[:MAX_PAIRS_PER_GROUP]
        for ni, item in enumerate(entries):
            for side in range(2):
                vector = torch.as_tensor(item["endpoints"][side], dtype=torch.float32)
                norm = torch.linalg.vector_norm(vector)
                if not torch.isfinite(vector).all() or not torch.isfinite(norm) or norm < 1e-8:
                    raise ValueError("invalid endpoint feature vector")
                if item["confidence"][side] < MIN_CONFIDENCE:
                    raise ValueError("endpoint confidence below configured threshold")
                endpoints[pi, ni, side] = vector / norm
                confidences[pi, ni, side] = float(item["confidence"][side])
            valid[pi, ni] = True
            common = {key: item[key] for key in ("patient", "timepoint", "distance_mm", "image_sha256",
                     "source_sha256", "feature_source", "class_pair")}
            group_meta.append([
                {**common, "endpoint": "c", "coord_xyz": item["coord_c_xyz"],
                 "world_mm": item["world_c_mm"], "confidence": item["confidence"][0]},
                {**common, "endpoint": "d", "coord_xyz": item["coord_d_xyz"],
                 "world_mm": item["world_d_mm"], "confidence": item["confidence"][1]},
            ])
        group_meta.extend([[{}, {}] for _ in range(MAX_PAIRS_PER_GROUP - len(entries))])
        entry_metadata.append(group_meta)
    metadata = {"schema": "self_audit.memory_bank.v1", "class_map": list(CLASSES),
                "class_pairs": [list(pair) for pair in PAIRS], "feature_dim": FEATURE_DIM,
                "selection": "patient-balanced-round-robin-stable-sha256", "entry_metadata": entry_metadata,
                "provenance": provenance}
    return {"endpoints": endpoints, "confidences": confidences, "valid": valid, "metadata": metadata}


def _prediction_from_student(model: torch.nn.Module, cine_tzyx: np.ndarray, timepoint: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Run a frozen v2 student on native XY planes; features are its final 32D fuse map."""
    probs_tzyx = np.empty((cine_tzyx.shape[1], 4, *cine_tzyx.shape[-2:]), dtype=np.float32)
    feats_tzyx = np.empty((cine_tzyx.shape[1], FEATURE_DIM, *cine_tzyx.shape[-2:]), dtype=np.float32)
    for z in range(cine_tzyx.shape[1]):
        zs = [max(0, z - 1), z, min(cine_tzyx.shape[1] - 1, z + 1)]
        x = torch.from_numpy(np.array(cine_tzyx[timepoint, zs], dtype=np.float32, copy=True))[None].to(device)
        with torch.inference_mode():
            features = model.features(x)
            logits = model.head(features)
            if features.shape[1] != FEATURE_DIM:
                raise ValueError("A5 student final fuse features must be 32D")
            probs_tzyx[z] = logits.softmax(1)[0].cpu().numpy()
            feats_tzyx[z] = features[0].cpu().numpy()
    # Prepared TZYX image arrays store XY in reversed (Y,X) native plane order.
    # Output plane tensors are [Y,X], while the native grid is [X,Y,Z].
    probs_xyz = np.transpose(probs_tzyx, (1, 3, 2, 0))
    feats_xyz = np.transpose(feats_tzyx, (1, 3, 2, 0))
    return probs_xyz, feats_xyz


def _require_a0_student(
    model: torch.nn.Module,
    checkpoint_meta: dict[str, Any],
    *,
    data_sha256: str,
    labeled_ids: list[str],
    unlabeled_ids: list[str],
) -> str:
    """Reject checkpoints whose training/model path contains any memory arm."""
    _validate_a0_checkpoint_metadata(
        checkpoint_meta, data_sha256=data_sha256,
        labeled_ids=labeled_ids, unlabeled_ids=unlabeled_ids)
    if getattr(model, "teacher_memory", None) is not None:
        raise ValueError("A5 requires a real frozen A0 student checkpoint with no bank or external cache ancestry")
    return "A0"


def _validate_a0_checkpoint_metadata(
    checkpoint_meta: dict[str, Any], *, data_sha256: str,
    labeled_ids: list[str], unlabeled_ids: list[str],
) -> None:
    """Validate A0 ancestry and exact data/label-budget matching before weights load."""
    config = checkpoint_meta.get("config")
    memory_config = config.get("teacher_memory") if isinstance(config, dict) else None
    checkpoint_labeled = checkpoint_meta.get("labeled_ids")
    checkpoint_unlabeled = checkpoint_meta.get("unlabeled_ids")
    audit_reads = checkpoint_meta.get("audit_label_reads")
    if (checkpoint_meta.get("synthetic") is not False or not isinstance(memory_config, dict)
            or memory_config.get("arm") != "A0"
            or any(memory_config.get(key) is not None for key in ("bank_path", "bank_sha256", "cache_path", "cache_sha256"))
            or checkpoint_meta.get("teacher_memory") is not None
            or checkpoint_meta.get("data_sha256") != data_sha256
            or not isinstance(checkpoint_labeled, list) or sorted(checkpoint_labeled) != labeled_ids
            or not isinstance(checkpoint_unlabeled, list) or sorted(checkpoint_unlabeled) != unlabeled_ids
            or checkpoint_meta.get("label_budget") != len(labeled_ids)
            or not isinstance(audit_reads, list) or not set(audit_reads) <= set(labeled_ids)):
        raise ValueError("A5 requires a real frozen A0 student checkpoint with no bank or external cache ancestry")


def build_memory_bank(
    native_manifest_path: str | Path,
    native_manifest_sha256: str,
    *, mode: str,
    adapter: Any | None = None,
    student_checkpoint: str | Path | None = None,
    student_manifest_sha256: str | None = None,
    device: str = "cpu",
    max_patients: int | None = None,
    max_timepoints: int | None = None,
    collect_a1_cache: bool = False,
    a1_cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build image-only paired features from the adapter (A1/A3) or frozen student (A5)."""
    if mode not in {"teacher", "a5"}:
        raise ValueError("mode must be teacher or a5")
    teacher_pins = None
    if mode == "teacher":
        if adapter is None:
            raise ValueError("teacher mode requires an image-only CineMA adapter")
        from .cinema_teacher import SOURCE_COMMIT, CONFIG_SHA256, WEIGHTS_SHA256, CLASS_MAP as TEACHER_CLASS_MAP
        teacher_pins = {
            "source_commit": SOURCE_COMMIT,
            "config_sha256": CONFIG_SHA256,
            "weights_sha256": WEIGHTS_SHA256,
            "class_map": dict(TEACHER_CLASS_MAP),
        }
        if getattr(adapter, "synthetic", None) is not False:
            raise ValueError("patient teacher banks require the real pinned CineMA adapter")
        if any(getattr(adapter, name, None) != expected for name, expected in teacher_pins.items() if name != "class_map"):
            raise ValueError("CineMA adapter source/config/weights do not match the pinned teacher")
    if mode == "a5" and (student_checkpoint is None or student_manifest_sha256 is None or adapter is not None):
        raise ValueError("A5 requires only a frozen student checkpoint and manifest hash")
    if collect_a1_cache and (mode != "teacher" or a1_cache_dir is None):
        raise ValueError("A1 cache collection requires teacher mode and a dedicated output directory")
    if not collect_a1_cache and a1_cache_dir is not None:
        raise ValueError("A1 cache directory supplied while cache collection is disabled")
    if max_patients is not None and (type(max_patients) is not int or max_patients < 1):
        raise ValueError("max_patients must be positive")
    if max_timepoints is not None and (type(max_timepoints) is not int or max_timepoints < 1):
        raise ValueError("max_timepoints must be positive")
    manifest_path = Path(native_manifest_path).resolve()
    manifest = load_native_manifest(manifest_path, native_manifest_sha256)
    source_train_manifest = json.loads(Path(manifest["train_manifest_path"]).read_text())
    source_records, _ = _train_records(source_train_manifest)
    source_labeled_ids = sorted(r["id"] for r in source_records if r["training_role"] == "labeled")
    source_unlabeled_ids = sorted(r["id"] for r in source_records if r["training_role"] == "unlabeled")
    records = sorted(manifest["patients"], key=lambda r: r["id"])
    if max_patients is not None:
        records = records[:max_patients]
    student = None
    student_source_sha = None
    student_checkpoint_meta = None
    if mode == "a5":
        from .engine import load_checkpoint
        checkpoint_dir = Path(student_checkpoint)
        checkpoint_manifest_path = checkpoint_dir / "MANIFEST.json"
        if sha256_file(checkpoint_manifest_path) != student_manifest_sha256:
            raise ValueError("A5 student checkpoint manifest hash mismatch")
        student_checkpoint_meta = json.loads(checkpoint_manifest_path.read_text())
        _validate_a0_checkpoint_metadata(
            student_checkpoint_meta, data_sha256=manifest["train_manifest_sha256"],
            labeled_ids=source_labeled_ids, unlabeled_ids=source_unlabeled_ids)
        student, student_checkpoint_meta = load_checkpoint(student_checkpoint, student_manifest_sha256, device)
        student_source_arm = _require_a0_student(
            student, student_checkpoint_meta,
            data_sha256=manifest["train_manifest_sha256"],
            labeled_ids=source_labeled_ids,
            unlabeled_ids=source_unlabeled_ids,
        )
        student.eval().requires_grad_(False)
        student_source_sha = student_checkpoint_meta["payload_sha256"]

    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError("nibabel is required for native image loading") from exc
    per_patient: dict[str, dict[tuple[int, int], list[dict[str, Any]]]] = {}
    a1_records = []
    timepoints_total = sum(int(r["shape_xyzt"][3]) for r in manifest["patients"])
    timepoints_used = 0
    teacher_lineage = {}
    teacher_geometry = {}
    a1_dir = None
    if collect_a1_cache:
        a1_dir = Path(a1_cache_dir)
        a1_dir.mkdir(parents=True, exist_ok=False)
    for record in records:
        pid = record["id"]
        affine = np.asarray(record["affine_xyz_to_world_mm"], dtype=np.float64)
        native_img = nib.load(record["native_image_path"], mmap=True)
        header_affine = np.asarray(record["native_affine_xyz_to_world_header_units"], dtype=np.float64)
        if not np.allclose(np.asarray(native_img.affine, dtype=np.float64), header_affine, atol=1e-8, rtol=0):
            raise ValueError(f"native affine differs from hash-bound manifest for {pid}")
        raw_xyzt = np.asarray(native_img.dataobj, dtype=np.float32)
        if list(raw_xyzt.shape) != record["shape_xyzt"] or not np.isfinite(raw_xyzt).all():
            raise ValueError(f"native image shape/content mismatch for {pid}")
        # NIfTI and prepared arrays share exact voxel ordering. Prepared values use
        # the canonical per-patient percentile normalization from prepare_acdc.py.
        lo, hi = np.percentile(raw_xyzt, [.5, 99.5])
        cine_tzyx = np.clip((raw_xyzt - lo) / max(float(hi - lo), 1e-6), 0, 1).transpose(3, 2, 1, 0).astype(np.float32)
        prepared = np.load(record["prepared_image_path"], mmap_mode="r", allow_pickle=False)
        if prepared.shape != cine_tzyx.shape or not np.allclose(prepared, cine_tzyx, atol=2e-6, rtol=0):
            raise ValueError(f"prepared/native normalized voxel mismatch for {pid}")
        if mode == "a5":
            # The checked prepared TRAIN array is the student input.
            cine_tzyx = prepared
        t_count = len(cine_tzyx)
        patient_probabilities = patient_support = None
        if collect_a1_cache:
            z, y, x = cine_tzyx.shape[1:]
            patient_probabilities = np.lib.format.open_memmap(
                a1_dir / f"{pid}_probabilities.npy", mode="w+", dtype=np.float32,
                shape=(t_count, z, 4, y, x))
            patient_support = np.lib.format.open_memmap(
                a1_dir / f"{pid}_valid_mask.npy", mode="w+", dtype=np.bool_,
                shape=(t_count, z, y, x))
        chosen_times = list(range(t_count))
        if max_timepoints is not None:
            chosen_times = chosen_times[:max_timepoints]
        per_patient[pid] = {pair: [] for pair in PAIRS}
        for timepoint in chosen_times:
            if mode == "teacher":
                volume_xyz = raw_xyzt[..., timepoint]
                prediction = adapter.predict(volume_xyz, affine, timepoint=timepoint)
                support_mask = np.asarray(prediction.valid_mask)
                if support_mask.dtype != np.bool_:
                    raise ValueError("teacher valid_mask must be boolean native XYZ support")
                probs, features = _validate_prediction(prediction.probabilities, prediction.features,
                    tuple(volume_xyz.shape), support_mask)
                source_hash = str(prediction.lineage.get("weights_sha256", prediction.lineage.get("source_sha256", "")))
                source_kind = "pinned_external_teacher_features"
                geometry = prediction.geometry
                lineage = prediction.lineage
                if (lineage.get("status") != "PINNED_SOURCE"
                        or any(lineage.get(name) != teacher_pins[name]
                               for name in ("source_commit", "config_sha256", "weights_sha256", "class_map"))):
                    raise ValueError("teacher prediction lineage does not match the pinned CineMA source/assets/classes")
                geometry = {**geometry, "valid_mask_fraction": float(support_mask.mean())}
                teacher_lineage[f"{pid}:{timepoint}"] = lineage
                teacher_geometry[f"{pid}:{timepoint}"] = geometry
            else:
                probs, features = _prediction_from_student(student, cine_tzyx, timepoint, device)
                source_hash = student_source_sha
                source_kind = "frozen_student_final_fuse_32d"
                geometry = {"native_shape_xyz": list(raw_xyzt.shape[:3]), "affine_xyz_to_world_mm": affine.tolist()}
                lineage = {"checkpoint_payload_sha256": student_source_sha, "mode": "A5_student_only"}
            support_mask = prediction.valid_mask if mode == "teacher" else None
            by_pair = boundary_pair_candidates(probs, features, affine, patient=pid, timepoint=timepoint,
                image_sha256=record["image_sha256"], source_sha256=source_hash,
                source_kind=source_kind, valid_mask=support_mask)
            for pair in PAIRS:
                per_patient[pid][pair].extend(by_pair[pair])
            timepoints_used += 1
            if mode == "teacher":
                # Store native probability output in the exact training TZYX layout.
                p_tzyx = probs.transpose(3, 0, 2, 1)  # [Z,4,Y,X]; invalid support remains zero
                support_tzyx = support_mask.transpose(2, 1, 0)  # [Z,Y,X]
                if collect_a1_cache:
                    patient_probabilities[timepoint] = p_tzyx
                    patient_support[timepoint] = support_tzyx
        if collect_a1_cache:
            patient_probabilities.flush()
            patient_support.flush()
            del patient_probabilities, patient_support
            a1_records.append({"id": pid, "probabilities_path": str((a1_dir / f"{pid}_probabilities.npy").resolve()),
                               "valid_support_path": str((a1_dir / f"{pid}_valid_mask.npy").resolve()),
                               "timepoints": chosen_times})
        del native_img, raw_xyzt

    selected = patient_balanced_select(per_patient)
    coverage = {"train_patients_total": int(manifest.get("coverage", {}).get("train_patients_total", len(manifest["patients"]))),
                "train_patients_in_manifest": len(manifest["patients"]), "train_patients_used": len(records),
                "timepoints_total": timepoints_total, "timepoints_used": timepoints_used,
                "complete": bool(manifest.get("coverage", {}).get("complete", True))
                    and len(records) == len(manifest["patients"]) and timepoints_used == timepoints_total,
                "bounded": bool(manifest.get("coverage", {}).get("bounded", False))
                    or max_patients is not None or max_timepoints is not None}
    geometry_by_patient = {
        r["id"]: {
            "shape_xyzt": r["shape_xyzt"],
            "shape_tzyx": [int(r["shape_xyzt"][3]), int(r["shape_xyzt"][2]),
                           int(r["shape_xyzt"][1]), int(r["shape_xyzt"][0])],
            "affine_xyz_to_world_mm": r["affine_xyz_to_world_mm"],
            "image_sha256": r["image_sha256"],
            "native_image_sha256": r["native_image_sha256"],
            "axis_mapping": "native_NIfTI_XYZT_to_prepared_TZYX=transpose(3,2,1,0)",
        }
        for r in records
    }
    provenance = {
        "mode": "A5_frozen_student_only" if mode == "a5" else "pinned_image_only_teacher",
        "native_manifest_sha256": native_manifest_sha256,
        "teacher_weights_sha256": None if mode == "a5" else getattr(adapter, "weights_sha256", None),
        "teacher_config_sha256": None if mode == "a5" else getattr(adapter, "config_sha256", None),
        "teacher_source_commit": None if mode == "a5" else getattr(adapter, "source_commit", None),
        "student_checkpoint_payload_sha256": student_source_sha,
        "student_checkpoint_manifest_sha256": student_manifest_sha256 if mode == "a5" else None,
        "student_source_arm": student_source_arm if mode == "a5" else None,
        "student_checkpoint_data_sha256": None if mode != "a5" else student_checkpoint_meta["data_sha256"],
        "student_checkpoint_labeled_ids": None if mode != "a5" else student_checkpoint_meta["labeled_ids"],
        "student_checkpoint_unlabeled_ids": None if mode != "a5" else student_checkpoint_meta["unlabeled_ids"],
        "student_checkpoint_label_budget": None if mode != "a5" else student_checkpoint_meta["label_budget"],
        "student_checkpoint_audit_label_reads": None if mode != "a5" else student_checkpoint_meta["audit_label_reads"],
        "source_sha256": student_source_sha if mode == "a5" else getattr(adapter, "weights_sha256", None),
        "student_checkpoint_synthetic": None if mode != "a5" else bool(student_checkpoint_meta.get("synthetic", False)),
        "student_input_contract": None if mode != "a5" else {
            "source": "prepared_image_path mmap after native-voxel comparison atol_2e-6_rtol_0",
            "normalization": "per_cine_image_only_percentiles_0.5_99.5_full_XYZT; clip_to_[0,1]",
            "layout": "native_NIfTI_XYZT_to_prepared_TZYX=transpose(3,2,1,0)",
            "comparison_atol": 2e-6,
        },
        "image_hashes": {r["id"]: r["image_sha256"] for r in records},
        "native_image_hashes": {r["id"]: r["native_image_sha256"] for r in records},
        "native_geometry_by_patient": geometry_by_patient,
        "teacher_lineage_by_patient_timepoint": teacher_lineage,
        "teacher_geometry_by_patient_timepoint": teacher_geometry,
        "source_train_manifest_sha256": manifest["train_manifest_sha256"],
        "source_labeled_ids": source_labeled_ids,
        "source_unlabeled_ids": source_unlabeled_ids,
        "geometry": "native NIfTI affine; endpoint positions store XYZ voxel and RAS/world mm coordinates",
        "pair_distance_mm": MAX_DISTANCE_MM, "endpoint_confidence_min": MIN_CONFIDENCE,
        "feature_dim": FEATURE_DIM, "coverage": coverage,
    }
    return {"bank": pack_bank(selected, provenance), "coverage": coverage,
            "a1_records": a1_records, "manifest": manifest}


def write_a1_cache(output_dir: str | Path, build_result: dict[str, Any], *, bank_sha256: str | None = None) -> Path:
    """Index streamed A1 TRAIN arrays without loading dense predictions into RAM."""
    if not build_result["a1_records"]:
        raise ValueError("A1 cache writing requires teacher-mode predictions")
    if not build_result["coverage"]["complete"]:
        raise ValueError("refusing to write a partial A1 cache; all TRAIN patients/timepoints are required")
    out = Path(output_dir).resolve()
    if not out.is_dir():
        raise ValueError("A1 streamed cache directory must already exist")
    records = []
    image_hashes = build_result["bank"]["metadata"]["provenance"]["image_hashes"]
    bank_provenance = build_result["bank"]["metadata"]["provenance"]
    native_geometry = {}
    native_by_id = {item["id"]: item for item in build_result["manifest"]["patients"]}
    output_items = build_result["a1_records"]
    output_ids = [item.get("id") for item in output_items if isinstance(item, dict)]
    if (len(output_ids) != len(output_items) or len(output_ids) != len(set(output_ids))
            or set(output_ids) != set(native_by_id)):
        raise ValueError("A1 cache must contain exactly one output record for every native TRAIN patient")
    lineage = {}
    teacher_lineage = bank_provenance.get("teacher_lineage_by_patient_timepoint", {})
    teacher_geometry = bank_provenance.get("teacher_geometry_by_patient_timepoint", {})
    for key, source_lineage in teacher_lineage.items():
        pid, timepoint = key.rsplit(":", 1)
        lineage.setdefault(pid, {})[timepoint] = {
            "geometry": teacher_geometry.get(key, {}), "lineage": source_lineage}
    for item in sorted(build_result["a1_records"], key=lambda r: r["id"]):
        pid = item["id"]
        path = Path(item["probabilities_path"]).resolve()
        support_path = Path(item["valid_support_path"]).resolve()
        if path.parent != out or support_path.parent != out:
            raise ValueError("streamed A1 arrays must reside beside the cache manifest")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        support_array = np.load(support_path, mmap_mode="r", allow_pickle=False)
        native = native_by_id[pid]
        expected_shape = (int(native["shape_xyzt"][3]), int(native["shape_xyzt"][2]), 4,
                          int(native["shape_xyzt"][1]), int(native["shape_xyzt"][0]))
        expected_support_shape = (expected_shape[0], expected_shape[1], expected_shape[3], expected_shape[4])
        expected_timepoints = list(range(expected_shape[0]))
        if item.get("timepoints") != expected_timepoints:
            raise ValueError("A1 cache must cover each native TRAIN timepoint exactly once")
        if array.dtype != np.float32 or array.shape != expected_shape:
            raise ValueError("streamed A1 probabilities do not match native T,Z,4,Y,X geometry")
        if support_array.dtype != np.bool_ or support_array.shape != expected_support_shape:
            raise ValueError("streamed A1 support does not match native boolean T,Z,Y,X geometry")
        flat_probs = array.reshape(-1, 4, expected_shape[-2], expected_shape[-1])
        flat_support = support_array.reshape(-1, expected_shape[-2], expected_shape[-1])
        for start in range(0, len(flat_probs), 8):
            block = np.asarray(flat_probs[start:start + 8])
            support_block = np.asarray(flat_support[start:start + 8])
            if not np.isfinite(block).all() or np.any(block < 0):
                raise ValueError("streamed A1 probabilities must be finite and nonnegative")
            sums = block.sum(axis=1)
            if support_block.any() and not np.allclose(sums[support_block], 1.0, rtol=0, atol=2e-3):
                raise ValueError("streamed A1 probabilities must sum to one on valid support")
            if (~support_block).any() and np.any(block.transpose(0, 2, 3, 1)[~support_block] != 0.0):
                raise ValueError("streamed A1 probabilities must be zero outside valid support")
        records.append({"id": pid, "image_sha256": image_hashes[pid], "probabilities": path.name,
                        "probabilities_sha256": sha256_file(path), "shape": list(array.shape)})
        native_geometry[pid] = {
            "shape_tzyx": [int(native["shape_xyzt"][3]), int(native["shape_xyzt"][2]),
                           int(native["shape_xyzt"][1]), int(native["shape_xyzt"][0])],
            "affine_xyz_to_world_mm": native["affine_xyz_to_world_mm"],
            "probability_shape_tzcyx": list(array.shape),
            "axis_mapping": "native_NIfTI_XYZT_to_prepared_TZYX=transpose(3,2,1,0); probability_TZCYX_plane=class,Y,X from the same native affine XYZ grid",
            "valid_support": {"path": support_path.name, "sha256": sha256_file(support_path),
                              "shape_tzyx": list(support_array.shape)},
        }
    cache = {
        "schema": "self_audit.external_pseudo.v1", "role": "training", "synthetic": False,
        "probability_layout": "TZCYX", "class_map": list(CLASSES), "records": records,
        "evaluation_ids": build_result["manifest"]["evaluation_ids"], "bank_sha256": bank_sha256,
        "provenance": {"writer": "self_audit.memory_bank.write_a1_cache", "lineage": lineage,
                       "native_geometry_by_patient": native_geometry,
                       "native_manifest_sha256": bank_provenance["native_manifest_sha256"],
                       "native_image_hashes": bank_provenance["native_image_hashes"],
                       "unsupported_voxel_target": "zero_probability_abstention; valid_support_mask_required_for_consumers",
                       "coverage": build_result["coverage"], "masks_read": 0},
    }
    cache_path = out / "external_pseudo.json"
    cache_path.write_text(json.dumps(cache, indent=2, allow_nan=False) + "\n")
    return cache_path
