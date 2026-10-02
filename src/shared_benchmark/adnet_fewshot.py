"""ADNet few-shot support contract helpers.

ADNet is not an image-only producer: it needs labeled support evidence at
inference time.  This module keeps that support evidence explicit and keeps
query records image-only, so runners can separate producer execution from
query-GT evaluation.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .provenance import canonical_json_bytes, sha256_bytes, sha256_json


SUPPORT_MANIFEST_SCHEMA = "adnet.support_manifest.v1"
QUERY_MANIFEST_SCHEMA = "adnet.query_manifest.v1"
GT_MANIFEST_SCHEMA = "adnet.gt_manifest.v1"
RAW_SCHEMA_VERSION = "adnet.fewshot_raw.v1"
SEMANTIC_SCHEMA_VERSION = "adnet.fewshot_semantic.v1"
EVALUATION_SCHEMA_VERSION = "adnet.fewshot_evaluation.v1"
ALLOWED_TIE_RULES = frozenset({"fail_closed"})

_FORBIDDEN_QUERY_KEYS = {
    "gt", "groundtruth", "ground_truth", "mask", "masks", "label", "labels",
    "annotation", "annotations", "dice", "iou", "oracle", "metric", "metrics",
}


class ADNetContractError(ValueError):
    """Raised when an ADNet few-shot manifest or output violates the contract."""


@dataclass(frozen=True)
class ADNetSupportClass:
    support_id: str
    class_name: str
    canonical_class_id: int
    image_path: Path
    image_sha256: str
    mask_path: Path
    mask_sha256: str
    slice_indices: tuple[int, ...]


@dataclass(frozen=True)
class ADNetSupportManifest:
    path: Path
    document: dict[str, Any]
    sha256: str
    class_mapping: dict[str, int]
    supports: tuple[ADNetSupportClass, ...]


@dataclass(frozen=True)
class ADNetQueryRecord:
    sample_id: str
    image_path: Path
    image_sha256: str
    split: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ADNetQueryManifest:
    path: Path
    document: dict[str, Any]
    sha256: str
    records: tuple[ADNetQueryRecord, ...]


@dataclass(frozen=True)
class ADNetGTRecord:
    sample_id: str
    gt_path: Path
    gt_sha256: str


@dataclass(frozen=True)
class ADNetGTManifest:
    path: Path
    document: dict[str, Any]
    sha256: str
    records: tuple[ADNetGTRecord, ...]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    _atomic_write_bytes(Path(path), canonical_json_bytes(_json_safe(value)) + b"\n")


def atomic_write_npy(path: str | Path, array: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, np.asarray(array), allow_pickle=False)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def load_support_manifest(path: str | Path, *, asset_root: str | Path | None = None) -> ADNetSupportManifest:
    location = Path(path).resolve()
    document = _read_object(location)
    if document.get("schema") != SUPPORT_MANIFEST_SCHEMA:
        raise ADNetContractError("unknown ADNet support manifest schema")
    raw_mapping = document.get("class_mapping")
    if not isinstance(raw_mapping, Mapping) or not raw_mapping:
        raise ADNetContractError("support manifest requires class_mapping")
    class_mapping = _parse_class_mapping(raw_mapping)
    raw_supports = document.get("supports")
    if not isinstance(raw_supports, list) or not raw_supports:
        raise ADNetContractError("support manifest requires non-empty supports")
    supports = tuple(_parse_support(row, location, asset_root, class_mapping) for row in raw_supports)
    class_ids = [support.canonical_class_id for support in supports]
    if len(set(class_ids)) != len(class_ids):
        raise ADNetContractError("support manifest has duplicate canonical_class_id")
    return ADNetSupportManifest(
        path=location,
        document=document,
        sha256=sha256_json(document),
        class_mapping=class_mapping,
        supports=supports,
    )


def load_query_manifest(path: str | Path, *, image_root: str | Path | None = None) -> ADNetQueryManifest:
    location = Path(path).resolve()
    document = _read_object(location)
    if document.get("schema") != QUERY_MANIFEST_SCHEMA:
        raise ADNetContractError("unknown ADNet query manifest schema")
    _reject_forbidden_query_evidence(document, where="query_manifest")
    raw_records = document.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ADNetContractError("query manifest requires non-empty records")
    records = tuple(_parse_query(row, location, image_root) for row in raw_records)
    sample_ids = [record.sample_id for record in records]
    if len(set(sample_ids)) != len(sample_ids):
        raise ADNetContractError("query manifest has duplicate sample_id")
    return ADNetQueryManifest(
        path=location,
        document=document,
        sha256=sha256_json(document),
        records=records,
    )


def load_gt_manifest(path: str | Path, *, gt_root: str | Path | None = None) -> ADNetGTManifest:
    """Load evaluator-only GT bindings.

    This manifest is intentionally separate from the producer query manifest.
    It may name GT files because only the post-seal evaluator consumes it.
    """
    location = Path(path).resolve()
    document = _read_object(location)
    if document.get("schema") != GT_MANIFEST_SCHEMA:
        raise ADNetContractError("unknown ADNet GT manifest schema")
    raw_records = document.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ADNetContractError("GT manifest requires non-empty records")
    records = tuple(_parse_gt(row, location, gt_root) for row in raw_records)
    sample_ids = [record.sample_id for record in records]
    if len(set(sample_ids)) != len(sample_ids):
        raise ADNetContractError("GT manifest has duplicate sample_id")
    return ADNetGTManifest(
        path=location,
        document=document,
        sha256=sha256_json(document),
        records=records,
    )


def merge_class_binary_predictions(
    predictions: Mapping[int, np.ndarray], *, tie_rule: str = "fail_closed",
) -> tuple[np.ndarray, np.ndarray]:
    """Merge class-wise binary ADNet predictions into one semantic map.

    The first audit-grade rule fails on overlap because upstream ADNet saves
    hard binary masks and does not preserve comparable per-class scores.
    """
    if tie_rule not in ALLOWED_TIE_RULES:
        raise ADNetContractError(f"unsupported ADNet tie rule: {tie_rule}")
    if not predictions:
        raise ADNetContractError("at least one class prediction is required")
    arrays: list[tuple[int, np.ndarray]] = []
    shape: tuple[int, ...] | None = None
    for class_id, value in sorted(predictions.items(), key=lambda item: int(item[0])):
        cid = int(class_id)
        if cid <= 0:
            raise ADNetContractError("foreground class ids must be positive")
        array = np.asarray(value)
        if array.ndim not in (2, 3):
            raise ADNetContractError("ADNet predictions must be rank-2 or rank-3")
        binary = array.astype(bool, copy=False)
        if shape is None:
            shape = tuple(int(dim) for dim in binary.shape)
        elif tuple(int(dim) for dim in binary.shape) != shape:
            raise ADNetContractError("class predictions have mismatched shapes")
        arrays.append((cid, binary))
    assert shape is not None
    votes = np.zeros(shape, dtype=np.uint8)
    semantic = np.zeros(shape, dtype=np.uint8)
    for cid, binary in arrays:
        votes += binary.astype(np.uint8)
        semantic[binary] = np.uint8(cid)
    if np.any(votes > 1):
        raise ADNetContractError("overlapping ADNet class predictions require an explicit tie policy")
    validity = np.ones(shape, dtype=bool)
    return semantic, validity


def seal_adnet_outputs(
    output_root: str | Path,
    *,
    sample_id: str,
    class_predictions: Mapping[int, np.ndarray],
    support_manifest: ADNetSupportManifest,
    query_manifest: ADNetQueryManifest,
    query_record: ADNetQueryRecord,
    checkpoint_sha256: str,
    code_identity: Mapping[str, Any],
    run_config: Mapping[str, Any],
    tie_rule: str = "fail_closed",
) -> dict[str, Any]:
    semantic, validity = merge_class_binary_predictions(class_predictions, tie_rule=tie_rule)
    root = Path(output_root) / _safe_slug(sample_id)
    raw_root = root / "class_binary"
    raw_root.mkdir(parents=True, exist_ok=True)
    class_payload: dict[str, Any] = {}
    for class_id, prediction in sorted(class_predictions.items(), key=lambda item: int(item[0])):
        binary = np.asarray(prediction).astype(np.uint8)
        filename = f"class_{int(class_id):03d}.npy"
        atomic_write_npy(raw_root / filename, binary)
        class_payload[str(int(class_id))] = {
            "path": str(Path("class_binary") / filename),
            "shape": [int(dim) for dim in binary.shape],
            "dtype": str(binary.dtype),
            "array_sha256": array_sha256(binary),
            "file_sha256": file_sha256(raw_root / filename),
        }
    atomic_write_npy(root / "semantic_map.npy", semantic)
    atomic_write_npy(root / "validity_map.npy", validity)
    metadata = {
        "schema": SEMANTIC_SCHEMA_VERSION,
        "completion_status": "SEMANTIC_COMPLETE",
        "sample_id": sample_id,
        "query_image_sha256": query_record.image_sha256,
        "query_manifest_sha256": query_manifest.sha256,
        "support_manifest_sha256": support_manifest.sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "tie_rule": tie_rule,
        "class_mapping": support_manifest.class_mapping,
        "class_predictions": class_payload,
        "semantic_map": {
            "path": "semantic_map.npy",
            "shape": [int(dim) for dim in semantic.shape],
            "dtype": str(semantic.dtype),
            "array_sha256": array_sha256(semantic),
            "file_sha256": file_sha256(root / "semantic_map.npy"),
        },
        "validity_map": {
            "path": "validity_map.npy",
            "shape": [int(dim) for dim in validity.shape],
            "dtype": str(validity.dtype),
            "array_sha256": array_sha256(validity),
            "file_sha256": file_sha256(root / "validity_map.npy"),
        },
        "code_identity": _json_safe(dict(code_identity)),
        "run_config": _json_safe(dict(run_config)),
    }
    metadata["scientific_payload_sha256"] = sha256_json(metadata)
    atomic_write_json(root / "metadata.json", metadata)
    return metadata


def verify_adnet_output(output_dir: str | Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    root = Path(output_dir)
    metadata = _read_object(root / "metadata.json")
    if metadata.get("schema") != SEMANTIC_SCHEMA_VERSION or metadata.get("completion_status") != "SEMANTIC_COMPLETE":
        raise ADNetContractError("ADNet output metadata is not SEMANTIC_COMPLETE")
    payload = dict(metadata)
    expected_payload_hash = payload.pop("scientific_payload_sha256", None)
    if expected_payload_hash != sha256_json(payload):
        raise ADNetContractError("ADNet output scientific payload hash mismatch")
    semantic_info = _required_mapping(metadata, "semantic_map")
    validity_info = _required_mapping(metadata, "validity_map")
    semantic = _load_bound_array(root, semantic_info, expected_dtype=np.uint8)
    validity = _load_bound_array(root, validity_info, expected_dtype=np.bool_)
    if semantic.shape != validity.shape:
        raise ADNetContractError("semantic and validity maps have mismatched shapes")
    if not np.isin(semantic, [0, 1, 2, 3, 4]).all():
        raise ADNetContractError("semantic map contains labels outside 0..4")
    for class_id, class_info in _required_mapping(metadata, "class_predictions").items():
        _load_bound_array(root, class_info, expected_dtype=np.uint8)
        if int(class_id) <= 0:
            raise ADNetContractError("class prediction key must be a foreground class id")
    return metadata, np.ascontiguousarray(semantic), np.ascontiguousarray(validity)


def evaluate_adnet_outputs(
    output_root: str | Path,
    gt_manifest: ADNetGTManifest,
    *,
    class_ids: tuple[int, ...] = (1, 2, 3),
) -> dict[str, Any]:
    rows = []
    per_class: dict[int, list[dict[str, Any]]] = {int(class_id): [] for class_id in class_ids}
    for gt_record in gt_manifest.records:
        metadata, semantic, validity = verify_adnet_output(Path(output_root) / _safe_slug(gt_record.sample_id))
        gt = _load_gt_array(gt_record)
        if gt.shape != semantic.shape:
            raise ADNetContractError("GT and semantic output shapes differ")
        if validity.shape != semantic.shape:
            raise ADNetContractError("validity and semantic output shapes differ")
        sample_scores = {}
        for class_id in class_ids:
            score = _binary_score(semantic == int(class_id), gt == int(class_id), validity)
            sample_scores[str(int(class_id))] = score
            per_class[int(class_id)].append(score)
        rows.append({
            "sample_id": gt_record.sample_id,
            "output_scientific_payload_sha256": metadata["scientific_payload_sha256"],
            "gt_sha256": gt_record.gt_sha256,
            "scores": sample_scores,
        })
    class_summary = {}
    for class_id, scores in per_class.items():
        defined = [score for score in scores if score["defined"]]
        class_summary[str(class_id)] = {
            "dice": _mean_or_none(score["dice"] for score in defined),
            "iou": _mean_or_none(score["iou"] for score in defined),
            "defined_sample_count": len(defined),
            "sample_count": len(scores),
        }
    defined_dice = [value["dice"] for value in class_summary.values() if value["dice"] is not None]
    defined_iou = [value["iou"] for value in class_summary.values() if value["iou"] is not None]
    return {
        "schema": EVALUATION_SCHEMA_VERSION,
        "gt_manifest_sha256": gt_manifest.sha256,
        "class_ids": [int(class_id) for class_id in class_ids],
        "samples": rows,
        "class_summary": class_summary,
        "macro_dice": _mean_or_none(defined_dice),
        "macro_iou": _mean_or_none(defined_iou),
    }


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(array))
    header = canonical_json_bytes({"shape": [int(dim) for dim in value.shape], "dtype": value.dtype.str})
    return sha256_bytes(header + value.tobytes(order="C"))


def _parse_class_mapping(raw_mapping: Mapping[str, Any]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for key, value in raw_mapping.items():
        if not isinstance(key, str) or not key:
            raise ADNetContractError("class_mapping keys must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, int) or int(value) < 0 or int(value) > 4:
            raise ADNetContractError("class_mapping values must be canonical cardiac ids 0..4")
        mapping[key] = int(value)
    return dict(sorted(mapping.items()))


def _parse_support(
    row: Any, manifest_path: Path, asset_root: str | Path | None, class_mapping: Mapping[str, int],
) -> ADNetSupportClass:
    if not isinstance(row, Mapping):
        raise ADNetContractError("support rows must be objects")
    expected = {
        "support_id", "class_name", "canonical_class_id", "image_path",
        "image_sha256", "mask_path", "mask_sha256", "slice_indices",
    }
    if set(row) != expected:
        raise ADNetContractError(f"support row fields must be exactly {sorted(expected)}")
    support_id = _safe_id(row["support_id"], "support_id")
    class_name = _safe_id(row["class_name"], "class_name")
    canonical_class_id = _int_field(row["canonical_class_id"], "canonical_class_id", minimum=1, maximum=4)
    if class_mapping.get(class_name) != canonical_class_id:
        raise ADNetContractError("support class mapping disagrees with canonical_class_id")
    image_path = _resolve_bound_path(row["image_path"], manifest_path, asset_root, field="support image")
    mask_path = _resolve_bound_path(row["mask_path"], manifest_path, asset_root, field="support mask")
    image_sha = _sha_field(row["image_sha256"], "image_sha256")
    mask_sha = _sha_field(row["mask_sha256"], "mask_sha256")
    if file_sha256(image_path) != image_sha:
        raise ADNetContractError("support image hash mismatch")
    if file_sha256(mask_path) != mask_sha:
        raise ADNetContractError("support mask hash mismatch")
    raw_indices = row["slice_indices"]
    if not isinstance(raw_indices, list) or not raw_indices:
        raise ADNetContractError("support slice_indices must be a non-empty list")
    indices = tuple(_int_field(value, "slice_indices", minimum=0) for value in raw_indices)
    return ADNetSupportClass(
        support_id=support_id,
        class_name=class_name,
        canonical_class_id=canonical_class_id,
        image_path=image_path,
        image_sha256=image_sha,
        mask_path=mask_path,
        mask_sha256=mask_sha,
        slice_indices=indices,
    )


def _parse_query(row: Any, manifest_path: Path, image_root: str | Path | None) -> ADNetQueryRecord:
    if not isinstance(row, Mapping):
        raise ADNetContractError("query rows must be objects")
    allowed = {"sample_id", "image_path", "image_sha256", "split", "metadata"}
    if set(row) - allowed:
        raise ADNetContractError(f"query row has unsupported fields: {sorted(set(row) - allowed)}")
    if not {"sample_id", "image_path", "image_sha256"} <= set(row):
        raise ADNetContractError("query row requires sample_id, image_path and image_sha256")
    _reject_forbidden_query_evidence(row, where="query_record")
    sample_id = _safe_id(row["sample_id"], "sample_id")
    path = _resolve_bound_path(row["image_path"], manifest_path, image_root, field="query image")
    sha = _sha_field(row["image_sha256"], "image_sha256")
    if file_sha256(path) != sha:
        raise ADNetContractError("query image hash mismatch")
    split = str(row.get("split", "dev"))
    metadata = row.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ADNetContractError("query metadata must be an object")
    return ADNetQueryRecord(
        sample_id=sample_id,
        image_path=path,
        image_sha256=sha,
        split=split,
        metadata=_json_safe(dict(metadata)),
    )


def _parse_gt(row: Any, manifest_path: Path, gt_root: str | Path | None) -> ADNetGTRecord:
    if not isinstance(row, Mapping):
        raise ADNetContractError("GT rows must be objects")
    expected = {"sample_id", "gt_path", "gt_sha256"}
    if set(row) != expected:
        raise ADNetContractError(f"GT row fields must be exactly {sorted(expected)}")
    sample_id = _safe_id(row["sample_id"], "sample_id")
    path = _resolve_bound_path(row["gt_path"], manifest_path, gt_root, field="GT")
    sha = _sha_field(row["gt_sha256"], "gt_sha256")
    if file_sha256(path) != sha:
        raise ADNetContractError("GT hash mismatch")
    return ADNetGTRecord(sample_id=sample_id, gt_path=path, gt_sha256=sha)


def _resolve_bound_path(
    raw_path: Any, manifest_path: Path, root: str | Path | None, *, field: str,
) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ADNetContractError(f"{field} path must be a non-empty string")
    path = Path(raw_path)
    if path.is_absolute():
        candidate = path
    else:
        base = Path(root).resolve() if root is not None else manifest_path.parent
        candidate = base / path
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ADNetContractError(f"{field} path is not readable") from exc
    if root is not None:
        root_path = Path(root).resolve()
        try:
            resolved.relative_to(root_path)
        except ValueError as exc:
            raise ADNetContractError(f"{field} path escapes declared root") from exc
    return resolved


def _reject_forbidden_query_evidence(value: Any, *, where: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower().replace("-", "_").replace(" ", "_")
            if normalized in _FORBIDDEN_QUERY_KEYS or any(token in normalized for token in ("mask", "label", "gt")):
                raise ADNetContractError(f"forbidden query evidence at {where}.{key}")
            _reject_forbidden_query_evidence(child, where=f"{where}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_forbidden_query_evidence(child, where=f"{where}[{index}]")


def _safe_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ADNetContractError(f"{field} must be a non-empty string")
    if any(ch in value for ch in "/\\\x00"):
        raise ADNetContractError(f"{field} contains unsafe path characters")
    return value


def _safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)[:120] or "sample"


def _int_field(value: Any, field: str, *, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ADNetContractError(f"{field} must be an integer")
    result = int(value)
    if result < minimum or (maximum is not None and result > maximum):
        raise ADNetContractError(f"{field} is outside the allowed range")
    return result


def _sha_field(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ADNetContractError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ADNetContractError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ADNetContractError("manifest must be a JSON object")
    return value


def _required_mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    candidate = value.get(key)
    if not isinstance(candidate, Mapping) or not candidate:
        raise ADNetContractError(f"metadata requires non-empty mapping {key}")
    return candidate


def _load_bound_array(root: Path, info: Mapping[str, Any], *, expected_dtype: np.dtype) -> np.ndarray:
    relative = info.get("path")
    if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ADNetContractError("array path must be a safe relative path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ADNetContractError("array path escapes output root") from exc
    if file_sha256(path) != info.get("file_sha256"):
        raise ADNetContractError("array file hash mismatch")
    array = np.load(path, allow_pickle=False)
    if array.dtype != expected_dtype:
        raise ADNetContractError("array dtype mismatch")
    if [int(dim) for dim in array.shape] != info.get("shape"):
        raise ADNetContractError("array shape metadata mismatch")
    if array_sha256(array) != info.get("array_sha256"):
        raise ADNetContractError("array scientific hash mismatch")
    return np.ascontiguousarray(array)


def _load_gt_array(record: ADNetGTRecord) -> np.ndarray:
    name = record.gt_path.name.lower()
    if name.endswith(".npy"):
        value = np.load(record.gt_path, allow_pickle=False)
    elif name.endswith(".nii") or name.endswith(".nii.gz"):
        try:
            import SimpleITK as sitk
        except Exception as exc:  # pragma: no cover - environment dependent
            raise ADNetContractError(f"SimpleITK is required to read {record.gt_path.name}: {exc}") from exc
        value = sitk.GetArrayFromImage(sitk.ReadImage(str(record.gt_path)))
    else:
        raise ADNetContractError("unsupported GT container")
    array = np.asarray(value)
    if array.ndim not in (2, 3) or not np.issubdtype(array.dtype, np.integer):
        raise ADNetContractError("GT must be an integer rank-2 or rank-3 array")
    if not np.isin(array, [0, 1, 2, 3]).all():
        raise ADNetContractError("GT contains labels outside canonical BG/RV/MYO/LV ids")
    return np.ascontiguousarray(array.astype(np.uint8, copy=False))


def _binary_score(prediction: np.ndarray, target: np.ndarray, validity: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(prediction, dtype=bool) & np.asarray(validity, dtype=bool)
    gt = np.asarray(target, dtype=bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    denom_dice = (2 * tp) + fp + fn
    denom_iou = tp + fp + fn
    defined = denom_iou > 0
    return {
        "dice": None if not defined else float((2 * tp) / denom_dice),
        "iou": None if not defined else float(tp / denom_iou),
        "defined": bool(defined),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _mean_or_none(values: Any) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return None if not clean else float(np.mean(clean))


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
