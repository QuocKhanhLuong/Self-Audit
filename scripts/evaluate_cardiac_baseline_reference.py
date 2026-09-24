#!/usr/bin/env python
"""Evaluate sealed ACDC semantic artifacts in a GT-only post-freeze process.

This is deliberately separate from CUTS/DFC generation.  It never creates a
partition, invokes an optimizer, or resolves anonymous components.  The named
semantic mapping is read only from the frozen semantic artifact: BG=0, RV=1,
MYO=2, LV=3, VOID=4.  GT is used only after all artifact seals have passed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT / "src"))

from self_audit_maskfree.evaluation.metrics import dice_iou_volume, patient_macro
from shared_benchmark.artifacts import ArtifactError, RawArtifact, verify_raw_partition, verify_semantic_partition
from shared_benchmark.provenance import sha256_file
from shared_benchmark.semantic_contract import BG, FROZEN_ADAPTER_SPEC_SHA256, LV, MYO, RV, VOID
from shared_benchmark.spatial import read_self_audit_context_stack, resize_values_to_grid


EVALUATOR_SCHEMA = "shared_benchmark.cardiac_reference_evaluator.v1"
SEMANTIC_SCHEMA = "shared_benchmark.cardiac-semantic.v2"
CLASS_NAMES = {BG: "BG", RV: "RV", MYO: "MYO", LV: "LV", VOID: "VOID"}
FOREGROUND = (RV, MYO, LV)


class ReferenceEvaluationError(ValueError):
    """Raised for a contract, artifact, geometry, or GT binding failure."""


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReferenceEvaluationError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ReferenceEvaluationError(f"JSON object required: {path}")
    return value


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    contract = _json(path)
    expected_classes = {"0": "BG", "1": "RV", "2": "MYO", "3": "LV", "4": "VOID"}
    expected_reference = {"0": "BG", "1": "RV", "2": "MYO", "3": "LV"}
    expected_void_policy = "VOID is never relabelled as BG; full-support foreground Dice/IoU treats VOID as no predicted named class and reports native-grid coverage separately"
    expected_baseline = {
        "name": "DFC", "mode": "DFC-Direct-2D-Default-MinL3",
        "baseline_config_sha256": "36fefade141b76290640a9feaf269abbe957239a5e085c73ec4f93e6d51a9c58",
    }
    if (
        contract.get("schema_version") != EVALUATOR_SCHEMA
        or contract.get("dataset") != "acdc"
        or contract.get("split") != "dev"
        or contract.get("semantic_schema_version") != SEMANTIC_SCHEMA
        or contract.get("baseline") != expected_baseline
        or contract.get("semantic_class_map") != expected_classes
        or contract.get("reference_label_map") != expected_reference
        or contract.get("prediction_inverse") != "nearest_exact_256_to_native_per_manifest_spatial_transform"
        or contract.get("void_policy") != expected_void_policy
        or contract.get("aggregation") != "per_patient_frame_3d_then_patient_macro_over_defined_ED_ES_frame_scores"
        or set(contract.get("metrics", [])) != {"dice", "iou", "coverage"}
    ):
        raise ReferenceEvaluationError("reference evaluator contract does not match the frozen ACDC v1 policy")
    forbidden = set(contract.get("forbidden", []))
    required_forbidden = {"per_case_matching", "hungarian_mapping", "gt_based_mapping", "hyperparameter_selection", "checkpoint_selection"}
    if not required_forbidden.issubset(forbidden):
        raise ReferenceEvaluationError("reference evaluator contract omits a required anti-leakage prohibition")
    return contract


def _dev_records(manifest: Mapping[str, Any], split: str) -> list[dict[str, Any]]:
    if manifest.get("schema_version") != "shared_benchmark_manifest.v1":
        raise ReferenceEvaluationError("unsupported manifest schema")
    records = sorted((dict(item) for item in manifest.get("records", []) if item.get("split") == split), key=lambda item: str(item.get("sample_id")))
    if not records:
        raise ReferenceEvaluationError(f"manifest has no {split} records")
    if split != "dev" or any(item.get("dataset") != "acdc" for item in records):
        raise ReferenceEvaluationError("reference evaluator accepts frozen ACDC dev records only")
    return records


def _contained(root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise ReferenceEvaluationError("reference path must be a safe relative locator")
    result = (root / relative).resolve()
    try:
        result.relative_to(root.resolve())
    except ValueError as exc:
        raise ReferenceEvaluationError("reference path escapes GT root") from exc
    return result


def gt_path_for_record(record: Mapping[str, Any], gt_root: Path) -> Path:
    source = record.get("source")
    if not isinstance(source, Mapping):
        raise ReferenceEvaluationError("record has no source locator")
    locator = Path(str(source.get("locator", "")))
    if locator.suffix != ".nii":
        raise ReferenceEvaluationError("ACDC reference contract requires an uncompressed .nii image locator")
    return _contained(gt_root, locator.with_name(f"{locator.stem}_gt.nii"))


def _load_gt(path: Path, expected_shape: tuple[int, int, int]) -> np.ndarray:
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ReferenceEvaluationError("nibabel is required by the GT-only evaluator") from exc
    if not path.is_file():
        raise ReferenceEvaluationError(f"missing ACDC GT file: {path}")
    value = np.asarray(nib.load(str(path)).dataobj)
    if value.shape != expected_shape:
        raise ReferenceEvaluationError(f"GT shape {value.shape} differs from frozen native shape {expected_shape}: {path}")
    if not np.isfinite(value).all() or not np.equal(value, np.rint(value)).all():
        raise ReferenceEvaluationError(f"GT labels are not finite integers: {path}")
    labels = np.asarray(np.rint(value), dtype=np.int16)
    if not np.isin(labels, [BG, RV, MYO, LV]).all():
        raise ReferenceEvaluationError(f"GT labels outside fixed ACDC 0/1/2/3 map: {path}")
    return labels


def _central_image(record: Mapping[str, Any], image_root: Path) -> np.ndarray:
    stack = read_self_audit_context_stack(record, source_root=image_root)
    center = resize_values_to_grid(torch.from_numpy(np.ascontiguousarray(stack[1:2])), record["shared_grid"])[0]
    return np.asarray(center, dtype=np.float32)


def restore_semantic_slice(semantic: np.ndarray, native_hw: tuple[int, int]) -> np.ndarray:
    """Apply the frozen nearest-exact label inverse; VOID remains class 4."""
    array = np.asarray(semantic)
    if array.dtype != np.uint8 or array.ndim != 2 or not np.isin(array, [BG, RV, MYO, LV, VOID]).all():
        raise ReferenceEvaluationError("semantic artifact is not a uint8 BG/RV/MYO/LV/VOID map")
    restored = F.interpolate(torch.from_numpy(array.astype(np.float32, copy=False))[None, None], size=native_hw, mode="nearest-exact")[0, 0].cpu().numpy()
    if not np.equal(restored, np.rint(restored)).all():
        raise ReferenceEvaluationError("nearest label inverse produced non-integer values")
    return np.asarray(np.rint(restored), dtype=np.uint8)


def _state(directory: Path, expected_stage: str) -> None:
    state = _json(directory / "state.json")
    if state.get("stage") != expected_stage:
        raise ReferenceEvaluationError(f"artifact is not {expected_stage}: {directory}")


def _index_raw(root: Path, baseline_name: str, baseline_mode: str) -> dict[str, RawArtifact]:
    indexed: dict[str, RawArtifact] = {}
    for metadata_path in sorted(root.rglob("metadata.json")):
        raw = verify_raw_partition(metadata_path.parent)
        metadata = raw.metadata
        if metadata.get("baseline_name") != baseline_name or metadata.get("baseline_mode") != baseline_mode:
            continue
        sample_id = str(metadata["sample_id"])
        if sample_id in indexed:
            raise ReferenceEvaluationError(f"duplicate raw artifact for {sample_id}")
        indexed[sample_id] = raw
    return indexed


def _index_semantic(root: Path, baseline_name: str, baseline_mode: str) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for metadata_path in sorted(root.rglob("metadata.json")):
        metadata = _json(metadata_path)
        if metadata.get("schema_version") != "cardiac_semantic_partition.v2":
            continue
        if metadata.get("baseline_name") != baseline_name or metadata.get("baseline_mode") != baseline_mode:
            continue
        _state(metadata_path.parent, "SEMANTIC_COMPLETE")
        sample_id = str(metadata.get("sample_id"))
        if sample_id in indexed:
            raise ReferenceEvaluationError(f"duplicate semantic artifact for {sample_id}")
        indexed[sample_id] = metadata_path.parent
    return indexed


def _native_volume_from_slices(records: list[dict[str, Any]], semantic_by_id: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    first = records[0]
    if int(first.get("depth_axis", -1)) != 2:
        raise ReferenceEvaluationError("ACDC evaluator expects frozen HWZ records")
    native_shape = tuple(int(item) for item in first["native_shape"])
    native_hw = (int(first["native_hw"][0]), int(first["native_hw"][1]))
    depth = int(first["depth"])
    if native_shape != (native_hw[0], native_hw[1], depth):
        raise ReferenceEvaluationError("record native shape/extent disagree")
    prediction = np.full(native_shape, VOID, dtype=np.uint8)
    seen: set[int] = set()
    for record in records:
        if tuple(int(item) for item in record["native_shape"]) != native_shape or int(record["depth_axis"]) != 2:
            raise ReferenceEvaluationError("a frozen volume mixes native geometry")
        index = int(record["slice_index"])
        if not 0 <= index < depth or index in seen:
            raise ReferenceEvaluationError("invalid or duplicate slice in frozen volume")
        restored = restore_semantic_slice(np.asarray(semantic_by_id[str(record["sample_id"])]), native_hw)
        prediction[:, :, index] = restored
        seen.add(index)
    if seen != set(range(depth)):
        raise ReferenceEvaluationError("frozen volume is missing slices")
    return prediction, np.asarray(prediction != VOID, dtype=bool)


def _patient_scores(per_volume: list[dict[str, Any]]) -> dict[str, dict[int, dict[str, Any]]]:
    grouped: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for volume in per_volume:
        for class_id, score in volume["scores"].items():
            grouped[str(volume["patient_id"])][int(class_id)].append(score)
    result: dict[str, dict[int, dict[str, Any]]] = {}
    for patient_id, classes in grouped.items():
        result[patient_id] = {}
        for class_id, values in classes.items():
            defined = [item for item in values if bool(item["defined"])]
            result[patient_id][class_id] = {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "dice": float(np.mean([float(item["dice"]) for item in defined])) if defined else None,
                "iou": float(np.mean([float(item["iou"]) for item in defined])) if defined else None,
                "defined": bool(defined),
                "both_empty": not bool(defined),
                "one_empty": any(bool(item["one_empty"]) for item in values),
                "volume_count": len(values),
                "defined_volume_count": len(defined),
            }
    return result


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise ReferenceEvaluationError(f"refusing to overwrite evaluator output: {args.output_dir}")
    contract = load_contract(args.contract)
    expected_baseline = contract["baseline"]
    if args.baseline_name != expected_baseline["name"] or args.baseline_mode != expected_baseline["mode"]:
        raise ReferenceEvaluationError("CLI baseline identity differs from the frozen evaluator contract")
    manifest = _json(args.manifest)
    records = _dev_records(manifest, args.split)
    expected_ids = {str(record["sample_id"]) for record in records}
    raw = _index_raw(args.raw_root, args.baseline_name, args.baseline_mode)
    semantic_dirs = _index_semantic(args.semantic_root, args.baseline_name, args.baseline_mode)
    if set(raw) != expected_ids or set(semantic_dirs) != expected_ids:
        raise ReferenceEvaluationError("raw/semantic artifacts do not exactly match frozen dev manifest")
    adapter_spec = _json(args.adapter_spec)
    if adapter_spec.get("adapter_version") != "cardiac_adapter_v2" or sha256_file(args.adapter_spec) != FROZEN_ADAPTER_SPEC_SHA256:
        raise ReferenceEvaluationError("adapter spec does not match the frozen cardiac_adapter_v2 identity")
    for sample_id, artifact in raw.items():
        metadata = artifact.metadata
        if (
            metadata.get("shared_manifest_hash") != manifest.get("manifest_hash")
            or metadata.get("shared_grid_hash") != manifest.get("shared_grid_hash")
            or metadata.get("baseline_config_hash") != expected_baseline["baseline_config_sha256"]
            or metadata.get("split") != args.split
        ):
            raise ReferenceEvaluationError(f"raw artifact provenance mismatch for {sample_id}")
    maps: dict[str, np.ndarray] = {}
    for record in records:
        sample_id = str(record["sample_id"])
        semantic = verify_semantic_partition(
            semantic_dirs[sample_id], raw_artifact=raw[sample_id], record=record,
            central_image=_central_image(record, args.image_root), adapter_spec=adapter_spec,
        )
        maps[sample_id] = semantic.semantic_map

    by_volume: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_volume[str(record["volume_id"])].append(record)
    per_volume: list[dict[str, Any]] = []
    for volume_id, volume_records in sorted(by_volume.items()):
        ordered = sorted(volume_records, key=lambda record: int(record["slice_index"]))
        prediction, validity = _native_volume_from_slices(ordered, maps)
        gt_path = gt_path_for_record(ordered[0], args.gt_root)
        truth = _load_gt(gt_path, tuple(int(item) for item in ordered[0]["native_shape"]))
        pred_zhw = np.moveaxis(prediction, 2, 0)
        truth_zhw = np.moveaxis(truth, 2, 0)
        scores = dice_iou_volume(pred_zhw, truth_zhw, classes=FOREGROUND)
        per_volume.append({
            "patient_id": str(ordered[0]["patient_id"]), "volume_id": volume_id,
            "frame_index": int(ordered[0]["frame_index"]), "slice_count": len(ordered),
            "native_shape_hwz": list(prediction.shape), "coverage": float(validity.mean()),
            "void_voxels": int((~validity).sum()), "gt_locator": str(gt_path.relative_to(args.gt_root)),
            "gt_sha256": sha256_file(gt_path), "scores": {str(key): value for key, value in scores.items()},
        })
    patients = _patient_scores(per_volume)
    coverage = [float(item["coverage"]) for item in per_volume]
    raw_config_hashes = {str(item.metadata.get("baseline_config_hash")) for item in raw.values()}
    semantic_metadata = [_json(path / "metadata.json") for path in semantic_dirs.values()]
    adapter_spec_hashes = {str(item.get("adapter_spec_sha256")) for item in semantic_metadata}
    adapter_impl_hashes = {str(item.get("adapter_implementation_sha256")) for item in semantic_metadata}
    if len(raw_config_hashes) != 1 or len(adapter_spec_hashes) != 1 or len(adapter_impl_hashes) != 1:
        raise ReferenceEvaluationError("baseline or adapter identity differs within the evaluated cohort")
    if next(iter(raw_config_hashes)) != expected_baseline["baseline_config_sha256"]:
        raise ReferenceEvaluationError("evaluated baseline config identity differs from frozen evaluator contract")
    reference_catalog = [{"volume_id": item["volume_id"], "gt_locator": item["gt_locator"], "gt_sha256": item["gt_sha256"]} for item in per_volume]
    result = {
        "schema_version": EVALUATOR_SCHEMA,
        "evaluator_contract": {"path": str(args.contract), "sha256": sha256_file(args.contract), "payload_sha256": _sha256_json(contract)},
        "baseline": {"name": args.baseline_name, "mode": args.baseline_mode},
        "manifest": {"hash": manifest.get("manifest_hash"), "shared_grid_hash": manifest.get("shared_grid_hash"), "split": args.split, "sample_count": len(records), "volume_count": len(per_volume), "patient_count": len(patients)},
        "baseline_config_sha256": next(iter(raw_config_hashes)),
        "adapter": {"spec_sha256": next(iter(adapter_spec_hashes)), "implementation_sha256": next(iter(adapter_impl_hashes))},
        "reference_catalog": {"entries": reference_catalog, "sha256": _sha256_json({"entries": reference_catalog})},
        "semantic_verification": "full_recomputed_from_verified_raw_plus_image_only_central_plane",
        "void_policy": contract["void_policy"],
        "coverage": {"mean": float(np.mean(coverage)), "min": float(np.min(coverage)), "max": float(np.max(coverage))},
        "dice_patient_macro": patient_macro(patients, key="dice", classes=FOREGROUND),
        "iou_patient_macro": patient_macro(patients, key="iou", classes=FOREGROUND),
        "per_volume": per_volume,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--semantic-root", type=Path, required=True)
    parser.add_argument("--adapter-spec", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=ROOT / "configs" / "cardiac_reference_evaluator_v1.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-name", required=True)
    parser.add_argument("--baseline-mode", required=True)
    parser.add_argument("--split", choices=("dev",), default="dev")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate(args)
    print(json.dumps({"baseline": result["baseline"], "sample_count": result["manifest"]["sample_count"], "volume_count": result["manifest"]["volume_count"], "patient_count": result["manifest"]["patient_count"], "output": str(args.output_dir / "summary.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
