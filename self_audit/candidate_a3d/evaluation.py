"""Leakage-safe full-volume A-3D evaluation.

The cache contract is native ``[T,D,H,W]``. A volume is passed to teacher or
student as ``[1,1,D,H,W]``; no slice context, crop, resize, or hidden resample is
introduced here. Prediction files and serialized codes are frozen and hashed
before any validation mask is opened.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import ndimage

from ..data import CLASS_MAPPING, geometry_map, validate_geometry, write_json
from ..metrics import class_metrics, confusion_matrix
from ..model import digest
from .profile import VolumeProfile, validate_device

CLASS_NAMES = ("BG", "RV", "MYO", "LV")
BOUNDARY_TOLERANCE_MM = 1.0


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _validated_manifest(data_manifest, data_sha, *, train_ids=(), synthetic=None):
    path = Path(data_manifest).resolve()
    if digest(path) != data_sha:
        raise ValueError("A-3D evaluation manifest hash mismatch")
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "self_audit.data.v2" or manifest.get("role") not in {"development", "external_frozen"}:
        raise ValueError("A-3D evaluation requires a development/external_frozen manifest")
    if manifest.get("class_mapping") != CLASS_MAPPING:
        raise ValueError("A-3D evaluation requires the explicit native class mapping")
    patients = manifest.get("patients", [])
    if not isinstance(patients, list) or not patients:
        raise ValueError("A-3D evaluation manifest must contain patient records")
    for record in patients:
        if (not isinstance(record, Mapping)
                or {"id", "image", "image_sha256"} - set(record)
                or not isinstance(record["id"], str) or not record["id"]):
            raise ValueError("invalid A-3D evaluation patient record")
    ids = [record["id"] for record in patients]
    if len(ids) != len(set(ids)):
        raise ValueError("A-3D evaluation patient IDs must be unique")
    if set(ids) & set(train_ids):
        raise ValueError("A-3D evaluation patients overlap with TRAIN")
    if synthetic is not None and bool(manifest.get("synthetic", False)) != bool(synthetic):
        raise ValueError("A-3D synthetic/patient identity mismatch")
    try:
        geometries = geometry_map(manifest)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid A-3D geometry map") from exc
    if set(geometries) != set(ids):
        raise ValueError("every A-3D evaluation patient requires geometry")
    resolve = lambda value: _resolve(path.parent, value)
    for record in patients:
        image_path = resolve(record["image"])
        if digest(image_path) != record["image_sha256"]:
            raise ValueError("A-3D evaluation image hash mismatch")
        image = np.load(image_path, mmap_mode="r", allow_pickle=False)
        if image.ndim != 4 or not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("A-3D cache image must be finite normalized [T,D,H,W]")
        validate_geometry(geometries[record["id"]], shape_zyx=image.shape[1:])
    return path, manifest, patients, ids, geometries, resolve


def _volume_input(image: np.ndarray, frame: int, device: str | torch.device) -> torch.Tensor:
    if image.ndim != 4 or not 0 <= frame < image.shape[0]:
        raise ValueError("volume cache must be [T,D,H,W]")
    return torch.from_numpy(np.array(image[frame], dtype=np.float32, copy=True))[None, None].to(device)


def _surface_metrics_3d(reference: np.ndarray, prediction: np.ndarray,
                        spacing_zyx: tuple[float, float, float], tolerance_mm=BOUNDARY_TOLERANCE_MM):
    ref, pred = np.asarray(reference, dtype=bool), np.asarray(prediction, dtype=bool)
    if not ref.any() and not pred.any():
        return {"status": "both_absent", "surface_dice": None, "assd_mm": None, "hd95_mm": None}
    if not ref.any() or not pred.any():
        return {"status": "one_absent", "surface_dice": 0.0, "assd_mm": None, "hd95_mm": None}
    structure = ndimage.generate_binary_structure(3, 1)
    ref_surface = ref ^ ndimage.binary_erosion(ref, structure=structure, border_value=0)
    pred_surface = pred ^ ndimage.binary_erosion(pred, structure=structure, border_value=0)
    if not ref_surface.any() or not pred_surface.any():
        return {"status": "no_surface", "surface_dice": None, "assd_mm": None, "hd95_mm": None}
    d_ref = ndimage.distance_transform_edt(~ref_surface, sampling=spacing_zyx)
    d_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing_zyx)
    ref_to_pred, pred_to_ref = d_pred[ref_surface], d_ref[pred_surface]
    distances = np.concatenate((ref_to_pred, pred_to_ref)).astype(np.float64)
    return {"status": "ok",
            "surface_dice": float((np.count_nonzero(ref_to_pred <= tolerance_mm)
                                    + np.count_nonzero(pred_to_ref <= tolerance_mm)) / len(distances)),
            "assd_mm": float(distances.mean()), "hd95_mm": float(np.percentile(distances, 95))}


def _surface_metrics_for_class(reference, prediction, class_id, spacing_zyx):
    reference = np.asarray(reference)
    if np.any(reference == 255):
        return {"status": "ignored_reference_voxels", "surface_dice": None,
                "assd_mm": None, "hd95_mm": None}
    return _surface_metrics_3d(reference == class_id, np.asarray(prediction) == class_id, spacing_zyx)


def _aggregate_metric(values: list[dict[str, Any]]) -> dict[str, Any]:
    result = {"status_counts": {}}
    for value in values:
        result["status_counts"][value["status"]] = result["status_counts"].get(value["status"], 0) + 1
    for key in ("surface_dice", "assd_mm", "hd95_mm"):
        numbers = [float(value[key]) for value in values
                   if value.get(key) is not None and np.isfinite(value[key])]
        result[key] = float(np.mean(numbers)) if numbers else None
    return result


def _aggregate_classes(rows: list[dict], class_index: int) -> dict[str, Any]:
    if not rows:
        return {"dice": None, "f1": None, "precision": None, "recall": None,
                "reference_present": False, "support": 0, "prediction_count": 0}
    values = [row["raw"][class_index] for row in rows]
    result = {}
    for key in ("dice", "f1", "precision", "recall"):
        numbers = [float(value[key]) for value in values
                   if value.get(key) is not None and np.isfinite(value[key])]
        result[key] = float(np.mean(numbers)) if numbers else None
    result["reference_present"] = any(bool(value.get("reference_present")) for value in values)
    result["support"] = int(sum(int(value.get("reference_count", 0)) for value in values))
    result["prediction_count"] = int(sum(int(value.get("prediction_count", 0)) for value in values))
    return result


def _patient_metrics(rows: list[dict], ids: list[str]) -> tuple[dict, dict]:
    patients = {
        patient_id: {name: _aggregate_classes([row for row in rows if row["patient"] == patient_id], index)
                     for index, name in enumerate(CLASS_NAMES)}
        for patient_id in ids
    }
    macro = {}
    for name in CLASS_NAMES:
        values = [patient[name] for patient in patients.values()]
        macro[name] = {
            key: (float(np.mean([float(value[key]) for value in values if value.get(key) is not None]))
                  if any(value.get(key) is not None for value in values) else None)
            for key in ("dice", "f1", "precision", "recall")
        }
        macro[name]["reference_present"] = any(bool(value.get("reference_present")) for value in values)
        macro[name]["present_patients"] = sum(bool(value.get("reference_present")) for value in values)
    return patients, macro


def _vendor_map(manifest: Mapping[str, Any], patients: list[dict]) -> dict[str, str]:
    """Return only explicitly declared vendor metadata.

    M&Ms vendor labels live in the manifest's patient metadata.  A missing
    declaration is kept as the explicit ``unknown`` group; filenames and
    paths are never inspected to infer a vendor.
    """

    metadata = manifest.get("patient_metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise TypeError("patient_metadata must be a mapping when present")
    vendors = {}
    for record in patients:
        patient_id = str(record["id"])
        entry = metadata.get(patient_id)
        vendor = None
        if entry is not None:
            if not isinstance(entry, Mapping):
                raise ValueError(f"patient_metadata[{patient_id!r}] must be a mapping")
            vendor = entry.get("vendor")
        # Keep compatibility with an already-materialized evaluation manifest,
        # but prefer the canonical top-level map whenever both are present.
        if vendor is None and "vendor" in record:
            vendor = record.get("vendor")
        if vendor is not None and not isinstance(vendor, str):
            raise ValueError(f"vendor for patient {patient_id!r} must be a string")
        vendor = vendor.strip() if vendor is not None else ""
        vendors[patient_id] = vendor or "unknown"
    return vendors


def _score_volume_predictions(prediction_dir: Path, patients: list[dict], ids: list[str],
                              geometries: Mapping[str, dict], resolve,
                              vendors: Mapping[str, str]) -> dict:
    rows, boundary_rows, total = [], [], np.zeros((4, 5), dtype=np.int64)
    vendors = dict(vendors)
    if set(vendors) != set(ids):
        raise ValueError("A-3D vendor metadata must cover exactly the evaluation patients")
    for record in patients:
        probabilities = np.load(prediction_dir / f"{record['id']}.npy", mmap_mode="r", allow_pickle=False)
        if probabilities.ndim != 5 or probabilities.shape[2] != len(CLASS_NAMES):
            raise ValueError("A-3D prediction volume must have shape [T,D,4,H,W]")
        spacing = tuple(float(x) for x in geometries[record["id"]]["spacing_zyx_mm"])
        for frame, mask_record in record.get("masks", {}).items():
            mask_path = resolve(mask_record["path"])
            if digest(mask_path) != mask_record["sha256"]:
                raise ValueError("A-3D reference mask hash mismatch")
            reference = np.load(mask_path, allow_pickle=False)
            frame_index = int(frame)
            expected_shape = (probabilities.shape[1], probabilities.shape[-2], probabilities.shape[-1])
            if reference.shape != expected_shape:
                raise ValueError("A-3D reference mask shape mismatch")
            predicted = probabilities[frame_index].argmax(1)
            matrix = confusion_matrix(reference, predicted)
            total += matrix
            rows.append({"patient": record["id"], "frame": frame_index,
                         "vendor": vendors.get(str(record["id"]), "unknown"), "raw": class_metrics(matrix),
                         "raw_confusion": matrix.tolist()})
            for class_id, class_name in enumerate(CLASS_NAMES[1:], 1):
                metric = _surface_metrics_for_class(reference, predicted, class_id, spacing)
                metric.update({"patient": record["id"], "frame": frame_index,
                               "vendor": vendors.get(str(record["id"]), "unknown"), "class": class_name})
                boundary_rows.append(metric)
    patients_out, macro = _patient_metrics(rows, ids)
    vendor_groups = {}
    for vendor in sorted(set(vendors.values())):
        vendor_rows = [row for row in rows if row.get("vendor") == vendor]
        vendor_groups[vendor] = {name: _aggregate_classes(vendor_rows, index)
                                 for index, name in enumerate(CLASS_NAMES)}
    boundary_groups = {
        vendor: {
            name: _aggregate_metric([
                row for row in boundary_rows
                if row.get("vendor") == vendor and row["class"] == name
            ])
            for name in CLASS_NAMES[1:]
        }
        for vendor in sorted(set(vendors.values()))
    }
    return {"per_class": {name: _aggregate_classes(rows, index) for index, name in enumerate(CLASS_NAMES)},
            "macro_patient": macro, "per_patient": patients_out, "per_vendor": vendor_groups,
            "frames": rows, "pooled_confusion": total.tolist(),
            "boundary": {name: _aggregate_metric([row for row in boundary_rows if row["class"] == name])
                         for name in CLASS_NAMES[1:]},
            "boundary_per_vendor": boundary_groups}


def _teacher_forward(teacher, x: torch.Tensor, geometry: Mapping[str, Any],
                     height: int, width: int, depth: int):
    output = teacher(x, geometry=geometry)
    if not isinstance(output, Mapping) or set(output) != {"region_logits", "boundary_logits", "sdf"}:
        raise ValueError("A-3D teacher must return exactly region_logits, boundary_logits and sdf")
    logits = {}
    for name in ("region", "boundary"):
        value = output[f"{name}_logits"]
        if value.shape != (1, 4, depth, height, width) or not torch.isfinite(value).all():
            raise ValueError(f"A-3D teacher {name} logits shape/value mismatch")
        logits[name] = value
    sdf = output["sdf"]
    if sdf.shape != (1, 4, depth, height, width) or not torch.isfinite(sdf).all():
        raise ValueError("A-3D teacher sdf shape/value mismatch")
    return logits


def evaluate_teacher_epoch(teacher, data_manifest, data_sha, out, *, epoch: int,
                           device="cuda", train_ids=(), synthetic=None):
    """Freeze and score region/boundary teacher predictions for one epoch."""

    if type(epoch) is not int or epoch < 0:
        raise ValueError("epoch must be a nonnegative integer")
    device = validate_device(device)
    path, manifest, patients, ids, geometries, resolve = _validated_manifest(
        data_manifest, data_sha, train_ids=train_ids, synthetic=synthetic)
    vendors = _vendor_map(manifest, patients)
    teacher = teacher.to(device).eval()
    out = Path(out).resolve()
    if out.exists() or out.is_relative_to(path.parent) or path.parent.is_relative_to(out):
        raise ValueError("new A-3D teacher output must be outside evaluation data")
    out.mkdir(parents=True)
    directories = {name: out / f"predictions_{name}" for name in ("region", "boundary")}
    for directory in directories.values():
        directory.mkdir()
    profile_rows = []
    with torch.inference_mode():
        for record in patients:
            image = np.load(resolve(record["image"]), mmap_mode="r", allow_pickle=False)
            t_count, depth, height, width = image.shape
            geometry_record = geometries[record["id"]]
            geometry = {
                "shape": (depth, height, width),
                "spacing_zyx": tuple(geometry_record["spacing_zyx_mm"]),
                "affine_zyx": geometry_record["affine_zyx"],
                "context": "full_volume",
            }
            volumes = {name: np.lib.format.open_memmap(directory / f"{record['id']}.npy", mode="w+",
                         dtype="float32", shape=(t_count, depth, 4, height, width))
                       for name, directory in directories.items()}
            try:
                for frame in range(t_count):
                    timer = VolumeProfile()
                    with timer.measure("teacher_forward_ms", device):
                            logits = _teacher_forward(
                                teacher, _volume_input(image, frame, device), geometry,
                                height, width, depth,
                            )
                    for name, value in logits.items():
                        volumes[name][frame] = value[0].softmax(0).cpu().numpy().transpose(1, 0, 2, 3)
                    profile_rows.append({"patient": record["id"], "frame": frame, **timer.fields})
            finally:
                for volume in volumes.values():
                    volume.flush(); del volume
    frozen = {
        name: [{"id": record["id"], "file": f"{record['id']}.npy",
                "sha256": digest(directories[name] / f"{record['id']}.npy")}
               for record in patients] for name in directories
    }
    write_json(out / "PREDICTIONS_FROZEN.json", {"epoch": epoch, "data_sha256": data_sha, "branches": frozen})
    result = {"epoch": epoch, "scope": manifest.get("scope", manifest.get("role")),
              "synthetic": bool(manifest.get("synthetic", False)), "branches": {}}
    for name, directory in directories.items():
        scored = _score_volume_predictions(directory, patients, ids, geometries, resolve, vendors)
        result["branches"][name] = scored
    result["profile"] = {"frames": profile_rows,
                          "total_ms": float(sum(row.get("teacher_forward_ms", 0.0) for row in profile_rows))}
    result["labels_used_for"] = "development scoring only; no target, gate, checkpoint or early-stop decision"
    if result["synthetic"]:
        result["warning"] = "Synthetic outputs are smoke evidence only; no patient Dice, SOTA, or 96-Dice claim."
    write_json(out / "TEACHER_EPOCH_METRICS.json", result)
    return result


def _student_rollout(model, x, bank, codec, geometry, rollout_config):
    """Call the explicit A-3D K1 rollout; missing implementation fails closed."""

    try:
        from .k1 import student_rollout
    except ImportError as exc:
        raise RuntimeError("A-3D student rollout module is not installed") from exc
    return student_rollout(model, x, bank, codec=codec, geometry=geometry,
                           config=rollout_config, labels=None)


def _serialize_student(codec, code, geometry, budget_bits):
    if not isinstance(code, Mapping) or {"ids", "values"} - set(code):
        raise ValueError("A-3D rollout code must contain ids and student values")
    values = torch.as_tensor(code["values"])
    if values.ndim > 1:
        values = values[0]
    quantized = codec.quantize_values(values, code["ids"], ste=False)
    blob = codec.serialize(code["ids"], quantized.levels, geometry["shape"], geometry["spacing_zyx"],
                           affine_zyx=geometry["affine_zyx"], budget_bits=budget_bits)
    if not isinstance(blob, (bytes, bytearray, memoryview)) or not blob:
        raise ValueError("A-3D codec must return a nonempty serialized packet")
    return bytes(blob)


def evaluate_epoch(model, bank, codec, rollout_config, data_manifest, data_sha, out, *,
                   epoch: int, device="cuda", train_ids=(), synthetic=None):
    """Freeze, serialize and score student full-volume predictions for one epoch."""

    if type(epoch) is not int or epoch < 0:
        raise ValueError("epoch must be a nonnegative integer")
    device = validate_device(device)
    path, manifest, patients, ids, geometries, resolve = _validated_manifest(
        data_manifest, data_sha, train_ids=train_ids, synthetic=synthetic)
    vendors = _vendor_map(manifest, patients)
    model = model.to(device).eval()
    out = Path(out).resolve()
    if out.exists() or out.is_relative_to(path.parent) or path.parent.is_relative_to(out):
        raise ValueError("new A-3D student output must be outside evaluation data")
    out.mkdir(parents=True); prediction_dir, code_dir = out / "predictions", out / "codes"
    prediction_dir.mkdir(); code_dir.mkdir()
    profiles, inventory = [], []
    with torch.inference_mode():
        for record in patients:
            image = np.load(resolve(record["image"]), mmap_mode="r", allow_pickle=False)
            t_count, depth, height, width = image.shape
            probability = np.lib.format.open_memmap(prediction_dir / f"{record['id']}.npy", mode="w+",
                                                    dtype="float32", shape=(t_count, depth, 4, height, width))
            geometry_record = geometries[record["id"]]
            geometry = {"shape": (depth, height, width), "spacing_zyx": tuple(geometry_record["spacing_zyx_mm"]),
                        "affine_zyx": geometry_record["affine_zyx"], "context": "full_volume"}
            try:
                for frame in range(t_count):
                    x = _volume_input(image, frame, device)
                    profile = VolumeProfile()
                    with profile.measure("rollout_ms", device):
                        result = _student_rollout(model, x, bank, codec, geometry, rollout_config)
                    output = result.final_logits
                    if output.shape != (1, 4, depth, height, width) or not torch.isfinite(output).all():
                        raise ValueError("A-3D student decoder output shape/value mismatch")
                    code = {"ids": tuple(result.code_ids), "values": result.code_values}
                    with profile.measure("packet_serialize_ms", device):
                        blob = _serialize_student(codec, code, geometry, rollout_config.bit_budget)
                    packet_path = code_dir / f"{record['id']}_t{frame:04d}.bin"
                    packet_path.write_bytes(blob)
                    with profile.measure("packet_roundtrip_ms", device):
                        packet = codec.deserialize(blob, budget_bits=rollout_config.bit_budget)
                        values = codec.dequantize(packet, device=output.device, dtype=output.dtype)
                    with profile.measure("decoder_roundtrip_ms", device):
                        decoded = model.decode(packet.ids, values.unsqueeze(0),
                                               {"shape": packet.shape, "spacing_zyx": packet.spacing_zyx,
                                                "affine_zyx": packet.affine_zyx, "context": "full_volume"})
                    if not torch.allclose(decoded, output, atol=1e-5, rtol=1e-5):
                        raise RuntimeError("A-3D serialized/dequantized code changed decoder output")
                    probability[frame] = output[0].softmax(0).cpu().numpy().transpose(1, 0, 2, 3)
                    profile.fields["total_ms"] = sum(profile.fields.values())
                    profile_rows = {"patient": record["id"], "frame": frame, **profile.fields}
                    profiles.append(profile_rows)
                    inventory.append({"patient": record["id"], "frame": frame,
                                      "bytes": len(blob), "query_count": int(result.query_count),
                                      "sha256": digest(packet_path), "file": packet_path.name})
            finally:
                probability.flush(); del probability
    frozen = [{"id": record["id"], "file": f"{record['id']}.npy",
               "sha256": digest(prediction_dir / f"{record['id']}.npy")} for record in patients]
    write_json(out / "PREDICTIONS_FROZEN.json", {"epoch": epoch, "data_sha256": data_sha,
                                                  "files": frozen, "codes": inventory})
    result = _score_volume_predictions(prediction_dir, patients, ids, geometries, resolve, vendors)
    result.update({"epoch": epoch, "scope": manifest.get("scope", manifest.get("role")),
                   "synthetic": bool(manifest.get("synthetic", False)), "profile": {
                       "timing_scope": "rollout_ms=encoder+query_value+policy+full_volume_decoder; packet roundtrip and decoder roundtrip are separate",
                       "frames": profiles, "total_ms": float(sum(row["total_ms"] for row in profiles)),
                       "mean_ms": float(np.mean([row["total_ms"] for row in profiles])) if profiles else None,
                   }, "code": {"packets": len(inventory),
                               "total_bytes": int(sum(row["bytes"] for row in inventory)),
                               "mean_bytes": float(np.mean([row["bytes"] for row in inventory])) if inventory else None,
                               "query_counts": [row["query_count"] for row in inventory]},
                   "labels_used_for": "development scoring only; no target, gate, checkpoint or early-stop decision"})
    if result["synthetic"]:
        result["warning"] = "Synthetic outputs are smoke evidence only; no patient Dice, SOTA, or 96-Dice claim."
    write_json(out / "EPOCH_METRICS.json", result)
    return result


def evaluate(checkpoint, checkpoint_sha, data_manifest, data_sha, out,
             device="cuda", threshold=None):
    """Evaluate a self-contained A-3D student checkpoint.

    This is the checkpoint-facing adapter used by the repository-level
    ``evaluate.py`` dispatcher.  A-3D uses argmax over four logits, so a 2-D
    foreground threshold has no defined meaning and is rejected explicitly.
    The model is loaded only after the read-only manifest/hash/geometry checks.
    """

    if threshold is not None:
        raise ValueError("A-3D evaluation does not support a foreground threshold")
    validate_device(device)
    path, manifest, _patients, _ids, _geometries, _resolve = _validated_manifest(data_manifest, data_sha)
    from . import io as candidate_io
    model, bank, codec, metadata = candidate_io.load_student(checkpoint, checkpoint_sha, device=device)
    train_ids = set(metadata.get("labeled_ids", [])) | set(metadata.get("unlabeled_ids", []))
    if set(_ids) & train_ids:
        raise ValueError("A-3D evaluation patients overlap with TRAIN")
    if bool(manifest.get("synthetic", False)) != bool(metadata.get("synthetic", False)):
        raise ValueError("synthetic/patient identity mismatch")
    try:
        from .k1 import K1Config
        rollout_config = K1Config(**metadata["config"]["rollout"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("A-3D student checkpoint is missing the exact K1 rollout config") from exc
    return evaluate_epoch(model, bank, codec, rollout_config, path, data_sha, out,
                          epoch=int(metadata.get("epoch", 0)), device=device,
                          train_ids=train_ids, synthetic=bool(metadata.get("synthetic", False)))


__all__ = [
    "_surface_metrics_3d",
    "_surface_metrics_for_class",
    "evaluate",
    "evaluate_epoch",
    "evaluate_teacher_epoch",
]
