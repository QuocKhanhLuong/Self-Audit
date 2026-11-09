"""Student-only evaluation for candidate A.

The evaluator deliberately has a different contract from ``self_audit.evaluation``:
candidate A predicts with a serialized measurement code and never needs a teacher or
reference mask while generating predictions.  All reference masks are opened only
after the prediction files and their hashes have been frozen.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import ndimage

from ..data import CLASS_MAPPING, TrainingData, geometry_map, validate_geometry, write_json
from ..model import digest
from ..evaluation import class_metrics, confusion_matrix


CLASS_NAMES = ("BG", "RV", "MYO", "LV")
BOUNDARY_TOLERANCE_MM = 1.0


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _synchronise(device: str | torch.device) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _load_student(checkpoint: str | Path, checkpoint_sha: str, device: str):
    """Load the opt-in student-only checkpoint without importing a teacher."""

    from . import io as candidate_io
    model, bank, codec, metadata = candidate_io.load_student(checkpoint, checkpoint_sha, device=device)
    model = model.to(device).eval()
    return model, bank, codec, metadata


def _student_infer(model, bank, codec, x: torch.Tensor, spacing_yx, rollout_config, *, chunk_size=4096):
    """Run one image through the public candidate-A inference API.

    The result is normalised to ``(probabilities, code, trace)``.  Training and
    inference share the exact K1 rollout, so no separate greedy or unit-spacing
    fallback is allowed here.
    """
    from .k1 import student_rollout
    geometry = {"shape": tuple(x.shape[-2:]), "spacing_yx": tuple(spacing_yx),
                "spacing": tuple(spacing_yx)}
    result = student_rollout(model, x, bank, labels=None, codec=codec,
                             geometry=geometry, config=rollout_config,
                             chunk_size=chunk_size)
    output = result.final_logits
    code = {"ids": tuple(result.code_ids), "values": result.code_values}
    trace = {"query_count": int(result.query_count), "code_bits": int(result.code_bits)}
    if output is None:
        raise ValueError("student inference returned no decoded segmentation")
    if not torch.is_tensor(output):
        output = torch.as_tensor(output)
    if output.ndim == 3:
        output = output.unsqueeze(0)
    if output.ndim != 4 or output.shape[0] != 1 or output.shape[1] != 4:
        raise ValueError("candidate-A decoder must return Bx4xHxW logits/probabilities")
    # Keep logits for packet parity.  The evaluator applies softmax only after
    # the serialized/dequantized packet has reproduced these exact logits.
    output = output.detach()
    if not torch.isfinite(output).all():
        raise ValueError("student decoder returned non-finite output")
    if code is None:
        raise ValueError("student inference must return the serialized measurement code")
    return output, code, trace


def _serialize(codec, code, spacing_yx, shape, *, budget_bits=None):
    """Serialize exactly the code used by the decoder and return its bytes."""

    if isinstance(code, (bytes, bytearray, memoryview)):
        blob = bytes(code)
    else:
        if not isinstance(code, dict) or "ids" not in code or "values" not in code:
            raise ValueError("student rollout code must contain integer IDs and values")
        ids = code["ids"]
        values = torch.as_tensor(code["values"])
        if values.ndim > 1:
            values = values[0]
        quantized = codec.quantize_values(values, ids, ste=False)
        levels = quantized.levels
        blob = codec.serialize(ids, levels, shape, spacing_yx, budget_bits=budget_bits)
        if isinstance(blob, dict):
            blob = blob.get("bytes", blob.get("payload"))
        if not isinstance(blob, (bytes, bytearray, memoryview)):
            raise TypeError("codec serialization must return bytes")
        blob = bytes(blob)
    if not blob:
        raise ValueError("empty serialized code")
    return blob


def _spacing_for(record: dict[str, Any], manifest: dict[str, Any], z: int) -> tuple[float, float]:
    """Read declared in-plane geometry in millimetres.

    The original v2 manifests have no spacing and therefore cannot certify a
    physical boundary metric.  Candidate-A evaluation fails closed for those
    manifests instead of silently substituting isotropic voxels.
    """

    source = manifest["geometry"]["by_patient"][record["id"]]
    spacing = source["spacing_yx_mm"]
    if len(spacing) != 2 or not np.isfinite(spacing).all() or min(float(spacing[0]), float(spacing[1])) <= 0:
        raise ValueError("invalid in-plane spacing_yx")
    return float(spacing[0]), float(spacing[1])


def _surface_metrics(reference: np.ndarray, prediction: np.ndarray, spacing_yx, tolerance_mm=BOUNDARY_TOLERANCE_MM):
    """Compute finite 2-D physical boundary metrics with explicit empty cases."""

    ref = np.asarray(reference, dtype=bool)
    pred = np.asarray(prediction, dtype=bool)
    if not ref.any() and not pred.any():
        return {"status": "both_absent", "surface_dice": None, "assd_mm": None, "hd95_mm": None}
    if not ref.any() or not pred.any():
        return {"status": "one_absent", "surface_dice": 0.0, "assd_mm": None, "hd95_mm": None}
    structure = ndimage.generate_binary_structure(2, 1)
    rs = ref ^ ndimage.binary_erosion(ref, structure=structure, border_value=0)
    ps = pred ^ ndimage.binary_erosion(pred, structure=structure, border_value=0)
    if not rs.any() or not ps.any():
        return {"status": "no_surface", "surface_dice": None, "assd_mm": None, "hd95_mm": None}
    d_ref = ndimage.distance_transform_edt(~ref, sampling=spacing_yx)
    d_pred = ndimage.distance_transform_edt(~pred, sampling=spacing_yx)
    a = np.asarray(d_pred[rs], dtype=np.float64)
    b = np.asarray(d_ref[ps], dtype=np.float64)
    all_dist = np.concatenate([a, b])
    return {"status": "ok", "surface_dice": float(((a <= tolerance_mm).sum() + (b <= tolerance_mm).sum()) / len(all_dist)),
            "assd_mm": float(all_dist.mean()), "hd95_mm": float(np.percentile(all_dist, 95))}


def _aggregate_metric(values: list[dict[str, Any]]) -> dict[str, Any]:
    out = {"status_counts": {}}
    for row in values:
        out["status_counts"][row["status"]] = out["status_counts"].get(row["status"], 0) + 1
    for key in ("surface_dice", "assd_mm", "hd95_mm"):
        nums = [float(r[key]) for r in values if r.get(key) is not None and np.isfinite(r[key])]
        out[key] = float(np.mean(nums)) if nums else None
    return out


def evaluate(checkpoint, checkpoint_sha, data_manifest, data_sha, out, device="cuda", threshold=None):
    """Evaluate a student-only export on a disjoint development/external manifest."""

    manifest_path = Path(data_manifest).resolve()
    out = Path(out).resolve()
    if digest(manifest_path) != data_sha:
        raise ValueError("evaluation manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "self_audit.data.v2" or manifest.get("role") not in {"development", "external_frozen"}:
        raise ValueError("separate development/external_frozen manifest required")
    patients = manifest.get("patients", [])
    ids = [r.get("id") for r in patients]
    if manifest.get("class_mapping") != CLASS_MAPPING:
        raise ValueError("evaluation manifest class_mapping must be the explicit native ontology")
    if len(set(ids)) != len(ids):
        raise ValueError("evaluation patient IDs must be unique")
    # Validate the complete native geometry map, affine, class ontology, and
    # cached image shape before loading a checkpoint or opening any masks.
    try:
        geometries = geometry_map(manifest)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid candidate-A evaluation geometry map") from exc
    if set(geometries) != set(ids):
        raise ValueError("every evaluation patient requires one native geometry record")
    resolve = lambda value: _resolve(manifest_path.parent, value)
    for record in patients:
        image_path = resolve(record["image"])
        if digest(image_path) != record["image_sha256"]:
            raise ValueError("evaluation image hash mismatch")
        image = np.load(image_path, mmap_mode="r", allow_pickle=False)
        if image.ndim != 4 or not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("evaluation image must be finite normalized TZYX")
        validate_geometry(geometries[record["id"]], shape_zyx=image.shape[1:])
    model, bank, codec, meta = _load_student(checkpoint, checkpoint_sha, device)
    from .k1 import K1Config
    try:
        rollout_config = K1Config(**meta["config"]["rollout"])
    except (KeyError, TypeError) as exc:
        raise ValueError("student-only checkpoint is missing the exact K1 rollout config") from exc
    train_ids = set(meta.get("labeled_ids", [])) | set(meta.get("unlabeled_ids", []))
    if set(ids) & train_ids:
        raise ValueError("evaluation patient overlap with TRAIN")
    if bool(manifest.get("synthetic", False)) != bool(meta.get("synthetic", False)):
        raise ValueError("synthetic/patient identity mismatch")
    if threshold is not None and not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0,1]")
    if out.exists() or out.is_relative_to(manifest_path.parent) or manifest_path.parent.is_relative_to(out):
        raise ValueError("new output must be outside evaluation data directory")
    out.mkdir(parents=True)
    pred_dir, code_dir = out / "predictions", out / "codes"
    pred_dir.mkdir(); code_dir.mkdir()
    inventory, timings, frames = [], [], []
    recipe = {"checkpoint_sha256": checkpoint_sha, "data_sha256": data_sha,
              "scope": manifest.get("scope", manifest.get("role")), "prediction_model": "student_code_only",
              "synthetic": bool(meta.get("synthetic", False)), "threshold": threshold,
              "boundary_tolerance_mm": BOUNDARY_TOLERANCE_MM}
    write_json(out / "RECIPE.json", recipe)
    # This first pass never opens a mask.  It freezes all probabilities and packets.
    with torch.inference_mode():
        for record in patients:
            image_path = resolve(record["image"])
            if digest(image_path) != record["image_sha256"]:
                raise ValueError("evaluation image hash mismatch")
            image = np.load(image_path, mmap_mode="r", allow_pickle=False)
            if image.ndim != 4 or not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                raise ValueError("evaluation image must be finite normalized TZYX")
            t_count, z_count, height, width = image.shape
            probability = np.lib.format.open_memmap(pred_dir / f"{record['id']}.npy", mode="w+",
                                                    dtype="float32", shape=(t_count, z_count, 4, height, width))
            for t in range(t_count):
                for z in range(z_count):
                    x = TrainingData.context(image, t, z).to(device)
                    spacing = _spacing_for(record, manifest, z)
                    _synchronise(device); started = time.perf_counter()
                    output, code, trace = _student_infer(model, bank, codec, x, spacing,
                                                         rollout_config, chunk_size=4096)
                    budget_bits = rollout_config.bit_budget
                    blob = _serialize(codec, code, spacing, (height, width), budget_bits=budget_bits)
                    # Decode the packet that was actually written and compare it
                    # to the rollout output.  This catches accidental use of
                    # unquantized float values or a decoder bypass.
                    packet = codec.deserialize(blob, budget_bits=budget_bits)
                    dequantized = codec.dequantize(packet, device=output.device, dtype=output.dtype)
                    decoded = model.decode(packet.ids, dequantized.unsqueeze(0),
                                           {"shape": (height, width), "spacing_yx": spacing,
                                            "spacing": spacing}, chunk_size=4096)
                    if not torch.allclose(decoded, output, atol=1e-5, rtol=1e-5):
                        raise RuntimeError("serialized/dequantized code does not reproduce decoder output")
                    _synchronise(device); elapsed = (time.perf_counter() - started) * 1000
                    probability[t, z] = output[0].softmax(0).cpu().numpy()
                    packet_path = code_dir / f"{record['id']}_t{t:04d}_z{z:04d}.bin"
                    packet_path.write_bytes(blob)
                    packet_hash = digest(packet_path)
                    count = trace.get("query_count", trace.get("num_queries", None)) if isinstance(trace, dict) else None
                    inventory.append({"patient": record["id"], "frame": t, "slice": z,
                                      "file": packet_path.name, "sha256": packet_hash,
                                      "bytes": len(blob), "query_count": int(count) if count is not None else None})
                    timings.append(elapsed)
            probability.flush(); del probability
    prediction_files = [{"id": r["id"], "file": f"{r['id']}.npy", "sha256": digest(pred_dir / f"{r['id']}.npy")}
                        for r in patients]
    write_json(out / "PREDICTIONS_FROZEN.json", {"recipe_sha256": digest(out / "RECIPE.json"),
                                                  "files": prediction_files, "codes": inventory})

    # Only now may reference masks be opened.
    total = np.zeros((4, 5), dtype=np.int64)
    rows, boundary_rows = [], []
    for record, prediction_file in zip(patients, prediction_files):
        probability = np.load(pred_dir / prediction_file["file"], mmap_mode="r", allow_pickle=False)
        for frame, mask_record in record.get("masks", {}).items():
            mask_path = resolve(mask_record["path"])
            if digest(mask_path) != mask_record["sha256"]:
                raise ValueError("reference mask hash mismatch")
            reference = np.load(mask_path, allow_pickle=False)
            frame_index = int(frame)
            if reference.shape != probability.shape[1:2] + probability.shape[-2:]:
                raise ValueError("evaluation mask shape mismatch")
            predicted = probability[frame_index].argmax(1)
            cm = confusion_matrix(reference, predicted)
            total += cm
            rows.append({"patient": record["id"], "frame": frame_index,
                         "raw": class_metrics(cm), "raw_confusion": cm.tolist()})
            for slice_index in range(reference.shape[0]):
                spacing = _spacing_for(record, manifest, slice_index)
                for cls, name in enumerate(CLASS_NAMES[1:], 1):
                    metric = _surface_metrics(reference[slice_index] == cls,
                                              predicted[slice_index] == cls, spacing)
                    metric.update({"patient": record["id"], "frame": frame_index,
                                   "slice": slice_index, "class": name})
                    boundary_rows.append(metric)
    per_patient = {}
    for patient_id in ids:
        subset = [r for r in rows if r["patient"] == patient_id]
        per_patient[patient_id] = {}
        for index, name in enumerate(CLASS_NAMES):
            vals = [r["raw"][index]["dice"] for r in subset if r["raw"][index]["dice"] is not None]
            per_patient[patient_id][name] = float(np.mean(vals)) if vals else None
    macro = {}
    for index, name in enumerate(CLASS_NAMES):
        vals = [p[name] for p in per_patient.values() if p[name] is not None]
        macro[name] = {"dice": float(np.mean(vals)) if vals else None, "present_patients": len(vals)}
    warmup = timings[0] if timings else None
    steady = timings[1:] if len(timings) > 1 else timings
    result = {"scope": recipe["scope"], "synthetic": recipe["synthetic"], "macro_patient": macro,
              "patients": per_patient, "frames": rows, "pooled_confusion": total.tolist(),
              "boundary": {name: _aggregate_metric([r for r in boundary_rows if r["class"] == name])
                           for name in CLASS_NAMES[1:]},
              "code": {"packets": len(inventory), "total_bytes": int(sum(r["bytes"] for r in inventory)),
                       "mean_bytes": float(np.mean([r["bytes"] for r in inventory])) if inventory else None,
                       "query_counts": [r["query_count"] for r in inventory if r["query_count"] is not None]},
              "latency_ms": {"cold_first": warmup, "warm_mean": float(np.mean(steady)) if steady else None,
                             "total": float(sum(timings)), "per_slice": float(np.mean(timings)) if timings else None,
                             "n": len(timings)},
              "warning": "Synthetic outputs are smoke evidence only; no patient Dice, SOTA, or 96-Dice claim."}
    write_json(out / "METRICS.json", result)
    return result


__all__ = ["evaluate", "_surface_metrics"]
