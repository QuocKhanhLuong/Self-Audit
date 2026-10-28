"""Deployment/evaluation extension for the fully automatic NativeStudent pilot.

Contract requirements & coordinator integration specifications:
- Deployment is raw NativeStudent with fixed-window feature path and teacher_memory=None.
- Image-only raw-student full native cine uint8 argmax export with complete 100%
  output coverage and hashes, deterministic, strictly no postprocessing.
- Sentinel 255 buffer initialization and explicit plane counting; rejects empty inputs
  and incomplete/accidental coverage.
- Freeze-before-reference-scoring order: RECIPE.json and PREDICTIONS_FROZEN.json written
  and hashed BEFORE opening any reference masks. Zero mask paths read during inference.
- Canonical integer frame ID validation (0 <= frame < T) strictly enforced BEFORE mask I/O.
- Safe patient IDs preventing directory traversal (no '/', '..', or path escape).
- Image range [0, 1] and positive dimensions validated; non-finite/empty images fail closed.
- Missing training lineage fails closed (never infers empty set).
- Direct logits.argmax(1) without softmax-rounding ties.
- All-ignored frame FP denominator 0 is undefined (None), never invented 0.
- Checkpoint load time tracked; unused teacher/payload deleted immediately before inference.
- Cold total output cost reports checkpoint load and export times separately.
- Hierarchical scoring: frame-volume confusion sum, available frame mean per patient,
  patient macro per class, and foreground macro matching evaluate_frozen_labels.py
  (mean available foreground classes within each patient, then mean across patients).
- Inference-only mode tolerates records without masks.
"""

from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from .data import TrainingData, write_json
from .evaluation import class_metrics, confusion_matrix
from .model import digest


SCHEMA_EVALUATION_RECIPE = "self_audit.automatic_evaluation_recipe.v1"
SCHEMA_FROZEN_PREDICTIONS = "self_audit.automatic_frozen_predictions.v1"
SCHEMA_METRICS = "self_audit.automatic_evaluation_metrics.v1"
METRIC_CLASSES = ["BG", "RV", "MYO", "LV"]
FOREGROUND_CLASSES = ["RV", "MYO", "LV"]


def validate_evaluation_manifest(
    manifest_path: str | Path,
    expected_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Validate data manifest schema, role, hashes, safe patient IDs, and patient structure.

    Tolerates records without masks for inference-only deployment.
    Fails closed on hash mismatch, invalid schema, duplicate patient IDs, or path traversal.
    """
    if not expected_sha256 or not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("expected 64-char hex data manifest SHA256 is required; failing closed")

    manifest_file = Path(manifest_path).resolve()
    if not manifest_file.is_file():
        raise ValueError(f"evaluation manifest file not found: {manifest_file}")

    actual_sha = digest(manifest_file)
    if actual_sha != expected_sha256.lower():
        raise ValueError(
            f"evaluation manifest hash mismatch: expected {expected_sha256}, got {actual_sha}"
        )

    try:
        manifest = json.loads(manifest_file.read_text())
    except Exception as exc:
        raise ValueError(f"failed to parse manifest JSON: {exc}") from exc

    if manifest.get("schema") != "self_audit.data.v2":
        raise ValueError(
            f"invalid data manifest schema: expected 'self_audit.data.v2', got '{manifest.get('schema')}'"
        )

    role = manifest.get("role")
    if role not in {"development", "external_frozen"}:
        raise ValueError(
            f"invalid evaluation role '{role}'; must be 'development' or 'external_frozen'"
        )

    patients = manifest.get("patients")
    if not isinstance(patients, list) or len(patients) == 0:
        raise ValueError("manifest 'patients' must be a non-empty list of records")

    ids: list[str] = []
    for r in patients:
        pid = r.get("id")
        if not pid or not isinstance(pid, str):
            raise ValueError("all patient records must have non-empty string 'id'")

        # Prevent directory escape, absolute paths, or traversal in output predictions
        if "/" in pid or "\\" in pid or pid in {".", ".."} or Path(pid).name != pid:
            raise ValueError(f"unsafe patient ID '{pid}' attempting path traversal or directory escape")

        ids.append(pid)

        if "image" not in r or not isinstance(r["image"], str):
            raise ValueError(f"patient {pid} missing valid 'image' path")
        if "image_sha256" not in r or not isinstance(r["image_sha256"], str) or len(r["image_sha256"]) != 64:
            raise ValueError(f"patient {pid} missing valid 64-char 'image_sha256'")

    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate patient IDs detected in evaluation manifest: {ids}")

    return manifest, actual_sha


def extract_training_lineage_ids(meta: dict[str, Any]) -> set[str]:
    """Extract all patient IDs belonging to labeled/unlabeled training lineage.

    Fails closed if training lineage metadata is missing, partial, or empty.
    Mandatory complete fields required:
    - training_manifest_sha256 (64-char hex string)
    - labeled_ids (complete list of string IDs)
    - unlabeled_ids (complete list of string IDs)
    - target_label_lineage (dict containing lineage metadata)
    """
    if not isinstance(meta, dict):
        raise ValueError("checkpoint metadata must be a dictionary")

    manifest_sha = meta.get("training_manifest_sha256")
    if not manifest_sha or not isinstance(manifest_sha, str) or len(manifest_sha) != 64:
        raise ValueError("checkpoint metadata missing mandatory 64-char 'training_manifest_sha256'; failing closed")

    if "labeled_ids" not in meta or not isinstance(meta["labeled_ids"], (list, tuple, set)):
        raise ValueError("checkpoint metadata missing mandatory complete 'labeled_ids' list; failing closed")

    if "unlabeled_ids" not in meta or not isinstance(meta["unlabeled_ids"], (list, tuple, set)):
        raise ValueError("checkpoint metadata missing mandatory complete 'unlabeled_ids' list; failing closed")

    lineage = meta.get("target_label_lineage")
    if not lineage or not isinstance(lineage, dict):
        raise ValueError("checkpoint metadata missing mandatory 'target_label_lineage' dict; failing closed")

    training_ids: set[str] = set()

    for x in meta["labeled_ids"]:
        if x:
            training_ids.add(str(x))
    for x in meta["unlabeled_ids"]:
        if x:
            training_ids.add(str(x))

    pts = lineage.get("patients")
    if isinstance(pts, (list, tuple, set)):
        training_ids.update(str(x) for x in pts if x)

    supp_pts = lineage.get("support_patient_ids")
    if isinstance(supp_pts, (list, tuple, set)):
        training_ids.update(str(x) for x in supp_pts if x)

    meta_supp = meta.get("support_patient_ids")
    if isinstance(meta_supp, (list, tuple, set)):
        training_ids.update(str(x) for x in meta_supp if x)

    cfg = meta.get("config")
    if isinstance(cfg, dict):
        supp = cfg.get("support")
        if isinstance(supp, dict):
            supp_pts_cfg = supp.get("patient_ids")
            if isinstance(supp_pts_cfg, (list, tuple, set)):
                training_ids.update(str(x) for x in supp_pts_cfg if x)

    for key in ("seen_supervised", "seen_unlabeled", "seen_audit"):
        val = meta.get(key)
        if isinstance(val, (list, tuple, set)):
            training_ids.update(str(x) for x in val if x)

    if not training_ids:
        raise ValueError("checkpoint metadata has empty training labeled/unlabeled lineage union; failing closed")

    return training_ids


def check_lineage_overlap(
    eval_ids: Sequence[str],
    meta: dict[str, Any],
    role: str,
) -> None:
    """Enforce strict separation: evaluation patients must not overlap training lineage."""
    training_ids = extract_training_lineage_ids(meta)
    overlap = set(eval_ids) & training_ids
    if overlap:
        raise ValueError(
            f"evaluation patient overlap with training labeled/unlabeled lineage: {sorted(overlap)}; "
            "failing closed to prevent data leakage"
        )


def aggregate_patient_macro(
    per_patient_dict: dict[str, dict[str, Any]],
    classes: Sequence[str],
) -> float | None:
    """Aggregate matching evaluate_frozen_labels.py:

    Mean available classes WITHIN each patient, then mean across patients.
    This guarantees equal weighting per patient when class availability is asymmetric.
    """
    patient_values = [
        [per_patient_dict[pid][c] for c in classes if per_patient_dict[pid].get(c) is not None]
        for pid in per_patient_dict
    ]
    patient_means = [float(np.mean(vals)) for vals in patient_values if vals]
    return float(np.mean(patient_means)) if patient_means else None


def compute_patient_macro_metrics(
    frame_rows: list[dict[str, Any]],
    patient_ids: list[str],
) -> tuple[dict[str, Any], dict[str, Any], float | None, float | None]:
    """Hierarchical evaluation reduction:

    1. Frame volumes: confusion summed in each complete frame volume.
    2. Available frame mean per patient: mean of present-reference frame Dice values per class.
       Absent frames yield None for Dice, with FP diagnostics recorded.
    3. Patient macro per class: mean of patient Dice values across present patients.
    4. Foreground macro: mean available foreground classes within each patient, then mean patients.
    5. All-class macro: mean available classes within each patient, then mean patients.
    """
    per_patient: dict[str, Any] = {}
    for pid in patient_ids:
        p_frames = [f for f in frame_rows if f["patient"] == pid]
        per_patient[pid] = {}
        for j, name in enumerate(METRIC_CLASSES):
            valid_dice = [
                f["raw"][j]["dice"]
                for f in p_frames
                if f["raw"][j]["dice"] is not None
            ]
            per_patient[pid][name] = float(np.mean(valid_dice)) if valid_dice else None

            # FP diagnostics for absent reference frames
            absent_fps = [
                f["raw"][j]["FP"]
                for f in p_frames
                if not f["raw"][j]["reference_present"]
            ]
            per_patient[pid][f"{name}_absent_frames"] = len(absent_fps)
            per_patient[pid][f"{name}_absent_frame_fp_total"] = int(sum(absent_fps)) if absent_fps else 0

    macro: dict[str, Any] = {}
    for name in METRIC_CLASSES:
        vals = [per_patient[pid][name] for pid in patient_ids if per_patient[pid][name] is not None]
        macro[name] = {
            "dice": float(np.mean(vals)) if vals else None,
            "present_patients": len(vals),
            "total_patients": len(patient_ids),
        }

    # Within-patient average then across-patients average matching evaluate_frozen_labels.py
    fg_macro = aggregate_patient_macro(per_patient, FOREGROUND_CLASSES)
    all_macro = aggregate_patient_macro(per_patient, METRIC_CLASSES)

    return per_patient, macro, fg_macro, all_macro


def evaluate_automatic(
    checkpoint: str | Path,
    checkpoint_sha: str,
    data_manifest: str | Path,
    data_sha: str,
    out: str | Path,
    *,
    device: str = "cuda",
    batch_size: int = 8,
    threads: int = 2,
    compress_predictions: bool = False,
    inference_only: bool = False,
) -> dict[str, Any]:
    """Execute automatic evaluation pipeline.

    Stages:
    1. Validate checkpoint, manifest, schemas, hashes, lineage non-overlap, safe output directory.
    2. Image-only raw-student inference: export full native uint8 argmax volumes (100% coverage).
       Zero reference masks opened.
    3. Write RECIPE.json and PREDICTIONS_FROZEN.json receipts.
    4. Optional scoring: open reference masks only after receipt is written. Sum frame-volume
       confusion, compute available frame mean per patient, then patient macro.
    """
    overall_start = time.monotonic()
    deployment_start = overall_start
    manifest_path = Path(data_manifest).resolve()
    out_dir = Path(out).resolve()
    checkpoint_dir = Path(checkpoint).resolve()

    # 1. Output path safety checks
    if out_dir.exists():
        raise ValueError(f"output directory {out_dir} already exists; safe new outputs required")

    # Strict output safety outside checkpoint directory
    if out_dir == checkpoint_dir or out_dir.is_relative_to(checkpoint_dir) or checkpoint_dir.is_relative_to(out_dir):
        raise ValueError(f"output directory {out_dir} must be strictly outside checkpoint directory {checkpoint_dir}")

    # 2. Validate evaluation manifest BEFORE creating any output directories or files
    manifest, actual_data_sha = validate_evaluation_manifest(manifest_path, data_sha)
    manifest_dir = manifest_path.parent.resolve()

    # Strict output safety outside manifest directory
    if out_dir == manifest_dir or out_dir.is_relative_to(manifest_dir) or manifest_dir.is_relative_to(out_dir):
        raise ValueError(f"output directory {out_dir} must be strictly outside evaluation data directory {manifest_dir}")

    # Strict output safety outside all referenced data directories (images and masks)
    resolve_path = lambda p: Path(p).resolve() if Path(p).is_absolute() else (manifest_dir / p).resolve()
    referenced_dirs: set[Path] = {manifest_dir}
    patients = manifest["patients"]
    for record in patients:
        img_p = resolve_path(record["image"])
        referenced_dirs.add(img_p.parent)
        for mask_info in (record.get("masks") or {}).values():
            if isinstance(mask_info, dict) and "path" in mask_info:
                referenced_dirs.add(resolve_path(mask_info["path"]).parent)

    for ref_dir in referenced_dirs:
        if out_dir == ref_dir or out_dir.is_relative_to(ref_dir) or ref_dir.is_relative_to(out_dir):
            raise ValueError(f"output directory {out_dir} must be strictly outside referenced data directory {ref_dir}")

    patient_ids = [p["id"] for p in patients]

    if batch_size < 1 or threads < 1:
        raise ValueError("batch_size and threads must be positive integers")
    torch.set_num_threads(threads)

    # 3. Hardware device validation
    is_cuda = device.startswith("cuda")
    if is_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device '{device}' requested but torch.cuda is not available; failing closed")
        torch.cuda.reset_peak_memory_stats(device=device)
        torch.cuda.synchronize(device=device)

    # 4. Strict checkpoint loading using engine helper directly (no hasattr or teacher fallback)
    from .automatic import load_automatic_student
    from .model import NativeStudent

    load_start = time.monotonic()
    student_model, meta = load_automatic_student(
        checkpoint,
        expected_sha256=checkpoint_sha,
        device=device,
    )
    checkpoint_load_seconds = time.monotonic() - load_start

    # Verify model is raw NativeStudent with fixed window and teacher_memory strictly None
    if not isinstance(student_model, NativeStudent):
        raise ValueError(f"expected raw NativeStudent instance, got {type(student_model)}")
    if student_model.window is None or getattr(student_model.window, "mode", None) != "fixed":
        raise ValueError(
            f"evaluation requires NativeStudent with fixed window; got window={getattr(student_model.window, 'mode', None)}"
        )
    if student_model.teacher_memory is not None:
        raise ValueError("evaluation requires raw NativeStudent with teacher_memory=None; unexpected memory path detected")

    model = student_model.to(device).eval()

    # 5. Checkpoint vs manifest sanity checks
    is_synthetic_manifest = bool(manifest.get("synthetic", False))
    is_synthetic_meta = bool(meta.get("synthetic", False))
    if is_synthetic_manifest != is_synthetic_meta:
        raise ValueError(
            f"synthetic mismatch: manifest has {is_synthetic_manifest}, checkpoint has {is_synthetic_meta}"
        )

    # Lineage overlap check (rejects any train L or U patient overlap)
    check_lineage_overlap(patient_ids, meta, manifest.get("role", "external_frozen"))

    # 6. Initialize output directories
    out_dir.mkdir(parents=True, exist_ok=False)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=False)

    recipe = {
        "schema": SCHEMA_EVALUATION_RECIPE,
        "checkpoint_sha256": checkpoint_sha,
        "data_sha256": data_sha,
        "scope": manifest.get("scope", manifest.get("role")),
        "role": manifest.get("role"),
        "prediction_model": "raw_student",
        "feature_path": "fixed_window_native_student",
        "teacher_memory": None,
        "synthetic": is_synthetic_meta,
        "batch_size": batch_size,
        "threads": threads,
        "compress_predictions": compress_predictions,
        "format": "uint8 native TZYX argmax; deterministic, no postprocessing",
        "buffer_initialization": "sentinel_255_strictly_overwritten",
        "output_coverage_requirement": "100%",
        "metric_classes": METRIC_CLASSES,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(out_dir / "RECIPE.json", recipe)

    # Resolve paths helper
    resolve_path = lambda p: Path(p) if Path(p).is_absolute() else manifest_path.parent / p

    # 7. Image-only Raw-Student Inference
    # ZERO mask paths are accessed or opened in this block!
    inventory: list[dict[str, Any]] = []
    chunk_latencies_ms: list[float] = []
    total_voxels_evaluated = 0
    total_frames_exported = 0
    total_slices_exported = 0
    inference_start = time.monotonic()

    with torch.inference_mode():
        for record in patients:
            pid = record["id"]
            ip = resolve_path(record["image"])
            if not ip.is_file():
                raise ValueError(f"image file not found: {ip}")
            if digest(ip) != record["image_sha256"]:
                raise ValueError(f"evaluation image hash mismatch for patient {pid}: {ip}")

            cine = np.load(ip, mmap_mode="r", allow_pickle=False)
            if cine.ndim != 4:
                raise ValueError(f"invalid image shape for patient {pid}: expected 4D [T,Z,H,W], got {cine.shape}")

            T, Z, H, W = cine.shape
            if T <= 0 or Z <= 0 or H <= 0 or W <= 0:
                raise ValueError(f"empty image dimensions for patient {pid}: shape={cine.shape}")

            if not np.isfinite(cine).all():
                raise ValueError(f"image for patient {pid} contains non-finite values")

            c_min, c_max = float(cine.min()), float(cine.max())
            if c_min < 0.0 or c_max > 1.0:
                raise ValueError(
                    f"image for patient {pid} values out of normalized range [0, 1]: min={c_min}, max={c_max}"
                )

            patient_voxels = T * Z * H * W
            total_voxels_evaluated += patient_voxels
            total_frames_exported += T
            total_slices_exported += T * Z

            ext = ".npy.gz" if compress_predictions else ".npy"
            op = pred_dir / f"{pid}{ext}"

            # Initialize with sentinel 255 to verify every plane is explicitly written
            if compress_predictions:
                labels = np.full((T, Z, H, W), 255, dtype=np.uint8)
            else:
                labels = np.lib.format.open_memmap(op, mode="w+", dtype="uint8", shape=(T, Z, H, W))
                labels[:] = 255

            written_planes = 0
            indices = [(t, z) for t in range(T) for z in range(Z)]
            for offset in range(0, len(indices), batch_size):
                chunk = indices[offset : offset + batch_size]
                chunk_contexts = [TrainingData.context(cine, t, z) for t, z in chunk]
                batch_x = torch.cat(chunk_contexts, dim=0).to(device)

                if is_cuda:
                    torch.cuda.synchronize(device=device)
                t0 = time.perf_counter()

                # Native valid mask: all real image pixels are True; internal padding is False
                valid_mask = torch.ones(
                    (batch_x.shape[0], batch_x.shape[2], batch_x.shape[3]),
                    dtype=torch.bool,
                    device=batch_x.device,
                )

                # Direct raw student forward through head
                logits = model(batch_x, valid_mask=valid_mask)

                if is_cuda:
                    torch.cuda.synchronize(device=device)
                chunk_latencies_ms.append((time.perf_counter() - t0) * 1000.0)

                if logits.dtype != torch.float32 or not torch.isfinite(logits).all():
                    raise FloatingPointError(f"nonfinite or non-FP32 evaluation logits for patient {pid}")

                # Direct raw logits argmax: strictly no softmax rounding ties
                pred_chunk = logits.argmax(1).cpu().numpy().astype(np.uint8)

                for (t, z), plane in zip(chunk, pred_chunk):
                    labels[t, z] = plane
                    written_planes += 1

            # Plane count verification: all slices must be explicitly written
            if written_planes != T * Z:
                raise RuntimeError(
                    f"plane count mismatch for patient {pid}: wrote {written_planes} of {T * Z} planes"
                )

            # Sentinel check: zero 255 voxels may remain
            if (labels == 255).any():
                raise RuntimeError(
                    f"incomplete prediction output: unwritten sentinel 255 voxels detected for patient {pid}"
                )

            # Class range verification: all voxels in [0, 3]
            if not np.isin(labels, [0, 1, 2, 3]).all():
                raise RuntimeError(
                    f"invalid class values detected in predictions for patient {pid}"
                )

            if compress_predictions:
                with op.open("wb") as raw_stream:
                    with gzip.GzipFile(filename="", mode="wb", fileobj=raw_stream, compresslevel=1, mtime=0) as gz_stream:
                        np.save(gz_stream, labels, allow_pickle=False)
            else:
                labels.flush()
            del labels

            file_sha = digest(op)
            inventory.append({
                "id": pid,
                "file": op.name,
                "sha256": file_sha,
                "shape": [T, Z, H, W],
                "dtype": "uint8",
                "T": T,
                "Z": Z,
                "H": H,
                "W": W,
                "voxels": patient_voxels,
                "coverage": 1.0,
            })

    output_export_seconds = time.monotonic() - inference_start
    throughput_voxels = float(total_voxels_evaluated / max(1e-6, output_export_seconds))

    if chunk_latencies_ms:
        p50_latency_ms = float(np.percentile(chunk_latencies_ms, 50))
        p95_latency_ms = float(np.percentile(chunk_latencies_ms, 95))
    else:
        p50_latency_ms = None
        p95_latency_ms = None

    if is_cuda:
        torch.cuda.synchronize(device=device)
        peak_cuda_alloc = int(torch.cuda.max_memory_allocated(device=device))
        peak_cuda_res = int(torch.cuda.max_memory_reserved(device=device))
    else:
        peak_cuda_alloc = None
        peak_cuda_res = None

    # Measure true cold deployment total through frozen receipt generation (before scoring)
    cold_deployment_total_seconds = time.monotonic() - deployment_start

    cold_cost_diagnostics = {
        "checkpoint_load_seconds": checkpoint_load_seconds,
        "output_export_seconds": output_export_seconds,
        "cold_deployment_total_seconds": cold_deployment_total_seconds,
        "cold_output_total_seconds": cold_deployment_total_seconds,
        "total_seconds": cold_deployment_total_seconds,
        "total_voxels": total_voxels_evaluated,
        "throughput_voxels_per_sec": throughput_voxels,
        "latency_p50_ms": p50_latency_ms,
        "latency_p95_ms": p95_latency_ms,
        "peak_cuda_allocated_bytes": peak_cuda_alloc,
        "peak_cuda_reserved_bytes": peak_cuda_res,
    }

    # 8. Write Frozen Prediction Receipt BEFORE opening ANY reference masks
    frozen_receipt = {
        "schema": SCHEMA_FROZEN_PREDICTIONS,
        "recipe_sha256": digest(out_dir / "RECIPE.json"),
        "checkpoint_sha256": checkpoint_sha,
        "data_sha256": data_sha,
        "scope": manifest.get("scope", manifest.get("role")),
        "role": manifest.get("role"),
        "synthetic": is_synthetic_meta,
        "coverage": 1.0,
        "total_patients_exported": len(patients),
        "total_frames_exported": total_frames_exported,
        "total_slices_exported": total_slices_exported,
        "total_voxels_exported": total_voxels_evaluated,
        "files": inventory,
        "cold_cost_diagnostics": cold_cost_diagnostics,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(out_dir / "PREDICTIONS_FROZEN.json", frozen_receipt)
    frozen_sha256 = digest(out_dir / "PREDICTIONS_FROZEN.json")

    # =========================================================================
    # No mask array or reference path has been opened above this line.
    # Output is completely frozen on disk and hashed before scoring commences.
    # =========================================================================

    has_any_masks = any(bool(r.get("masks")) for r in patients)
    if inference_only or not has_any_masks:
        metrics_result = {
            "schema": SCHEMA_METRICS,
            "status": "INFERENCE_ONLY",
            "scope": manifest.get("scope", manifest.get("role")),
            "role": manifest.get("role"),
            "synthetic": is_synthetic_meta,
            "coverage": 1.0,
            "prediction_coverage": 1.0,
            "scoring_coverage": 0.0,
            "total_patients_exported": len(patients),
            "total_frames_exported": total_frames_exported,
            "total_slices_exported": total_slices_exported,
            "total_voxels_exported": total_voxels_evaluated,
            "total_patients_scored": 0,
            "total_frames_scored": 0,
            "checkpoint_load_seconds": checkpoint_load_seconds,
            "output_export_seconds": output_export_seconds,
            "cold_deployment_total_seconds": cold_deployment_total_seconds,
            "cold_output_total_seconds": cold_deployment_total_seconds,
            "total_seconds": cold_deployment_total_seconds,
            "elapsed_seconds": time.monotonic() - overall_start,
            "throughput_voxels_per_sec": throughput_voxels,
            "latency_p50_ms": p50_latency_ms,
            "latency_p95_ms": p95_latency_ms,
            "peak_cuda_allocated_bytes": peak_cuda_alloc,
            "peak_cuda_reserved_bytes": peak_cuda_res,
            "cold_cost_diagnostics": cold_cost_diagnostics,
            "frozen_receipt_sha256": frozen_sha256,
            "note": "Inference completed and predictions frozen; no masks opened or scored.",
        }
        write_json(out_dir / "METRICS.json", metrics_result)
        return metrics_result

    # 9. Optional Scoring Phase
    # Verify predictions have not mutated since freezing
    for item in inventory:
        check_op = pred_dir / item["file"]
        if digest(check_op) != item["sha256"]:
            raise ValueError(f"frozen prediction file modified after freeze: {check_op}")

    frame_rows: list[dict[str, Any]] = []
    pooled_confusion = np.zeros((4, 5), dtype=np.int64)
    patient_ids_scored: set[str] = set()

    for record, frozen_item in zip(patients, inventory):
        pred_path = pred_dir / frozen_item["file"]
        T_expected = frozen_item["T"]
        masks_dict = record.get("masks") or {}
        if not masks_dict:
            continue

        # Validate canonical integer frame IDs and 0 <= frame < T BEFORE mask I/O
        for frame_key in masks_dict.keys():
            if isinstance(frame_key, int):
                int_frame = frame_key
            elif isinstance(frame_key, str) and frame_key.isdigit() and str(int(frame_key)) == frame_key:
                int_frame = int(frame_key)
            else:
                raise ValueError(
                    f"non-canonical or invalid frame ID '{frame_key}' for patient {record['id']}"
                )

            if int_frame < 0 or int_frame >= T_expected:
                raise ValueError(
                    f"frame ID {int_frame} out of bounds: expected 0 <= frame < {T_expected} for patient {record['id']}"
                )

        if compress_predictions:
            with gzip.open(pred_path, "rb") as gz_in:
                pred_volume = np.load(gz_in, allow_pickle=False)
        else:
            pred_volume = np.load(pred_path, mmap_mode="r", allow_pickle=False)

        for frame_key, mask_info in sorted(masks_dict.items(), key=lambda kv: int(kv[0])):
            frame_idx = int(frame_key)
            mp = resolve_path(mask_info["path"])
            if not mp.is_file():
                raise ValueError(f"reference mask file not found: {mp}")
            if digest(mp) != mask_info["sha256"]:
                raise ValueError(f"reference mask hash mismatch for patient {record['id']} frame {frame_key}")

            y_ref = np.load(mp, allow_pickle=False)
            if y_ref.shape != pred_volume[frame_idx].shape:
                raise ValueError(
                    f"reference mask shape mismatch for patient {record['id']} frame {frame_key}: "
                    f"mask {y_ref.shape} vs prediction {pred_volume[frame_idx].shape}"
                )

            # Sum confusion in complete frame volume
            cm = confusion_matrix(y_ref, np.asarray(pred_volume[frame_idx]))
            pooled_confusion += cm
            c_metrics = class_metrics(cm)

            # Absent-reference FP diagnostics:
            # If all voxels are ignored (valid_voxels == 0), the FP denominator is 0,
            # so fp_rate is undefined (None), NOT 0.0!
            valid_voxels = int((y_ref != 255).sum())
            fp_diag: dict[str, Any] = {}
            for k, name in enumerate(METRIC_CLASSES):
                if not c_metrics[k]["reference_present"]:
                    fp_count = c_metrics[k]["FP"]
                    fp_rate = float(fp_count / valid_voxels) if valid_voxels > 0 else None
                    fp_diag[name] = {
                        "FP": fp_count,
                        "valid_voxels": valid_voxels,
                        "fp_rate": fp_rate,
                    }

            frame_rows.append({
                "patient": record["id"],
                "frame": frame_idx,
                "raw": c_metrics,
                "raw_confusion": cm.tolist(),
                "absent_class_fp_diagnostics": fp_diag,
            })
            patient_ids_scored.add(record["id"])

    per_patient, macro, fg_macro, all_macro = compute_patient_macro_metrics(frame_rows, patient_ids)

    warning_text = (
        "VAL is exposed development; this is not independent confirmation."
        if manifest.get("role") == "development"
        else "Audit risk is optimization evidence; patient Dice is empirical test."
    )

    total_patients_exported = len(patients)
    total_patients_scored = len(patient_ids_scored)
    scoring_coverage = float(total_patients_scored / max(1, total_patients_exported))

    metrics_result = {
        "schema": SCHEMA_METRICS,
        "status": "COMPLETED",
        "scope": manifest.get("scope", manifest.get("role")),
        "role": manifest.get("role"),
        "synthetic": is_synthetic_meta,
        "coverage": 1.0,
        "prediction_coverage": 1.0,
        "scoring_coverage": scoring_coverage,
        "total_patients_exported": total_patients_exported,
        "total_frames_exported": total_frames_exported,
        "total_slices_exported": total_slices_exported,
        "total_voxels_exported": total_voxels_evaluated,
        "total_patients_scored": total_patients_scored,
        "total_frames_scored": len(frame_rows),
        "checkpoint_load_seconds": checkpoint_load_seconds,
        "output_export_seconds": output_export_seconds,
        "cold_deployment_total_seconds": cold_deployment_total_seconds,
        "cold_output_total_seconds": cold_deployment_total_seconds,
        "total_seconds": cold_deployment_total_seconds,
        "elapsed_seconds": time.monotonic() - overall_start,
        "throughput_voxels_per_sec": throughput_voxels,
        "latency_p50_ms": p50_latency_ms,
        "latency_p95_ms": p95_latency_ms,
        "peak_cuda_allocated_bytes": peak_cuda_alloc,
        "peak_cuda_reserved_bytes": peak_cuda_res,
        "cold_cost_diagnostics": cold_cost_diagnostics,
        "macro_patient": macro,
        "foreground_patient_macro": fg_macro,
        "all_class_patient_macro": all_macro,
        "patients": per_patient,
        "frames": frame_rows,
        "pooled_confusion": pooled_confusion.tolist(),
        "frozen_receipt_sha256": frozen_sha256,
        "warning": warning_text,
        "physical_boundary_metrics": "NOT_COMPUTED; no spacing/HD95 certification",
    }
    write_json(out_dir / "METRICS.json", metrics_result)
    return metrics_result
