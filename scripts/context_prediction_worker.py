#!/usr/bin/env python3
"""Freeze adjacent-slice context predictions for the VAL18 diagnostic.

This worker deliberately has no scoring path.  It reads only the declared image
arrays and the *metadata* of the reference frames in the evaluation manifest;
ground-truth mask payloads are never opened.  A pinned runtime is selected
before importing ``self_audit`` so checkpoint source checks remain meaningful.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "self_audit.context_diagnostic.v1"
ROLES = {"teacher", "student"}
LAMBDAS = (0.0, 0.5, 1.0)
SCRIPT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_FILES = (
    "scripts/context_prediction_worker.py",
    "scripts/run_context_diagnostic.py",
    "self_audit/context_metrics.py",
)


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def protocol_snapshot(device: str, threads: int) -> dict[str, Any]:
    """Capture the execution contract used by one diagnostic worker.

    This function intentionally hashes the worker, coordinator and scoring
    module from the repository containing this script, before any pinned model
    runtime is imported.  The snapshot contains only JSON-native values.
    """
    import numpy as np
    import torch

    files = {}
    for relative in PROTOCOL_FILES:
        path = SCRIPT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"protocol source file not found: {path}")
        files[relative] = _sha256(path)
    snapshot: dict[str, Any] = {
        "schema": "self_audit.context_diagnostic.protocol.v1",
        "code_sha256": files,
        "python": {
            "executable": str(Path(sys.executable).absolute()),
            "resolved_executable": str(Path(sys.executable).resolve()),
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "platform": platform.platform(),
        "device": device,
        "threads": {
            "requested": int(threads),
            "actual": int(torch.get_num_threads()),
            "interop": int(torch.get_num_interop_threads()),
        },
        "torch": {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        },
        "numpy_version": np.__version__,
    }
    if device == "cuda":
        cuda = {
            "available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
        }
        if cuda["available"]:
            index = int(torch.cuda.current_device())
            props = torch.cuda.get_device_properties(index)
            cuda["current_device"] = index
            cuda["device"] = {
                "name": props.name,
                "total_memory": int(props.total_memory),
                "multi_processor_count": int(props.multi_processor_count),
                "major": int(props.major),
                "minor": int(props.minor),
            }
        snapshot["cuda"] = cuda
    return snapshot


def protocol_sha256(snapshot: Mapping[str, Any]) -> str:
    """Hash a protocol snapshot using canonical JSON serialization."""
    if not isinstance(snapshot, Mapping):
        raise ValueError("protocol snapshot must be a mapping")
    payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character SHA-256 hex string")
    clean = value.lower()
    if any(ch not in "0123456789abcdef" for ch in clean):
        raise ValueError(f"{name} must be a hexadecimal SHA-256 string")
    return clean


def _resolve_ref(base: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _overlap(left: Path, right: Path) -> bool:
    """Whether either path contains the other (after lexical resolution)."""
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


def _lambda_token(value: float) -> str:
    if value == 0.0:
        return "0"
    if value == 0.5:
        return "0p5"
    if value == 1.0:
        return "1"
    raise ValueError(f"unsupported lambda {value!r}")


def interpolate_context(x, lam: float):
    """Interpolate from repeated-center context to full adjacent-slice context.

    Lambda one intentionally returns a clone of the original tensor.  This is
    needed for the parity check: constructing ``center + 1 * (x-center)`` can
    preserve values mathematically while still changing floating-point bits.
    """
    import torch

    if not isinstance(x, torch.Tensor) or x.ndim != 4 or x.shape[1] != 3:
        raise ValueError("context must be a Bx3xHxW tensor")
    if isinstance(lam, bool) or not isinstance(lam, (int, float)) or not math.isfinite(float(lam)):
        raise ValueError("lambda must be finite numeric")
    lam = float(lam)
    if lam < 0.0 or lam > 1.0:
        raise ValueError("lambda must lie in [0, 1]")
    if lam == 1.0:
        return x.clone()
    center = x[:, 1:2]
    if lam == 0.0:
        return center.expand(-1, 3, -1, -1).clone()
    return center + lam * (x - center)


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")
    required = {"schema", "lambdas", "device", "threads", "parity", "val_manifest", "teacher", "student"}
    if set(config) != required:
        raise ValueError(f"config keys must be exactly {sorted(required)}")
    if config["schema"] != SCHEMA:
        raise ValueError(f"unsupported config schema: {config.get('schema')!r}")
    if config["lambdas"] != [0, 0.5, 1] and config["lambdas"] != [0.0, 0.5, 1.0]:
        raise ValueError("lambdas must be exactly [0, 0.5, 1]")
    if config["device"] not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if type(config["threads"]) is not int or config["threads"] < 1:
        raise ValueError("threads must be a positive integer")
    parity = config["parity"]
    if not isinstance(parity, Mapping) or set(parity) != {"logit_atol", "logit_rtol"}:
        raise ValueError("parity must contain exactly logit_atol and logit_rtol")
    for key in ("logit_atol", "logit_rtol"):
        value = parity[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"parity.{key} must be finite and non-negative")
    vm = config["val_manifest"]
    if not isinstance(vm, Mapping) or set(vm) != {"path", "sha256"}:
        raise ValueError("val_manifest must contain exactly path and sha256")
    _require_sha(vm["sha256"], "val_manifest.sha256")
    for role in ROLES:
        spec = config[role]
        required_model = {"path", "manifest_sha256", "payload_sha256", "runtime_root", "runtime_source_sha256"}
        if not isinstance(spec, Mapping) or set(spec) != required_model:
            raise ValueError(f"{role} must contain exactly {sorted(required_model)}")
        _require_sha(spec["manifest_sha256"], f"{role}.manifest_sha256")
        _require_sha(spec["payload_sha256"], f"{role}.payload_sha256")
        _require_sha(spec["runtime_source_sha256"], f"{role}.runtime_source_sha256")
        if not isinstance(spec["runtime_root"], str) or not spec["runtime_root"]:
            raise ValueError(f"{role}.runtime_root must be a non-empty path")
    return dict(config)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _validate_model_artifact(spec: Mapping[str, Any], role: str) -> tuple[Path, dict[str, Any], Path]:
    folder = Path(spec["path"]).resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"{role} artifact directory not found: {folder}")
    manifest_path = folder / "MANIFEST.json"
    payload_path = folder / "weights.pt"
    expected_manifest = _require_sha(spec["manifest_sha256"], f"{role}.manifest_sha256")
    expected_payload = _require_sha(spec["payload_sha256"], f"{role}.payload_sha256")
    if not manifest_path.is_file() or _sha256(manifest_path) != expected_manifest:
        raise ValueError(f"{role} manifest hash mismatch")
    manifest = _load_json(manifest_path)
    manifest_payload = _require_sha(manifest.get("payload_sha256"), f"{role}.MANIFEST.json.payload_sha256")
    if manifest_payload != expected_payload or not payload_path.is_file() or _sha256(payload_path) != expected_payload:
        raise ValueError(f"{role} payload hash mismatch")
    return folder, manifest, payload_path


def _load_runtime_model(spec: Mapping[str, Any], role: str, device: str):
    """Select pinned runtime first, verify provenance, then invoke strict loader."""
    import importlib

    runtime_root = Path(spec["runtime_root"]).resolve()
    if not runtime_root.is_dir():
        raise FileNotFoundError(f"{role} runtime_root not found: {runtime_root}")
    _validate_model_artifact(spec, role)
    # No self_audit import occurs before this insertion.
    runtime_text = str(runtime_root)
    sys.path.insert(0, runtime_text)
    try:
        engine = importlib.import_module("self_audit.engine")
        source_fn = getattr(engine, "source_digest", None)
        if source_fn is None:
            compact_artifacts = importlib.import_module("self_audit.compact_artifacts")
            source_fn = getattr(compact_artifacts, "source_digest", None)
        if source_fn is None:
            raise ValueError(f"{role} runtime has no source_digest")
        actual_source = _require_sha(source_fn(), f"{role}.runtime source_digest()")
        expected_source = _require_sha(spec["runtime_source_sha256"], f"{role}.runtime_source_sha256")
        if actual_source != expected_source:
            raise ValueError(f"{role} runtime source hash mismatch")
        if role == "teacher":
            model, metadata = engine.load_checkpoint(spec["path"], spec["manifest_sha256"], device=device)
        else:
            compact_artifacts = importlib.import_module("self_audit.compact_artifacts")
            model, metadata = compact_artifacts.load_compact_export(spec["path"], spec["manifest_sha256"], device=device)
        return model.eval(), metadata
    finally:
        # Keep the selected runtime available for already imported modules, but
        # do not leave duplicate insertions when tests invoke the helper twice.
        while sys.path.count(runtime_text) > 1:
            sys.path.remove(runtime_text)


def _model_training_ids(metadata: Mapping[str, Any]) -> set[str]:
    ids: set[str] = set()
    for key in ("labeled_ids", "unlabeled_ids", "lids", "uids"):
        value = metadata.get(key)
        if isinstance(value, list):
            ids.update(v for v in value if isinstance(v, str))
    lineage = metadata.get("lineage")
    if isinstance(lineage, Mapping):
        for key in ("lids", "uids", "labeled_ids", "unlabeled_ids"):
            value = lineage.get(key)
            if isinstance(value, list):
                ids.update(v for v in value if isinstance(v, str))
    data_provenance = metadata.get("data_provenance")
    if isinstance(data_provenance, Mapping):
        for key in ("lids", "uids", "labeled_ids", "unlabeled_ids"):
            value = data_provenance.get(key)
            if isinstance(value, list):
                ids.update(v for v in value if isinstance(v, str))
    return ids


def _load_evaluation_manifest(config: Mapping[str, Any], metadata: Mapping[str, Any]):
    import numpy as np

    vm = config["val_manifest"]
    manifest_path = Path(vm["path"]).resolve()
    expected = _require_sha(vm["sha256"], "val_manifest.sha256")
    if not manifest_path.is_file() or _sha256(manifest_path) != expected:
        raise ValueError("validation manifest hash mismatch")
    manifest = _load_json(manifest_path)
    if manifest.get("schema") != "self_audit.data.v2" or manifest.get("role") != "development":
        raise ValueError("context diagnostic requires a development self_audit.data.v2 manifest")
    synthetic = bool(manifest.get("synthetic", False))
    model_synthetic = metadata.get("synthetic")
    if model_synthetic is None and isinstance(metadata.get("lineage"), Mapping):
        model_synthetic = metadata["lineage"].get("synthetic")
    if model_synthetic is not None and bool(model_synthetic) != synthetic:
        raise ValueError("validation/model synthetic identity mismatch")
    patients = manifest.get("patients")
    if not isinstance(patients, list) or not patients:
        raise ValueError("development manifest must contain non-empty patients")
    ids = [p.get("id") for p in patients if isinstance(p, Mapping)]
    if len(ids) != len(patients) or any(not isinstance(pid, str) or not pid for pid in ids) or len(ids) != len(set(ids)):
        raise ValueError("development patients must have unique non-empty IDs")
    for patient_id in ids:
        if patient_id in {".", ".."} or Path(patient_id).is_absolute() or Path(patient_id).name != patient_id or "/" in patient_id or "\\" in patient_id:
            raise ValueError(f"unsafe patient ID for output filename: {patient_id!r}")
        if any(ord(char) < 32 for char in patient_id):
            raise ValueError(f"unsafe control character in patient ID: {patient_id!r}")
    overlap = set(ids) & _model_training_ids(metadata)
    if overlap:
        raise ValueError(f"validation/TRAIN patient overlap: {sorted(overlap)}")
    for patient in patients:
        required_patient = {"id", "image", "image_sha256", "masks"}
        if not required_patient.issubset(patient):
            raise ValueError(f"invalid development patient fields for {patient.get('id')!r}")
        # VAL18 native manifests carry geometry/provenance metadata.  It is
        # descriptive only for this diagnostic.  A geometric valid-mask field
        # is deliberately rejected because this pilot has no valid-mask path.
        if "valid_mask" in patient or "geometric_valid" in patient:
            raise ValueError("valid-mask metadata is unsupported in the context diagnostic pilot")
        image_path = _resolve_ref(manifest_path.parent, patient["image"], "patient.image")
        expected_image = _require_sha(patient["image_sha256"], "patient.image_sha256")
        if not image_path.is_file() or _sha256(image_path) != expected_image:
            raise ValueError(f"image hash mismatch for patient {patient['id']}")
        image = np.load(image_path, mmap_mode="r", allow_pickle=False)
        if not isinstance(image, np.ndarray) or image.ndim != 4 or min(image.shape) < 1:
            raise ValueError(f"prepared image must be non-empty TZYX for patient {patient['id']}")
        if not np.issubdtype(image.dtype, np.floating) or not np.isfinite(image).all() or image.min() < 0.0 or image.max() > 1.0:
            raise ValueError(f"prepared image must be finite floating-point data in [0,1] for patient {patient['id']}")
        axes = patient.get("axes")
        if axes is not None and axes not in (("time", "slice", "height", "width"), ["time", "slice", "height", "width"], "TZYX"):
            raise ValueError(f"patient {patient['id']} axes are not prepared TZYX")
        masks = patient["masks"]
        if not isinstance(masks, Mapping) or not masks:
            raise ValueError(f"patient {patient['id']} has no declared reference frames")
        for frame, mask_meta in masks.items():
            if not isinstance(frame, str):
                raise ValueError("mask frame IDs must be strings")
            try:
                frame_index = int(frame)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid frame ID {frame!r}") from exc
            if str(frame_index) != frame or not 0 <= frame_index < image.shape[0]:
                raise ValueError(f"frame {frame!r} out of range for patient {patient['id']}")
            if not isinstance(mask_meta, Mapping) or set(mask_meta) != {"path", "sha256"}:
                raise ValueError("mask metadata must contain path and sha256")
            _require_sha(mask_meta["sha256"], "mask metadata sha256")
            # Intentionally do not resolve, hash, or open this path.  It is GT.
    return manifest_path, manifest


def _extract_context(role: str, image, t: int, z: int):
    if role == "teacher":
        from self_audit.data import TrainingData

        return TrainingData.context(image, t, z)
    from self_audit.compact_inference import extract_slice_context

    return extract_slice_context(image, t, z)


def _as_float32_logits(logits, role: str):
    import torch

    if not isinstance(logits, torch.Tensor) or logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != 4:
        raise ValueError(f"{role} model must return Bx4xHxW logits")
    if logits.dtype != torch.float32 or not torch.isfinite(logits).all():
        raise FloatingPointError(f"{role} logits must be finite float32")
    return logits


def _parity_pass(model, role: str, manifest: Mapping[str, Any], manifest_path: Path, atol: float, rtol: float, device: str):
    import numpy as np
    import torch

    max_abs = 0.0
    max_rel = 0.0
    mismatches = 0
    slices = 0
    for patient in manifest["patients"]:
        image = np.load(_resolve_ref(manifest_path.parent, patient["image"], "patient.image"), mmap_mode="r", allow_pickle=False)
        for frame in sorted(patient["masks"], key=int):
            t = int(frame)
            for z in range(image.shape[1]):
                x_ref = _extract_context(role, image, t, z)
                x_diag = interpolate_context(x_ref, 1.0)
                if not torch.equal(x_ref, x_diag):
                    raise RuntimeError(f"lambda=1 input parity failed at {patient['id']} frame={t} slice={z}")
                with torch.inference_mode():
                    reference = _as_float32_logits(model(x_ref.to(device)), role)
                    diagnostic = _as_float32_logits(model(x_diag.to(device)), role)
                difference = (reference - diagnostic).abs()
                abs_error = float(difference.max().item()) if difference.numel() else 0.0
                denominator = reference.abs().clamp_min(torch.finfo(reference.dtype).tiny)
                rel_error = float((difference / denominator).max().item()) if difference.numel() else 0.0
                max_abs = max(max_abs, abs_error)
                max_rel = max(max_rel, rel_error)
                close = torch.allclose(reference, diagnostic, atol=atol, rtol=rtol)
                same_argmax = torch.equal(reference.argmax(1), diagnostic.argmax(1))
                if not close or not same_argmax:
                    mismatches += 1
                slices += 1
    if mismatches:
        raise RuntimeError(f"lambda=1 logit parity failed on {mismatches}/{slices} slices")
    return {"slices": slices, "mismatches": mismatches, "max_abs_error": max_abs, "max_rel_error": max_rel}


def _counterfactual_pass(model, role: str, manifest: Mapping[str, Any], manifest_path: Path, output: Path, lambdas: list[float], image_hashes: dict[str, str], device: str):
    import numpy as np
    import torch

    records: list[dict[str, Any]] = []
    for patient in manifest["patients"]:
        image = np.load(_resolve_ref(manifest_path.parent, patient["image"], "patient.image"), mmap_mode="r", allow_pickle=False)
        for frame in sorted(patient["masks"], key=int):
            t = int(frame)
            for lam in lambdas:
                predicted = np.empty((image.shape[1], image.shape[2], image.shape[3]), dtype=np.uint8)
                for z in range(image.shape[1]):
                    x = _extract_context(role, image, t, z)
                    x = interpolate_context(x, lam)
                    with torch.inference_mode():
                        logits = _as_float32_logits(model(x.to(device)), role)
                    predicted[z] = logits.argmax(1)[0].cpu().numpy().astype(np.uint8, copy=False)
                path = output / f"{patient['id']}_frame{t}_lam{_lambda_token(lam)}.npy"
                np.save(path, predicted, allow_pickle=False)
                records.append({
                    "patient": patient["id"],
                    "frame": t,
                    "lambda": float(lam),
                    "path": str(path.resolve()),
                    "sha256": _sha256(path),
                    "shape_zyx": list(predicted.shape),
                    "image_sha256": image_hashes[patient["id"]],
                })
    return records


def run_worker(
    config: Mapping[str, Any],
    role: str,
    output: str | Path,
    expected_protocol_sha256: str | None = None,
) -> dict[str, Any]:
    """Run one pinned teacher or student diagnostic process and freeze masks."""
    config = _validate_config(config)
    if role not in ROLES:
        raise ValueError("role must be teacher or student")
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f"output must be a new directory: {output}")
    val_path = Path(config["val_manifest"]["path"]).resolve()
    model_path = Path(config[role]["path"]).resolve()
    overlap_sources = [val_path.parent.resolve(), model_path]
    overlap_sources.extend(Path(config[other]["path"]).resolve() for other in ROLES)
    overlap_sources.extend(Path(config[other]["runtime_root"]).resolve() for other in ROLES)
    for source in overlap_sources:
        if _overlap(output, source):
            raise ValueError("output overlaps an input directory")

    import torch

    torch.set_num_threads(config["threads"])
    protocol = protocol_snapshot(config["device"], config["threads"])
    protocol_digest = protocol_sha256(protocol)
    if expected_protocol_sha256 is not None:
        expected_protocol = _require_sha(expected_protocol_sha256, "expected_protocol_sha256")
        if protocol_digest != expected_protocol:
            raise ValueError(f"protocol hash mismatch: expected {expected_protocol}, got {protocol_digest}")
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model, metadata = _load_runtime_model(config[role], role, config["device"])
    manifest_path, manifest = _load_evaluation_manifest(config, metadata)
    image_hashes = {p["id"]: p["image_sha256"] for p in manifest["patients"]}
    parity = _parity_pass(model, role, manifest, manifest_path, float(config["parity"]["logit_atol"]), float(config["parity"]["logit_rtol"]), config["device"])
    output.mkdir(parents=False)
    records = _counterfactual_pass(model, role, manifest, manifest_path, output, [float(v) for v in config["lambdas"]], image_hashes, config["device"])
    if protocol_snapshot(config["device"], config["threads"]) != protocol:
        raise RuntimeError("diagnostic code/environment changed during inference")
    for patient in manifest["patients"]:
        if _sha256(_resolve_ref(manifest_path.parent, patient["image"], "image")) != image_hashes[patient["id"]]:
            raise RuntimeError("image payload changed during inference")
    coverage = {
        "patients": len(manifest["patients"]),
        "frames": sum(len(p["masks"]) for p in manifest["patients"]),
        "slices": parity["slices"],
        "lambdas": [float(v) for v in config["lambdas"]],
        "prediction_files": len(records),
    }
    receipt = {
        "schema": "self_audit.context_diagnostic.predictions.v1",
        "role": role,
        "config_schema": SCHEMA,
        "val_manifest_sha256": _require_sha(config["val_manifest"]["sha256"], "val_manifest.sha256"),
        "model_manifest_sha256": _require_sha(config[role]["manifest_sha256"], f"{role}.manifest_sha256"),
        "model_payload_sha256": _require_sha(config[role]["payload_sha256"], f"{role}.payload_sha256"),
        "runtime_source_sha256": _require_sha(config[role]["runtime_source_sha256"], f"{role}.runtime_source_sha256"),
        "device": config["device"],
        "threads": config["threads"],
        "protocol": protocol,
        "protocol_sha256": protocol_digest,
        "ground_truth_opened_before_freeze": False,
        "input": "prepared float NPY TZYX; no renormalization; standard adjacent-slice context",
        "output": "uint8 ZYX argmax masks; no probabilities, thresholding, crop, TTA, or postprocessing",
        "parity": parity,
        "coverage": coverage,
        "records": records,
    }
    (output / "PREDICTIONS_FROZEN.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def _read_config(path: Path, expected_sha: str) -> dict[str, Any]:
    expected = _require_sha(expected_sha, "config_sha256")
    if not path.is_file() or _sha256(path) != expected:
        raise ValueError("config hash mismatch")
    return _load_json(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--role", choices=sorted(ROLES), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--protocol-sha256")
    args = parser.parse_args(argv)
    config = _read_config(Path(args.config).resolve(), args.config_sha256)
    receipt = run_worker(config, args.role, args.out, expected_protocol_sha256=args.protocol_sha256)
    print(json.dumps({"role": receipt["role"], "coverage": receipt["coverage"], "parity": receipt["parity"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
