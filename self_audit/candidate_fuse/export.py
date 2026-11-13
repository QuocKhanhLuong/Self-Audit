"""Minimal inference exports and fresh-interpreter cold-load verification."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import numpy as np
from torch import nn

EXPORT_SCHEMA = "self_audit.candidate_fuse.export.v1"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().contiguous() for k, v in module.state_dict().items()}


def _sidecar_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".manifest.json")


def _validate_sidecar(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError("FUSE export sidecar manifest is missing")
    artifact_sha = _sha(path)
    if expected_sha256 is not None and artifact_sha != expected_sha256:
        raise ValueError("FUSE export SHA-256 differs from expected_sha256")
    try:
        sidecar = json.loads(_sidecar_path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("FUSE export sidecar manifest is missing or invalid") from exc
    if (sidecar.get("schema") != EXPORT_SCHEMA
            or sidecar.get("artifact") != path.name
            or sidecar.get("artifact_sha256") != artifact_sha
            or sidecar.get("artifact_bytes") != path.stat().st_size):
        raise ValueError("FUSE export sidecar does not identify the requested artifact bytes")
    return sidecar


def export_model(student, teacher, config, path, *, mode, metadata=None) -> dict[str, Any]:
    if mode not in {"teacher", "student", "fusion"}:
        raise ValueError("FUSE export mode must be teacher, student or fusion")
    if mode in {"teacher", "fusion"} and teacher is None:
        raise ValueError(f"{mode} export requires the trained teacher")
    if mode in {"student", "fusion"} and student is None:
        raise ValueError(f"{mode} export requires the trained student")
    chosen = {}
    model = None
    if mode in {"student", "fusion"}:
        from .student import FuseStudent
        model = FuseStudent(config, teacher_channels=getattr(teacher, "feature_channels", None),
                            teacher_phases=getattr(teacher, "feature_phases", None),
                            include_fusion=(mode == "fusion"), include_mimic=False)
        target_keys = set(model.state_dict())
        source = student.state_dict()
        chosen = {key: value.detach().cpu() for key, value in source.items() if key in target_keys}
        missing = target_keys - set(chosen)
        if missing:
            raise ValueError(f"student is missing export weights: {sorted(missing)[:5]}")
        model.load_state_dict(chosen, strict=True)
    try:
        from .config import config_sha256
        config_sha = config_sha256(config)
    except (KeyError, TypeError, ValueError):
        config_sha = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"),
                                               allow_nan=False).encode()).hexdigest()
    export_metadata = dict(metadata or {})
    export_metadata.setdefault("config_sha256", config_sha)
    data_config = config.get("data", {}) if isinstance(config, Mapping) else {}
    export_metadata.setdefault("data_manifest_sha256", {
        key: data_config.get(key) for key in
        ("train_manifest_sha256", "val_manifest_sha256", "test_manifest_sha256")
    })
    parameter_count = sum(p.numel() for p in model.parameters()) if model is not None else 0
    if mode in {"teacher", "fusion"}:
        teacher_model = getattr(teacher, "model", teacher)
        parameter_count += sum(p.numel() for p in teacher_model.parameters())
    export_metadata.setdefault("parameter_count", parameter_count)
    export_metadata.setdefault("measured_latency_ms", None)
    export_metadata.setdefault("latency_status", "NOT_MEASURED")
    payload: dict[str, Any] = {"schema": EXPORT_SCHEMA, "mode": mode,
                                "config": config, "metadata": export_metadata}
    if mode in {"student", "fusion"}:
        payload["student_state"] = chosen
    if mode in {"teacher", "fusion"}:
        teacher_model = getattr(teacher, "model", teacher)
        payload["teacher_state"] = _state(teacher_model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    artifact_sha = _sha(path)
    manifest = {"schema": EXPORT_SCHEMA, "artifact": path.name,
                "artifact_sha256": artifact_sha, "artifact_bytes": path.stat().st_size,
                "mode": mode, "metadata": export_metadata}
    sidecar = _sidecar_path(path)
    sidecar.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"path": str(path), "sha256": artifact_sha, "mode": mode,
            "manifest_path": str(sidecar),
            "metadata": export_metadata}


class _FuseInference(nn.Module):
    def __init__(self, student: nn.Module | None, teacher: nn.Module | None, mode: str):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.mode = mode
        self.eval()

    def forward(self, image, geometry):
        if self.mode == "teacher":
            return self.teacher(image, geometry, return_features=False)["region_logits"]
        if self.teacher is None:
            return self.student.infer_logits(image, geometry)
        output = self.teacher(image, geometry, return_features=True)
        if isinstance(output, tuple):
            _, features = output
        else:
            features = output.get("features")
        return self.student(image, geometry, teacher_features=features,
                            keep_mask=torch.ones((image.shape[0], 1, 1, 1, 1),
                                                 device=image.device))["logits"]


def _build_teacher_architecture(config: Mapping[str, Any]):
    """Construct the exact architecture without consulting pretrained paths."""
    from .teacher import FuseTeacher
    from self_audit.candidate_q3d.teacher_resenc import Q3DResEncTeacher, build_res_enc_l

    teacher_config = config.get("model", {}).get("teacher")
    if not isinstance(teacher_config, Mapping):
        raise ValueError("FUSE export config lacks model.teacher architecture contract")
    strides = teacher_config.get("encoder_strides")
    if not isinstance(strides, (list, tuple)) or len(strides) != 6:
        raise ValueError("FUSE teacher export requires six declared encoder strides")
    phase_offsets = teacher_config.get("phase_offsets")
    synthetic = bool(config.get("runtime", {}).get("synthetic", False))
    if synthetic:
        core = Q3DResEncTeacher(strides=strides, phase_offsets=phase_offsets,
                                synthetic=True, num_classes=4)
    else:
        wrapper = build_res_enc_l(strides)
        encoder = getattr(wrapper, "encoder", wrapper)
        core = Q3DResEncTeacher(encoder=encoder, phase_offsets=phase_offsets, num_classes=4)
    return FuseTeacher(core)


def load_export(path, *, device="cpu", expected_sha256: str | None = None) -> nn.Module:
    path = Path(path).resolve()
    sidecar = _validate_sidecar(path, expected_sha256)
    # Verify artifact bytes against the sidecar before any deserialization.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or payload.get("schema") != EXPORT_SCHEMA:
        raise ValueError("not a valid FUSE inference export")
    mode, config = payload.get("mode"), payload.get("config")
    if mode not in {"teacher", "student", "fusion"} or sidecar.get("mode") != mode or not isinstance(config, Mapping):
        raise ValueError("FUSE export mode/config is invalid")
    teacher = None
    student = None
    if mode in {"teacher", "fusion"}:
        teacher = _build_teacher_architecture(config)
        teacher_state = payload.get("teacher_state")
        if not isinstance(teacher_state, Mapping):
            raise ValueError("FUSE teacher/fusion export lacks a complete teacher state")
        prefixed = {f"model.{key}": value for key, value in teacher_state.items()}
        teacher.load_state_dict(prefixed, strict=True)
        teacher.freeze_for_joint()
    if mode in {"student", "fusion"}:
        from .student import FuseStudent
        student = FuseStudent(config, teacher_channels=getattr(teacher, "feature_channels", None),
                              teacher_phases=getattr(teacher, "feature_phases", None),
                              include_fusion=(mode == "fusion"), include_mimic=False)
        student.load_state_dict(payload["student_state"], strict=True)
    model = _FuseInference(student, teacher, mode)
    model.to(device).eval()
    return model


def _cold_worker(artifact, volume_file, result_file, device, threads, dtype, deterministic):
    torch.set_num_threads(threads)
    torch.set_default_dtype(getattr(torch, dtype))
    torch.use_deterministic_algorithms(deterministic)
    volumes = torch.load(volume_file, map_location="cpu", weights_only=True)
    model = load_export(artifact, device=device)
    results = []
    with torch.inference_mode():
        for image, geometry in volumes:
            results.append(model(image.to(device), geometry).detach().cpu())
    torch.save(results, result_file)


def _reference_logits(original_model, mode, image, geometry):
    if mode == "fusion" and isinstance(original_model, Mapping):
        teacher, student = original_model.get("teacher"), original_model.get("student")
        if teacher is None or student is None:
            raise ValueError("fusion cold-load reference requires teacher and student")
        teacher_out = teacher(image, geometry, return_features=True)
        features = teacher_out.get("features") if isinstance(teacher_out, Mapping) else teacher_out[1]
        return student(image, geometry, teacher_features=features,
                       keep_mask=torch.ones((image.shape[0], 1, 1, 1, 1), device=image.device))["logits"]
    if mode == "student" and callable(getattr(original_model, "infer_logits", None)):
        return original_model.infer_logits(image, geometry)
    if mode == "teacher":
        try:
            return original_model(image, geometry, return_features=False)["region_logits"]
        except TypeError:
            pass
    output = original_model(image, geometry)
    if isinstance(output, Mapping):
        return output.get("logits", output.get("region_logits"))
    return output


def verify_cold_load(path, volumes, *, original_model, device="cpu") -> dict[str, Any]:
    """Compare original trained model, artifact reload, and fresh-process outputs."""
    if len(volumes) != 3:
        raise ValueError("cold-load verification requires exactly three synthetic native volumes")
    threads = torch.get_num_threads()
    default_dtype = torch.get_default_dtype()
    deterministic = torch.are_deterministic_algorithms_enabled()
    model = load_export(path, device=device)
    mode = model.mode
    original_roots = list(original_model.values()) if isinstance(original_model, Mapping) else [original_model]
    original_flags = []
    seen = set()
    for root in original_roots:
        if isinstance(root, nn.Module):
            for module in root.modules():
                if id(module) not in seen:
                    seen.add(id(module))
                    original_flags.append((module, module.training))
    expected = []
    latency_ms = []
    try:
        for root in original_roots:
            if isinstance(root, nn.Module):
                root.eval()
        with torch.inference_mode():
            for image, geometry in volumes:
                tensor = torch.as_tensor(image, device=device)
                expected.append(_reference_logits(original_model, mode, tensor, geometry).detach().cpu())
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                model_output = model(tensor, geometry).detach().cpu()
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize(device)
                latency_ms.append((time.perf_counter() - started) * 1000.0)
                if not torch.equal(expected[-1], model_output):
                    raise RuntimeError("FUSE loaded export differs bitwise from the original model")
    finally:
        for module, flag in original_flags:
            module.training = flag
    with tempfile.TemporaryDirectory(prefix="fuse-cold-load-") as directory:
        volume_file = Path(directory) / "volumes.pt"
        result_file = Path(directory) / "outputs.pt"
        torch.save([(torch.as_tensor(image).cpu(), dict(geometry)) for image, geometry in volumes], volume_file)
        code = (
            "from self_audit.candidate_fuse.export import _cold_worker;"
            f"_cold_worker({str(Path(path).resolve())!r},{str(volume_file)!r},{str(result_file)!r},"
            f"{str(device)!r},{threads},{str(default_dtype).split('.')[-1]!r},{deterministic!r})"
        )
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = "0"
        subprocess.run([sys.executable, "-c", code], check=True, env=env,
                       cwd=str(Path(__file__).resolve().parents[2]))
        actual = torch.load(result_file, map_location="cpu", weights_only=True)
    if len(actual) != 3 or any(not torch.equal(a, b) for a, b in zip(expected, actual)):
        raise RuntimeError("FUSE fresh-interpreter outputs are not bitwise identical")
    sidecar_path = Path(path).with_suffix(Path(path).suffix + ".manifest.json")
    if sidecar_path.is_file():
        sidecar = json.loads(sidecar_path.read_text())
        sidecar.setdefault("metadata", {})["measured_latency_ms"] = float(np.mean(latency_ms))
        sidecar["metadata"]["latency_per_volume_ms"] = latency_ms
        sidecar["metadata"]["latency_status"] = "MEASURED_IN_PROCESS_AFTER_LOAD"
        temporary = sidecar_path.with_suffix(sidecar_path.suffix + ".tmp")
        temporary.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
        temporary.replace(sidecar_path)
    return {"status": "PASS", "volumes": 3, "bitwise_equal": True,
            "sha256": _sha(Path(path)),
            "mean_latency_ms": float(np.mean(latency_ms)),
            "latency_per_volume_ms": latency_ms,
            "mode": mode}
