#!/usr/bin/env python3
"""Profile one native-volume A-3D update before a campaign.

This is a preflight measurement, not a training runner.  Candidate A-3D's
K1 rollout is deliberately single-volume (``B=1``), so larger ordinary batch
sizes are reported as unsupported rather than being simulated.  The script
measures one teacher update and one student/K1 update on the largest labelled
TRAIN volume selected from image geometry only.  No checkpoint or optimiser
state from a campaign is written.
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from self_audit.candidate_a3d.config import validate_config
from self_audit.candidate_a3d.data import CandidateA3DData
from self_audit.candidate_a3d.engine import _sample_parts, _teacher_values, _validate_teacher_output
from self_audit.candidate_a3d.io import build_student, source_inventory
from self_audit.candidate_a3d.teacher import build_teacher3d, set_teacher_stage, teacher_loss
from self_audit.candidate_a.k1 import K1Config, student_rollout


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _memory(device: torch.device) -> dict[str, int | None]:
    if device.type != "cuda":
        return {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
    return {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _finite_grads(parameters: Iterable[torch.nn.Parameter]) -> bool:
    return all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
               for parameter in parameters)


def _time_call(device: torch.device, fn) -> tuple[Any, float, dict[str, int | None]]:
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    result = fn()
    _sync(device)
    return result, float(time.perf_counter() - started), _memory(device)


def _phase(name: str, device: torch.device, fn) -> dict[str, Any]:
    result: dict[str, Any] = {"name": name, "status": "FAILED"}
    try:
        _, seconds, memory = _time_call(device, fn)
        result.update(status="PASS", elapsed_seconds=seconds, **memory)
    except torch.cuda.OutOfMemoryError as exc:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        result.update(status="OOM", error=repr(exc), **_memory(device))
    except Exception as exc:  # noqa: BLE001 - receipt must retain the failing stage
        result.update(status="FAILED", error=repr(exc), **_memory(device))
    return result


def _largest_sample(data: CandidateA3DData) -> tuple[tuple[int, int], dict[str, Any]]:
    """Select by native image volume size, never by a mask or label statistic."""

    candidates: list[tuple[int, tuple[int, int], dict[str, Any]]] = []
    for key in data.labeled_frame_keys():
        record_index, frame = key
        record = data.labeled[record_index]
        image = data.base.array(record["image"], record["image_sha256"])
        shape = tuple(int(value) for value in image.shape[1:])
        candidates.append((int(np.prod(shape, dtype=np.int64)), key, {
            "patient": record["id"], "frame": int(frame), "shape": list(shape),
            "voxel_count": int(np.prod(shape, dtype=np.int64)),
            "spacing_zyx": list(data.geometry[record["id"]].spacing_zyx),
        }))
    if not candidates:
        raise ValueError("TRAIN manifest has no labelled A-3D frame")
    _, key, metadata = max(candidates, key=lambda item: (item[0], str(item[1])))
    return key, metadata


def _overlay_model(config: dict[str, Any], model_config: str | None) -> dict[str, Any]:
    if model_config is None:
        return config
    model_path = Path(model_config).resolve()
    if not model_path.is_file():
        raise ValueError(f"model config does not exist: {model_path}")
    external = json.loads(model_path.read_text())
    if not isinstance(external, dict):
        raise ValueError("model config must contain one JSON object")
    config["model"] = external
    return config


def _load_config(path: str | Path, model_config: str | None) -> dict[str, Any]:
    source = Path(path).resolve()
    config = json.loads(source.read_text())
    if not isinstance(config, dict):
        raise ValueError("A-3D profile config must contain one JSON object")
    return validate_config(_overlay_model(config, model_config))


def _teacher_trial(config: dict[str, Any], x: torch.Tensor, y: torch.Tensor,
                   geometry: dict[str, Any], device: torch.device, *, synthetic: bool,
                   bank: Any) -> tuple[dict[str, Any], dict[str, Any], torch.nn.Module,
                                        torch.optim.Optimizer]:
    teacher_config = {**config["model"], "class_mapping": config["class_mapping"]}
    teacher, provenance = build_teacher3d(teacher_config, synthetic=synthetic)
    teacher = teacher.to(device)
    set_teacher_stage(teacher, "teacher_pretrain", freeze=False)
    parameters = [parameter for parameter in teacher.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["teacher_lr"], weight_decay=config["weight_decay"])
    shape = tuple(int(value) for value in x.shape[-3:])

    def update() -> None:
        teacher.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = _validate_teacher_output(teacher(x, geometry=geometry), shape)
        sdf = bank.sdf_targets(y, geometry["spacing_zyx"])
        loss = teacher_loss(outputs, y, sdf, config["lambda_distance"])
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite teacher profile loss")
        loss.backward()
        if not _finite_grads(parameters):
            raise FloatingPointError("nonfinite teacher gradients")
        optimizer.step()
        # Match engine.py: release update gradients after the optimizer step,
        # while retaining AdamW state for the subsequent student/J1 trial.
        optimizer.zero_grad(set_to_none=True)
        if not all(torch.isfinite(parameter).all() for parameter in teacher.parameters()):
            raise FloatingPointError("nonfinite teacher parameters after optimizer step")
        _PROFILE_STATE["loss"] = float(loss.detach())

    phase = _phase("teacher_forward_backward_optimizer", device, update)
    phase["loss"] = _PROFILE_STATE.pop("loss", None)
    return phase, provenance, teacher, optimizer


def _teacher_snapshot(teacher: torch.nn.Module, x: torch.Tensor, geometry: dict[str, Any],
                      bank: Any, device: torch.device) -> tuple[dict[str, Any], torch.Tensor | None]:
    """Measure the detached teacher target pass before any student update."""

    values: torch.Tensor | None = None

    def snapshot() -> None:
        nonlocal values
        with torch.inference_mode():
            outputs = _validate_teacher_output(teacher(x, geometry=geometry), tuple(x.shape[-3:]))
            values = _teacher_values(bank, outputs).detach()

    phase = _phase("teacher_snapshot_for_student", device, snapshot)
    return phase, values


def _student_trial(config: dict[str, Any], x: torch.Tensor, y: torch.Tensor,
                   geometry: dict[str, Any], device: torch.device, bank: Any, codec: Any,
                   teacher_values: torch.Tensor, chunk_size: int,
                   teacher_optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    # Keep the teacher AdamW state alive while measuring the joint student
    # update.  This matches the real engine's memory footprint.
    if teacher_optimizer is None:
        raise ValueError("student profile requires a live teacher optimizer")
    trial_config = copy.deepcopy(config)
    trial_config["student"]["chunk_size"] = int(chunk_size)
    torch.manual_seed(config["seed"] + 2000)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"] + 2000)
    student, _, _ = build_student(trial_config)
    student = student.to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=config["student_lr"],
                                  weight_decay=config["weight_decay"])
    rollout_config = K1Config(**config["rollout"])
    targets = bank.targets(y, geometry["spacing_zyx"])
    generator = torch.Generator(device="cpu").manual_seed(config["rollout"]["seed"])

    def update() -> None:
        student.train()
        optimizer.zero_grad(set_to_none=True)
        result = student_rollout(
            student, x, bank, labels=y, codec=codec, geometry=geometry,
            teacher_values=teacher_values.detach(), query_targets=targets,
            config=rollout_config, generator=generator,
        )
        loss = (result.segmentation_loss
                + config["loss"]["region"] * result.region_query_loss
                + config["loss"]["boundary"] * result.boundary_query_loss
                + config["loss"]["policy"] * result.policy_loss)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite student profile loss")
        loss.backward()
        if not _finite_grads(student.parameters()):
            raise FloatingPointError("nonfinite student gradients")
        optimizer.step()
        if not all(torch.isfinite(parameter).all() for parameter in student.parameters()):
            raise FloatingPointError("nonfinite student parameters after optimizer step")
        _PROFILE_STATE["loss"] = float(loss.detach())
        _PROFILE_STATE["query_count"] = int(result.query_count)
        _PROFILE_STATE["code_bits"] = int(result.code_bits)

    phase = _phase(f"student_k1_chunk_{chunk_size}", device, update)
    phase.update(chunk_size=int(chunk_size), loss=_PROFILE_STATE.pop("loss", None),
                 query_count=_PROFILE_STATE.pop("query_count", None),
                 code_bits=_PROFILE_STATE.pop("code_bits", None))
    return phase


_PROFILE_STATE: dict[str, Any] = {}


def _profile_impl(config: dict[str, Any], manifest: str | Path, manifest_sha256: str,
                  out: str | Path, *, device: str = "cuda", synthetic: bool = False,
                  chunk_sizes: Iterable[int] | None = None) -> dict[str, Any]:
    if synthetic and config["model"]["backbone"] != "tiny_test3d":
        raise ValueError("synthetic profile requires tiny_test3d")
    if not synthetic and config["model"]["backbone"] != "mednext_l_k3":
        raise ValueError("patient profile requires the MedNeXt-L k3 adapter")
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if target_device.type not in {"cpu", "cuda"}:
        raise ValueError("profile device must be CPU or CUDA")
    data = CandidateA3DData(manifest, manifest_sha256, synthetic=synthetic)
    key, sample_meta = _largest_sample(data)
    x, y, geometry, sample = _sample_parts(data, key)
    x, y = x.to(target_device), y.to(target_device)
    bank_config = config["bank"]
    from self_audit.candidate_a3d.bank import BankConfig3D, QueryBank3D
    from self_audit.candidate_a3d.codec import MeasurementCodec3D
    bank = QueryBank3D(BankConfig3D(**bank_config))
    codec = MeasurementCodec3D(bank, **config["codec"])
    _PROFILE_STATE.clear()
    torch.manual_seed(config["seed"])
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
    teacher_phase, teacher_provenance, teacher, teacher_optimizer = _teacher_trial(
        config, x, y, geometry, target_device, synthetic=synthetic, bank=bank,
    )
    # The student sees a detached snapshot exactly as the engine does.
    snapshot_phase: dict[str, Any]
    teacher_values: torch.Tensor | None
    if teacher_phase["status"] == "PASS":
        teacher.eval()
        snapshot_phase, teacher_values = _teacher_snapshot(
            teacher, x, geometry, bank, target_device,
        )
    else:
        snapshot_phase = {
            "name": "teacher_snapshot_for_student", "status": "NOT_RUN",
            "reason": "teacher update failed; student target snapshot was not attempted",
        }
        teacher_values = None
    sizes = tuple(int(value) for value in (chunk_sizes or (config["student"]["chunk_size"],)))
    if not sizes or any(value < 1 for value in sizes):
        raise ValueError("chunk sizes must be positive")
    student_phases: list[dict[str, Any]] = []
    if snapshot_phase["status"] == "PASS" and teacher_values is not None:
        student_phases = [
            _student_trial(config, x, y, geometry, target_device, bank, codec,
                           teacher_values, value, teacher_optimizer)
            for value in sizes
        ]
    successful = [phase for phase in student_phases if phase["status"] == "PASS"]
    recommendation = None
    if successful:
        headroom = 0.9 * (torch.cuda.get_device_properties(target_device).total_memory
                          if target_device.type == "cuda" else float("inf"))
        eligible = [phase for phase in successful
                    if phase["peak_reserved_bytes"] is None or phase["peak_reserved_bytes"] <= headroom]
        if eligible:
            recommendation = min(eligible, key=lambda phase: phase["elapsed_seconds"])["chunk_size"]
    batches = []
    b1_status = "SUPPORTED_PROTOCOL" if teacher_phase["status"] == "PASS" and snapshot_phase["status"] == "PASS" else "NOT_RUN"
    for value in (1, 2, 4, 8, 16, 32, 64):
        batches.append({"batch_size": value, "status": b1_status if value == 1 else "UNSUPPORTED",
                        "reason": None if value == 1 and b1_status == "SUPPORTED_PROTOCOL"
                        else ("teacher/snapshot preflight failed; B=1 was not run" if value == 1
                              else "A-3D trainer and K1 rollout require B=1; no batch-size path exists")})
    result = {
        "schema": "self_audit.candidate_a3d.profile.v1",
        "status": "PASS_PREFLIGHT" if teacher_phase["status"] == "PASS" and successful else "FAILED_PREFLIGHT",
        "synthetic": bool(synthetic), "device": str(target_device), "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "python": sys.version, "platform": platform.platform(),
        "source_files": source_inventory(), "data_sha256": manifest_sha256,
        "config": config, "sample": {**sample_meta, **sample},
        "selection": "largest native image volume among labelled TRAIN frames; no mask statistics",
        "teacher_provenance": teacher_provenance, "teacher_phase": teacher_phase,
        "teacher_snapshot_phase": snapshot_phase,
        "student_phases": student_phases, "batch_trials": batches,
        "recommended_chunk_size": recommendation,
        "protocol": {"batch_size": 1, "full_volume": True, "crop": False, "resample": False,
                      "note": "batch powers above one are unsupported and were not simulated"},
    }
    destination = Path(out).resolve()
    (destination / "PROFILE.json").write_text(json.dumps(result, indent=2, allow_nan=False, default=str) + "\n")
    return result


def profile(config: dict[str, Any], manifest: str | Path, manifest_sha256: str,
            out: str | Path, *, device: str = "cuda", synthetic: bool = False,
            chunk_sizes: Iterable[int] | None = None) -> dict[str, Any]:
    """Run a profile and persist a failure receipt for every exception."""

    destination = Path(out).resolve()
    data_path = Path(manifest).resolve()
    if destination.exists():
        raise ValueError("profile output must be a new directory")
    if destination.is_relative_to(data_path.parent) or data_path.parent.is_relative_to(destination):
        raise ValueError("profile output must be outside the source-data directory")
    destination.mkdir(parents=True)
    try:
        return _profile_impl(config, manifest, manifest_sha256, destination,
                             device=device, synthetic=synthetic, chunk_sizes=chunk_sizes)
    except Exception as exc:
        failure = {
            "schema": "self_audit.candidate_a3d.profile.v1",
            "status": "FAILED_PREFLIGHT", "error": repr(exc),
            "synthetic": bool(synthetic), "device": str(device),
            "data_sha256": manifest_sha256,
        }
        (destination / "PROFILE.json").write_text(
            json.dumps(failure, indent=2, allow_nan=False, default=str) + "\n"
        )
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-config")
    parser.add_argument("--data", required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--chunk-sizes", type=int, nargs="+", help="decoder chunk sizes to benchmark")
    args = parser.parse_args(argv)
    config = _load_config(args.config, args.model_config)
    try:
        result = profile(config, args.data, args.data_sha256, args.out, device=args.device,
                         synthetic=args.synthetic, chunk_sizes=args.chunk_sizes)
    except Exception as exc:  # noqa: BLE001 - CLI must return nonzero after receipt
        print(json.dumps({"status": "FAILED_PREFLIGHT", "error": repr(exc)}), file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(result, indent=2, default=str))
    if result["status"] != "PASS_PREFLIGHT":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
