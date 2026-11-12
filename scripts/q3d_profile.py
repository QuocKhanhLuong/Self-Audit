"""Measure student-only Q-3D CPU/GPU inference costs on one full volume."""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from self_audit.candidate_q3d.evaluation import validate_native_image  # noqa: E402
from self_audit.candidate_q3d.export import load_export  # noqa: E402


def _synthetic_input(shape_zyx: tuple[int, int, int]) -> tuple[torch.Tensor, dict[str, Any]]:
    if len(shape_zyx) != 3 or any(type(size) is not int or size <= 0 for size in shape_zyx):
        raise ValueError("profile shape must contain positive D,H,W")
    spacing = (8.0, 1.25, 1.25)
    affine = np.diag([*spacing, 1.0]).astype(np.float64)
    geometry = {"shape": shape_zyx, "spacing_zyx": spacing,
                "affine_zyx": affine.tolist(), "context": "full_volume"}
    generator = torch.Generator(device="cpu").manual_seed(4123)
    image = torch.rand((1, 1, *shape_zyx), generator=generator)
    return validate_native_image(image, geometry)


def _input_file(path: str | Path) -> tuple[torch.Tensor, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        required = {"image", "shape_zyx", "spacing_zyx", "affine_zyx", "context"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"profile input is missing fields: {sorted(missing)}")
        geometry = {"shape": tuple(int(v) for v in data["shape_zyx"].tolist()),
                    "spacing_zyx": data["spacing_zyx"].tolist(),
                    "affine_zyx": data["affine_zyx"].tolist(),
                    "context": str(np.asarray(data["context"]).item())}
        image = np.array(data["image"], copy=True)
    return validate_native_image(image, geometry)


def _count_known_module_flops(model: nn.Module, image: torch.Tensor,
                              geometry: dict[str, Any]) -> dict[str, Any]:
    total = 0
    module_calls = 0
    operator_macs = {"torch.matmul": 0, "torch.bmm": 0, "scaled_dot_product_attention": 0}
    operator_calls = {key: 0 for key in operator_macs}
    handles = []

    def record_macs(name: str, macs: int) -> None:
        operator_macs[name] += macs
        operator_calls[name] += 1

    def matmul_macs(left: torch.Tensor, right: torch.Tensor) -> int:
        if left.ndim == 0 or right.ndim == 0:
            return 0
        if left.ndim == 1 and right.ndim == 1:
            return int(left.shape[0])
        if left.ndim == 1:
            batch_shape = right.shape[:-2]
            output_shape = (*batch_shape, right.shape[-2])
            reduction = left.shape[0]
        elif right.ndim == 1:
            batch_shape = left.shape[:-2]
            output_shape = (*batch_shape, left.shape[-2])
            reduction = left.shape[-1]
        else:
            batch_shape = torch.broadcast_shapes(left.shape[:-2], right.shape[:-2])
            output_shape = (*batch_shape, left.shape[-2], right.shape[-1])
            reduction = left.shape[-1]
        return int(math.prod(output_shape) * reduction)

    def counted_matmul(left: torch.Tensor, right: torch.Tensor, *args: Any,
                       **kwargs: Any) -> torch.Tensor:
        record_macs("torch.matmul", matmul_macs(left, right))
        return original_matmul(left, right, *args, **kwargs)

    def counted_bmm(left: torch.Tensor, right: torch.Tensor, *args: Any,
                    **kwargs: Any) -> torch.Tensor:
        if left.ndim == right.ndim == 3:
            record_macs("torch.bmm", matmul_macs(left, right))
        return original_bmm(left, right, *args, **kwargs)

    def counted_sdpa(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                     *args: Any, **kwargs: Any) -> torch.Tensor:
        # QK^T and attention-weighted V each use one MAC per multiply-add.
        batch_shape = torch.broadcast_shapes(query.shape[:-2], key.shape[:-2], value.shape[:-2])
        qk_macs = math.prod(batch_shape) * query.shape[-2] * key.shape[-2] * query.shape[-1]
        av_macs = math.prod(batch_shape) * query.shape[-2] * key.shape[-2] * value.shape[-1]
        record_macs("scaled_dot_product_attention", int(qk_macs + av_macs))
        return original_sdpa(query, key, value, *args, **kwargs)

    def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        nonlocal total, module_calls
        if not torch.is_tensor(output):
            return
        if isinstance(module, nn.Conv3d):
            kernel = int(np.prod(module.kernel_size))
            total += int(output.numel() * (module.in_channels // module.groups) * kernel * 2)
            module_calls += 1
        elif isinstance(module, nn.Linear):
            total += int(output.numel() * module.in_features * 2)
            module_calls += 1
    for module in model.modules():
        if isinstance(module, (nn.Conv3d, nn.Linear)):
            handles.append(module.register_forward_hook(hook))
    original_matmul = torch.matmul
    original_bmm = torch.bmm
    original_sdpa = F.scaled_dot_product_attention
    try:
        torch.matmul = counted_matmul
        torch.bmm = counted_bmm
        F.scaled_dot_product_attention = counted_sdpa
        model.eval()
        with torch.inference_mode():
            model.infer_logits(image, geometry=geometry)
    finally:
        torch.matmul = original_matmul
        torch.bmm = original_bmm
        F.scaled_dot_product_attention = original_sdpa
        for handle in handles:
            handle.remove()
    operator_flops = sum(operator_macs.values()) * 2
    return {
        "status": "PARTIAL_LOWER_BOUND",
        "value": total + operator_flops,
        "module_flops": total,
        "functional_operator_flops": operator_flops,
        "mac_convention": "2 floating point operations per multiply-accumulate",
        "unit": "floating_point_operations_per_volume",
        "covered_modules": ["nn.Conv3d", "nn.Linear"],
        "covered_operators": list(operator_macs),
        "module_calls_counted": module_calls,
        "operator_calls_counted": operator_calls,
        "omitted": ["interpolation", "normalization", "softmax", "elementwise ops"],
        "total_flops_status": "NOT_RUN_INCOMPLETE_OPERATOR_COVERAGE",
    }


def _cpu_profile(model: nn.Module, image: torch.Tensor, geometry: dict[str, Any], *,
                 warmup: int, repeats: int) -> dict[str, Any]:
    cpu_model = copy.deepcopy(model).to(device="cpu", memory_format=torch.channels_last_3d).eval()
    cpu_image = image.to(device="cpu", memory_format=torch.channels_last_3d)
    known_flops = _count_known_module_flops(cpu_model, cpu_image, geometry)
    with torch.inference_mode():
        for _ in range(warmup):
            cpu_model.infer_logits(cpu_image, geometry=geometry)
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            cpu_model.infer_logits(cpu_image, geometry=geometry)
            timings.append((time.perf_counter() - start) * 1000.0)
    return {
        "cpu_latency_ms_per_volume": {
            "status": "COMPUTED", "value": float(np.median(timings)), "unit": "ms",
            "repeats": repeats, "warmup": warmup, "dtype": "float32",
            "summary": {"mean": float(np.mean(timings)), "p95": float(np.percentile(timings, 95))},
        },
        "flops": known_flops,
        "cpu_hardware": {"processor": platform.processor() or "UNDECLARED",
                         "platform": platform.platform(), "torch_threads": torch.get_num_threads()},
    }


def _gpu_profile(model: nn.Module, image: torch.Tensor, geometry: dict[str, Any], *,
                 warmup: int, repeats: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"status": "NOT_RUN_NO_CUDA", "gpu_latency_ms_per_volume": {
            "status": "NOT_RUN_NO_CUDA"}, "peak_vram_mb": {"status": "NOT_RUN_NO_CUDA"}}
    if not torch.cuda.is_bf16_supported():
        return {"status": "NOT_RUN_BF16_UNSUPPORTED", "gpu_latency_ms_per_volume": {
            "status": "NOT_RUN_BF16_UNSUPPORTED"}, "peak_vram_mb": {"status": "NOT_RUN_BF16_UNSUPPORTED"}}
    gpu_model = copy.deepcopy(model).to(device="cuda", memory_format=torch.channels_last_3d).eval()
    gpu_image = image.to(device="cuda", dtype=torch.float32, memory_format=torch.channels_last_3d)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    with torch.inference_mode():
        for _ in range(warmup):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                gpu_model.infer_logits(gpu_image, geometry=geometry)
        torch.cuda.synchronize()
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                gpu_model.infer_logits(gpu_image, geometry=geometry)
            end.record()
            torch.cuda.synchronize()
            times.append(float(start.elapsed_time(end)))
    peak = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    return {
        "status": "COMPUTED",
        "gpu_latency_ms_per_volume": {"status": "COMPUTED", "value": float(np.median(times)),
                                      "unit": "ms", "repeats": repeats, "warmup": warmup,
                                      "dtype": "bfloat16", "summary": {
                                          "mean": float(np.mean(times)),
                                          "p95": float(np.percentile(times, 95))}},
        "peak_vram_mb": {"status": "COMPUTED", "value": peak, "unit": "MiB",
                          "scope": "torch.cuda.max_memory_allocated"},
        "gpu_hardware": {"name": torch.cuda.get_device_name(),
                         "capability": list(torch.cuda.get_device_capability())},
    }


def profile_artifact(artifact_path: str | Path, output_path: str | Path, *,
                     input_path: str | Path | None = None,
                     shape_zyx: tuple[int, int, int] = (10, 64, 64),
                     request_gpu: bool = False, warmup: int = 2, repeats: int = 5,
                     exploratory_quality_waiver: str | None = None) -> dict[str, Any]:
    if warmup < 0 or repeats < 1:
        raise ValueError("profile warmup must be nonnegative and repeats must be positive")
    model, metadata = load_export(
        artifact_path, device="cpu",
        **({"exploratory_quality_waiver": exploratory_quality_waiver}
           if exploratory_quality_waiver is not None else {}),
    )
    image, geometry = _input_file(input_path) if input_path is not None else _synthetic_input(shape_zyx)
    cpu = _cpu_profile(model, image, geometry, warmup=warmup, repeats=repeats)
    gpu = _gpu_profile(model, image, geometry, warmup=warmup, repeats=repeats) if request_gpu else {
        "status": "NOT_RUN_NOT_REQUESTED",
        "gpu_latency_ms_per_volume": {"status": "NOT_RUN_NOT_REQUESTED"},
        "peak_vram_mb": {"status": "NOT_RUN_NOT_REQUESTED"},
    }
    parameters = int(sum(parameter.numel() for parameter in model.parameters()))
    result = {
        "schema": "q3d.profile.v1",
        "artifact_sha256": metadata["artifact_sha256"],
        **{key: metadata[key] for key in (
            "exploratory", "exploratory_quality_waiver", "measured_quality_gates",
        ) if key in metadata},
        "arm": metadata["arm"],
        "parameters": {"status": "COMPUTED", "value": parameters, "unit": "parameters"},
        "input": {"source": "EXTERNAL_NATIVE_NPZ" if input_path else "SYNTHETIC_SOFTWARE_FIXTURE",
                  "shape_zyx": list(geometry["shape"]), "spacing_zyx": list(geometry["spacing_zyx"]),
                  "dtype": "float32", "batch": 1},
        **cpu,
        **gpu,
        "gpu_requested": bool(request_gpu),
        "evidence_status": "SOFTWARE_ONLY_SYNTHETIC" if input_path is None else "PROFILE_ONLY_DECLARED_VOLUME",
    }
    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input", help="native input NPZ; otherwise use the explicit synthetic shape")
    parser.add_argument("--shape", nargs=3, type=int, default=(10, 64, 64), metavar=("D", "H", "W"))
    parser.add_argument("--gpu", action="store_true", help="request bf16 CUDA latency and peak VRAM")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--q3d-exploratory-quality-waiver", metavar="REASON",
                        help="exact explicit reason recorded in an exploratory artifact")
    args = parser.parse_args(argv)
    result = profile_artifact(args.artifact, args.output, input_path=args.input,
                              shape_zyx=tuple(args.shape), request_gpu=args.gpu,
                              warmup=args.warmup, repeats=args.repeats,
                              **({"exploratory_quality_waiver": args.q3d_exploratory_quality_waiver}
                                 if args.q3d_exploratory_quality_waiver is not None else {}))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
