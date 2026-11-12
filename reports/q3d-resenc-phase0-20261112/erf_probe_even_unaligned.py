#!/usr/bin/env python3
"""CPU-only empirical receptive-field probe for the external ResEnc-L checkpoint.

This probe does not read patient images. It loads only the encoder weights from the
locked checkpoint, feeds deterministic random Z-score volumes, and differentiates
the mean activation of the center cell in encoder stage 3 with respect to the
input. The channel reduction is an explicitly reported signed mean over all 256
stage-3 channels. Absolute input gradients are used for the centroid and mass
quantiles.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
import types
from pathlib import Path

import torch
from torch import nn


DEFAULT_ROOT = Path("/Users/alvinluong/Self-Audit-runs/q3d-resenc-20261112")
DEFAULT_CHECKPOINT = DEFAULT_ROOT / "checkpoint_final.pth"
DEFAULT_REGISTRY = DEFAULT_ROOT / "architecture_registry.py"
NATIVE_PLANNER_STRIDES = (
    (1, 1, 1),
    (2, 2, 2),
    (1, 2, 2),
    (1, 2, 2),
    (1, 2, 2),
    (1, 2, 2),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_registry(path: Path):
    # The external registry imports an optional no-skip class even though this
    # probe uses only get_res_enc_l. Keep the stub local to this process.
    root = types.ModuleType("nnssl")
    arch = types.ModuleType("nnssl.architectures")
    noskip = types.ModuleType("nnssl.architectures.noskipResEncUNet")
    noskip.ResidualEncoderUNet_noskip = object
    sys.modules.update({
        "nnssl": root,
        "nnssl.architectures": arch,
        "nnssl.architectures.noskipResEncUNet": noskip,
    })
    spec = importlib.util.spec_from_file_location("q3d_resenc_external_registry", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import architecture registry: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_encoder(model, checkpoint_path: Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint["network_weights"]
    encoder_state = {key: value for key, value in state.items() if key.startswith("encoder.")}
    model_state = model.state_dict()
    expected_encoder = {key for key in model_state if key.startswith("encoder.")}
    missing_encoder = sorted(expected_encoder.difference(encoder_state))
    unexpected_encoder = sorted(set(encoder_state).difference(expected_encoder))
    if missing_encoder or unexpected_encoder:
        raise RuntimeError(
            f"encoder checkpoint mismatch: missing={missing_encoder[:4]} "
            f"unexpected={unexpected_encoder[:4]}"
        )
    missing, unexpected = model.load_state_dict(encoder_state, strict=False)
    if unexpected or any(key.startswith("encoder.") for key in missing):
        raise RuntimeError(f"unexpected encoder load result: missing={missing} unexpected={unexpected}")
    return {
        "top_level_keys": sorted(checkpoint),
        "encoder_state_key_count": len(encoder_state),
        "missing_encoder_keys": missing_encoder,
        "unexpected_encoder_keys": unexpected_encoder,
        "non_encoder_missing_keys": len(missing),
    }


def build_model(registry, strides: tuple[tuple[int, int, int], ...]):
    """Build ResEnc-L with the exact planned stride schedule.

    The registry's public factory is the all-axis pretraining architecture. For
    the native Phase-0 probe we instantiate the same official ResEnc-L blocks
    with the stride list obtained from the pinned planner on the K8 median
    native metadata. Convolution weight shapes are unchanged, so encoder
    weights remain loadable; this is a geometry probe, not a checkpoint claim
    for a fully fine-tuned downstream model.
    """
    if len(strides) != 6:
        raise ValueError(f"expected six stage strides, got {strides}")
    # Read the factory once to keep the registry itself as the architecture
    # provenance, then construct the planner-derived variant explicitly.
    reference = registry.get_res_enc_l(1, 3, False)
    model_type = type(reference)
    return model_type(
        input_channels=1,
        n_stages=6,
        features_per_stage=[32, 64, 128, 256, 320, 320],
        conv_op=nn.Conv3d,
        kernel_sizes=[[3, 3, 3] for _ in range(6)],
        strides=[list(value) for value in strides],
        n_blocks_per_stage=[1, 3, 4, 6, 6, 6],
        num_classes=3,
        n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=False,
    )


def cumulative_stride(model, stage_index: int) -> tuple[int, int, int]:
    stride = [1, 1, 1]
    for stage in model.encoder.stages[: stage_index + 1]:
        block = stage.blocks[0]
        current = tuple(int(value) for value in block.conv1.conv.stride)
        stride = [old * new for old, new in zip(stride, current)]
    return tuple(stride)


def quantile_radius(mass: torch.Tensor, coordinates: torch.Tensor, center: torch.Tensor, q: float) -> float:
    radius = torch.sqrt(((coordinates - center[:, None, None, None]) ** 2).sum(dim=0)).flatten()
    weights = mass.flatten()
    order = torch.argsort(radius)
    cumulative = torch.cumsum(weights[order], dim=0)
    target = float(weights.sum()) * q
    index = int(torch.searchsorted(cumulative, torch.tensor(target, dtype=cumulative.dtype)).item())
    index = min(index, order.numel() - 1)
    return float(radius[order[index]].item())


def one_sample(model, shape: tuple[int, int, int], seed: int, stage_index: int) -> dict:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    image = torch.randn((1, 1, *shape), generator=generator, dtype=torch.float32)
    image = (image - image.mean()) / image.std(unbiased=False).clamp_min(1e-6)
    image.requires_grad_(True)
    started = time.perf_counter()
    feature = model.encoder.stem(image)
    for index, stage in enumerate(model.encoder.stages[: stage_index + 1]):
        feature = stage(feature)
    forward_seconds = time.perf_counter() - started
    index = tuple(int(size // 2) for size in feature.shape[-3:])
    # Explicit channel aggregation: signed arithmetic mean over all stage-3 channels.
    scalar = feature[0, :, index[0], index[1], index[2]].mean()
    started = time.perf_counter()
    gradient = torch.autograd.grad(scalar, image, retain_graph=False, create_graph=False)[0]
    backward_seconds = time.perf_counter() - started
    mass = gradient.detach().abs()[0, 0]
    if not torch.isfinite(mass).all() or float(mass.sum()) <= 0:
        raise RuntimeError("non-finite or zero absolute-gradient mass")
    grid = torch.meshgrid(
        *(torch.arange(size, dtype=torch.float32) for size in shape), indexing="ij"
    )
    coordinates = torch.stack(grid, dim=0)
    stride = torch.tensor(cumulative_stride(model, stage_index), dtype=torch.float32)
    phase_center = stride * torch.tensor(index, dtype=torch.float32)
    centroid = (coordinates * mass.unsqueeze(0)).flatten(1).sum(dim=1) / mass.sum()
    offset = centroid - phase_center
    return {
        "seed": seed,
        "input_shape_zyx": list(shape),
        "feature_shape": list(feature.shape),
        "cell_index_zyx": list(index),
        "channel_aggregation": "signed_mean_all_stage_channels",
        "channel_count": int(feature.shape[1]),
        "nominal_main_phase_center_voxels": [float(value) for value in phase_center],
        "gradient_centroid_voxels": [float(value) for value in centroid],
        "centroid_offset_from_nominal_main_phase_voxels": [float(value) for value in offset],
        "max_abs_offset_voxels": float(offset.abs().max()),
        "r50_voxels": quantile_radius(mass, coordinates, phase_center, 0.50),
        "r90_voxels": quantile_radius(mass, coordinates, phase_center, 0.90),
        "r95_voxels": quantile_radius(mass, coordinates, phase_center, 0.95),
        "gradient_mass": float(mass.sum()),
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
    }


def run(args: argparse.Namespace) -> dict:
    torch.set_num_threads(args.threads)
    checkpoint = args.checkpoint.resolve()
    registry_path = args.registry.resolve()
    checkpoint_sha256 = sha256_file(checkpoint)
    registry_sha256 = sha256_file(registry_path)
    registry = load_registry(registry_path)
    model = build_model(registry, tuple(NATIVE_PLANNER_STRIDES))
    load_result = load_encoder(model, checkpoint)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    stage_index = 3
    stride = cumulative_stride(model, stage_index)
    samples = []
    for sample_index in range(args.samples):
        samples.append(one_sample(model, tuple(args.shape), args.seed + sample_index, stage_index))
        print(f"sample={sample_index + 1}/{args.samples} max_abs_offset={samples[-1]['max_abs_offset_voxels']:.4f}", flush=True)
    offsets = torch.tensor([
        row["centroid_offset_from_nominal_main_phase_voxels"] for row in samples
    ])
    abs_offsets = offsets.abs()
    threshold = float(args.offset_threshold)
    mean_offset = offsets.mean(dim=0)
    abs_mean_offset = mean_offset.abs()
    result = {
        "schema": "self_audit.q3d_resenc.erf_probe.v1",
        "status": "PASS" if float(abs_offsets.max()) <= threshold else "REVIEW_OFFSET_OVER_THRESHOLD",
        "decision_rule": {
            "metric": "absolute mean centroid offset per axis from nominal main-convolution phase",
            "threshold_voxels": threshold,
            "gate": "all(abs(mean_offset_axis)) <= threshold; maximum individual offset is reported only",
            "interpretation": "A measured mean offset above threshold requires using the measured manifest/phase contract; it is not replaced by a theoretical skip-path offset."
        },
        "compute": {
            "device": "cpu",
            "cuda_available": bool(torch.cuda.is_available()),
            "threads": args.threads,
            "samples": args.samples,
            "seed_start": args.seed,
            "random_zscore_input": True,
            "patient_data": "NOT_USED",
            "shape_zyx": list(args.shape),
            "bounded_interior_rationale": "The declared bounded probe shape is used to keep 16 CPU samples feasible; the center cell is used and boundary truncation is reported as a limitation. This is not a patient-volume or full-patch claim."
        },
        "model": {
            "architecture": "ResEncL",
            "registry_path": str(registry_path),
            "registry_sha256": registry_sha256,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "stage_index": stage_index,
            "stage_name": "encoder.stages[3]",
            "stride_schedule_zyx": [list(value) for value in NATIVE_PLANNER_STRIDES],
            "cumulative_stride_zyx": list(stride),
            "phase_reference": "nominal main convolution path; o=0 with symmetric k3/p1 and cumulative stride; actual residual/AvgPool paths are included in the gradient",
            "checkpoint_load": load_result
        },
        "summary": {
            "mean_centroid_offset_voxels": [float(value) for value in mean_offset],
            "absolute_mean_centroid_offset_voxels": [float(value) for value in abs_mean_offset],
            "mean_abs_centroid_offset_voxels": [float(value) for value in abs_offsets.mean(dim=0)],
            "max_abs_centroid_offset_voxels": float(abs_offsets.max()),
            "p95_abs_centroid_offset_voxels": float(torch.quantile(abs_offsets.flatten(), 0.95)),
            "mean_r50_voxels": float(sum(row["r50_voxels"] for row in samples) / len(samples)),
            "mean_r90_voxels": float(sum(row["r90_voxels"] for row in samples) / len(samples)),
            "mean_r95_voxels": float(sum(row["r95_voxels"] for row in samples) / len(samples)),
            "max_r95_voxels": float(max(row["r95_voxels"] for row in samples)),
        },
        "samples": samples,
        "limitations": [
            "The input is synthetic random Z-score noise, not ACDC anatomy; this is software/architecture evidence only.",
            "The probe differentiates the encoder stage-3 center cell and does not measure decoder or patient-dependent effective fields.",
            "InstanceNorm and residual skip paths are included in the actual external architecture; their mixed paths can shift the empirical centroid relative to the nominal main phase.",
            "The bounded shape is smaller than the full theoretical stage-3 support on some axes, so edge truncation can bias the empirical mass and must be checked again on the declared deployment patch/volume policy.",
            "This CPU run does not provide GPU VRAM or throughput evidence."
        ]
    }
    result["status"] = (
        "PASS" if bool(torch.all(abs_mean_offset <= threshold))
        else "REVIEW_MEAN_OFFSET_OVER_THRESHOLD"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--out", type=Path, default=DEFAULT_ROOT / "erf_probe.json")
    parser.add_argument("--md-out", type=Path, default=DEFAULT_ROOT / "erf_probe.md")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--offset-threshold", type=float, default=2.0)
    # Even/divisible dimensions avoid the external residual block's known
    # ceil-vs-floor mismatch for odd stride-2 inputs. This is a bounded probe,
    # not a claim about the variable native patient shapes.
    parser.add_argument("--shape", type=int, nargs=3, default=[32, 96, 96], metavar=("Z", "Y", "X"))
    args = parser.parse_args()
    if args.samples < 1 or args.threads < 1 or any(value < 32 for value in args.shape):
        parser.error("samples, threads, and shape must be positive; each shape axis must be >= 32")
    return args


def main() -> None:
    args = parse_args()
    result = run(args)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = result["summary"]
    args.md_out.write_text(
        "# ResEnc-L empirical receptive-field probe\n\n"
        "This CPU-only probe used deterministic random Z-score inputs and the locked encoder checkpoint. "
        "It is architecture evidence, not a patient experiment.\n\n"
        f"Status: **{result['status']}**. Samples: {result['compute']['samples']}. "
        f"Absolute mean centroid offset from nominal main phase (per-axis max): "
        f"**{max(summary['absolute_mean_centroid_offset_voxels']):.4f} voxels**; threshold: "
        f"{result['decision_rule']['threshold_voxels']:.4f}.\n\n"
        f"Mean centroid offset (Z,Y,X): `{summary['mean_centroid_offset_voxels']}`; "
        f"maximum individual absolute offset (reported, not gated): `{summary['max_abs_centroid_offset_voxels']:.4f}`. "
        f"Mean r90: `{summary['mean_r90_voxels']:.4f}` voxels; mean r95: "
        f"`{summary['mean_r95_voxels']:.4f}` voxels.\n\n"
        "Channel aggregation is the signed mean of all stage-3 channels before differentiation; "
        "the centroid uses absolute input gradients. The full per-sample receipt, checkpoint hash, "
        "registry hash, limitations, and exact command parameters are in `erf_probe.json`.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": result["status"], "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
