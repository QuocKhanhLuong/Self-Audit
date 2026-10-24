#!/usr/bin/env python3
"""One-volume CineMA adapter smoke/profile with pinned asset verification.

Default mode uses a manufactured mock model and is marked SYNTHETIC. Supplying
all three pinned assets runs one manufactured input volume through the real
strict-loaded upstream model; no masks, patient data, or optimizer are used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from self_audit.cinema_teacher import CineMATeacher, PATCH_SIZE_XYZ


class _MockCineMA(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder_dict = nn.ModuleDict({"sax": nn.Identity()})
        self.pred_head_dict = nn.ModuleDict({"sax": nn.Conv3d(32, 4, kernel_size=1)})

    def set_grad_ckpt(self, _enabled: bool) -> None:
        pass

    def forward(self, image_dict):
        features = self.decoder_dict["sax"](image_dict["sax"].expand(-1, 32, -1, -1, -1))
        return {"sax": self.pred_head_dict["sax"](features)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--timepoint", default="synthetic-0")
    args = parser.parse_args()
    supplied = [args.source_root, args.config, args.weights]
    if any(supplied) and not all(supplied):
        parser.error("--source-root, --config, and --weights must be supplied together")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA device requested but CUDA is unavailable")

    if all(supplied):
        teacher = CineMATeacher(args.config, args.weights, source_root=args.source_root, device=args.device)
        mode = "PINNED_MANUFACTURED_SMOKE"
    else:
        teacher = CineMATeacher(model=_MockCineMA(), device=args.device)
        mode = "SYNTHETIC_MOCK_SMOKE"

    # Manufacture a shape that forces one exact model tile and includes a
    # non-axis-aligned direction so geometry inversion is exercised.
    theta = np.deg2rad(13.0)
    direction = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, -1.0],
    ])
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = direction @ np.diag((1.0, 1.0, 10.0))
    shape = (min(32, PATCH_SIZE_XYZ[0]), min(32, PATCH_SIZE_XYZ[1]), min(3, PATCH_SIZE_XYZ[2]))
    volume = np.random.default_rng(17).normal(size=shape).astype(np.float32)

    if str(teacher.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(teacher.device)
        torch.cuda.synchronize(teacher.device)
    start = time.perf_counter()
    result = teacher.predict(volume, affine, timepoint=args.timepoint)
    if str(teacher.device).startswith("cuda"):
        torch.cuda.synchronize(teacher.device)
    elapsed = time.perf_counter() - start
    is_cuda = teacher.device.type == "cuda"
    peak_allocated = torch.cuda.max_memory_allocated(teacher.device) if is_cuda else None
    peak_reserved = torch.cuda.max_memory_reserved(teacher.device) if is_cuda else None

    report = {
        "status": "PASS",
        "mode": mode,
        "seconds": elapsed,
        "device": str(teacher.device),
        "device_name": torch.cuda.get_device_name(teacher.device) if is_cuda else "CPU",
        "torch_version": torch.__version__,
        "inference_dtype": "float32",
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
        "cuda_memory_measurement": "torch.cuda.max_memory_allocated/max_memory_reserved after synchronize"
        if is_cuda else "NOT_APPLICABLE_CPU",
        "source_commit": result.lineage["source_commit"],
        "probabilities_shape": list(result.probabilities.shape),
        "features_shape": list(result.features.shape),
        "probability_sum_min": float(result.probabilities.sum(axis=0).min()),
        "probability_sum_max": float(result.probabilities.sum(axis=0).max()),
        "geometry": result.geometry,
        "lineage": result.lineage,
        "manufactured_input": True,
        "optimizer_steps": 0,
        "mask_reads": 0,
        "test_data_claim": "NONE",
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
