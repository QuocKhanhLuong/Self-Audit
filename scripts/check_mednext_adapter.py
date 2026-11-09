#!/usr/bin/env python3
"""Optional CPU check of the actual local MedNeXt adapter; no pretrained weights.

Requires an already available, clean official checkout. Never downloads source,
weights or data. The random tensor and one backward pass are software checks.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.nn import functional as F
from self_audit.candidate_a.io import source_inventory
from self_audit.candidate_a3d.teacher import UPSTREAM_MAIN_SHA, build_teacher3d, set_teacher_stage
from self_audit.model import digest


def check(source_repo, out):
    out = Path(out).resolve()
    if out.exists():
        raise ValueError("verification receipt must be a new file")
    torch.set_num_threads(1)
    torch.manual_seed(7)
    started = time.perf_counter()
    teacher, provenance = build_teacher3d({"backbone": "mednext_l_k3",
        "source_repo": str(Path(source_repo).resolve()), "source_commit": UPSTREAM_MAIN_SHA,
        "weights": None, "weights_sha256": None, "class_mapping": ["BG", "RV", "MYO", "LV"]})
    native = teacher.encoder.model
    geometry = {"shape": (16, 16, 16), "spacing_zyx": (3.0, 1.0, 0.8),
        "affine_zyx": [[3., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., .8, 0.], [0., 0., 0., 1.]],
        "context": "full_volume"}
    image = torch.rand(1, 1, 16, 16, 16)
    teacher.eval()
    with torch.no_grad():
        outputs = teacher(image, geometry=geometry)
        reference = native(F.pad(image, (0, 16, 0, 16, 0, 16), mode="replicate"))[:, :, :16, :16, :16]
    error = float((outputs["region_logits"] - reference).abs().max())
    if not torch.equal(outputs["region_logits"], reference):
        raise AssertionError(f"native region path differs: {error}")
    if not all(torch.isfinite(value).all() for value in outputs.values()):
        raise AssertionError("nonfinite teacher output")
    set_teacher_stage(teacher, "teacher_pretrain", freeze=False)
    teacher.train()
    image.requires_grad_(True)
    loss = sum(value.square().mean() for value in teacher(image, geometry=geometry).values())
    loss.backward()
    gradients = {"input": image.grad, "stem": native.stem.weight.grad}
    if any(value is None or not torch.isfinite(value).all() for value in gradients.values()):
        raise AssertionError("missing or nonfinite native checkpointed gradients")
    state_hash = hashlib.sha256()
    for name, value in native.state_dict().items():
        state_hash.update(name.encode())
        state_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
    result = {"status": "PASS_RANDOM_INITIALIZATION_SOFTWARE_ONLY", "seed": 7,
        "source_commit": UPSTREAM_MAIN_SHA, "source_repo": str(Path(source_repo).resolve()),
        "source_files": source_inventory(), "check_script_sha256": digest(__file__),
        "native_random_state_sha256": state_hash.hexdigest(),
        "native_parameters": sum(p.numel() for p in native.parameters()),
        "torch": torch.__version__, "python": sys.version, "hardware": platform.platform(),
        "device": "cpu", "input_shape": list(image.shape), "padded_shape": [1, 1, 32, 32, 32],
        "padding_mode": "replicate",
        "output_shapes": {key: list(value.shape) for key, value in outputs.items()},
        "region_parity_exact": True, "region_parity_max_abs": error,
        "checkpointed_backward_finite": True, "provenance": provenance,
        "elapsed_seconds": time.perf_counter() - started,
        "pretrained_weights": "NOT_AVAILABLE", "patient_training": "NOT_RUN", "gpu": "NOT_RUN"}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.source_repo, args.out), indent=2))
