# SPDX-License-Identifier: GPL-3.0
"""Official-code-derived per-image optimization, preserving stopping/update order."""
import hashlib
import random
import time

import numpy as np
import torch

from .model import CSNet
from .losses import loss_terms


def sample_seed(base_seed, sample_id):
    material = f"{base_seed}\0{sample_id}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def predict_native(image_bgr, settings, *, seed, device="cpu"):
    if image_bgr.dtype != np.uint8 or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("native SGSCN input must be OpenCV uint8 BGR")
    if min(image_bgr.shape[:2]) < 2:
        raise ValueError("native image needs spatial neighbors")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    data = torch.from_numpy(np.array([image_bgr.transpose(2, 0, 1).astype("float32") / 255.])).to(device)
    model = CSNet(3, n_channels=settings["n_channels"], n_conv=settings["n_conv"]).to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=settings["learning_rate"], momentum=settings["momentum"])
    trajectory = []
    started = time.perf_counter()
    stop_reason = "max_iterations"
    for index in range(settings["max_iterations"]):
        optimizer.zero_grad()
        logits = model(data)[0]
        loss, target, terms = loss_terms(logits, settings)
        preupdate_count = int(torch.unique(target).numel())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite original SGSCN loss at iteration {index}; no repair applied")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError(f"nonfinite original gradient at iteration {index}; no repair applied")
        optimizer.step()
        trajectory.append({"iteration": index, "preupdate_active_labels": preupdate_count,
                           "loss": float(loss.detach().cpu()), **{k: float(v.detach().cpu()) for k, v in terms.items()}})
        if preupdate_count <= settings["min_labels"]:
            stop_reason = "min_labels"
            break
    # Official final forward is train-mode and happens AFTER the last optimizer update.
    with torch.no_grad():
        final = model(data)[0]
    if not torch.isfinite(final).all():
        raise FloatingPointError("nonfinite final output; no repair applied")
    partition = final.argmax(dim=0).cpu().numpy().astype(np.int32)
    receipt = {"seed": seed, "iterations": len(trajectory), "stop_reason": stop_reason,
               "final_active_labels": int(len(np.unique(partition))), "trajectory": trajectory,
               "final_forward_mode": "train", "seconds": time.perf_counter() - started,
               "torch_version": torch.__version__, "device": str(device),
               "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
               "cudnn_deterministic": torch.backends.cudnn.deterministic,
               "cudnn_benchmark": torch.backends.cudnn.benchmark}
    return partition, receipt
