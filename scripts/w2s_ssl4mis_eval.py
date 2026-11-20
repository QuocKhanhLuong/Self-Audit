"""Shared evaluation helpers for the SSL4MIS-split campaign (native metrics + SSL4MIS-compatible metrics + complexity)."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from w2s_common import ROOT, clean  # noqa: E402,F401


def predict_records(model: Any, frames: list[Any], device: Any, *, kind: str) -> list[dict[str, Any]]:
    """Argmax predictions on the native grid (fp32, no autocast, as the engine's epoch evaluation); ``kind`` student or teacher."""
    import torch

    from self_audit.candidate_pg import engine

    was_training = model.training
    model.eval()
    records = []
    with torch.no_grad():
        for frame in frames:
            image, target, geometry = engine._frame_tensors(frame, device, labeled=True)
            if kind == "teacher":
                logits = model(image, geometry, return_features=True)["region_logits"]
            else:
                logits = engine._extract_logits(model(image, geometry))
            records.append({"patient": str(frame.patient_id), "frame": int(frame.frame_index), "target": target[0].detach().cpu(),
                            "prediction": logits.argmax(1)[0].detach().cpu(), "geometry": geometry})
    model.train(was_training)
    return records


def native_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """The repo's own metrics (Dice, HD95 in mm with true spacing, ASSD), patient macro, as logged during training."""
    from self_audit.candidate_pg import evaluation

    result = evaluation.evaluate_records(records, path="student_only")
    per = result.get("per_class", {})
    return {"fg_dice": result.get("fg_dice"), "fg_hd95_mm": result.get("fg_hd95"), "fg_assd_mm": result.get("fg_assd"),
            "patient_count": result.get("patient_count"),
            "RV": per.get("RV"), "MYO": per.get("MYO"), "LV": per.get("LV"), "per_patient": result.get("per_patient", {})}


def compat_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    from self_audit.candidate_pg.evaluation_ssl4mis_compat import evaluate_volumes

    rows = [{"patient": r["patient"], "frame": r["frame"], "target": r["target"].numpy(), "prediction": r["prediction"].numpy()} for r in records]
    return evaluate_volumes(rows)


def complexity(model: Any, frame: Any, device: Any) -> dict[str, Any]:
    """#params, FLOPs for one fixed native input, GPU latency (and CPU if requested by availability)."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    from self_audit.candidate_pg import engine

    model.eval()
    image, _, geometry = engine._frame_tensors(frame, device, labeled=True)
    params = sum(p.numel() for p in model.parameters())
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model(image, geometry)
    out: dict[str, Any] = {"parameters": int(params), "flops": int(counter.get_total_flops()), "input_shape": list(image.shape),
                           "flop_counter": "torch.utils.flop_counter (multiply-add counted as 2 FLOPs)"}

    def timed(dev: Any, runs: int) -> float:
        x, _, g = engine._frame_tensors(frame, dev, labeled=True)
        net = model.to(dev)
        with torch.no_grad():
            net(x, g)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            start = time.perf_counter()
            for _ in range(runs):
                net(x, g)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
        return (time.perf_counter() - start) / runs

    if device.type == "cuda":
        out["latency_gpu_s"] = timed(device, 10)
        out["gpu"] = torch.cuda.get_device_name(device)
    cpu = torch.device("cpu")
    out["latency_cpu_s"] = timed(cpu, 2)
    model.to(device)
    return out
