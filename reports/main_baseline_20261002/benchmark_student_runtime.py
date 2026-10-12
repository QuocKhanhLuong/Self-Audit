"""Untrained student engineering probe; never evidence of segmentation quality.

Run from repository root with PYTHONPATH=src. GPU timing synchronizes each
forward. Inputs are already resident; loading, preprocessing, transfers, argmax,
native reconstruction and teacher cost are excluded and must be timed separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from environment_contract import check_environment
from self_audit_pseudolabel.system_v3 import AdaptiveAnnotationStudent, CinePseudoTeacher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--sizes", type=int, nargs="+", default=[224, 256])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started_at = datetime.now(timezone.utc).isoformat()
    if min(args.threads, args.warmup, args.repeats, *args.sizes) < 1:
        raise ValueError("positive probe dimensions and budgets required")
    if args.out.exists():
        raise FileExistsError(args.out)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no fallback")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("probe supports explicit CPU or CUDA only")
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    repo = Path(__file__).resolve().parents[2]
    config_path = repo / "configs/pseudolabel_v3.json"
    cfg = json.loads(config_path.read_text())
    teacher_args = dict(cfg["teacher"])
    teacher_args["k"] = teacher_args.pop("anonymous_prototypes")
    teacher_args.pop("registration_max_displacement")
    teacher = CinePseudoTeacher(**teacher_args)
    deployment = cfg["deployment"]
    model = AdaptiveAnnotationStudent(width=deployment["student_width"],
                                      window_k=deployment["window_k"]).eval().to(device)
    count = lambda module: sum(p.numel() for p in module.parameters())
    parameters = {"teacher_offline": count(teacher), "student_resident": count(model),
                  "compact_encoder_and_a0": count(model.encoder) + count(model.a0_head),
                  "optional_refiner": count(model.refiner)}
    del teacher
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    rows = []
    with torch.inference_mode():
        for size in args.sizes:
            x = torch.randn(1, 3, size, size, device=device)
            for profile, passes in (("compact", 0), ("balanced", 1), ("accurate", 3)):
                for _ in range(args.warmup):
                    model(x, profile=profile)
                sync()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                timings = []
                peak_allocated = peak_reserved = None
                for _ in range(args.repeats):
                    sync()
                    start = time.perf_counter_ns()
                    output = model(x, profile=profile)
                    sync()
                    timings.append((time.perf_counter_ns() - start) / 1e6)
                    assert output["final_logits"].shape == (1, 4, size, size)
                    if device.type == "cuda":
                        peak_allocated = torch.cuda.max_memory_allocated(device)
                        peak_reserved = torch.cuda.max_memory_reserved(device)
                    # Do not keep a previous prediction resident during the next
                    # forward or charge this diagnostic allocation to the scope.
                    if _ == args.repeats - 1:
                        assert torch.isfinite(output["final_logits"]).all()
                    del output
                rows.append({"shape": [1, 3, size, size], "profile": profile,
                             "internal_window_passes": passes,
                             "milliseconds": timings,
                             "p50_ms": float(np.percentile(timings, 50)),
                             "p95_ms": float(np.percentile(timings, 95)),
                             "cuda_peak_allocated_bytes": peak_allocated,
                             "cuda_peak_reserved_bytes": peak_reserved})
    tracked = subprocess.check_output(["git", "ls-files", "src/self_audit", "src/self_audit_pseudolabel",
                                       "configs/pseudolabel_v3.json"], cwd=repo, text=True).splitlines()
    result = {"status": "MEASURED_ENGINEERING_ONLY", "quality_status": "NOT_MEASURED",
              "started_at_utc": started_at,
              "finished_at_utc": datetime.now(timezone.utc).isoformat(),
              "weights": "seed_42_random_untrained", "dice": None,
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
              "source_sha256": {p: hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in tracked},
              "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "environment": check_environment(), "platform": platform.platform(),
              "processor": platform.processor(), "torch": torch.__version__,
              "device": str(device), "device_name": (torch.cuda.get_device_name(device) if device.type == "cuda" else platform.machine()),
              "threads": args.threads, "dtype": "float32", "batch_size": 1,
              "warmup": args.warmup, "repeats": args.repeats, "parameters": parameters,
              "student_fp32_parameter_bytes": parameters["student_resident"]*4,
              "exclusions": ["training", "teacher execution", "I/O", "preprocessing", "host/device transfer",
                             "argmax", "native volume reconstruction", "quality evaluation"],
              "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"parameters": parameters, "device": str(device), "dice": None,
                      "rows": [{k:r[k] for k in ("shape", "profile", "p50_ms", "p95_ms")} for r in rows]}, indent=2))


if __name__ == "__main__":
    main()
