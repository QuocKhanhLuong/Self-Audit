#!/usr/bin/env python3
"""Run the canonical two-update GPU pipeline in an isolated profile directory."""
from __future__ import annotations
import argparse
import hashlib
import json
import runpy
import subprocess
import sys
import time
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError("GPU profile requires a new output directory")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU profile requires CUDA")
    args.out.mkdir(parents=True)
    sys.path.insert(0, str(ROOT))
    command = [str(ROOT / "train.py"), "--config", str(args.config), "--out", str(args.out), "--device", "cuda", "--pg-stage", "all", "--pg-max-updates", "2"]
    receipt = {"schema":"self_audit.pg_gpu_pipeline_profile.v1", "command":[sys.executable,*command], "source_sha":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(), "config_sha256":hashlib.sha256(args.config.read_bytes()).hexdigest(), "gpu":torch.cuda.get_device_name(), "torch":torch.__version__, "cuda":torch.version.cuda, "status":"RUNNING", "limits":["Two updates per stage, warmup only for S2 and arms; not peak memory after warmup", "S1 includes its unchanged full TRAIN image-only cosine evaluation", "Separate profile artifacts must never seed the full campaign"]}
    path=args.out/"GPU_PROFILE.json"
    path.write_text(json.dumps(receipt,indent=2)+"\n")
    torch.cuda.reset_peak_memory_stats()
    started=time.monotonic()
    previous=sys.argv
    try:
        sys.argv=command
        runpy.run_path(command[0],run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (None,0):
            receipt["status"]="FAILED";receipt["error"]=str(exc);raise
        receipt["status"]="PASS"
    except BaseException as exc:
        receipt["status"]="FAILED";receipt["error"]=f"{type(exc).__name__}: {exc}";raise
    else:
        receipt["status"]="PASS"
    finally:
        sys.argv=previous
        receipt.update(elapsed_seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        path.write_text(json.dumps(receipt,indent=2)+"\n")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
