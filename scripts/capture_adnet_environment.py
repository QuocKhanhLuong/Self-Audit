#!/usr/bin/env python3
"""Capture ADNet audit environment provenance for GPU execution receipts."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shared_benchmark.adnet_fewshot import atomic_write_json, file_sha256  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adnet-root", type=Path, default=ROOT / "baseline" / "ADNet")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--asset-spec", type=Path)
    parser.add_argument("--query-manifest", type=Path)
    parser.add_argument("--support-manifest", type=Path)
    parser.add_argument("--gt-manifest", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt = capture_environment(
        adnet_root=args.adnet_root,
        checkpoint=args.checkpoint,
        asset_spec=args.asset_spec,
        query_manifest=args.query_manifest,
        support_manifest=args.support_manifest,
        gt_manifest=args.gt_manifest,
    )
    atomic_write_json(args.output, receipt)
    print(json.dumps({"status": "COMPLETE", "output": str(args.output)}, sort_keys=True))
    return 0


def capture_environment(
    *,
    adnet_root: Path,
    checkpoint: Path | None = None,
    asset_spec: Path | None = None,
    query_manifest: Path | None = None,
    support_manifest: Path | None = None,
    gt_manifest: Path | None = None,
) -> dict[str, Any]:
    torch_info = _torch_info()
    receipt: dict[str, Any] = {
        "schema": "adnet.environment_receipt.v1",
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
        },
        "repository": _git_identity(ROOT),
        "adnet_source": _git_identity(adnet_root),
        "torch": torch_info,
        "nvidia_smi": _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        "artifacts": {},
    }
    for key, path in {
        "checkpoint": checkpoint,
        "asset_spec": asset_spec,
        "query_manifest": query_manifest,
        "support_manifest": support_manifest,
        "gt_manifest": gt_manifest,
    }.items():
        if path is not None:
            resolved = Path(path).resolve()
            receipt["artifacts"][key] = {
                "path": str(resolved),
                "exists": resolved.exists(),
                "sha256": file_sha256(resolved) if resolved.is_file() else None,
            }
    return receipt


def _torch_info() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"importable": False, "error": str(exc)}
    cuda_available = bool(torch.cuda.is_available())
    devices = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append({
                "index": index,
                "name": props.name,
                "total_memory": int(props.total_memory),
                "capability": [int(value) for value in props.major_minor] if hasattr(props, "major_minor") else [
                    int(props.major),
                    int(props.minor),
                ],
            })
    return {
        "importable": True,
        "version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "cudnn_version": torch.backends.cudnn.version(),
        "devices": devices,
    }


def _git_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(Path(path).resolve()),
        "commit": _run(["git", "-C", str(path), "rev-parse", "HEAD"])["stdout"].strip() or None,
        "branch": _run(["git", "-C", str(path), "branch", "--show-current"])["stdout"].strip() or None,
        "dirty": bool(_run(["git", "-C", str(path), "status", "--porcelain"])["stdout"].strip()),
        "remote_origin": _run(["git", "-C", str(path), "config", "--get", "remote.origin.url"])["stdout"].strip() or None,
    }


def _run(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=15)
    except Exception as exc:
        return {"command": command, "returncode": None, "stdout": "", "stderr": str(exc)}
    return {
        "command": command,
        "returncode": int(completed.returncode),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


if __name__ == "__main__":
    raise SystemExit(main())
