#!/usr/bin/env python3
"""Standalone preflight verification tool for scalar transfer-utility distillation study.

Read-only GPU occupancy check, workspace active jobs check, and immutable protocol asset validation.
Failclosed: records clear NOT_RUN status with exact reasons if assets/endpoints are missing or busy.
Never overwrites existing user output directories.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_transfer_utility_pilot import check_preflight, read_json


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify GPU occupancy, active training processes, and protocol asset integrity before launch."
    )
    parser.add_argument(
        "--protocol",
        default=str(ROOT / "configs/transfer_utility/protocol.json"),
        help="Path to immutable protocol JSON",
    )
    parser.add_argument(
        "--data",
        help="Path to TRAIN dataset manifest JSON",
    )
    parser.add_argument(
        "--data-sha256",
        help="Expected SHA-256 hex string of TRAIN manifest JSON",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        help="Path to teacher checkpoint directory",
    )
    parser.add_argument(
        "--teacher-checkpoint-sha256",
        help="Expected SHA-256 of teacher MANIFEST.json",
    )
    parser.add_argument(
        "--teacher-payload-sha256",
        help="Expected SHA-256 of teacher physical weights payload",
    )
    parser.add_argument(
        "--teacher-cache",
        help="Path to teacher cache directory",
    )
    parser.add_argument(
        "--teacher-cache-sha256",
        help="Expected SHA-256 of CACHE_MANIFEST.json",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Target output directory for preflight receipt or planned pilot run",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to check occupancy on (cuda or cpu)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Permit synthetic mock/smoke validation without physical patient dataset",
    )
    parser.add_argument(
        "--skip-jobs-check",
        action="store_true",
        help="Bypass active jobs preflight check",
    )
    parser.add_argument("--val-manifest")
    parser.add_argument("--val-manifest-sha256")
    args = parser.parse_args()
    if args.skip_jobs_check and not args.synthetic:
        parser.error("real preflight job checks cannot be bypassed")

    protocol_path = Path(args.protocol).resolve()
    if not protocol_path.is_file():
        print(json.dumps({"status": "FAIL", "error": f"Protocol file not found: {protocol_path}"}, indent=2))
        return 2

    out_p = Path(args.out).resolve()
    protocol = read_json(protocol_path)

    receipt = check_preflight(
        protocol=protocol,
        data_manifest=args.data,
        data_manifest_sha256=args.data_sha256,
        out_dir=out_p,
        device=args.device,
        synthetic=args.synthetic,
        check_jobs=not args.skip_jobs_check,
        teacher_checkpoint=args.teacher_checkpoint,
        teacher_checkpoint_sha256=args.teacher_checkpoint_sha256,
        teacher_payload_sha256=args.teacher_payload_sha256,
        teacher_cache=args.teacher_cache,
        teacher_cache_sha256=args.teacher_cache_sha256,
        val_manifest=args.val_manifest, val_manifest_sha256=args.val_manifest_sha256,
    )

    print(json.dumps(receipt, indent=2))
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
