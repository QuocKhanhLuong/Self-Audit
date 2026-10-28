#!/usr/bin/env python3
"""Export a standalone compact student model from a training checkpoint.

Requires only PyTorch and standard library; does not import scipy, DINO, or training assets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path for standalone invocation
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from self_audit.compact_artifacts import export_compact_student


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a standalone, teacher/DINO-independent compact student model."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to compact training checkpoint directory",
    )
    parser.add_argument(
        "--checkpoint-sha256",
        required=True,
        help="Mandatory expected 64-hex SHA-256 hash of the checkpoint MANIFEST.json",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Destination directory for exported student artifact",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    out_path = Path(args.out).resolve()

    manifest_sha = export_compact_student(
        checkpoint=checkpoint_path,
        expected_sha=args.checkpoint_sha256,
        out=out_path,
    )

    manifest_file = out_path / "MANIFEST.json"
    manifest_data = json.loads(manifest_file.read_text())

    receipt = {
        "status": "EXPORTED",
        "schema": manifest_data.get("schema"),
        "manifest_sha256": manifest_sha,
        "checkpoint": str(checkpoint_path),
        "out": str(out_path),
        "weights_only_bytes": manifest_data.get("weights_only_bytes"),
        "serialization_bytes": manifest_data.get("serialization_bytes"),
        "total_parameters": manifest_data.get("parameters_total"),
        "trainable_parameters": manifest_data.get("parameters_trainable"),
        "teacher_included": manifest_data.get("teacher_included", False),
        "dino_required": manifest_data.get("dino_required", False),
        "model_file_sha256": manifest_data.get("model_file_sha256"),
    }
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
