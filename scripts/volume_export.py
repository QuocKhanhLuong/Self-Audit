#!/usr/bin/env python3
"""Teacher/cache/pretrained-weight-independent native student export."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from self_audit.volume.evaluation import export_student, load_checkpoint_student_only


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--synthetic", action="store_true",
                        help="declare the checkpoint as manufactured synthetic provenance")
    args = parser.parse_args(argv)
    model, manifest = load_checkpoint_student_only(args.checkpoint, args.checkpoint_sha256, device=args.device)
    declared_synthetic = bool((manifest.get("data_provenance") or {}).get("synthetic", False))
    if declared_synthetic != bool(args.synthetic):
        raise ValueError("--synthetic must exactly match checkpoint data provenance")
    sha = export_student(model, args.out, config=manifest["config"],
                         source_checkpoint_sha256=args.checkpoint_sha256,
                         synthetic=declared_synthetic)
    result = {"schema": "self_audit.volume.student_export.v2", "manifest_sha256": sha,
              "output": str(args.out), "prediction_model": "student", "teacher_included": False}
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
