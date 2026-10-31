#!/usr/bin/env python3
"""CLI for building compact teacher target cache on unlabeled TRAIN images.

Enforces:
  1. Image-only U access (no mask files or labels ever opened).
  2. Streaming open_memmap output batch-by-batch (no full volume RAM allocation).
  3. Bounded memmapped array files with strict SHA256 integrity.
  4. Explicit --historical-source path for historical baseline loading.
  5. Latency, VRAM, and disk bytes reporting.
  6. Real missing assets BLOCKED (fails closed on unverified/missing assets).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import torch

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from self_audit.compact_data import CompactTrainingData
from self_audit.compact_teacher import (
    HISTORICAL_BASELINE_PINS,
    build_teacher_cache,
    load_teacher_for_cache,
    validate_reproduction_receipt,
    validate_teacher_identity,
    verify_historical_source,
    verify_reproduction_source,
)
from self_audit.model import build_model, digest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build compact teacher probability cache.")
    parser.add_argument("--data-manifest", type=str, required=True, help="Path to training data manifest.")
    parser.add_argument("--data-sha256", type=str, required=True, help="Expected training manifest SHA256.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save cache files.")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Path to teacher checkpoint directory.")
    parser.add_argument("--use-historical-baseline", action="store_true", help="Use pinned historical baseline teacher.")
    parser.add_argument("--historical-source", type=str, default=None, help="Explicit path to historical source checkout.")
    parser.add_argument("--synthetic-teacher", action="store_true", help="Use synthetic tiny teacher (for testing).")
    parser.add_argument("--precision", type=str, default="float32", choices=["float32", "float16"], help="Cache dtype.")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device (e.g. cpu).")
    parser.add_argument("--batch-size", type=int, default=8, help="Slice batch size.")
    parser.add_argument("--allow-k1", action="store_true", default=True, help="Allow K=1 training data.")

    parser.add_argument("--teacher-identity", help="Pinned identity JSON for a matching-budget canonical teacher")
    parser.add_argument("--teacher-identity-sha256")
    parser.add_argument("--reproduction-receipt", type=str, default=None, help="Path to reproduction teacher receipt JSON.")
    parser.add_argument("--reproduction-receipt-sha256", type=str, default=None, help="Expected SHA256 of reproduction receipt.")
    parser.add_argument("--reproduction-source", type=str, default=None, help="Explicit path to reproduction source repository.")
    args = parser.parse_args()

    manifest_path = Path(args.data_manifest).resolve()
    data_sha = args.data_sha256

    # Read manifest to detect synthetic mode
    m_raw = json.loads(manifest_path.read_text())
    is_synthetic = bool(m_raw.get("synthetic", False))

    data = CompactTrainingData(
        manifest=manifest_path,
        expected_sha256=data_sha,
        synthetic=is_synthetic,
        allow_k1=args.allow_k1,
    )

    if args.synthetic_teacher:
        if not is_synthetic:
            raise ValueError("synthetic teacher requires explicitly synthetic TRAIN data")
        torch.manual_seed(217)
        # Synthetic teacher for testing
        teacher_model, _ = build_model({"backbone": "tiny_test"}, synthetic=True)
        for p in teacher_model.parameters():
            p.requires_grad_(False)
        teacher_model.eval()

        teacher_id = {
            "schema": "self_audit.compact_teacher_identity.v1",
            "teacher_kind": "synthetic",
            "inference_state": "student",
            "label_budget": len(data.lids),
            "labeled_ids": list(data.lids),
        }
    elif args.reproduction_receipt:
        if not args.reproduction_receipt_sha256:
            parser.error("--reproduction-receipt requires --reproduction-receipt-sha256")
        if not args.checkpoint_dir:
            parser.error("--reproduction-receipt requires --checkpoint-dir")
        if not args.reproduction_source:
            parser.error("--reproduction-receipt requires explicit --reproduction-source")
        receipt_path = Path(args.reproduction_receipt).resolve()
        teacher_id = validate_reproduction_receipt(
            receipt_path,
            expected_receipt_sha256=args.reproduction_receipt_sha256,
            checkpoint_dir=args.checkpoint_dir,
        )
        teacher_id["teacher_kind"] = "reproduction"
        validate_teacher_identity(teacher_id, data=data)
        teacher_model = load_teacher_for_cache(
            args.checkpoint_dir,
            teacher_identity=teacher_id,
            device=args.device,
            reproduction_source_path=args.reproduction_source,
            synthetic=is_synthetic,
        )
    elif args.teacher_identity:
        if not args.teacher_identity_sha256 or digest(args.teacher_identity) != args.teacher_identity_sha256:
            raise ValueError("teacher identity requires its exact SHA256")
        if not args.checkpoint_dir:
            parser.error("--teacher-identity requires --checkpoint-dir")
        teacher_id = json.loads(Path(args.teacher_identity).read_text())
        validate_teacher_identity(teacher_id, data=data)
        teacher_model = load_teacher_for_cache(
            args.checkpoint_dir, teacher_identity=teacher_id, device=args.device,
            historical_source_path=args.historical_source, synthetic=is_synthetic)
    elif args.use_historical_baseline or args.checkpoint_dir:
        ckpt_dir = args.checkpoint_dir or "reports/window_pilot_test50_20261024/baseline"
        if not args.historical_source:
            raise ValueError(
                "loading historical baseline requires explicit --historical-source path; "
                "historical source closure must be verified"
            )

        teacher_id = dict(HISTORICAL_BASELINE_PINS)
        teacher_id["teacher_kind"] = "historical_baseline"
        teacher_id["historical_bridge"] = True

        validate_teacher_identity(teacher_id, data=data)
        teacher_model = load_teacher_for_cache(
            checkpoint_dir=ckpt_dir,
            teacher_identity=teacher_id,
            device=args.device,
            historical_source_path=args.historical_source,
            synthetic=is_synthetic,
        )
    else:
        parser.error("must specify --synthetic-teacher, --use-historical-baseline, or --checkpoint-dir")

    receipt = build_teacher_cache(
        model=teacher_model,
        data=data,
        output_dir=args.output_dir,
        teacher_identity=teacher_id,
        device=args.device,
        batch_size=args.batch_size,
        precision=args.precision,
    )

    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
