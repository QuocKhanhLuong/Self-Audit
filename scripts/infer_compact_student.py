#!/usr/bin/env python3
"""Run streaming inference with an exported compact student model.

Owned by Worker inference per COMPACT_IMPLEMENTATION_CONTRACT.md:
  - Consumes agreed compact export artifact API from compact_artifacts.
  - Mandatory SHA-256 validation on all manifests, images, and export artifacts.
  - Native manifest-driven input with real invertible affine metadata.
  - Streaming uint8 masks (no full probability cache) with valid_mask ignore-padding support.
  - Explicit normalization contract (raw 0.5/99.5 percentiles vs prepared [0, 1]).
  - Freezes predictions before any mask scoring.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from self_audit.compact_inference import (
    digest,
    infer_cine_stream,
    load_compact_export,
    load_native_manifest,
    load_patient_cine,
    require_sha256,
    validate_export_preprocessing,
    VALID_EVAL_ROLES,
    write_prediction_file,
    write_json,
)


def run_infer(args: argparse.Namespace) -> None:
    clean_export_sha = require_sha256(args.export_sha256, "export_sha256")
    clean_manifest_sha = require_sha256(args.manifest_sha256, "manifest_sha256")

    out = Path(args.out).resolve()
    if out.exists():
        raise ValueError(f"Output directory already exists: {out}")

    manifest_p = Path(args.manifest).resolve()
    manifest_dir = manifest_p.parent
    if out == manifest_dir or out.is_relative_to(manifest_dir) or manifest_dir.is_relative_to(out):
        raise ValueError(f"Output directory cannot overlap with manifest directory: {out}")
    export_dir = Path(args.export).resolve()
    if out == export_dir or out.is_relative_to(export_dir) or export_dir.is_relative_to(out):
        raise ValueError(f"Output directory cannot overlap with export directory: {export_dir}")

    manifest_data, actual_manifest_sha = load_native_manifest(
        args.manifest, clean_manifest_sha, expected_roles=list(VALID_EVAL_ROLES)
    )
    if type(manifest_data.get("synthetic")) is not bool:
        raise ValueError("evaluation manifest synthetic flag must be explicitly boolean")
    patient_records = manifest_data.get("patients", [])
    if not patient_records:
        raise ValueError("Input manifest contains no patients")
    model, export_meta = load_compact_export(args.export, clean_export_sha, device=args.device)
    preprocessing_recipe = validate_export_preprocessing(export_meta, args.normalization)

    out.mkdir(parents=True)
    pred_dir = out / "predictions"
    pred_dir.mkdir(parents=True)

    receipts = []
    frozen_inventory = []

    for prec in patient_records:
        pid = prec["id"]
        cine, valid_mask, cine_meta = load_patient_cine(prec, manifest_dir, normalization=args.normalization)

        pred = infer_cine_stream(
            model=model,
            cine=cine,
            batch_size=args.batch_size,
            device=args.device,
            valid_mask=valid_mask,
        )

        pred_file = f"{pid}_prediction.npy"
        pred_path = (pred_dir / pred_file).resolve()
        if not pred_path.is_relative_to(pred_dir.resolve()):
            raise ValueError(f"Prediction path escapes pred_dir: {pred_path}")

        written = write_prediction_file(pred_path, pred)
        pred_sha = written["sha256"]

        receipt = {
            "patient": pid,
            "dataset": prec.get("domain") or prec.get("dataset") or manifest_data.get("domain") or manifest_data.get("dataset") or "unspecified",
            "image_sha256": cine_meta["image_sha256"],
            "prediction": pred_file,
            "prediction_sha256": pred_sha,
            "shape_tzyx": written["shape_tzyx"],
            "affine": cine_meta["affine"],
            "axes": cine_meta["axes"],
            "normalization": args.normalization,
            "has_valid_mask": cine_meta["has_valid_mask"],
        }
        receipts.append(receipt)
        frozen_inventory.append({
            "id": pid,
            "file": pred_file,
            "sha256": pred_sha,
            "shape_tzyx": written["shape_tzyx"],
            "affine": cine_meta["affine"],
            "axes": cine_meta["axes"],
            "dataset": receipt["dataset"],
            "image_sha256": cine_meta["image_sha256"],
            "has_valid_mask": cine_meta["has_valid_mask"],
        })

    expected_patient_ids = [p["id"] for p in patient_records]
    predicted_patient_ids = [p["id"] for p in frozen_inventory]
    if len(predicted_patient_ids) != len(set(predicted_patient_ids)) or predicted_patient_ids != expected_patient_ids:
        raise ValueError("Inference patient coverage is duplicate or incomplete")

    inference_summary = {
        "export_manifest_sha256": clean_export_sha,
        "export_payload_sha256": export_meta.get("payload_sha256"),
        "class_map": export_meta["class_map"],
        "native_axes": export_meta["native_axes"],
        "manifest_sha256": actual_manifest_sha,
        "synthetic": manifest_data["synthetic"],
        "device": args.device,
        "batch_size": args.batch_size,
        "normalization": args.normalization,
        "preprocessing_recipe": preprocessing_recipe,
        "total_patients": len(receipts),
        "coverage": {
            "expected_patients": expected_patient_ids,
            "predicted_patients": predicted_patient_ids,
            "complete": True,
        },
        "patients": receipts,
    }
    write_json(out / "PREDICTIONS_FROZEN.json", {
        "export_manifest_sha256": clean_export_sha,
        "export_payload_sha256": export_meta.get("payload_sha256"),
        "manifest_sha256": actual_manifest_sha,
        "synthetic": manifest_data["synthetic"],
        "coverage": inference_summary["coverage"],
        "predictions": frozen_inventory,
    })
    write_json(out / "INFERENCE.json", inference_summary)
    print(f"Inference completed for {len(receipts)} patients. Results written to {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export", type=Path, required=True, help="Path to exported compact student directory")
    p.add_argument("--export-sha256", type=str, required=True, help="Mandatory expected SHA-256 of export MANIFEST.json")
    p.add_argument("--manifest", type=Path, required=True, help="Path to input manifest JSON")
    p.add_argument("--manifest-sha256", type=str, required=True, help="Mandatory expected SHA-256 of input manifest JSON")
    p.add_argument("--out", type=Path, required=True, help="New output directory for predictions")
    p.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu or cuda)")
    p.add_argument("--batch-size", type=int, default=16, help="Inference batch size")
    p.add_argument("--normalization", choices=["prepared", "raw"], default="prepared", help="Image normalization mode")
    args = p.parse_args()

    if args.batch_size < 1:
        p.error("batch-size must be at least 1")

    run_infer(args)


if __name__ == "__main__":
    main()
