#!/usr/bin/env python3
"""Evaluate an exported compact student model on native evaluation data.

Owned by Worker inference per COMPACT_IMPLEMENTATION_CONTRACT.md:
  - Two-phase frozen evaluation: freeze all predictions before opening reference masks.
  - Mandatory TRAIN provenance verification: ID disjointness and physical byte/voxel overlap checks.
  - Mandatory SHA-256 validation for export, evaluation manifest, and TRAIN manifest.
  - Exact historical class-first patient macro metrics:
      frame -> patient/class -> classwise patientmeans -> meanclass Macro4 & FG.
      Omits Dice when GT is absent even with false positives.
  - Reports extra patient-first aggregate (equal on full cohort, distinct when class absent).
  - Explicit absent-class FP tracking with fixed observed valid-voxel count denominator.
  - Strict coverage and failure case tracking.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from self_audit.compact_inference import require_sha256, run_two_phase_evaluation


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export", type=Path, required=True, help="Path to exported compact student directory")
    p.add_argument("--export-sha256", type=str, required=True, help="Mandatory expected SHA-256 of export MANIFEST.json")
    p.add_argument("--manifest", type=Path, required=True, help="Path to evaluation manifest JSON")
    p.add_argument("--manifest-sha256", type=str, required=True, help="Mandatory expected SHA-256 of evaluation manifest JSON")
    p.add_argument("--train-manifest", type=Path, required=True, help="Mandatory TRAIN manifest to verify disjointness and voxel overlap")
    p.add_argument("--train-manifest-sha256", type=str, required=True, help="Mandatory expected SHA-256 of TRAIN manifest")
    p.add_argument("--out", type=Path, required=True, help="New output directory for evaluation results")
    p.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu or cuda)")
    p.add_argument("--batch-size", type=int, default=16, help="Inference batch size")
    p.add_argument("--normalization", choices=["prepared", "raw"], default="prepared", help="Image normalization mode")
    args = p.parse_args()

    if args.batch_size < 1:
        p.error("batch-size must be at least 1")

    # Validate mandatory SHA-256 pins
    require_sha256(args.export_sha256, "export_sha256")
    require_sha256(args.manifest_sha256, "manifest_sha256")
    require_sha256(args.train_manifest_sha256, "train_manifest_sha256")

    metrics = run_two_phase_evaluation(
        export_path=args.export,
        export_sha256=args.export_sha256,
        manifest_path=args.manifest,
        manifest_sha256=args.manifest_sha256,
        train_manifest_path=args.train_manifest,
        train_manifest_sha256=args.train_manifest_sha256,
        out_dir=args.out,
        device=args.device,
        batch_size=args.batch_size,
        normalization=args.normalization,
    )

    print("\n=== Evaluation Results ===")
    print(f"Class-First Macro4 (historical exact): {metrics.get('class_first_macro4')}")
    print(f"Class-First FG Macro (RV, MYO, LV):   {metrics.get('class_first_foreground_macro')}")
    print(f"Patient-First Macro4:                  {metrics.get('patient_first_all_class_macro')}")
    print(f"Patient-First FG Macro:                {metrics.get('patient_first_foreground_macro')}")
    print(f"Synthetic evidence:                     {metrics.get('synthetic')}")
    print("\nPer-Dataset Class-First Macro4:")
    for dataset, report in metrics.get("datasets", {}).items():
        print(f"  {dataset}: {report.get('class_first_macro4')}")
    print("\nPer-Class Macro Metrics:")
    for c_name, c_data in metrics.get("macro_patient", {}).items():
        print(f"  {c_name:4s}: Dice = {c_data.get('dice')}, Patients = {c_data.get('present_patients')}")
    print(f"\nFull metrics written to {args.out / 'METRICS.json'}")


if __name__ == "__main__":
    main()
