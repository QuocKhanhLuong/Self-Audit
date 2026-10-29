#!/usr/bin/env python3
"""Fair isolated-process benchmarking comparing compact student and baseline.

Owned by Worker inference per COMPACT_IMPLEMENTATION_CONTRACT.md:
  - Streams ALL manifest cases (no preloading all contexts).
  - Actual E2E includes load + preprocess + context extraction + H2D + forward + argmax + D2H + mask write.
  - Separate forward-only timing with CUDA synchronization.
  - Cold load timed separately.
  - Steady-state RAM sampled via psutil outside optimizer payload contamination.
  - Reference model uses deployed weights only (never optimizer buffer bytes).
  - Subprocess isolation per model to cleanly measure process RAM without pollution.
  - Missing real baseline assets explicitly recorded as NOT_RUN.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from self_audit.compact_inference import require_sha256, run_isolated_benchmark


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--compact-export", type=Path, required=True, help="Path to exported compact student directory")
    p.add_argument("--compact-export-sha256", type=str, required=True, help="Mandatory expected SHA-256 of compact export manifest")
    p.add_argument("--manifest", type=Path, required=True, help="Path to input manifest JSON")
    p.add_argument("--manifest-sha256", type=str, required=True, help="Mandatory expected SHA-256 of input manifest JSON")
    p.add_argument("--baseline-checkpoint", type=Path, default=None, help="Optional path to baseline checkpoint directory")
    p.add_argument("--baseline-checkpoint-sha256", type=str, default=None, help="Expected SHA-256 of baseline checkpoint manifest")
    p.add_argument("--baseline-state", choices=["student"], default="student", help="Locked historical raw-student inference state")
    p.add_argument("--historical-source", type=Path, default=None, help="Explicit verified historical source checkout required for reference benchmarking")
    p.add_argument("--baseline-teacher-identity", type=Path, default=None, help="Verified canonical teacher identity JSON for a recovered EMA reference")
    p.add_argument("--baseline-teacher-identity-sha256", type=str, default=None, help="Exact SHA-256 of canonical teacher identity JSON")
    p.add_argument("--out", type=Path, required=True, help="Output JSON path for benchmark report")
    p.add_argument("--device", type=str, default="cpu", help="Device to benchmark on (cpu or cuda)")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size for 2.5D slices")
    p.add_argument("--warmup", type=int, default=2, help="Number of warmup iterations across manifest")
    p.add_argument("--repeats", type=int, default=5, help="Number of steady state repeats across manifest")
    p.add_argument("--normalization", choices=["prepared", "raw"], default="prepared", help="Image normalization mode")
    args = p.parse_args()

    if args.batch_size < 1:
        p.error("batch-size must be at least 1")
    if args.warmup < 0:
        p.error("warmup must be non-negative")
    if args.repeats < 1:
        p.error("repeats must be at least 1")

    require_sha256(args.compact_export_sha256, "compact_export_sha256")
    require_sha256(args.manifest_sha256, "manifest_sha256")
    if args.baseline_checkpoint_sha256 is not None:
        require_sha256(args.baseline_checkpoint_sha256, "baseline_checkpoint_sha256")
    if args.baseline_teacher_identity is not None or args.baseline_teacher_identity_sha256 is not None:
        if args.baseline_teacher_identity is None or args.baseline_teacher_identity_sha256 is None:
            p.error("canonical baseline requires both identity path and SHA-256")
        require_sha256(args.baseline_teacher_identity_sha256, "baseline_teacher_identity_sha256")

    report = run_isolated_benchmark(
        compact_export_path=args.compact_export,
        compact_export_sha=args.compact_export_sha256,
        manifest_path=args.manifest,
        manifest_sha=args.manifest_sha256,
        out_path=args.out,
        baseline_checkpoint_path=args.baseline_checkpoint,
        baseline_checkpoint_sha=args.baseline_checkpoint_sha256,
        baseline_state=args.baseline_state,
        historical_source_path=args.historical_source,
        baseline_teacher_identity_path=args.baseline_teacher_identity,
        baseline_teacher_identity_sha=args.baseline_teacher_identity_sha256,
        device=args.device,
        batch_size=args.batch_size,
        warmup=args.warmup,
        repeats=args.repeats,
        normalization=args.normalization,
    )

    print("\n=== Benchmark Summary ===")
    compact_res = report.get("compact_student", {})
    if compact_res.get("status") == "COMPLETED":
        print("Compact Student:")
        print(f"  Params:           {compact_res['model_size']['total_parameters']:,}")
        print(f"  Deployed Weights: {compact_res['model_size']['deployed_weights_bytes']:,} bytes")
        print(f"  Peak RAM:         {compact_res['memory']['peak_inference_process_tree_ram_mb']:.2f} MB")
        if compact_res["memory"].get("cuda_peak_allocated_mb") is not None:
            print(f"  CUDA Alloc:       {compact_res['memory']['cuda_peak_allocated_mb']:.2f} MB")
        print(f"  Cold Load:        {compact_res['latency']['cold']['load_seconds']:.4f} s")
        print(f"  Cold Model/Vol:   {compact_res['latency']['cold']['model_only_median_seconds_per_volume']:.4f} s")
        print(f"  Cold E2E/Vol:     {compact_res['latency']['cold']['e2e_median_seconds_per_volume']:.4f} s")
        print(f"  Steady Model/Vol: {compact_res['latency']['steady']['model_only_median_seconds_per_volume']:.4f} s (p95: {compact_res['latency']['steady']['model_only_p95_seconds_per_volume']:.4f} s)")
        print(f"  Steady E2E/Vol:   {compact_res['latency']['steady']['e2e_median_seconds_per_volume']:.4f} s (p95: {compact_res['latency']['steady']['e2e_p95_seconds_per_volume']:.4f} s)")
        print(f"  Throughput:       {compact_res['latency']['steady']['volumes_per_second']:.2f} vol/s ({compact_res['latency']['steady']['slices_per_second']:.1f} slices/s)")

    base_res = report.get("baseline_student", {})
    print(f"\nBaseline Student Status: {base_res.get('status')}")
    if base_res.get("status") == "COMPLETED":
        print(f"  Params:           {base_res['model_size']['total_parameters']:,}")
        print(f"  Deployed Weights: {base_res['model_size']['deployed_weights_bytes']:,} bytes")
        print(f"  Steady RAM:       {base_res['memory']['peak_inference_process_tree_ram_mb']:.2f} MB")
        print(f"  Steady Model/Vol: {base_res['latency']['steady']['model_only_median_seconds_per_volume']:.4f} s")
    else:
        print(f"  Reason:           {base_res.get('reason')}")

    comp = report.get("comparison", {})
    if comp:
        print("\nComparison (Speedup & Reduction):")
        for k, v in comp.items():
            if v is not None:
                print(f"  {k}: {v:.2f}x")

    print(f"\nFull report written to {args.out}")


if __name__ == "__main__":
    main()
