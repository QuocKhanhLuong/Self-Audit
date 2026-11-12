#!/usr/bin/env python3
"""Phase D: bounded real-data Q3D T/J GPU profile.

Example (run only after the corrected K8 manifests and real T/Q/P artifacts
are available, along with separate TRAIN/VAL geometry repair evidence and a
geometry-bound ERF phase receipt):

    python scripts/q3d_profile_gpu.py --config /path/to/acdc-k8-v1.json \
      --stage-artifact T=/path/to/T.json --stage-artifact Q=/path/to/Q.json \
      --stage-artifact P=/path/to/P.json \
      --stage-artifact-sha T=<sha256> --stage-artifact-sha Q=<sha256> \
      --stage-artifact-sha P=<sha256> \
      --train-source-manifest /path/to/source-train.json \
      --train-source-manifest-sha256 <sha256> --train-repair-report /path/to/train-repair.json \
      --train-repair-report-sha256 <sha256> \
      --dev-source-manifest /path/to/source-dev.json --dev-source-manifest-sha256 <sha256> \
      --dev-repair-report /path/to/dev-repair.json --dev-repair-report-sha256 <sha256> \
      --phase-receipt /path/to/geometry-reprobe.json --phase-receipt-sha256 <sha256> \
      --report /tmp/q3d-phase-d.json \
      --scratch-root /tmp --device cuda:0 --allow-real-profile

This command retains the recipe's full 8,000-update T/J budgets and applies
the engine's update_limit=20 API separately to each measured stage. ``--stage
T`` can profile the teacher before T/Q/P artifacts exist; ``--stage J`` and
the default ``--stage all`` require exact completed T/Q/P artifacts. It never
runs Q/P or accepts synthetic gate overrides.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from self_audit.candidate_q3d.profiling import (  # noqa: E402
    ProfilePrerequisiteError,
    run_profile,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="real ACDC protocol-A K8 V1 recipe with pinned corrected manifests")
    parser.add_argument("--stage", choices=("T", "J", "all"), default="all",
                        help="profile T before T/Q/P, J after verified T/Q/P, or both (default: all)")
    parser.add_argument("--stage-artifact", action="append", default=[], metavar="STAGE=PATH",
                        help="SHA-pinned real stage manifest; supply T, Q and P")
    parser.add_argument("--stage-artifact-sha", action="append", default=[], metavar="STAGE=SHA256",
                        help="manifest SHA-256; supply T, Q and P")
    parser.add_argument("--train-source-manifest", required=True,
                        help="source K8 TRAIN manifest used by the geometry repair")
    parser.add_argument("--train-source-manifest-sha256", required=True,
                        help="source K8 TRAIN manifest SHA-256")
    parser.add_argument("--train-repair-report", required=True,
                        help="successful, hash-bound K8 TRAIN geometry repair report")
    parser.add_argument("--train-repair-report-sha256", required=True,
                        help="K8 TRAIN repair report SHA-256")
    parser.add_argument("--dev-source-manifest", required=True,
                        help="source VAL18 manifest used by its independent geometry repair")
    parser.add_argument("--dev-source-manifest-sha256", required=True,
                        help="source VAL18 manifest SHA-256")
    parser.add_argument("--dev-repair-report", required=True,
                        help="successful, hash-bound VAL18 geometry repair report")
    parser.add_argument("--dev-repair-report-sha256", required=True,
                        help="VAL18 repair report SHA-256")
    parser.add_argument("--phase-receipt", required=True,
                        help="canonical ERF receipt with geometry_reprobe lineage for corrected TRAIN")
    parser.add_argument("--phase-receipt-sha256", required=True,
                        help="geometry-bound ERF receipt SHA-256; must match config phase measurement")
    parser.add_argument("--report", required=True, help="new JSON report path outside the repository and temporary outputs")
    parser.add_argument("--scratch-root", default=tempfile.gettempdir(),
                        help="temporary parent outside the repository/main run (default: system temp directory)")
    parser.add_argument("--device", default="cuda:0", help="CUDA device used for both stages")
    parser.add_argument("--allow-real-profile", action="store_true",
                        help="explicitly authorize the bounded real-data GPU profile")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    measured_stages = {"T": ("T",), "J": ("J",), "all": ("T", "J")}[args.stage]
    try:
        report = run_profile(
            config_path=args.config,
            artifact_paths=args.stage_artifact,
            artifact_hashes=args.stage_artifact_sha,
            source_train_manifest_path=args.train_source_manifest,
            source_train_manifest_sha256=args.train_source_manifest_sha256,
            train_repair_report_path=args.train_repair_report,
            train_repair_report_sha256=args.train_repair_report_sha256,
            source_development_manifest_path=args.dev_source_manifest,
            source_development_manifest_sha256=args.dev_source_manifest_sha256,
            development_repair_report_path=args.dev_repair_report,
            development_repair_report_sha256=args.dev_repair_report_sha256,
            phase_receipt_path=args.phase_receipt,
            phase_receipt_sha256=args.phase_receipt_sha256,
            report_path=args.report,
            scratch_root=args.scratch_root,
            device=args.device,
            allow_real_profile=args.allow_real_profile,
            measured_stages=measured_stages,
        )
    except ProfilePrerequisiteError as exc:
        print(f"Q3D Phase D NOT_RUN: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": report.get("status"),
        "evidence_status": report.get("evidence_status"),
        "report": str(Path(args.report).expanduser().resolve()),
        "failure": report.get("failure"),
    }, sort_keys=True))
    return 0 if report.get("status") == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
