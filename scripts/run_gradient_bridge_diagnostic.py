#!/usr/bin/env python3
"""Run the locked image-only gradient-bridge diagnostic B lifecycle.

Run after context diagnostic A has created and frozen its predictions:
preflight -> fit-p -> lock-b -> predict -> score -> report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from self_audit.gradient_bridge_runtime import (  # noqa: E402
    fit_p_stage,
    lock_b_stage,
    predict_stage,
    preflight_stage,
    report_stage,
    score_stage,
)


def _common(parser: argparse.ArgumentParser, *, config: bool = True) -> None:
    if config:
        parser.add_argument("--config", required=True, help="validated gradient bridge runtime JSON")
        parser.add_argument("--config-sha256", help="optional expected SHA-256 for config bytes")
    parser.add_argument("--round-dir", required=True, help="shared A/B diagnostic round directory")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (
        ("preflight", "verify hashes, lineage, splits, and image overlap without reading GT payloads"),
        ("fit-p", "fit the bounded projector using TRAIN images only"),
        ("lock-b", "lock B after verifying A is frozen and fit-P is complete"),
        ("predict", "stream frozen D0-D5/rho logits from images only"),
        ("score", "authorize the common A/B freeze before opening any GT masks"),
    ):
        sub = commands.add_parser(name, help=help_text)
        _common(sub)
    report = commands.add_parser("report", help="write the shared round JSON and Markdown report")
    _common(report, config=False)
    fixture = commands.add_parser("fixture-smoke", help="run both actual CLIs on serialized synthetic models, CPU only")
    _common(fixture, config=False)

    args = parser.parse_args(argv)
    try:
        if args.command == "fixture-smoke":
            from self_audit.diagnostic_fixture import run_fixture_smoke
            print(json.dumps(run_fixture_smoke(args.round_dir), sort_keys=True, allow_nan=False))
            return 0
        if args.command == "preflight":
            result = preflight_stage(args.config, args.round_dir, expected_config_sha256=args.config_sha256)
        elif args.command == "fit-p":
            result = fit_p_stage(args.config, args.round_dir, expected_config_sha256=args.config_sha256)
        elif args.command == "lock-b":
            result = lock_b_stage(args.config, args.round_dir, expected_config_sha256=args.config_sha256)
        elif args.command == "predict":
            result = predict_stage(args.config, args.round_dir, expected_config_sha256=args.config_sha256)
        elif args.command == "score":
            result = score_stage(args.config, args.round_dir, expected_config_sha256=args.config_sha256)
        else:
            result = report_stage(args.round_dir)
    except FloatingPointError as exc:
        print(json.dumps({"status": "INVALID_DIAGNOSTIC", "reason": "INVALID_NONFINITE",
                          "command": args.command, "error": str(exc)}, sort_keys=True))
        return 2
    except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        print(json.dumps({"status": "BLOCKED", "command": args.command, "error": str(exc)}, sort_keys=True))
        return 2

    if args.command == "preflight":
        summary = {
            "status": result["status"],
            "profile": result["profile"],
            "strict_weights_loaded": result["strict_weights_loaded"],
            "ground_truth_payloads_opened": result["ground_truth_payloads_opened"],
            "train_patients": len(result["lineage"]["train_patient_ids"]),
            "evaluation_patients": len(result["lineage"]["evaluation_patient_ids"]),
        }
    elif args.command == "fit-p":
        summary = {key: result[key] for key in ("status", "artifact_path", "artifact_sha256", "artifact_identity")}
    elif args.command == "lock-b":
        summary = {"status": "LOCKED", "stage": "B", "record_sha256": result["record_sha256"]}
    elif args.command == "predict":
        summary = {key: result[key] for key in ("status", "prediction_receipt_sha256", "freeze_sha256", "shards", "frames", "D0_parity", "h_dependence")}
    elif args.command == "score":
        summary = {
            "status": result["status"],
            "primary_rho": result["rho_primary"],
            "gate_status": result.get("fixture_gate_status") or result.get("rho_scores", {}).get(str(result["rho_primary"]), {}).get("gate", {}).get("status"),
            "exposure_ledger_sha256": result["exposure_ledger_sha256"],
            "synthetic_fixture": result["synthetic_fixture"],
        }
    else:
        summary = {
            "status": result["status"],
            "report_json": str((Path(args.round_dir).resolve() / "REPORT.json")),
            "report_markdown": str((Path(args.round_dir).resolve() / "REPORT.md")),
        }
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
