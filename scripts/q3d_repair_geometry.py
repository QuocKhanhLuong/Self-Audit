#!/usr/bin/env python3
"""Audit Q3D manifest geometry from original CINE NIfTI headers.

This command is header-only. It hashes source files but never loads voxel or
mask arrays. The default mode is audit-only and writes no corrected manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from self_audit.candidate_q3d.geometry_repair import repair_manifest_geometry


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(path: Path) -> dict[str, Path]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value:
        raise ValueError("CINE mapping JSON must be a non-empty patient_id -> path object")
    result = {}
    for patient_id, raw_path in value.items():
        if not isinstance(patient_id, str) or not patient_id:
            raise ValueError("CINE mapping patient IDs must be non-empty strings")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"CINE path for {patient_id} must be a non-empty string")
        candidate = Path(raw_path)
        result[patient_id] = (candidate if candidate.is_absolute() else path.parent / candidate).resolve()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--cine-map", required=True, type=Path,
                        help="JSON object mapping manifest patient IDs to original CINE NIfTI paths")
    parser.add_argument("--cine-map-provenance", required=True,
                        help="declared provenance for how patient IDs were mapped to original CINE files")
    parser.add_argument("--axis-map", required=True, choices=("XYZ->ZYX",))
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument("--report-csv", required=True, type=Path)
    parser.add_argument("--unit-declarations-json", type=Path,
                        help="JSON patient map to {unit, provenance}; required for unknown header units")
    parser.add_argument("--write-corrected-manifest", action="store_true",
                        help="explicitly emit a new manifest after this manifest passes audit")
    parser.add_argument("--output-manifest", type=Path,
                        help="new output path required with --write-corrected-manifest")
    args = parser.parse_args(argv)

    if args.write_corrected_manifest and args.output_manifest is None:
        parser.error("--output-manifest is required with --write-corrected-manifest")
    if not args.write_corrected_manifest and args.output_manifest is not None:
        parser.error("--output-manifest requires --write-corrected-manifest")

    mapping_path = args.cine_map.resolve()
    declarations: dict[str, Any] = {}
    if args.unit_declarations_json is not None:
        declarations_value = json.loads(args.unit_declarations_json.read_text(encoding="utf-8"))
        if not isinstance(declarations_value, dict):
            parser.error("--unit-declarations-json must contain a patient object")
        declarations = declarations_value
    report = repair_manifest_geometry(
        args.source_manifest,
        args.source_manifest_sha256,
        _mapping(mapping_path),
        axis_map=args.axis_map,
        cine_mapping_source=str(mapping_path),
        cine_mapping_sha256=_sha256(mapping_path),
        cine_mapping_provenance=args.cine_map_provenance,
        report_json_path=args.report_json,
        report_csv_path=args.report_csv,
        output_manifest_path=args.output_manifest,
        unit_declarations_by_patient=declarations,
        audit_only=not args.write_corrected_manifest,
    )
    print(json.dumps({
        "status": report["status"],
        "audit_rows": len(report["audit_rows"]),
        "errors": len(report["errors"]),
        "report_json": str(args.report_json),
        "report_csv": str(args.report_csv),
        "output_manifest": str(args.output_manifest) if report["status"] == "success" else None,
        "receipt": report.get("receipt"),
    }, indent=2, allow_nan=False))
    return 0 if report["status"] in {"success", "audit_only_pass"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
