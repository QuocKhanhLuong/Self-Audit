#!/usr/bin/env python3
"""CLI for compact teacher lineage and identity preflight.

Enforces:
  1. Teacher lineage <= K across all labeled IDs before IO.
  2. Exact match with declared IDs and binding to actual student manifest.
  3. Frozen full-EMA teacher identity and raw student state validation.
  4. Historical source checkout closure verification.
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

from self_audit.compact_data import CompactTrainingData
from self_audit.compact_teacher import (
    HISTORICAL_BASELINE_PINS,
    preflight_teacher_lineage,
    validate_teacher_identity,
    verify_historical_source,
)
from self_audit.model import digest


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight compact teacher lineage and identity.")
    parser.add_argument("--identity-json", type=str, help="Path to teacher identity JSON file.")
    parser.add_argument("--use-historical-baseline", action="store_true", help="Use pinned historical baseline.")
    parser.add_argument("--historical-source", type=str, default=None, help="Path to historical source checkout.")
    parser.add_argument("--student-manifest", type=str, help="Path to student training data manifest.")
    parser.add_argument("--teacher-labeled-ids", nargs="*", default=None, help="Teacher labeled patient IDs.")
    parser.add_argument("--student-labeled-ids", nargs="*", default=None, help="Student labeled patient IDs.")
    parser.add_argument("--audit-labeled-ids", nargs="*", default=None, help="Audit labeled patient IDs.")
    parser.add_argument("--warmstart-labeled-ids", nargs="*", default=None, help="Warm-start labeled patient IDs.")
    parser.add_argument("--auxiliary-labeled-ids", nargs="*", default=None, help="Auxiliary labeled patient IDs.")
    parser.add_argument("--label-budget", type=int, help="Declared total label budget K.")
    parser.add_argument("--declared-labeled-ids", nargs="*", default=None, help="Exact declared labeled IDs.")

    parser.add_argument("--student-manifest-sha256")
    parser.add_argument("--identity-sha256")
    args = parser.parse_args()

    # 1. Historical source verification if supplied
    historical_receipt = None
    if args.historical_source:
        historical_receipt = verify_historical_source(args.historical_source)

    # 2. Identity validation
    identity = None
    if args.use_historical_baseline:
        identity = dict(HISTORICAL_BASELINE_PINS)
        identity["teacher_kind"] = "historical_baseline"
        identity["historical_bridge"] = True
        identity = validate_teacher_identity(identity, allow_historical_bridge=True)
    elif args.identity_json:
        if not args.identity_sha256 or digest(args.identity_json) != args.identity_sha256:
            parser.error("identity-json requires matching --identity-sha256")
        raw_id = json.loads(Path(args.identity_json).read_text())
        identity = validate_teacher_identity(raw_id, allow_historical_bridge=True)

    # 3. Load actual data if manifest provided
    data = None
    student_lids = args.student_labeled_ids
    budget = args.label_budget
    if args.student_manifest:
        sm_path = Path(args.student_manifest).resolve()
        if not args.student_manifest_sha256 or digest(sm_path) != args.student_manifest_sha256:
            parser.error("student-manifest requires matching --student-manifest-sha256")
        sm = json.loads(sm_path.read_text())
        data = CompactTrainingData(
            manifest=sm_path,
            expected_sha256=args.student_manifest_sha256,
            synthetic=bool(sm.get("synthetic", False)),
            allow_k1=True,
        )
        if student_lids is None:
            student_lids = list(data.lids)
        if budget is None:
            budget = sm.get("label_budget", len(student_lids))

    teacher_lids = args.teacher_labeled_ids
    if teacher_lids is None and identity is not None:
        teacher_lids = identity.get("labeled_ids", [])

    if teacher_lids is None:
        parser.error("must provide --teacher-labeled-ids, --identity-json, or --use-historical-baseline")
    if student_lids is None and data is None:
        parser.error("must provide --student-labeled-ids or --student-manifest")
    if budget is None and data is None:
        parser.error("must provide --label-budget or --student-manifest")

    receipt = preflight_teacher_lineage(
        teacher_labeled_ids=teacher_lids,
        student_labeled_ids=student_lids,
        data=data,
        warmstart_labeled_ids=args.warmstart_labeled_ids,
        audit_labeled_ids=args.audit_labeled_ids,
        auxiliary_labeled_ids=args.auxiliary_labeled_ids,
        label_budget=budget,
        declared_labeled_ids=args.declared_labeled_ids,
        teacher_kind=identity.get("teacher_kind") if identity else None,
    )

    if identity is not None:
        receipt["teacher_identity_verified"] = identity.get("verified", False)
        receipt["teacher_kind"] = identity["teacher_kind"]
        receipt["inference_state"] = identity["inference_state"]

    if historical_receipt is not None:
        receipt["historical_source_bridge"] = historical_receipt

    receipt["scope"] = "metadata_label_union_only"
    receipt["physical_teacher_status"] = "NOT_VERIFIED; cache builder verifies checkpoint and payload"
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
